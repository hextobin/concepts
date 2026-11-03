# concepts

**Does a small story model write better arcs if it plans each clause with a
discrete "concept" code?** At this scale and with this setup, no. The planning
codes lowered word-level loss, but not by enough to pay for themselves. They
also didn't improve a next-event test beyond seed noise. The setup likely let
the codes carry only "free" information, so this is a weak negative result,
not a definitive one. Details below.

## Setup

- **Model:** 30M-parameter GPT (8 layers, d=512, context 512) on TinyStories
  (2.7M stories, 536M tokens), with an 8k BPE tokenizer. Each run is 6000 steps
  (about 197M tokens).
- **Segments:** `prep.py` marks clause and paragraph boundaries using only the
  current and previous token, so the same rule works during generation.
- **Plan mode:** at each boundary, a clause encoder reads the *upcoming* segment
  and gives a distribution q over K codes. A sampled code's vector is added to
  the residual stream (at the middle layer) while that segment is written. A
  code predictor gives its own distribution p from the story so far, which is
  what generation uses. The code costs KL(q‖p), so the honest score is
  **fair loss = token loss + KL**, an upper bound on the true loss (as in a VAE).
- **Free bits = 2:** during training, the first 2 bits per segment of KL cost
  nothing, so that the codes start being used at all. The fair loss still
  charges for them.
- **Controls:**
  - `baseline`: no codes.
  - `summary`: segment g's code is only shown while segment g+1 is written,
    so it carries nothing the model hasn't already seen.
  - `random`: boundaries at random positions, at the same rate as clauses.
  - `paragraph`: one code per paragraph instead of per clause.
- **Arc test (`eval_arc.py`):** cut a validation story after a sentence, then
  rank the true next sentence against three distractors:
  - a later sentence from the same story
  - an earlier sentence (repeating instead of moving on)
  - a sentence from another story

  Scoring uses PMI, log p(s | prefix) − log p(s), which removes the bias toward
  short, generic sentences. Plan models never see the candidate when choosing
  codes; they mix their predictor's top-8 codes. The test has 1000 items, the
  same items for every run.
- **Grid:** 7 variants × 2 seeds on one H100. It took 50 minutes and cost
  about $5.

## Results (mean of 2 seeds)

| variant | fair loss | token loss | code bits / segment | codes used | arc acc (4-way) | beats "later" |
|---|---|---|---|---|---|---|
| baseline | **1.222** | 1.222 | – | – | 0.635 | 0.725 |
| plan, clause, K=64 | 1.252 | **1.109** | 1.99 | 44 / 64 | 0.649 | 0.730 |
| plan, clause, K=256 | 1.255 | 1.111 | 1.99 | 115 / 256 | 0.645 | 0.722 |
| plan, clause, K=1024 | 1.257 | 1.114 | 1.99 | 269 / 1024 | 0.649 | 0.725 |
| plan, paragraph, K=256 | 1.231 | 1.199 | 1.99 | 134 / 256 | 0.634 | 0.721 |
| plan, random, K=256 | 1.296 | 1.149 | 1.99 | 81 / 256 | 0.629 | 0.718 |
| summary, clause, K=256 | **1.221** | 1.221 | 0 | 183 / 256 | 0.637 | 0.728 |

Losses are in nats per token. Per-seed numbers, training curves and all 14
checkpoints are on the [`results`](../../tree/results) branch.

## What it shows

1. **The codes don't pay for themselves.** Clause plans cut token loss by about
   0.11 nats/token, but the codes cost about 0.14, so the fair loss is worse
   than the baseline. Summary codes, which add no new information, match the
   baseline exactly. That's a useful sanity check that the plumbing is fair.
2. **No reliable arc gain.** Plan models score about +0.01 on 4-way accuracy.
   One run's standard error is about 0.015, and the two seeds of the same
   variant differ by up to 0.022. The unweighted accuracy and the "beats
   later" rate show no gap at all.
3. **The code rate is pinned at the free-bits floor.** Every plan variant sends
   1.99 bits per segment, whatever K is (64, 256 or 1024). Below 2 bits the
   code is free in training, so the encoder uses exactly the free allowance:
   about 2 bits about the upcoming clause, which helps token loss. Above 2
   bits, the predictor can't anticipate the code, so going further costs more
   than it saves. In other words, the training setup bought the token-loss
   gain with information the predictor never learned to supply. The test of
   whether *predictable* plans help was never really run.
4. Clause-level codes help token loss much more than paragraph-level ones.
   Random boundaries are the worst, as expected: the segmentation matters.

## What I'd try next

- **free_bits = 0 (and 0.5, 1):** make the codes earn their keep from step one.
  This is the real test, and it is likely to collapse the codes, which would
  itself be an answer.
- **A paired significance test:** save per-item results over the shared 1000
  items (McNemar or bootstrap), and use more items. The +0.01 is probably noise.
- **A larger model and longer training**, where planning has more room to matter.
- **Look at what the codes mean, then try steering:** swap a code mid-story and
  see whether the story changes.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy tokenizers
.venv/bin/python prep.py --train-truncated     # tokenizer + data into data/
.venv/bin/python train.py --name base --mode baseline
.venv/bin/python train.py --name plan-clause-256 --mode plan --level clause --codes 256 --free-bits 2
.venv/bin/python eval_arc.py runs/plan-clause-256 runs/base
```

`box/run_grid.sh` has the exact grid. `test_causal.py` is a leak test: it checks
that the baseline, the summary codes and the predictor's codes never see the
future, and that the encoder's codes do (that is the leak the KL pays for).

## How it ran

`launch.sh` creates a GPU droplet on DigitalOcean. The box then:

- runs the grid unattended
- pushes logs, metrics and checkpoints to the `results` branch every 15 minutes
- deletes itself when the grid finishes, at a hard deadline, or when a
  `control/stop` file is pushed to `main`

Nothing needs to stay connected. A generalized version of this setup lives in a
separate template repo.
