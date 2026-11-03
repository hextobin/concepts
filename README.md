# concepts

**Question:** does a small story model predict story arcs better if it plans
each clause with a discrete "concept" code?

**Answer (at this scale):** no. The codes lowered word-level loss, but not by
enough to pay for themselves, and they didn't help a next-event test beyond
seed noise. The setup likely let the codes carry only "free" information, so
this is a weak negative result.

## Setup

- **Model:** a 30M-parameter GPT trained on TinyStories for 197M tokens.
- **Plan codes:** at each clause boundary, an encoder reads the upcoming clause
  and picks one of K codes, which is added to the model while that clause is
  written. A predictor learns to guess the code from the story so far; it is
  what the model would use when generating. A code costs the KL between the
  encoder's and the predictor's choices, so the honest score is
  **fair loss = word loss + code cost**.
- **Free bits:** in training, the first 2 bits of code cost per clause were
  free, to get the codes used at all. The fair loss still charges for them.
- **Controls:**
  - no codes (baseline)
  - codes that only describe the past (summary)
  - codes at random positions
  - one code per paragraph
- **Arc test:** cut a story after a sentence. Then rank the true next sentence
  against a later sentence from the same story, an earlier one, and one from
  another story. There are 1000 items, the same for every model.
- 7 variants × 2 seeds, on one H100, in 50 minutes, for about $5.

## Results (mean of 2 seeds)

| variant | fair loss | word loss | code bits per clause | arc accuracy |
|---|---|---|---|---|
| baseline | **1.222** | 1.222 | – | 0.635 |
| plan, K=64 | 1.252 | **1.109** | 1.99 | 0.649 |
| plan, K=256 | 1.255 | 1.111 | 1.99 | 0.645 |
| plan, K=1024 | 1.257 | 1.114 | 1.99 | 0.649 |
| plan per paragraph | 1.231 | 1.199 | 1.99 | 0.634 |
| plan at random positions | 1.296 | 1.149 | 1.99 | 0.629 |
| summary codes | 1.221 | 1.221 | 0 | 0.637 |

Losses are in nats per token. Arc accuracy is 4-way, so chance is 0.25.

## Findings

1. **The codes don't pay for themselves.** They cut word loss by 0.11, but cost
   0.14, so fair loss is worse than the baseline.
2. **No reliable arc gain.** The +0.01 is within noise: one run's standard
   error is about 0.015, and two seeds of the same model differ by up to 0.022.
3. **The code rate sits exactly at the free allowance.** Every plan model uses
   1.99 bits per clause, whatever K is. The codes carry about 2 free bits about
   the upcoming clause, which helps word loss. Beyond that, the predictor can't
   anticipate them, so they stop paying. The real question, whether
   *predictable* plans help, was never tested.
4. Clause boundaries matter: per-paragraph and random-position codes help less.

## Next, if revisited

- Retrain with free bits at 0 to 1, so the codes must earn their keep.
- A paired significance test over the shared items, with more items.
- A larger model, then inspecting and swapping codes to steer stories.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy tokenizers
.venv/bin/python prep.py --train-truncated
.venv/bin/python train.py --name base --mode baseline
.venv/bin/python train.py --name plan --mode plan --level clause --codes 256 --free-bits 2
.venv/bin/python eval_arc.py runs/plan runs/base
```

The full grid is in `box/run_grid.sh`. Per-run metrics and all checkpoints are
on the `results` branch.
