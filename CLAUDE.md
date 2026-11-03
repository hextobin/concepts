# concepts: handoff

Read this first. See `README.md` for the idea, and `box/` for how the GPU run worked.

## Status (Nov 3, 2026)

The 14-run grid finished on Nov 2. It ran on a DigitalOcean H100, took 50 min,
and cost about $5. The box destroyed itself. Everything is on the `results`
branch:

- `runs/<name>/{args.json,log.jsonl,arc.json}`: config, training curve, next-event eval
- `checkpoints/<name>.pt.gz`: all 14 models (bf16, gzipped, about 55MB each)
- `data/{meta.json,tokenizer.json}`, `logs/grid.log`

To fetch the results without the checkpoints:
`git fetch --depth 1 --filter=blob:limit=2m origin results`.
Get single checkpoints with `git show origin/results:checkpoints/<name>.pt.gz`.

Each run is 6000 steps (about 197M tokens) with the small preset (30M params).
Each variant has seeds 0 and 1. The arc eval is 1000 items, seed 0, so every
run is scored on the same items.

## Results (means over 2 seeds)

| variant | fair_loss | token_loss | code bits/seg | codes used | acc_4way | beats_later |
|---|---|---|---|---|---|---|
| base | 1.222 | 1.222 | - | - | 0.635 | 0.725 |
| plan-clause-64 | 1.252 | 1.109 | 1.99 | 44/64 | 0.649 | 0.730 |
| plan-clause-256 | 1.255 | 1.111 | 1.99 | 115/256 | 0.645 | 0.722 |
| plan-clause-1024 | 1.257 | 1.114 | 1.99 | 269/1024 | 0.649 | 0.725 |
| plan-paragraph-256 | 1.231 | 1.199 | 1.99 | 134/256 | 0.634 | 0.721 |
| plan-random-256 | 1.296 | 1.149 | 1.99 | 81/256 | 0.629 | 0.718 |
| summary-clause-256 | 1.221 | 1.221 | 0 | 183/256 | 0.637 | 0.728 |

Per-seed numbers are in `runs/*/`. `fair_loss = token_loss + code_loss`.

## What it says so far

- **No reliable arc gain.** Plan-clause beats base by about +0.01 on acc_4way.
  One run's standard error is about 0.015, and seeds differ by up to 0.022
  (plan-clause-256: 0.656 vs 0.634). acc_4way_raw and beats_later show no gap.
- **The codes help token prediction but cost more than they save.**
  Plan-clause lowers token_loss by 0.11 nats/token, but the code costs about
  0.14, so fair_loss is worse than base (1.255 vs 1.222).
- **The code rate sits exactly at the free-bits floor.** Every plan variant
  sends about 1.99 bits/segment with free_bits=2. The KL is probably pinned
  there: the model uses the free allowance and nothing more, whatever K is.
  This is the main thing to understand before reading meaning into the codes.
- Clause-level codes beat paragraph-level ones on token_loss. Random boundaries
  (the control) are worst, as expected.
- Summary codes aren't predicted (0 bits), so their fair_loss is just
  token_loss. They match base.

## Next phases (agreed order)

1. **Analysis: is the arc difference real?**
   - Change `eval_arc.py` to save per-item correctness. Then do a paired test
     (McNemar or bootstrap) over the 1000 shared items, pooling seeds and the
     three plan-clause sizes against base.
   - Re-run the eval with more items (e.g. 5000) and with `--plans 1` vs `8`.
   - The checkpoints are small, so this can probably run on CPU. It needs
     `prep.py` run on the validation file only, using the tokenizer from the
     results branch.
2. **What do the codes mean?**
   - For plan-clause-64: collect the clauses assigned to each code. Look at the
     top tokens and examples per code, and code usage over story position.
   - Check whether the predictor's code matches the encoder's (agreement rate).
3. **Later, maybe: steering.** Swap the code at one boundary during generation
   and see whether the story changes. Only worth it if phase 2 finds meaning.

Open question for phase 1 or 2: rerun with free_bits 0 or 1 (and maybe 4) to see
whether the codes carry information beyond the free allowance. That needs a GPU
run, using the template below.

## Working rules

- New GPU runs: use hextobin/gpu-experiment-template (`launch.sh`, `box/`).
  This repo's `box/` scripts are the older version it was generalized from.
- Keep each phase in its own session. Update this file before ending one.
