# concepts

A small experiment: does a story model write better arcs if it plans each clause
with a discrete "concept" code?

A ~30M-parameter transformer is trained on TinyStories. At each clause boundary
it picks one of K learned codes from a codebook C and adds that vector to the
residual stream until the next boundary. During training a clause encoder picks
the code (snapped to the codebook with vector quantization); a code predictor
learns to guess it from the story so far and picks it when generating.

Main questions:
1. Does it predict what happens next better? (next-event ranking, LLM-judged sample)
2. Do the codes mean something, and can swapping one steer the story?

## Data prep

```bash
python3 -m venv .venv
.venv/bin/pip install numpy tokenizers
.venv/bin/python prep.py --train-truncated
```

`prep.py` trains an 8k byte-level BPE tokenizer, tokenizes TinyStories, and
marks clause boundaries using only the current and previous token, so the same
rule works while generating. Outputs go to `data/` (not tracked in git).
