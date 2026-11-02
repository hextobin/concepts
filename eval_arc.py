"""Next-event ranking: does the model know what happens next?

For each item, a validation story is cut after a sentence. The model scores the
true next sentence against three distractors:
  later    a sentence from later in the same story (right story, wrong order)
  earlier  a sentence already told in the prefix (repeats instead of moving on)
  other    a sentence from a different story (wrong story)

Each candidate gets a PMI score: log p(candidate | prefix) - log p(candidate | no prefix),
i.e. how much the story so far raises its likelihood. This removes the bias toward
short, generic sentences. Plan models write with the codes their predictor picks
(no peeking at the candidate).

  python eval_arc.py runs/plan-clause-256 runs/base
"""

import argparse
import dataclasses
import json
import os

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, Config
from prep import SENTENCE, boundary_tables, mark_boundaries
from train import LEVEL_BITS

KINDS = ("later", "earlier", "other")


def load_stories(data_dir, tok):
    tokens = np.fromfile(os.path.join(data_dir, "valid_tokens.bin"), np.uint16)
    flags = np.fromfile(os.path.join(data_dir, "valid_flags.bin"), np.uint8)
    eot = tok.token_to_id("<|endoftext|>")
    stories, sentences, start = [], [], 0
    for i, (t, f) in enumerate(zip(tokens, flags)):
        if t == eot:
            if sentences:
                stories.append(sentences)
            sentences, start = [], i + 1
        elif f & SENTENCE:
            text = tok.decode(tokens[start : i + 1].tolist()).strip()
            if text:
                sentences.append(text)
            start = i + 1
    return stories


def make_items(stories, n, seed):
    rng = np.random.default_rng(seed)
    pool = [s for s in stories if len(s) >= 6]
    items = []
    for k in rng.permutation(len(pool))[:n]:
        s = pool[k]
        cut = int(rng.integers(2, len(s) - 2))  # leaves room for a later sentence
        other = pool[(k + 1 + int(rng.integers(len(pool) - 1))) % len(pool)]
        items.append({
            "prefix": " ".join(s[: cut + 1]),
            "true": s[cut + 1],
            "later": s[int(rng.integers(cut + 2, len(s)))],
            "earlier": s[int(rng.integers(0, cut + 1))],
            "other": other[int(rng.integers(1, len(other)))],
        })
    return items


class Scorer:
    def __init__(self, run_dir, data_dir, device):
        ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
        known = {f.name for f in dataclasses.fields(Config)}  # older runs saved since-removed fields
        self.model = GPT(Config(**{k: v for k, v in ckpt["config"].items() if k in known})).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.args = ckpt["args"]
        self.tok = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
        self.tables = boundary_tables(self.tok)
        self.eot = self.tok.token_to_id("<|endoftext|>")
        self.device = device

    def logprob(self, contexts, candidates, batch=32):
        """Summed log-prob of each candidate after its context (both lists of text)."""
        T = self.model.c.block_size
        out = []
        for b in range(0, len(contexts), batch):
            rows, spans = [], []
            for ctx, cand in zip(contexts[b : b + batch], candidates[b : b + batch]):
                c_ids = self.tok.encode(" " + cand).ids
                p_ids = self.tok.encode(ctx).ids if ctx else []
                p_ids = p_ids[-(T - len(c_ids)) :] if len(p_ids) + len(c_ids) >= T else p_ids
                rows.append([self.eot] + p_ids + c_ids)
                spans.append(len(c_ids))
            L = max(map(len, rows))
            seq = np.full((len(rows), L), self.eot, np.int64)
            for i, r in enumerate(rows):
                seq[i, : len(r)] = r
            bnd = np.stack([self.boundaries(r) for r in seq[:, :-1]])
            x, y = torch.from_numpy(seq[:, :-1]), torch.from_numpy(seq[:, 1:])
            with torch.no_grad():
                logits, _, _ = self.model(x.to(self.device), bnd=torch.from_numpy(bnd).to(self.device),
                                          code_source="prior")
            lp = logits.float().log_softmax(-1).gather(-1, y.to(self.device)[..., None])[..., 0].cpu()
            for i, (r, n) in enumerate(zip(rows, spans)):
                out.append(float(lp[i, len(r) - 1 - n : len(r) - 1].sum()))
        return np.array(out)

    def boundaries(self, ids):
        level = self.args.get("level", "clause")
        if level == "random":
            bnd = np.random.default_rng(0).random(len(ids)) < self.args["random_rate"]
        else:
            bnd = (mark_boundaries(ids, self.tables, self.eot) & LEVEL_BITS[level]) > 0
        bnd[0] = True
        return bnd


def evaluate(scorer, items):
    names = ("true",) + KINDS
    ctx = [it["prefix"] for it in items for _ in names]
    cand = [it[k] for it in items for k in names]
    cond = scorer.logprob(ctx, cand).reshape(len(items), len(names))
    uncond = scorer.logprob([""] * len(cand), cand).reshape(len(items), len(names))
    pmi = cond - uncond
    res = {"items": len(items),
           "acc_4way": round(float((pmi.argmax(1) == 0).mean()), 3),
           "acc_4way_raw": round(float((cond.argmax(1) == 0).mean()), 3)}
    for j, k in enumerate(KINDS, start=1):
        res[f"beats_{k}"] = round(float((pmi[:, 0] > pmi[:, j]).mean()), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--items", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    items = make_items(load_stories(args.data_dir, tok), args.items, args.seed)
    for run in args.runs:
        res = evaluate(Scorer(run, args.data_dir, device), items)
        print(os.path.basename(run.rstrip("/")), json.dumps(res), flush=True)
        json.dump(res, open(os.path.join(run, "arc.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
