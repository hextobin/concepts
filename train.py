"""Train one variant and log validation metrics to runs/<name>/log.jsonl.

Examples:
  python train.py --name base --mode baseline
  python train.py --name plan-clause-256 --mode plan --level clause --codes 256
  python train.py --name smoke --preset tiny --steps 300 --mode plan
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, Config

LEVEL_BITS = {"clause": 1, "paragraph": 4}
PRESETS = {
    "tiny": dict(n_layer=4, n_head=4, d_model=128, block_size=256),
    "small": dict(n_layer=8, n_head=8, d_model=512, block_size=512),
}


class Data:
    def __init__(self, data_dir, split, block_size, eot_id):
        self.tokens = np.memmap(os.path.join(data_dir, f"{split}_tokens.bin"), np.uint16, mode="r")
        self.flags = np.memmap(os.path.join(data_dir, f"{split}_flags.bin"), np.uint8, mode="r")
        starts = np.flatnonzero(self.tokens == eot_id)  # windows open on a story boundary
        self.starts = starts[starts + block_size + 1 <= len(self.tokens)]
        self.T = block_size

    def batch(self, B, rng, level, random_rate, device):
        picks = self.starts[rng.integers(len(self.starts), size=B)]
        x = np.stack([self.tokens[p : p + self.T] for p in picks]).astype(np.int64)
        y = np.stack([self.tokens[p + 1 : p + self.T + 1] for p in picks]).astype(np.int64)
        if level == "random":
            bnd = rng.random((B, self.T)) < random_rate
        else:
            f = np.stack([self.flags[p : p + self.T] for p in picks])
            bnd = (f & LEVEL_BITS[level]) > 0
        bnd[:, 0] = True
        to = lambda a: torch.from_numpy(a).to(device)
        return to(x), to(y), to(bnd)


@torch.no_grad()
def evaluate(model, data, args, device, n_batches):
    model.eval()
    rng = np.random.default_rng(1234)  # same eval windows for every run
    sums, codes = {}, []
    for _ in range(n_batches):
        x, y, bnd = data.batch(args.batch, rng, args.level, args.random_rate, device)
        with autocast(device):
            _, aux, _ = model(x, y, bnd)
        for k in ("token_loss", "code_loss", "fair_loss", "vq"):
            if k in aux:
                sums[k] = sums.get(k, 0.0) + float(aux[k])
        if "codes" in aux:
            codes.append(aux["codes"].cpu())
    out = {k: round(v / n_batches, 4) for k, v in sums.items()}
    if codes:
        counts = torch.bincount(torch.cat(codes), minlength=model.c.n_codes).float()
        p = counts / counts.sum()
        out["codes_used"] = int((counts > 0).sum())
        out["code_perplexity"] = round(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())), 1)
    model.train()
    return out


def autocast(device):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="baseline", choices=["baseline", "plan", "summary"])
    ap.add_argument("--level", default="clause", choices=["clause", "paragraph", "random"])
    ap.add_argument("--codes", type=int, default=256)
    ap.add_argument("--preset", default="small", choices=PRESETS)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--restart-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    # Random boundaries get the same average spacing as real clauses.
    args.random_rate = meta["train"]["clause"]["count"] / meta["train"]["tokens"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    c = Config(vocab_size=meta["vocab_size"], mode=args.mode, n_codes=args.codes, **PRESETS[args.preset])
    model = GPT(c).to(device)
    train = Data(args.data_dir, "train", c.block_size, meta["eot_id"])
    valid = Data(args.data_dir, "valid", c.block_size, meta["eot_id"])

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    step_model = torch.compile(model) if args.compile else model

    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.jsonl"), "a")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {n_params / 1e6:.1f}M params on {device}")
    json.dump({**vars(args), "params": n_params, "config": c.__dict__}, open(os.path.join(out, "args.json"), "w"), indent=2)

    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            row = {"step": step, "tokens_seen": step * args.batch * c.block_size,
                   "minutes": round((time.time() - t0) / 60, 2),
                   **evaluate(model, valid, args, device, args.eval_batches)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        x, y, bnd = train.batch(args.batch, rng, args.level, args.random_rate, device)
        with autocast(device):
            _, aux, loss = step_model(x, y, bnd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if model.roadmap is not None and (step + 1) % args.restart_every == 0:
            model.roadmap.restart_dead_codes()

    torch.save({"model": model.state_dict(), "config": c.__dict__, "args": vars(args)},
               os.path.join(out, "model.pt"))


if __name__ == "__main__":
    main()
