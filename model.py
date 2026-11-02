"""Small GPT with an optional concept roadmap.

Segments (clauses or paragraphs) are marked by a boolean `bnd` that is True on
the last token of each segment; position 0 must be a boundary (training windows
start on <|endoftext|>). Code g applies to the positions that *predict* segment
g's tokens: from the boundary that opens it up to the token before the next one.

Modes:
  baseline  plain transformer
  plan      the clause encoder reads segment g and gives a distribution q over
            codes; a sampled code is added while segment g is written. The code
            predictor gives its own distribution p from the story so far. The
            KL(q || p) is the cost of the code and is part of the fair score
            (token loss + KL is an upper bound on the true loss, like a VAE).
  summary   segment g's code is only added while segment g+1 is written, so it
            carries no information the model has not already seen.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    mode: str = "baseline"
    n_codes: int = 256
    d_code: int = 64

    @property
    def inject_layer(self):
        return self.n_layer // 2


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(c.d_model, 4 * c.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * c.d_model, c.d_model, bias=False),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


def segment_ids(bnd):
    """Global id (across the batch) of the code that applies at each position."""
    counts = bnd.sum(1)
    offset = torch.cumsum(counts, 0) - counts
    return bnd.long().cumsum(1) - 1 + offset[:, None], int(counts.sum())


def mean_pool(h, ids, n):
    D = h.shape[-1]
    ids = ids.reshape(-1)
    total = h.new_zeros(n, D).index_add_(0, ids, h.reshape(-1, D))
    count = h.new_zeros(n).index_add_(0, ids, h.new_ones(ids.shape))
    return total / count.clamp(min=1)[:, None], count


class Roadmap(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.enc = nn.Sequential(nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.n_codes))
        self.codebook = nn.Parameter(torch.randn(c.n_codes, c.d_code))
        self.up = nn.Linear(c.d_code, c.d_model, bias=False)
        if c.mode == "plan":
            self.prior = nn.Sequential(nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.n_codes))
        else:
            self.null = nn.Parameter(torch.zeros(c.d_code))  # code for a window's first segment

    def forward(self, h, bnd, code_source):
        ids, n = segment_ids(bnd)
        # Segment g's tokens are predicted at positions with id g, so they sit one step later.
        pooled, count = mean_pool(h[:, 1:], ids[:, :-1], n)
        valid = count > 0
        log_q = self.enc(pooled).float().log_softmax(-1)
        if self.c.mode == "plan":
            # Sample a code; the gradient flows through the soft probabilities.
            pick = F.gumbel_softmax(log_q, tau=1.0, hard=True)
        else:
            soft = log_q.exp()
            pick = F.one_hot(soft.argmax(-1), self.c.n_codes).float() + soft - soft.detach()
        aux = {"codes": pick.argmax(-1)[valid], "n_segments": int(valid.sum())}

        if self.c.mode == "plan":
            log_p = self.prior(h[bnd]).float().log_softmax(-1)  # boundary h only sees the past
            kl = (log_q.exp() * (log_q - log_p)).sum(-1)
            aux["code_nll"] = kl[valid].sum()
            if code_source == "prior":
                pick = F.one_hot(log_p.argmax(-1), self.c.n_codes).float()
        z = pick.to(h.dtype) @ self.codebook.to(h.dtype)
        if self.c.mode == "summary":
            first_segment = bnd.long().cumsum(1) == 1
            z = torch.cat([z, self.null[None].to(h.dtype)])
            ids = torch.where(first_segment, n, ids - 1)
        return h + self.up(z)[ids], aux


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.block_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.roadmap = Roadmap(c) if c.mode != "baseline" else None
        self.apply(self._init)
        if self.roadmap is not None:
            nn.init.zeros_(self.roadmap.up.weight)  # starts as an exact copy of the baseline
            nn.init.normal_(self.roadmap.codebook)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, bnd=None, code_source="encoder", kl_weight=1.0, free_bits=0.0):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        aux = {}
        for i, block in enumerate(self.blocks):
            if i == self.c.inject_layer and self.roadmap is not None:
                x, aux = self.roadmap(x, bnd, code_source)
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, aux, None
        token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        code_loss = aux.get("code_nll", token_loss.new_zeros(())) / targets.numel()
        aux["token_loss"] = token_loss
        aux["code_loss"] = code_loss
        aux["fair_loss"] = token_loss + code_loss  # nats per token, comparable to baseline
        train_code_loss = code_loss
        if free_bits and aux.get("n_segments"):
            # Free bits: codes may carry this much information per segment at no charge in
            # training, so the model has a reason to start using them. The fair loss is unchanged.
            n = aux["n_segments"]
            per_segment = aux["code_nll"] / n
            floor = free_bits * math.log(2)
            train_code_loss = torch.clamp(per_segment, min=floor) * n / targets.numel()
        return logits, aux, token_loss + kl_weight * train_code_loss
