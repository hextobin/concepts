"""Leak test: changing tokens after position t must not change outputs at <= t.

The plan mode with encoder codes is *expected* to leak (that is what the code
loss pays for); the test checks that the leak is real, so we know it can detect one.
"""

import torch

from model import GPT, Config


def outputs(model, x, bnd, code_source):
    with torch.no_grad():
        return model(x, bnd=bnd, code_source=code_source)[0]


def leaks(mode, code_source, T=64, t=30, seed=0):
    torch.manual_seed(seed)
    c = Config(vocab_size=100, block_size=T, n_layer=4, n_head=2, d_model=32, mode=mode, n_codes=16, d_code=8)
    model = GPT(c).eval()
    if model.roadmap is not None:
        torch.nn.init.normal_(model.roadmap.up.weight, std=0.5)  # zero init would hide any leak
    x = torch.randint(100, (2, T))
    bnd = torch.rand(2, T) < 0.15
    bnd[:, 0] = True
    x2, bnd2 = x.clone(), bnd.clone()
    x2[:, t + 1:] = torch.randint(100, (2, T - t - 1))
    bnd2[:, t + 1:] = torch.rand(2, T - t - 1) < 0.15
    a, b = outputs(model, x, bnd, code_source), outputs(model, x2, bnd2, code_source)
    return not torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-5)


if __name__ == "__main__":
    cases = [
        ("baseline", "encoder", False),
        ("summary", "encoder", False),
        ("plan", "prior", False),
        ("plan", "encoder", True),
    ]
    ok = True
    for mode, source, expected in cases:
        got = any(leaks(mode, source, seed=s) for s in range(5))
        status = "ok" if got == expected else "FAIL"
        ok &= got == expected
        print(f"{status:4}  {mode:8} codes from {source:7}  leaks future: {got}")
    raise SystemExit(0 if ok else 1)
