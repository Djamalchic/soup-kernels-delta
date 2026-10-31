"""C0 / v8 C1: Triton compile probe.

How many kernel binaries (*.cubin) and gcc-built launchers (__triton_launcher*.so) an empty Triton
cache gets from one round of calls of every kernel the training path launches, and that a second
round with other sizes adds none (no recompile, no new launcher).

Each configuration runs in a fresh Python process with its own empty TRITON_CACHE_DIR, so the
in-memory JIT caches of earlier tests cannot hide a compile. CUDA only (Kaggle T4):

    python -m pytest solution/tests/test_compile_probe.py -q -s     # -s prints the timings

Expected (solution/kernels/__init__.py):
- USE_UNIFIED_SIG + USE_SOUP_SWIGLU (training default): 5 cubins (addnorm fwd / bwd, head_ce, nf4,
  swiglu) and exactly 1 launcher.
- both off (v7): 6 cubins and 6 launchers, as in the v7 Kaggle run's triton_cache.
- C1 only: 6 cubins, 3 launchers (ours + the example's two signatures).
The second round uses training-like sizes (multiples of 16 where a slot stays specialised: head_ce
n_cols, the SwiGLU element count); the first-call times are printed, not asserted.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]

PROBE = r'''
import glob, json, os, sys, time, types
import torch

repo, unified, soup = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1"
sys.path.insert(0, repo)
from solution import swiglu as example
from solution.kernels import addnorm, head_ce, nf4
from solution.kernels import swiglu as ours

for mod in (addnorm, head_ce, nf4):
    mod.USE_UNIFIED_SIG = unified
cache = os.environ["TRITON_CACHE_DIR"]
dev = "cuda"


def files():
    def count(pattern):
        return len(glob.glob(os.path.join(cache, "**", pattern), recursive=True))
    return {"cubin": count("*.cubin"), "launcher": count("__triton_launcher*.so"),
            "cuda_utils": count("cuda_utils*.so")}


def nf4_state(n_out, k_in):
    count = n_out * k_in
    blocks = count // 64
    gen = torch.Generator().manual_seed(count)
    packed = torch.randint(0, 256, (count // 2,), dtype=torch.uint8, generator=gen).to(dev)
    absmax = torch.randint(0, 256, (blocks,), dtype=torch.uint8, generator=gen).to(dev)
    state2 = types.SimpleNamespace(absmax=torch.rand((blocks + 255) // 256, generator=gen).to(dev),
                                   code=torch.randn(256, generator=gen).to(dev))
    state = types.SimpleNamespace(absmax=absmax, state2=state2,
                                  offset=torch.rand(1, generator=gen).to(dev))
    return packed, state, count


def one_round(rows, vocab, hidden, tokens, nk):
    took = {}

    def timed(name, fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        took[name] = round(time.perf_counter() - start, 3)

    width = addnorm.HIDDEN_SIZE
    x = torch.randn(rows, width, device=dev)
    d16 = torch.randn(rows, width, device=dev, dtype=torch.float16)
    w = torch.rand(width, device=dev) + 0.5
    timed("addnorm_fwd", lambda: (addnorm._launch_fwd(x, d16, w, 1e-6),
                                  addnorm._launch_fwd(x, None, w, 1e-6),
                                  addnorm._launch_fwd(x, d16, w, 1e-6, store_r=False)))
    _, _, rstd = addnorm._launch_fwd(x, None, w, 1e-6)
    dy = torch.randn_like(x)
    timed("addnorm_bwd", lambda: (addnorm._launch_bwd(dy, dy, x, w, rstd),
                                  addnorm._launch_bwd(dy, None, x, w, rstd, delta=d16)))

    hid = torch.randn(2, tokens, hidden, device=dev, requires_grad=True)
    weight = torch.randn(vocab, hidden, device=dev) * 0.05
    labels = torch.randint(0, vocab, (2, tokens), device=dev)
    labels[:, :3] = -100
    mask = torch.ones(2, tokens, device=dev, dtype=torch.long)
    mask[1, -5:] = 0

    def head():
        head_ce.SoupHeadCE.apply(hid, weight, labels, mask, None)[0].backward()
    timed("head_ce", head)

    packed, state, count = nf4_state(*nk)
    dense = torch.empty(nk, device=dev, dtype=torch.float16)
    timed("nf4", lambda: nf4._launch(packed, state, count, dense))

    g = torch.randn(tokens, 8960, device=dev, dtype=torch.float16, requires_grad=True)
    u = torch.randn(tokens, 8960, device=dev, dtype=torch.float16, requires_grad=True)
    dh = torch.randn(tokens, 8960, device=dev, dtype=torch.float16)
    act = ours.swiglu if soup else example.swiglu
    timed("swiglu", lambda: act(g, u).backward(dh))
    g2, u2, dh2 = g.detach().clone(), u.detach().clone(), dh.clone()
    if soup:
        timed("swiglu_inplace", lambda: ours.swiglu_backward_into(dh2, g2, u2))
    else:
        kernel, block, n = example.example_swiglu_bwd_kernel, example.BLOCK, g2.numel()
        timed("swiglu_inplace",
              lambda: kernel[((n + block - 1) // block,)](dh2, g2, u2, dh2, g2, n, BLOCK=block))
    return took


first = one_round(rows=77, vocab=2048, hidden=64, tokens=37, nk=(256, 1536))
after_first = files()
second = one_round(rows=300, vocab=4096, hidden=128, tokens=129, nk=(1536, 1536))
after_second = files()
print("PROBE " + json.dumps({"first": first, "second": second, "after_first": after_first,
                             "after_second": after_second}))
'''

# (USE_UNIFIED_SIG, USE_SOUP_SWIGLU) -> (cubins, launchers) after the first round
EXPECTED = {(1, 1): (5, 1), (0, 0): (6, 6), (1, 0): (6, 3)}


def _probe(tmp_path, unified, soup):
    cache = tmp_path / f"triton_cache_{unified}{soup}"
    cache.mkdir()
    env = dict(os.environ, TRITON_CACHE_DIR=str(cache))
    proc = subprocess.run([sys.executable, "-c", PROBE, str(REPO), str(unified), str(soup)],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=900,
                          check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [s for s in proc.stdout.splitlines() if s.startswith("PROBE ")][-1]
    return json.loads(line[len("PROBE "):])


@pytest.mark.parametrize("unified, soup", sorted(EXPECTED, reverse=True))
def test_launchers_and_no_recompile(tmp_path, unified, soup):
    result = _probe(tmp_path, unified, soup)
    first, second = result["after_first"], result["after_second"]
    total = sum(result["first"].values())
    print(f"\nC1={unified} C2={soup}: {first['cubin']} cubin, {first['launcher']} launcher, "
          f"{first['cuda_utils']} cuda_utils; first round {total:.2f} s {result['first']}; "
          f"second round {sum(result['second'].values()):.2f} s")
    cubins, launchers = EXPECTED[(unified, soup)]
    assert first["cubin"] == cubins
    assert first["launcher"] == launchers
    assert first["cuda_utils"] <= 1
    # Later sizes compile nothing and build no launcher (no hidden compile after step 1).
    assert second == first
