"""v9 M5 (USE_BWD_CHUNKING): row-chunked S4 / B backward and K2's 8192-row vocabulary tile.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
(add -s to see the measured peaks). The wiring and the chunk boundaries are tested without a GPU
in tests/test_v6_apply.py (test_m5_flags_reach_the_modules, test_m5_row_bounds).

What has to hold before the flag stays on:
- S4 "rows" gives "both"'s input gradient bit for bit (torch.equal), alone, through SoupMLPRecompute
  on [B, T, C] and through the fused block B. A failure here means cuBLAS ran the u GEMM of a
  chunk (M = chunk rows) with other bits than the full one (M = N): then either accept ulp-level
  gradient differences (checks 5/6 have tolerances) or set USE_BWD_CHUNKING = False.
- "rows" lowers the MLP backward peak by >= 90 % of W + U - max(U_c, dx) and runs 3 dequants.
- K2 with 8192-row tiles: the same forward numbers as 16384 (loss, entropy, token losses,
  correct, labelled) bit for bit if cuBLAS's logits GEMM is N-invariant; the hidden gradient sums
  19 fp16 tile partials instead of 10, so it is compared with K2's own tolerance; peak lower by
  >= 90 % of the two halved buffers.
"""

import gc

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

pytest.importorskip("bitsandbytes")
pytest.importorskip("triton")
pytest.importorskip("peft")
pytest.importorskip("trl")

from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.fused_block as fb  # noqa: E402
import solution.mlp as k4  # noqa: E402
import solution.mlp_recompute as rc  # noqa: E402
from solution.kernels import head_ce  # noqa: E402
from solution.tests import test_head_ce as th  # noqa: E402
from solution.tests import test_peak_v6 as tp  # noqa: E402
from solution.tests.test_mlp_recompute import (  # noqa: E402,F401  (fixtures reused)
    C,
    I,
    _count_dequants,
    _Lora,
    base_forward,
    real_mlp,
    set_k1,
    uses_s4,
)
from solution.tests.test_peak_v6 import s4  # noqa: E402,F401  (fixture reused)

MiB = 1 << 20


def _rows_on(monkeypatch, chunks, min_rows=0):
    monkeypatch.setattr(rc, "S4_DEQUANT_REUSE", "both")
    monkeypatch.setattr(rc, "BWD_ROW_CHUNKS", chunks)
    monkeypatch.setattr(rc, "BWD_CHUNK_MIN_ROWS", min_rows)
    monkeypatch.setattr(rc, "BWD_ROW_ALIGN", 128)


def _grad_helper(mlp, x16, dh):
    with torch.no_grad():  # the backward runs without grad
        return rc._gate_up_input_grad([dh.clone()], x16, mlp.gate_proj, mlp.up_proj)


# ---------------------------------------------------------------------------------------------
# S4 helper: "rows" against "both"
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [513, 600, 1350, 1351, 2048])
@pytest.mark.parametrize("chunks", [2, 4])
@pytest.mark.parametrize("soup", [True, False])  # v8 C2's kernel (the run) / the example's
def test_rows_helper_bit_exact_with_three_dequants(real_mlp, base_forward, monkeypatch, n, chunks,
                                                   soup):
    set_k1(monkeypatch, True)
    monkeypatch.setattr(k4, "USE_SOUP_SWIGLU", soup)
    torch.manual_seed(n)
    x16 = torch.randn(n, C, device="cuda", dtype=torch.float16)
    dh = torch.randn(n, I, device="cuda", dtype=torch.float16)

    _rows_on(monkeypatch, 1)
    with _count_dequants(monkeypatch) as calls:
        ref = _grad_helper(real_mlp, x16, dh)
    assert len(calls) == 2  # v7 I2 "both"

    _rows_on(monkeypatch, chunks)
    rows_before = rc.ROWS_COUNTS["rows"]
    both_before = rc.REUSE_COUNTS["both"]
    with _count_dequants(monkeypatch) as calls:
        got = _grad_helper(real_mlp, x16, dh)
    torch.cuda.synchronize()
    assert rc.ROWS_COUNTS["rows"] == rows_before + 1
    assert rc.REUSE_COUNTS["both"] == both_before + 1  # still the dequant-reuse path
    assert len(calls) == 3  # W_gate twice, W_up once
    assert got.dtype == torch.float16 and got.shape == ref.shape
    if not torch.equal(got, ref):
        diff = (got.float() - ref.float()).abs()
        pytest.fail(f"rows != both at n={n}, k={chunks}: {int((diff > 0).sum())} elements, "
                    f"max {float(diff.max()):.3e} (cuBLAS u GEMM not M-invariant)")


def test_rows_helper_lowers_the_peak(real_mlp, base_forward, monkeypatch):
    """At N = 1350 (the peak batch), k = 4: "both" holds W_gate, g, W_up, u (2W + 2U above dh and
    x16); "rows" holds g, W_up and one 384-row u chunk, or g, W_up and dx (U + W + max(U_c, dx))."""
    set_k1(monkeypatch, True)
    n = 1350
    torch.manual_seed(3)
    x16 = torch.randn(n, C, device="cuda", dtype=torch.float16)
    dh = torch.randn(n, I, device="cuda", dtype=torch.float16)
    peaks = {}
    for chunks in (1, 4, 1, 4):  # second round measured (the first warms cuBLAS and the caches)
        _rows_on(monkeypatch, chunks, min_rows=512)
        box = [dh.clone()]
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            dx = rc._gate_up_input_grad(box, x16, real_mlp.gate_proj, real_mlp.up_proj)
        torch.cuda.synchronize()
        peaks[chunks] = torch.cuda.max_memory_allocated() - base
        del dx, box
    u_bytes, w_bytes = n * I * 2, I * C * 2
    chunk_bytes, dx_bytes = 384 * I * 2, n * C * 2
    expect = w_bytes + u_bytes - max(chunk_bytes, dx_bytes)  # 44.8 MB
    print(f"\nS4 backward peak above entry: both {peaks[1] / MiB:.1f} MiB, "
          f"rows {peaks[4] / MiB:.1f} MiB, expected saving {expect / MiB:.1f} MiB")
    assert peaks[1] - peaks[4] >= 0.9 * expect, (peaks[1] / MiB, peaks[4] / MiB)


def test_rows_end_to_end_on_batch_shape(real_mlp, base_forward, monkeypatch):
    """SoupMLPRecompute on [B, T, C] (U's unpacked path): outputs, x and LoRA grads equal."""
    set_k1(monkeypatch, True)
    torch.manual_seed(5)
    lora = _Lora().cuda()
    x = torch.randn(2, 700, C, device="cuda")
    grad = torch.randn(2, 700, C, device="cuda")
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
    res = {}
    for chunks in (1, 4):
        _rows_on(monkeypatch, chunks, min_rows=512)
        lora.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        rows_before = rc.ROWS_COUNTS["rows"]
        with torch.autocast("cuda", dtype=torch.float16):
            out = real_mlp(lora(xi))
        assert uses_s4(out)
        out.backward(grad)
        assert rc.ROWS_COUNTS["rows"] - rows_before == (1 if chunks > 1 else 0)
        res[chunks] = (out.detach(), xi.grad, lora.a.weight.grad.clone(),
                       lora.b.weight.grad.clone())
    for new, ref in zip(res[4], res[1]):
        assert torch.equal(new, ref)


@pytest.mark.parametrize("rows", [1350, 2000])
@pytest.mark.parametrize("soup", [True, False])
def test_fused_block_rows_bit_exact(s4, monkeypatch, rows, soup):  # noqa: F811
    """B with "rows" against B with "both" and v5's ops with "off": same output, dr and da."""
    monkeypatch.setattr(fb, "USE_FUSED_BLOCK", True)
    monkeypatch.setattr(k4, "USE_SOUP_SWIGLU", soup)
    layer = tp._Layer()
    torch.manual_seed(11)
    r0 = 2.0 * torch.randn(rows, tp.H, device="cuda")
    a0 = torch.randn(rows, tp.H, device="cuda", dtype=torch.float16)
    g = torch.randn(rows, tp.H, device="cuda")
    out = {}
    for fused, mode, chunks in ((False, "off", 1), (True, "both", 1), (True, "both", 4)):
        monkeypatch.setattr(rc, "S4_DEQUANT_REUSE", mode)
        monkeypatch.setattr(rc, "BWD_ROW_CHUNKS", chunks)
        monkeypatch.setattr(rc, "BWD_CHUNK_MIN_ROWS", 512)
        r = r0.clone().requires_grad_(True)
        a = a0.clone().requires_grad_(True)
        rows_before = rc.ROWS_COUNTS["rows"]
        with torch.autocast("cuda", dtype=torch.float16):
            if fused:
                assert fb.fused_ok(layer, r, a, mq.Qwen2MLP, mq.Qwen2RMSNorm)
                nxt = fb.add_norm_mlp(r, a, layer)
            else:
                nxt = tp._v5_ops(layer, r, a)
        nxt.backward(g)
        assert rc.ROWS_COUNTS["rows"] - rows_before == (1 if chunks > 1 else 0)
        out[(fused, mode, chunks)] = (nxt.detach(), r.grad, a.grad)
    ref = out[(False, "off", 1)]
    for key, got in out.items():
        for new, old in zip(got, ref):
            assert torch.equal(new, old), key


def test_rows_skips_small_batches_and_saved_gate(real_mlp, base_forward, monkeypatch):
    set_k1(monkeypatch, True)
    _rows_on(monkeypatch, 4, min_rows=512)
    x16 = torch.randn(400, C, device="cuda", dtype=torch.float16)
    dh = torch.randn(400, I, device="cuda", dtype=torch.float16)
    before = rc.ROWS_COUNTS["rows"]
    _grad_helper(real_mlp, x16, dh)  # 400 < 512 rows: "both"
    assert rc.ROWS_COUNTS["rows"] == before
    g = torch.empty(400, I, device="cuda", dtype=torch.float16)
    monkeypatch.setattr(rc, "BWD_CHUNK_MIN_ROWS", 0)
    with torch.no_grad():  # RECOMPUTE_GATE False keeps g: "both" refuses, so does "rows"
        rc._gate_up_input_grad([dh.clone()], x16, real_mlp.gate_proj, real_mlp.up_proj, g)
    assert rc.ROWS_COUNTS["rows"] == before


# ---------------------------------------------------------------------------------------------
# K2: VOCAB_TILE 8192 against 16384 at the run's widths
# ---------------------------------------------------------------------------------------------

V2 = 3 * 16384 + 4480  # tiles: 16384 x 3 + 4480 (v8b) vs 8192 x 6 + 4480 (M5), as with 151936


@pytest.fixture(scope="module")
def k2_case():
    gen = torch.Generator().manual_seed(0)
    weight = th.frozen(torch.randn(V2, C, generator=gen) * 0.02)
    hidden = torch.randn(4, 512, C, generator=gen).to("cuda")
    lengths = torch.tensor([512, 400, 250, 188])  # 1350 real tokens, the peak batch
    labels, mask = th._labels_and_mask(gen, 4, 512, V2, lengths=lengths)
    return weight, hidden, labels, mask


def _k2_run(monkeypatch, case, tile):
    monkeypatch.setattr(head_ce, "VOCAB_TILE", tile)
    weight, hidden, labels, mask = case
    num_items = labels[..., 1:].ne(-100).sum()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out, grad = th.ours(weight, hidden, labels, mask, num_items)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return [t.detach() for t in out], grad, peak


def test_k2_tile_8192_matches_16384_and_lowers_the_peak(monkeypatch, k2_case):
    _k2_run(monkeypatch, k2_case, 16384)  # warm-up: compile, cuBLAS workspace
    _k2_run(monkeypatch, k2_case, 8192)
    out16, grad16, peak16 = _k2_run(monkeypatch, k2_case, 16384)
    out8, grad8, peak8 = _k2_run(monkeypatch, k2_case, 8192)
    names = ("loss", "token_entropy", "token_loss", "correct", "labelled")
    for name, a, b in zip(names, out8, out16):
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs().max()
            pytest.fail(f"K2 {name} differs with 8192-row tiles (max {float(diff):.3e}): cuBLAS "
                        f"logits GEMM not N-invariant; ulp-level, within checks 5/6")
    th.assert_grad_close(grad8, grad16)
    labels = k2_case[2]
    n_lab = int(labels[..., 1:].ne(-100).sum())
    expect = 8192 * C * 2 + n_lab * 8192 * 2  # half the weight tile + half the backward logits tile
    print(f"\nK2 peak above entry: tile 16384 {peak16 / MiB:.1f} MiB, tile 8192 {peak8 / MiB:.1f} "
          f"MiB, expected saving >= {expect / MiB:.1f} MiB")
    assert peak16 - peak8 >= 0.9 * expect, (peak16 / MiB, peak8 / MiB)


def test_k2_tile_8192_one_compiled_variant(monkeypatch, k2_case):
    """8192 / 4480-wide tiles and 8192 x 1536 casts stay in the variant 16384 uses (n_cols and the
    cast length divisible by 16): no compile is added inside the measured run."""
    _k2_run(monkeypatch, k2_case, 16384)
    kernel = head_ce.soup_head_ce_usig if head_ce.USE_UNIFIED_SIG else head_ce.soup_head_ce_kernel
    before = th._compiled_count(kernel)
    _k2_run(monkeypatch, k2_case, 8192)
    assert th._compiled_count(kernel) == before
