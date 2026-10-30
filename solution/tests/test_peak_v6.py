"""v6 peak-timing: B (fused_block), A + D (saved_compact), K3's fp16 delta / recompute flags and
S4's in-place SwiGLU backward. Each must give v5's numbers (bit for bit where v5 itself is
deterministic) while autograd keeps less.

Run from the repository root on a T4: python -m pytest solution -q
"""

import contextlib
import gc

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("triton")
pytest.importorskip("peft")

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.integrations import sdpa_attention as sa  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # noqa: E402

import solution.fused_block as fb  # noqa: E402
import solution.kernels.nf4 as nf4  # noqa: E402
import solution.layers as k3_layers  # noqa: E402
import solution.mlp_recompute as rc  # noqa: E402
import solution.saved_compact as sc  # noqa: E402
import solution.unpad as unpad  # noqa: E402
from solution.kernels import addnorm  # noqa: E402
from solution.tests import test_unpad as tu  # noqa: E402

forwards = tu.forwards  # the U fixture (restores every patched class forward after each test)

H = addnorm.HIDDEN_SIZE
EPS = 1e-6
MiB = 1 << 20


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def _assert_matches(ref_a, ref_b, new, name=""):
    """new == ref_a bit for bit when the reference is itself reproducible (ref_a == ref_b);
    otherwise (a non-deterministic kernel in the reference path, e.g. an attention backward with
    atomics) no farther from ref_a than ref_b is, up to a small factor."""
    if _same(ref_a, ref_b):
        assert _same(new, ref_a), name
    else:
        noise = tu.rel(ref_b, ref_a)
        assert tu.rel(new, ref_a) <= 4 * noise + 1e-7, name


@pytest.fixture
def sdpa_patch():
    """A's registry wrapper installed for the test, the registry put back afterwards."""
    had = "sdpa" in ALL_ATTENTION_FUNCTIONS._local_mapping
    before = ALL_ATTENTION_FUNCTIONS._local_mapping.get("sdpa")
    assert sc.patch_sdpa_kv_compact()
    assert sc.patch_sdpa_kv_compact()  # idempotent
    wrapper = ALL_ATTENTION_FUNCTIONS["sdpa"]
    assert getattr(wrapper, sc._MARK, False)
    yield wrapper
    if had:
        ALL_ATTENTION_FUNCTIONS._local_mapping["sdpa"] = before
    else:
        ALL_ATTENTION_FUNCTIONS._local_mapping.pop("sdpa", None)


# ---------------------------------------------------------------------------------------------
# K3 with an fp16 delta and the recompute flags
# ---------------------------------------------------------------------------------------------

def _w(seed=0):
    torch.manual_seed(seed)
    return (1.0 + 0.1 * torch.randn(H, device="cuda")).requires_grad_(False)


# The fp16 delta pointer is a different compiled variant of the fwd kernel: the add is the same
# single fp32 add (r bit for bit), but the fp16 load is vectorised differently and tl.sum adds the
# 1536 squares in another order, so rstd and y differ from the fp32-pointer launch by a few fp32
# ulp (Kaggle T4 run of 30.10: rows 77 and 2048 differed, rows 1 did not). That is a summation-order
# difference, not a different formula; the tolerance below is the fp32 reduction bound we accept
# (2e-6 relative ~ 17-34 fp32 ulp, far above the expected few ulp and 250x below the 4.9e-4
# step of the fp16 cast that follows y). Within v6 every launch uses the fp16 variant, so the exact checks
# (store_r, the bwd rebuild) stay exact.
RTOL_ORDER = 2e-6


@pytest.mark.parametrize("rows", [1, 77, 2048])
def test_k3_fp16_delta_pointer_matches_fp32_cast(rows):
    w = _w()
    r = 3.0 * torch.randn(rows, H, device="cuda")
    d16 = torch.randn(rows, H, device="cuda", dtype=torch.float16)
    r16, y16, s16 = addnorm._launch_fwd(r, d16, w, EPS)  # v6: fp16 pointer
    r32, y32, s32 = addnorm._launch_fwd(r, d16.float(), w, EPS)  # v5: Python-side fp32 cast
    assert torch.equal(r16, r + d16.float())
    assert torch.equal(r16, r32)  # the add itself is exact in both variants
    torch.testing.assert_close(s16, s32, rtol=RTOL_ORDER, atol=0)
    torch.testing.assert_close(y16, y32, rtol=RTOL_ORDER, atol=RTOL_ORDER)
    # Both are the same RMSNorm: against an fp64 reference neither is more than a few ulp off.
    ref_s = (r32.double().pow(2).mean(-1) + EPS).rsqrt()
    torch.testing.assert_close(s16.double(), ref_s, rtol=RTOL_ORDER, atol=0)
    torch.testing.assert_close(s32.double(), ref_s, rtol=RTOL_ORDER, atol=0)
    # store_r = 0: the same y and rstd, r not materialised.
    none, y_nr, s_nr = addnorm._launch_fwd(r, d16, w, EPS, store_r=False)
    assert none is None and torch.equal(y_nr, y16) and torch.equal(s_nr, s16)
    # bwd with has_delta rebuilds r_mid exactly: the same dx as from the saved r_mid.
    dy = torch.randn(rows, H, device="cuda")
    dr = torch.randn(rows, H, device="cuda")
    saved = addnorm._launch_bwd(dy, dr, r16, w, s16)
    rebuilt = addnorm._launch_bwd(dy, dr, r, w, s16, delta=d16)
    assert torch.equal(saved, rebuilt)


def test_add_rmsnorm_fp16_delta_grad_dtype_and_value():
    w = _w(1)
    r0 = torch.randn(3, 41, H, device="cuda", requires_grad=True)
    d0 = torch.randn(3, 41, H, device="cuda", dtype=torch.float16, requires_grad=True)
    res, y = addnorm.add_rmsnorm(r0, d0, w, EPS)
    gr, gy = torch.randn_like(res), torch.randn_like(y)
    torch.autograd.backward((res, y), (gr, gy))
    # Reference: v5's fp32 delta, then autograd's ToCopyBackward to fp16.
    r1 = r0.detach().clone().requires_grad_(True)
    d1 = d0.detach().clone().requires_grad_(True)
    res1, y1 = addnorm.add_rmsnorm(r1, d1.float(), w, EPS)
    torch.autograd.backward((res1, y1), (gr, gy))
    assert torch.equal(res, res1)  # the add: exact
    # y and the grads see rstd from the other fwd variant: summation order only (see RTOL_ORDER).
    torch.testing.assert_close(y, y1, rtol=RTOL_ORDER, atol=RTOL_ORDER)
    assert d0.grad.dtype == torch.float16
    torch.testing.assert_close(r0.grad, r1.grad, rtol=4 * RTOL_ORDER, atol=4 * RTOL_ORDER)
    # The fp16 delta grad is the fp16 rounding of (nearly) the same fp32 value: equal, or one fp16
    # step apart where the tiny fp32 difference crosses a rounding boundary.
    torch.testing.assert_close(d0.grad, d1.grad, rtol=2.0 ** -10, atol=2.0 ** -24)
    assert (d0.grad != d1.grad).float().mean().item() < 0.01
    # Same variant on both sides: v6's add_rmsnorm is deterministic bit for bit.
    r2 = r0.detach().clone().requires_grad_(True)
    d2 = d0.detach().clone().requires_grad_(True)
    res2, y2 = addnorm.add_rmsnorm(r2, d2, w, EPS)
    torch.autograd.backward((res2, y2), (gr, gy))
    assert torch.equal(y2, y) and torch.equal(r2.grad, r0.grad) and torch.equal(d2.grad, d0.grad)


# ---------------------------------------------------------------------------------------------
# S4: in-place SwiGLU backward
# ---------------------------------------------------------------------------------------------

def fp16_linear4bit(n_in, n_out, bias=False):
    """bnb Linear4bit quantized from an fp16 weight, as from_pretrained(torch_dtype=fp16) leaves it:
    quantize_4bit records the input dtype in quant_state.dtype (bitsandbytes functional.py:668-677,
    Params4bit._quantize nn/modules.py:381-391, run by .cuda() via to() :424-428), and K1 only takes
    fp16 states (solution/kernels/nf4.py:196-197). Built from the default fp32 weight, the state is
    fp32, K1 refuses it and S4 falls back to bnb's fp32 dequant + .to(fp16), whose 52 MiB fp32
    weight then sets the backward peak (that is why the in-place SwiGLU test saw no difference)."""
    lin = bnb.nn.Linear4bit(n_in, n_out, bias=bias, compute_dtype=torch.float16,
                            compress_statistics=True, quant_type="nf4")
    lin.weight.data = lin.weight.data.to(torch.float16)  # as Params4bit.__new__ sets .data
    return lin


def _mlp(hidden=H, inter=8960, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=hidden, intermediate_size=inter, hidden_act="silu")
    mlp = mq.Qwen2MLP(cfg)
    for name, (n_in, n_out) in {"gate_proj": (hidden, inter), "up_proj": (hidden, inter),
                                "down_proj": (inter, hidden)}.items():
        setattr(mlp, name, fp16_linear4bit(n_in, n_out))
    mlp = mlp.cuda()
    assert all(getattr(mlp, n).weight.quant_state.dtype == torch.float16
               for n in ("gate_proj", "up_proj", "down_proj"))
    return mlp


@pytest.fixture
def s4(monkeypatch):
    monkeypatch.setattr(mq.Qwen2MLP, "forward", mq.Qwen2MLP.forward)
    monkeypatch.setattr(bnb, "matmul_4bit", bnb.matmul_4bit)
    nf4.patch_bnb_matmul_4bit()
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
    monkeypatch.setattr(rc, "RECOMPUTE_GATE", True)
    monkeypatch.setattr(rc, "SWIGLU_BACKWARD", "example")
    rc.patch_qwen2_mlp_recompute()
    assert rc._swiglu_example_kernel is not None


def test_inplace_swiglu_backward_bit_exact_and_smaller_peak(s4, monkeypatch):
    mlp = _mlp()
    x = torch.randn(1350, H, device="cuda")
    grad = torch.randn(1350, H, device="cuda")
    results = {}
    for inplace in (False, True):
        monkeypatch.setattr(rc, "INPLACE_SWIGLU_BACKWARD", inplace)
        xi = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = mlp(xi)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out.backward(grad)
        torch.cuda.synchronize()
        results[inplace] = (out.detach(), xi.grad, torch.cuda.max_memory_allocated() - base)
    (o0, g0, p0), (o1, g1, p1) = results[False], results[True]
    assert torch.equal(o0, o1) and torch.equal(g0, g1)
    # Buffers of U = [1350, 8960] fp16 and one dense weight W = [8960, 1536] fp16. Out of place
    # the peak is the SwiGLU backward (dh, g, u, dg, du = 5U); in place it is the recompute of u
    # (dh, g, u + K1's dequantised W = 3U + W). Difference 2U - W = ~20 MiB here.
    u_bytes, w_bytes = 1350 * 8960 * 2, 8960 * H * 2
    assert p0 - p1 >= 0.8 * (2 * u_bytes - w_bytes), (p0 / MiB, p1 / MiB)


def test_inplace_needs_fresh_distinct_buffers():
    a = torch.zeros(4, 8, device="cuda", dtype=torch.float16)
    assert not rc._inplace_ok(a, a, torch.zeros_like(a))
    assert not rc._inplace_ok(a, torch.zeros_like(a), torch.zeros(4, 8, device="cuda"))


# ---------------------------------------------------------------------------------------------
# B: the fused add + norm + MLP node against v5's three ops
# ---------------------------------------------------------------------------------------------

class _Layer(torch.nn.Module):
    """The two submodules fused_block reads; built on the GPU (no .cuda() on the 4-bit MLP)."""

    def __init__(self):
        super().__init__()
        norm = mq.Qwen2RMSNorm(H, eps=EPS).cuda()
        with torch.no_grad():
            norm.weight.copy_(1.0 + 0.1 * torch.randn(H, device="cuda"))
        norm.weight.requires_grad_(False)
        self.post_attention_layernorm = norm
        self.mlp = _mlp()


def _v5_ops(layer, r, a):
    """unpad._layer_packed's K3 branch in v5: add_rmsnorm, then layer.mlp (S4), residual add."""
    ln2 = layer.post_attention_layernorm
    r_mid, y = addnorm.add_rmsnorm(r, a, ln2.weight, ln2.variance_epsilon)
    return r_mid + layer.mlp(y)


@pytest.mark.parametrize("rows", [300, 1350])
@pytest.mark.parametrize("r_grad", [True, False])  # False: layer 0 (frozen embeddings)
def test_fused_block_bit_exact_and_saves_less(s4, monkeypatch, rows, r_grad):
    monkeypatch.setattr(fb, "USE_FUSED_BLOCK", True)
    layer = _Layer()
    torch.manual_seed(7)
    r0 = 2.0 * torch.randn(rows, H, device="cuda")
    a0 = torch.randn(rows, H, device="cuda", dtype=torch.float16)
    g = torch.randn(rows, H, device="cuda")
    out = {}
    for fused in (False, True):
        r = r0.clone().requires_grad_(r_grad)
        a = a0.clone().requires_grad_(True)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16):
            if fused:
                assert fb.fused_ok(layer, r, a, mq.Qwen2MLP, mq.Qwen2RMSNorm)
                nxt = fb.add_norm_mlp(r, a, layer)
            else:
                nxt = _v5_ops(layer, r, a)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before  # output + what autograd keeps
        nxt.backward(g)
        out[fused] = (nxt.detach(), r.grad, a.grad, held)
    (n0, dr0, da0, h0), (n1, dr1, da1, h1) = out[False], out[True]
    assert torch.equal(n0, n1)
    assert torch.equal(da0, da1) and da1.dtype == torch.float16
    if r_grad:
        assert torch.equal(dr0, dr1)
    else:
        assert dr0 is None and dr1 is None
    # v5 keeps r_mid (fp32) + x16 (fp16); B keeps a (fp16) only (r is the caller's tensor).
    expect = rows * H * (4 + 2 - 2)
    assert h0 - h1 >= 0.9 * expect, (h0, h1, expect)


def test_fused_ok_guards(s4, monkeypatch):
    layer = _Layer()
    r = torch.randn(200, H, device="cuda", requires_grad=True)
    a = torch.randn(200, H, device="cuda", dtype=torch.float16, requires_grad=True)
    args = (mq.Qwen2MLP, mq.Qwen2RMSNorm)
    monkeypatch.setattr(fb, "USE_FUSED_BLOCK", False)
    assert not fb.fused_ok(layer, r, a, *args)
    monkeypatch.setattr(fb, "USE_FUSED_BLOCK", True)
    with torch.autocast("cuda", dtype=torch.float16):
        assert fb.fused_ok(layer, r, a, *args)
        assert not fb.fused_ok(layer, r, a.float(), *args)  # fp32 delta
        assert not fb.fused_ok(layer, r[:, :H // 2].contiguous(), a, *args)  # width
        with torch.no_grad():
            assert not fb.fused_ok(layer, r, a, *args)
        monkeypatch.setattr(rc, "RECOMPUTE_GATE", False)
        assert not fb.fused_ok(layer, r, a, *args)
        monkeypatch.setattr(rc, "RECOMPUTE_GATE", True)
        monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", False)
        assert not fb.fused_ok(layer, r, a, *args)
        monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
        h = layer.mlp.register_forward_hook(lambda *_: None)
        assert not fb.fused_ok(layer, r, a, *args)
        h.remove()
        h = layer.post_attention_layernorm.register_forward_pre_hook(lambda *_: None)
        assert not fb.fused_ok(layer, r, a, *args)
        h.remove()


# ---------------------------------------------------------------------------------------------
# A: SDPA key/value compaction, the attention function alone
# ---------------------------------------------------------------------------------------------

def _attn_inputs(batch=4, seq=512, lengths=(512, 300, 37, 129), seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=H, num_attention_heads=12, num_key_value_heads=2,
                      intermediate_size=256, num_hidden_layers=1, attn_implementation="sdpa")
    attn = mq.Qwen2Attention(cfg, 0).cuda().train()
    q0 = torch.randn(batch, seq, 12, 128, device="cuda", requires_grad=True)
    k0 = torch.randn(batch, seq, 2, 128, device="cuda", requires_grad=True)
    v0 = torch.randn(batch, seq, 2, 128, device="cuda", requires_grad=True)
    keep = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
    for b, n in enumerate(lengths):
        keep[b, :n] = True
    pos = torch.arange(seq, device="cuda")
    mask = (pos[None, :, None] >= pos[None, None, :]) & keep[:, None, :]  # [B, T(q), T(k)]
    return attn, (q0, k0, v0), mask[:, None], keep


def _run_attn(fn, attn, leaves, mask, grad, scope=None):
    for t in leaves:
        t.grad = None
    q, k, v = (t.transpose(1, 2) for t in leaves)  # the [B, H, T, D] views Qwen2 passes
    # Collect earlier tests' cyclic garbage first: otherwise Python's GC may free it in the middle
    # of the measured forward (the T4 run measured ref held = -78.9 MB).
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with scope if scope is not None else contextlib.nullcontext():
        with torch.autocast("cuda", dtype=torch.float16):
            out, w = fn(attn, q, k, v, mask, dropout=0.0, scaling=attn.scaling,
                        sliding_window=None)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - before
    assert w is None
    out.backward(grad)
    return out.detach(), [t.grad.clone() for t in leaves], held


def test_sdpa_kv_compact_matches_original_and_saves_less(sdpa_patch, monkeypatch):
    monkeypatch.setattr(sc, "USE_SDPA_KV_COMPACT", True)
    monkeypatch.setattr(sc, "USE_LORA_ROW_COMPACT", False)
    attn, leaves, mask, keep = _attn_inputs()
    batch, seq = keep.shape
    grad = torch.randn(batch, seq, 12, 128, device="cuda", dtype=torch.float16)
    idx = keep.reshape(-1).nonzero().view(-1)
    original = sa.sdpa_attention_forward
    ref_a = _run_attn(original, attn, leaves, mask, grad)
    ref_b = _run_attn(original, attn, leaves, mask, grad)
    before = dict(sc.COUNTS)
    new = _run_attn(sdpa_patch, attn, leaves, mask, grad,
                    scope=sc.attention_scope(idx, batch, seq, H))
    assert sc.COUNTS["sdpa_compact"] == before["sdpa_compact"] + 1
    assert sc.COUNTS["kv_packed"] == before["kv_packed"] + 2, "SDPA did not save k16/v16 as given"
    _assert_matches(ref_a[0], ref_b[0], new[0], "out")
    for name, ga, gb, gn in zip("qkv", ref_a[1], ref_b[1], new[1]):
        _assert_matches(ga, gb, gn, name)
    # Each of k16, v16 [B, 12, T, 128] fp16 is kept as [B, 2, T, 128]. Before the fix the pack
    # hook closed over the full k16 / v16, so A held both copies (new = 29 458 432 B on the T4).
    expect = 2 * batch * 10 * seq * 128 * 2
    assert ref_a[2] > 0 and new[2] > 0, (ref_a[2], new[2])  # a sane measurement
    assert ref_a[2] - new[2] >= 0.9 * expect, (ref_a[2], new[2], expect)


def test_kv_hooks_hold_no_tensor():
    k16 = torch.randn(2, 12, 8, 128, device="cuda", dtype=torch.float16)
    v16 = torch.randn(2, 12, 8, 128, device="cuda", dtype=torch.float16)
    pack, unpack = sc._kv_hooks(k16, v16, 6)

    def tensors(obj):
        if isinstance(obj, torch.Tensor):
            return 1
        if isinstance(obj, (tuple, list)):
            return sum(tensors(o) for o in obj)
        return 0

    for fn in (pack, unpack):
        assert not any(tensors(c.cell_contents) for c in (fn.__closure__ or ()))
    handle = pack(k16)
    assert isinstance(handle, sc._Rep) and handle.small.shape == (2, 2, 8, 128)
    other = torch.randn(2, 12, 8, 128, device="cuda", dtype=torch.float16, requires_grad=True) * 1
    kept = pack(other)
    assert kept.grad_fn is None and kept.data_ptr() == other.data_ptr()


def test_sdpa_wrapper_falls_back_outside_scope_and_on_odd_inputs(sdpa_patch, monkeypatch):
    monkeypatch.setattr(sc, "USE_SDPA_KV_COMPACT", True)
    attn, leaves, mask, keep = _attn_inputs(batch=2, seq=64, lengths=(64, 20))
    grad = torch.randn(2, 64, 12, 128, device="cuda", dtype=torch.float16)
    idx = keep.reshape(-1).nonzero().view(-1)
    c = dict(sc.COUNTS)
    _run_attn(sdpa_patch, attn, leaves, mask, grad)  # no scope
    assert sc.COUNTS["sdpa_original"] == c["sdpa_original"] + 1
    c = dict(sc.COUNTS)
    _run_attn(sdpa_patch, attn, leaves, mask.float(), grad,
              scope=sc.attention_scope(idx, 2, 64, H))  # float mask: not ours
    assert sc.COUNTS["sdpa_original"] == c["sdpa_original"] + 1
    monkeypatch.setattr(sc, "USE_SDPA_KV_COMPACT", False)
    c = dict(sc.COUNTS)
    _run_attn(sdpa_patch, attn, leaves, mask, grad, scope=sc.attention_scope(idx, 2, 64, H))
    assert sc.COUNTS["sdpa_original"] == c["sdpa_original"] + 1
    # Fake objects (Soup's tests) go to the original, which then fails or not on its own.
    assert not sc._sdpa_ok(object(), None, None, None, None, 0.0, {})


def test_patch_refuses_a_foreign_sdpa(monkeypatch):
    def foreign(*a, **k):
        raise AssertionError

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._local_mapping, "sdpa", foreign)
    assert not sc.patch_sdpa_kv_compact()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is foreign


# ---------------------------------------------------------------------------------------------
# A + D (+ B) on U's packed path, whole model against v5
# ---------------------------------------------------------------------------------------------

def _set_v6(monkeypatch, on):
    monkeypatch.setattr(sc, "USE_SDPA_KV_COMPACT", on)
    monkeypatch.setattr(sc, "USE_LORA_ROW_COMPACT", on)
    monkeypatch.setattr(fb, "USE_FUSED_BLOCK", on)


def _held_run(pm, batch, forward, **extra):
    """tu.run plus the memory autograd holds right after the forward."""
    held = {}
    real_backward = torch.Tensor.backward

    def backward(self, *a, **k):
        torch.cuda.synchronize()
        held["bytes"] = torch.cuda.memory_allocated() - held["start"]
        return real_backward(self, *a, **k)

    gc.collect()  # no earlier garbage freed inside the measured window
    torch.cuda.synchronize()
    held["start"] = torch.cuda.memory_allocated()
    torch.Tensor.backward = backward
    try:
        result = tu.run(pm, batch, forward, **extra)
    finally:
        torch.Tensor.backward = real_backward
    return result, held["bytes"]


def _compare_runs(ref_a, ref_b, new):
    (la, reca, rnga, ga), (lb, recb, rngb, gb), (ln, recn, rngn, gn) = ref_a, ref_b, new
    assert torch.equal(rnga, rngn)  # A, B and D draw no random numbers
    for s0, s1 in zip(reca["rng_pre"], recn["rng_pre"], strict=True):
        assert torch.equal(s0, s1)
    _assert_matches(reca["hidden"], recb["hidden"], recn["hidden"], "hidden")
    _assert_matches(la, lb, ln, "loss")
    for name in ga:
        _assert_matches(ga[name], gb[name], gn[name], name)


def test_small_model_a_and_d_match_v5(forwards, sdpa_patch, monkeypatch):
    base, patched = forwards
    pm = tu.build()  # hidden 128, 4 heads, 2 kv heads, LoRA q/v with dropout 0.05
    batch = tu.make_batch(pm.config.vocab_size)
    _set_v6(monkeypatch, False)
    ref_a = tu.run(pm, batch, patched)
    ref_b = tu.run(pm, batch, patched)
    _set_v6(monkeypatch, True)
    c = dict(sc.COUNTS)
    new = tu.run(pm, batch, patched)
    n_layers = pm.config.num_hidden_layers
    assert sc.COUNTS["sdpa_compact"] - c["sdpa_compact"] == n_layers
    # Per layer: q and v LoRA-A inputs (always), plus the q and v keep-masks when torch saves them
    # as bool (the fused native_dropout path). Layer 0's attention input comes from the frozen
    # embeddings and needs no grad, so its dropout builds no node and keeps no mask: 2 there, 4 in
    # every later layer (the T4 run counted 6 = 2 + 4 for 2 layers). Any other count means
    # something unexpected was packed.
    assert sc.COUNTS["rows_packed"] - c["rows_packed"] in (2 * n_layers, 4 * n_layers - 2)
    _compare_runs(ref_a, ref_b, new)


def _install_k1_k3_s4(monkeypatch):
    monkeypatch.setattr(bnb, "matmul_4bit", bnb.matmul_4bit)
    nf4.patch_bnb_matmul_4bit()
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
    monkeypatch.setattr(rc, "RECOMPUTE_GATE", True)
    monkeypatch.setattr(rc, "SWIGLU_BACKWARD", "example")
    rc.patch_qwen2_mlp_recompute()
    monkeypatch.delitem(unpad._REFS, "k3", raising=False)
    for cls in (mq.Qwen2DecoderLayer, mq.Qwen2RMSNorm):
        fn = cls.forward
        while getattr(fn, unpad._K3_MARK, False):
            fn = fn.__wrapped__
        cls.forward = fn  # the forwards fixture restores the classes afterwards
    k3_layers.patch_qwen2_rmsnorm()
    k3_layers.patch_qwen2_decoder_layer()


def test_real_width_stack_matches_v5_and_holds_less(forwards, sdpa_patch, monkeypatch):
    """K1 + K3 + K4 + S4 + U as in v5, then with B + A + D: same loss, hidden states and LoRA
    gradients, and less memory held by autograd after the forward."""
    base, patched = forwards
    _install_k1_k3_s4(monkeypatch)
    calls = []
    real_fused = fb.add_norm_mlp

    def counting(*a, **k):
        calls.append(1)
        return real_fused(*a, **k)

    monkeypatch.setattr(fb, "add_norm_mlp", counting)
    pm = tu.build(hidden=H, heads=12, kv_heads=2, inter=8960, layers=2, vocab=1024,
                  four_bit_mlp=True)
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    _set_v6(monkeypatch, False)
    ref_a, held_a = _held_run(pm, batch, patched, **extra)
    ref_b, _ = _held_run(pm, batch, patched, **extra)
    assert not calls
    _set_v6(monkeypatch, True)
    new, held_n = _held_run(pm, batch, patched, **extra)
    n_layers = pm.config.num_hidden_layers
    assert len(calls) == n_layers
    _compare_runs(ref_a, ref_b, new)

    mask = batch["attention_mask"]
    rows, n = mask.numel(), int(mask.sum())
    # Layer 0 (frozen embeddings, no grad into the stack) saves less in v5 already: no norm keeps
    # its r, so B's (r, a) costs what v5's (r_mid, x16) costs there, and its dropouts keep no mask.
    expect = (n_layers - 1) * 4 * n * H              # B: r_mid fp32 + x16 fp16 - a fp16, real rows
    expect += n_layers * 2 * rows * (H - 256) * 2    # A: k16, v16 keep 2 of 12 heads
    expect += n_layers * 4 * (rows - n) * H          # D: two fp16 LoRA-A inputs, padding rows
    expect += (n_layers - 1) * 2 * (rows - n) * H    # D: two bool masks, padding rows (layer >= 1)
    assert held_a - held_n >= 0.8 * expect, (held_a, held_n, expect)


# ---------------------------------------------------------------------------------------------
# D's hooks alone
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bool])
@pytest.mark.parametrize("two_d", [False, True])
def test_row_hooks_roundtrip(dtype, two_d):
    batch, seq, width = 4, 33, 64
    keep = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
    keep[0, :33], keep[1, :5], keep[2, :1], keep[3, :20] = True, True, True, True
    idx = keep.reshape(-1).nonzero().view(-1)
    pack, unpack = sc._row_hooks(idx, batch, seq, width)
    if dtype == torch.bool:
        t = torch.rand(batch, seq, width, device="cuda") > 0.5
        t[~keep] = False
    else:
        t = torch.randn(batch, seq, width, device="cuda").to(dtype)
        t[~keep] = 0
    if two_d:
        t = t.reshape(batch * seq, width)
    handle = pack(t)
    assert isinstance(handle, sc._Rows) and handle.small.shape == (idx.numel(), width)
    back = unpack(handle)
    assert back.shape == t.shape and back.dtype == t.dtype and back.is_contiguous()
    assert torch.equal(back, t)
    # Anything else is kept as a detached alias: same storage and values, but never the tensor
    # itself, whose grad_fn would close a reference cycle for a saved output (the v6 OOM).
    other = (torch.randn(batch, 12, seq, 128, device="cuda", dtype=torch.float16)
             .requires_grad_(True) * 1)  # a non-leaf with a grad_fn, like a saved output
    kept = pack(other)
    assert kept is not other and kept.grad_fn is None and not kept.requires_grad
    assert kept.data_ptr() == other.data_ptr() and kept.shape == other.shape
    assert unpack(kept) is kept
    f32 = torch.randn(batch, seq, width, device="cuda")
    kept = pack(f32)
    assert kept.data_ptr() == f32.data_ptr() and kept.dtype == f32.dtype
