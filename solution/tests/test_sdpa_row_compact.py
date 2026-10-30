"""v9 M3 (solution/sdpa_recompute.py USE_ROW_COMPACT + solution/unpad.py _row_scope): on U's packed
path A1 saves q16 / k16 / v16 for the real rows only and rebuilds the padded tensors with zero pad
rows in its backward recompute. Checks:
(a) pack / unpack round trip: real rows equal, pad rows +0, the original shape and strides;
(b) one SDPA call (Qwen2.5-1.5B head shapes, transformers' own causal+padding mask, dout zero on
    pad rows as after U's pack_rows), dense and I1-causal, BTHD-strided and contiguous inputs:
    forward bit for bit, gradients equal to v8b's full saves (bit for bit whenever the reference
    itself is, else within the mem-efficient backward's own reorder noise), pad-row gradients
    exactly zero, no NaN; outer hooks see [N, heads, 128] saves; fewer bytes held after forward;
(c) every guard keeps v8b's full saves;
(d) a real-width NF4 + LoRA stack with every v6 patch under U, I1 off and on: loss, hidden states
    and RNG bit for bit, LoRA gradients as v8b, M3 fired once per layer each way, bytes held lower,
    and a foreign mask builder keeps the full saves.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import contextlib
import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from transformers import masking_utils as mu  # noqa: E402
from transformers.integrations import sdpa_attention as sa  # noqa: E402
from transformers.modeling_utils import AttentionInterface  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.sdpa_recompute as a1  # noqa: E402

DEV = "cuda"
B, HEADS, KV_HEADS, HEAD_DIM = 4, 12, 2, 128  # Qwen2.5-1.5B attention
SCALE = HEAD_DIM ** -0.5
KEYS = ("rows", "rows_rebuilt", "causal", "recompute")


@pytest.fixture
def patched(monkeypatch):
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward
    assert a1.patch_sdpa_recompute()
    monkeypatch.setattr(a1, "USE_SDPA_RECOMPUTE", True)
    for name in ("USE_CAUSAL_PADDED", "USE_ROW_COMPACT", "_CAUSAL_MASK", "_ROW_SCOPE"):
        monkeypatch.setattr(a1, name, getattr(a1, name))
    a1._CAUSAL_MASK = a1._ROW_SCOPE = None
    yield
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward


def attn_module(n_rep=HEADS // KV_HEADS):
    return types.SimpleNamespace(num_key_value_groups=n_rep, is_causal=True, training=True)


def setup_batch(seq, seed):
    """(real [B, T] bool, 4D mask from transformers' create_causal_mask, idx as U builds it)."""
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, seq + 1, (B,), generator=gen).tolist()
    lengths[0], lengths[1] = 1, min(lengths[1], seq - 1)  # a length-1 row and a padded row
    mask2d = torch.zeros(B, seq, dtype=torch.long, device=DEV)
    for b, n in enumerate(lengths):
        mask2d[b, :n] = 1
    cfg = mq.Qwen2Config(hidden_size=HEADS * HEAD_DIM, num_attention_heads=HEADS,
                         num_key_value_heads=KV_HEADS, num_hidden_layers=1,
                         attn_implementation="sdpa")
    embeds = torch.zeros(B, seq, 8, device=DEV)  # shape / device only
    pos = torch.arange(seq, device=DEV).unsqueeze(0)
    mask = mu.create_causal_mask(config=cfg, inputs_embeds=embeds, attention_mask=mask2d,
                                 past_key_values=None, position_ids=pos)
    assert isinstance(mask, torch.Tensor) and tuple(mask.shape) == (B, 1, seq, seq)
    idx = mask2d.eq(1).reshape(-1).nonzero().view(-1)  # unpad._row_indices on a valid mask
    return mask2d.bool(), mask, idx


def leaves(seq, seed, bthd=True, dtype=torch.float32):
    """fp32 q/k/v leaves. bthd: [B, T, H, D] leaves passed transposed (Qwen2Attention's layout);
    else contiguous [B, H, T, D] leaves."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    out = []
    for h in (HEADS, KV_HEADS, KV_HEADS):
        shape = (B, seq, h, HEAD_DIM) if bthd else (B, h, seq, HEAD_DIM)
        out.append(torch.randn(shape, device=DEV, generator=gen, dtype=dtype).requires_grad_(True))
    return out


def as_bhtd(t, bthd):
    return t.transpose(1, 2) if bthd else t


def call(qkv, mask, dout, *, causal, rows, idx=None, bthd=True, hook=None, scope_mask=None,
         scope_bt=None, autocast=True, **kw):
    """One A1 call as Qwen2Attention makes it. Returns (out, grads of the leaves, counter deltas)."""
    a1.USE_CAUSAL_PADDED = causal
    a1.USE_ROW_COMPACT = rows
    for t in qkv:
        t.grad = None
    q, k, v = (as_bhtd(t, bthd) for t in qkv)
    seq = q.shape[2]
    kwargs = dict(dropout=0.0, scaling=SCALE, sliding_window=None)
    kwargs.update(kw)
    before = dict(a1.COUNTS)
    with contextlib.ExitStack() as stack:
        stack.enter_context(a1.causal_mask_scope(mask))
        if idx is not None:
            bt = scope_bt or (B, seq)
            stack.enter_context(a1.row_compact_scope(mask if scope_mask is None else scope_mask,
                                                     idx, *bt))
        if hook is not None:
            stack.enter_context(torch.autograd.graph.saved_tensors_hooks(hook, lambda t: t))
        if autocast:
            stack.enter_context(torch.autocast("cuda", dtype=torch.float16))
        out, weights = a1.sdpa_forward(attn_module(), q, k, v, mask, **kwargs)
    assert weights is None
    assert a1._ROW_SCOPE is None and a1._CAUSAL_MASK is None  # scopes left
    out.backward(dout)
    delta = {key: a1.COUNTS[key] - before[key] for key in KEYS}
    return out.detach(), [t.grad.clone() for t in qkv], delta


def make_dout(real, seq, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    dout = 0.1 * torch.randn(B, seq, HEADS, HEAD_DIM, device=DEV, generator=gen, dtype=torch.float16)
    dout[~real] = 0  # U's pack_rows backward gives exact zeros on the pad rows
    return dout


def _report(name, a, b):
    diff = (a.float() - b.float()).abs()
    print(f"{name}: max|diff| {diff.max().item():.3e}  exact fraction "
          f"{(a == b).float().mean().item():.4f}")


def _grad_match(new, refs, name):
    """new == refs[0] bit for bit when all reference runs agree bit for bit (the backward then is
    reproducible here); otherwise within the mem-efficient backward's reorder noise. True if exact."""
    ref = refs[0]
    assert new.dtype == ref.dtype and new.shape == ref.shape, name
    reproducible = all(torch.equal(r, ref) for r in refs[1:])
    _report(name, new, ref)
    if reproducible:
        assert torch.equal(new, ref), name
        return True
    torch.testing.assert_close(new, ref, atol=2e-3, rtol=1e-3, msg=name)
    return False


# ---------------------------------------------------------------------------------------------
# (a) round trip
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bthd", [True, False])
def test_pack_unpack_round_trip(bthd):
    seq = 37
    real, _, idx = setup_batch(seq, seed=5)
    bi, ti = a1._RowScope(None, idx, B, seq).rows()
    assert bi.dtype == ti.dtype == torch.int64 and bi.shape == ti.shape == idx.shape
    assert torch.equal(bi * seq + ti, idx)
    base = torch.randn(B, seq, HEADS, HEAD_DIM, device=DEV, dtype=torch.float16)
    x = base.transpose(1, 2) if bthd else base.transpose(1, 2).contiguous()  # [B, H, T, D]
    packed = a1._pack_rows(x, bi, ti)
    assert packed is not None
    small, layout = packed
    assert layout == ("bthd" if bthd else "bhtd")
    assert small.shape == (idx.numel(), HEADS, HEAD_DIM) and small.is_contiguous()
    full = a1._unpack_rows(small, bi, ti, B, seq, layout)
    assert full.shape == x.shape and full.stride() == x.stride() and full.dtype == x.dtype
    rows_new, rows_old = full.transpose(1, 2), x.transpose(1, 2)  # [B, T, H, D]
    assert torch.equal(rows_new[real], rows_old[real])
    pad = rows_new[~real]
    assert torch.count_nonzero(pad) == 0 and not torch.signbit(pad).any()  # +0, not -0
    # A layout we do not rebuild (neither BTHD-strided nor contiguous) is refused.
    odd = torch.randn(B, seq, HEADS, 2 * HEAD_DIM, device=DEV)[..., :HEAD_DIM].transpose(1, 2)
    assert a1._pack_rows(odd, bi, ti) is None


# ---------------------------------------------------------------------------------------------
# (b) one call
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bthd", [True, False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq", [200, 512])
def test_real_rows_match_full_saves(patched, seq, causal, bthd):
    real, mask, idx = setup_batch(seq, seed=seq)
    qkv = leaves(seq, seed=seq, bthd=bthd)
    dout = make_dout(real, seq, seed=seq + 1)

    refs = [call(qkv, mask, dout, causal=causal, rows=False) for _ in range(3)]  # v8b
    new = call(qkv, mask, dout, causal=causal, rows=True, idx=idx)
    for _, _, d in refs:
        assert d["rows"] == 0 and d["rows_rebuilt"] == 0
    assert new[2]["rows"] == 1 and new[2]["rows_rebuilt"] == 1
    assert new[2]["causal"] == refs[0][2]["causal"] == (2 if causal else 0)

    out0, g0, _ = refs[0]
    out1, g1, _ = new
    assert torch.equal(out1, out0)  # the forward call is unchanged (full tensors)
    print(f"T={seq} causal={causal} bthd={bthd} N={idx.numel()}")
    for i, name in enumerate(("dq", "dk", "dv")):
        assert torch.isfinite(g1[i]).all(), name
        _grad_match(g1[i], [r[1][i] for r in refs], name)
        rows_new = g1[i] if bthd else g1[i].transpose(1, 2)
        rows_old = g0[i] if bthd else g0[i].transpose(1, 2)
        assert torch.count_nonzero(rows_new[~real]) == 0, name  # pad rows: exactly zero
        assert torch.count_nonzero(rows_old[~real]) == 0, name


@pytest.mark.parametrize("causal", [False, True])
def test_outer_hook_sees_real_rows_only(patched, causal):
    seq = 200
    real, mask, idx = setup_batch(seq, seed=7)
    qkv = leaves(seq, seed=7)
    dout = make_dout(real, seq, seed=8)
    seen = []

    def keep(t):
        seen.append((tuple(t.shape), t.dtype))
        return t.detach()

    call(qkv, mask, dout, causal=causal, rows=True, idx=idx, hook=keep)
    n = idx.numel()
    f16, i64 = torch.float16, torch.int64
    expected = [((n, HEADS, HEAD_DIM), f16), ((n, KV_HEADS, HEAD_DIM), f16),
                ((n, KV_HEADS, HEAD_DIM), f16), ((n,), i64), ((n,), i64)]
    if not causal:
        expected.append(((B, 1, seq, seq), torch.bool))
    assert sorted(seen, key=str) == sorted(expected, key=str), seen


def test_holds_fewer_bytes_after_forward(patched):
    seq = 512
    real, mask, idx = setup_batch(seq, seed=21)
    qkv = leaves(seq, seed=21)
    q, k, v = (t.transpose(1, 2) for t in qkv)
    held = {}
    for rows in (False, True):
        a1.USE_ROW_COMPACT = rows
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        with a1.causal_mask_scope(mask), a1.row_compact_scope(mask, idx, B, seq), \
                torch.autocast("cuda", dtype=torch.float16):
            out, _ = a1.sdpa_forward(attn_module(), q, k, v, mask, dropout=0.0, scaling=SCALE,
                                     sliding_window=None)
        torch.cuda.synchronize()
        held[rows] = torch.cuda.memory_allocated() - start
        del out
    pad, n = real.numel() - int(real.sum()), idx.numel()
    expect = pad * (HEADS + 2 * KV_HEADS) * HEAD_DIM * 2 - 2 * n * 8  # fp16 rows - (b, t) vectors
    saved = held[False] - held[True]
    print(f"held v8b {held[False]} M3 {held[True]} saved {saved} expected {expect}")
    assert saved >= 0.9 * expect, (held, expect)


# ---------------------------------------------------------------------------------------------
# (c) guards
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["flag_off", "no_scope", "equal_values_other_object",
                                  "scope_other_batch_seq", "sliding_window", "int32_idx",
                                  "fp16_odd_layout"])
def test_guards_keep_full_saves(patched, case):
    seq = 200
    real, mask, idx = setup_batch(seq, seed=3)
    qkv = leaves(seq, seed=3)
    dout = make_dout(real, seq, seed=4)
    kw = dict(causal=False, rows=True, idx=idx)
    used = mask
    if case == "flag_off":
        kw["rows"] = False
    elif case == "no_scope":
        kw["idx"] = None
    elif case == "equal_values_other_object":
        used, kw["scope_mask"] = mask.clone(), mask  # same values, other storage
    elif case == "scope_other_batch_seq":
        kw["scope_bt"] = (B, seq - 1)
    elif case == "sliding_window":
        kw["sliding_window"] = 64
    elif case == "int32_idx":
        kw["idx"] = idx.int()
    elif case == "fp16_odd_layout":
        # fp16 inputs under autocast are used as they are: a strided view neither BTHD-dense nor
        # contiguous cannot be rebuilt, so the full tensors are saved.
        gen = torch.Generator(device=DEV).manual_seed(3)
        qkv = []
        for h in (HEADS, KV_HEADS, KV_HEADS):
            wide = torch.randn(B, seq, h, 2 * HEAD_DIM, device=DEV, generator=gen,
                               dtype=torch.float16)
            qkv.append(wide[..., :HEAD_DIM].detach().requires_grad_(True))
    seen = []

    def keep(t):
        seen.append(tuple(t.shape))
        return t.detach()

    out, g, d = call(qkv, used, dout, hook=keep, **kw)
    assert d["rows"] == 0 and d["rows_rebuilt"] == 0 and d["recompute"] == 1, case
    q_shape, kv_shape = (B, HEADS, seq, HEAD_DIM), (B, KV_HEADS, seq, HEAD_DIM)
    assert sorted(seen) == sorted([q_shape, kv_shape, kv_shape, (B, 1, seq, seq)]), seen
    for t in [out, *g]:
        assert torch.isfinite(t).all()


def test_scope_restores_on_error():
    marker = torch.zeros(1)
    assert a1._ROW_SCOPE is None
    with pytest.raises(RuntimeError):
        with a1.row_compact_scope(marker, marker.long(), 1, 1):
            assert a1._ROW_SCOPE is not None and a1._ROW_SCOPE.mask is marker
            raise RuntimeError("boom")
    assert a1._ROW_SCOPE is None


# ---------------------------------------------------------------------------------------------
# (d) real-width stack with every v6 patch under U
# ---------------------------------------------------------------------------------------------

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
pytest.importorskip("triton")

from solution.tests import test_peak_v6 as tp  # noqa: E402
from solution.tests import test_unpad as tu  # noqa: E402  (also hides torchao from peft)
from solution.tests import test_v6_merged as tm  # noqa: E402

forwards = tu.forwards
v6_patches = tm.v6_patches


@pytest.mark.parametrize("causal", [False, True])
def test_stack_matches_v8b_and_holds_less(v6_patches, causal):
    patched_forward = v6_patches
    pm = tm.build(layers=2)
    n_layers = pm.config.num_hidden_layers
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    real = batch["attention_mask"].bool()

    tm._set(True, causal=causal)
    a1.USE_ROW_COMPACT = False  # v8b
    ref_a = tm._run(pm, batch, patched_forward, extra)
    ref_b = tm._run(pm, batch, patched_forward, extra)
    a1.USE_ROW_COMPACT = True  # M3
    before = dict(a1.COUNTS)
    new = tm._run(pm, batch, patched_forward, extra)
    assert a1.COUNTS["rows"] - before["rows"] == n_layers
    assert a1.COUNTS["rows_rebuilt"] - before["rows_rebuilt"] == n_layers
    assert a1.COUNTS["causal"] - before["causal"] == (2 * n_layers if causal else 0)

    # The forward is unchanged: bit for bit (when v8b is itself reproducible), the same RNG stream.
    tp._assert_matches(ref_a["loss"], ref_b["loss"], new["loss"], "loss")
    tp._assert_matches(ref_a["hidden"], ref_b["hidden"], new["hidden"], "hidden")
    assert torch.count_nonzero(new["hidden"][~real]) == 0
    assert torch.equal(new["rng_fwd"], ref_a["rng_fwd"]) and torch.equal(new["rng_bwd"], ref_a["rng_bwd"])
    assert new["grads"].keys() == ref_a["grads"].keys()
    exact = [tp._assert_grad_close(ref_a["grads"][k], ref_b["grads"][k], new["grads"][k], k)
             for k in ref_a["grads"]]
    print(f"LoRA grads bit for bit: {sum(exact)}/{len(exact)}")

    mask = batch["attention_mask"]
    rows, n = mask.numel(), int(mask.sum())
    expect = n_layers * (rows - n) * (HEADS + 2 * KV_HEADS) * HEAD_DIM * 2 - 2 * n * 8
    saved = ref_a["held"] - new["held"]
    print(f"held v8b {ref_a['held']} M3 {new['held']} saved {saved} expected {expect}")
    assert saved >= 0.8 * expect, (ref_a["held"], new["held"], expect)


def test_foreign_mask_builder_keeps_full_saves(v6_patches, monkeypatch):
    """U enters the row scope only for transformers' genuine create_causal_mask / sdpa_mask."""
    patched_forward = v6_patches
    genuine = mq.create_causal_mask

    def foreign(*args, **kwargs):  # same values, but not provably causal AND padding
        return genuine(*args, **kwargs)

    monkeypatch.setattr(mq, "create_causal_mask", foreign)
    pm = tm.build(layers=2)
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    tm._set(True, causal=True)
    a1.USE_ROW_COMPACT = True
    before = dict(a1.COUNTS)
    tm._run(pm, batch, patched_forward, extra)
    assert a1.COUNTS["recompute"] - before["recompute"] == pm.config.num_hidden_layers
    assert a1.COUNTS["rows"] == before["rows"] and a1.COUNTS["causal"] == before["causal"]
