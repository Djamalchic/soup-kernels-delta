"""v7 I1 (solution/sdpa_recompute.py USE_CAUSAL_PADDED + solution/unpad.py _causal_scope): on U's
packed path A1 runs SDPA with is_causal=True and no mask instead of the [B, 1, T, T] causal+padding
bool mask. Checks:
(a) one SDPA call, Qwen2.5-1.5B head shapes, right-padded masks from transformers'
    create_causal_mask: real query rows of out, dq / dk / dv (dout zero on pad rows, as after U's
    pack_rows) match the dense-mask call; pad key rows get exactly zero dk / dv in both modes; no
    NaN; outer hooks see only q16, k16, v16 (no mask kept); every guard falls back to the dense call;
(b) the profiler shows torch's mem-efficient (cutlass) kernels, not the math path;
(c) a real-width 2-layer NF4 + LoRA stack with every v6 patch under U: loss, hidden real rows and
    LoRA gradients match v6.1 (I1 off), the RNG stream is untouched, and I1 fired once per layer
    in forward and once per layer in backward.
The flat-memory regression with I1 on is test_v6_no_leak.py::test_training_steps_hold_flat_memory.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from transformers import masking_utils as mu  # noqa: E402
from transformers.integrations import sdpa_attention as sa  # noqa: E402
from transformers.modeling_utils import AttentionInterface  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.sdpa_recompute as a1  # noqa: E402
from solution.tests import test_sdpa_recompute as ts  # noqa: E402

DEV = "cuda"
B, HEADS, KV_HEADS, HEAD_DIM = 4, 12, 2, 128  # Qwen2.5-1.5B attention
SCALE = HEAD_DIM ** -0.5


@pytest.fixture
def patched(monkeypatch):
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward
    assert a1.patch_sdpa_recompute()
    monkeypatch.setattr(a1, "USE_SDPA_RECOMPUTE", True)
    monkeypatch.setattr(a1, "USE_CAUSAL_PADDED", a1.USE_CAUSAL_PADDED)
    monkeypatch.setattr(a1, "_CAUSAL_MASK", None)
    yield
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward


def lengths_for(seq, seed):
    """Random right-padded lengths in [1, seq]; always one length-1 row and one shorter than seq."""
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, seq + 1, (B,), generator=gen).tolist()
    lengths[0] = 1
    lengths[1] = min(lengths[1], seq - 1)
    return lengths


def hf_mask(lengths, seq):
    """(2D long mask, the 4D bool mask transformers' own create_causal_mask builds for sdpa)."""
    mask2d = torch.zeros(B, seq, dtype=torch.long, device=DEV)
    for b, n in enumerate(lengths):
        mask2d[b, :n] = 1
    cfg = mq.Qwen2Config(hidden_size=HEADS * HEAD_DIM, num_attention_heads=HEADS,
                         num_key_value_heads=KV_HEADS, num_hidden_layers=1,
                         attn_implementation="sdpa")
    embeds = torch.zeros(B, seq, 8, device=DEV, dtype=torch.float32)  # shape / device only
    pos = torch.arange(seq, device=DEV).unsqueeze(0)
    mask = mu.create_causal_mask(config=cfg, inputs_embeds=embeds, attention_mask=mask2d,
                                 past_key_values=None, position_ids=pos)
    assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
    assert tuple(mask.shape) == (B, 1, seq, seq)
    return mask2d.bool(), mask


def leaves(seq, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, seq, HEADS, HEAD_DIM, device=DEV, generator=gen)
    k = torch.randn(B, seq, KV_HEADS, HEAD_DIM, device=DEV, generator=gen)
    v = torch.randn(B, seq, KV_HEADS, HEAD_DIM, device=DEV, generator=gen)
    return [t.requires_grad_(True) for t in (q, k, v)]


def call(qkv, mask, dout, causal, scope_mask="same", hook=None, **kw):
    """One A1 call as Qwen2Attention makes it (fp32 leaves under fp16 autocast). Returns
    (out, grads, number of causal calls in forward + backward)."""
    a1.USE_CAUSAL_PADDED = causal
    for t in qkv:
        t.grad = None
    q, k, v = (t.transpose(1, 2) for t in qkv)
    scoped = mask if isinstance(scope_mask, str) else scope_mask
    before = a1.COUNTS["causal"]
    hooks = (torch.autograd.graph.saved_tensors_hooks(hook, lambda t: t) if hook is not None
             else _null())
    kwargs = dict(dropout=0.0, scaling=SCALE, sliding_window=None)
    kwargs.update(kw)
    scope = a1.causal_mask_scope(scoped) if scoped is not None else _null()
    with scope, hooks, torch.autocast("cuda", dtype=torch.float16):
        out, weights = a1.sdpa_forward(ts.module(), q, k, v, mask, **kwargs)
    assert weights is None
    assert a1._CAUSAL_MASK is None  # the scope is left
    out.backward(dout)
    return out.detach(), [t.grad.clone() for t in qkv], a1.COUNTS["causal"] - before


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _report(name, a, b):
    diff = (a.float() - b.float()).abs()
    exact = (a == b).float().mean().item()
    print(f"{name}: max|diff| {diff.max().item():.3e}  exact fraction {exact:.4f}")


# ---------------------------------------------------------------------------------------------
# (a) one call
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seq", [200, 512])
def test_causal_matches_dense_on_real_rows(patched, seq):
    lengths = lengths_for(seq, seed=seq)
    real, mask = hf_mask(lengths, seq)
    qkv = leaves(seq, seed=seq)
    gen = torch.Generator(device=DEV).manual_seed(seq + 1)
    dout = 0.1 * torch.randn(B, seq, HEADS, HEAD_DIM, device=DEV, generator=gen,
                             dtype=torch.float16)
    dout[~real] = 0  # U's pack_rows backward gives exact zeros on the pad rows

    out0, g0, n0 = call(qkv, mask, dout, causal=False)
    out1, g1, n1 = call(qkv, mask, dout, causal=True)
    assert n0 == 0 and n1 == 2  # forward + recompute in backward

    for t in [out0, out1, *g0, *g1]:
        assert torch.isfinite(t).all()  # also on pad rows: no fully masked row under causal
    print(f"T={seq} lengths={lengths}")
    _report("out real rows", out1[real], out0[real])
    torch.testing.assert_close(out1[real], out0[real], atol=2e-3, rtol=1e-3)
    for name, a, b in zip(("dq", "dk", "dv"), g1, g0):
        assert a.dtype == b.dtype == torch.float32
        _report(f"{name} real rows", a[real], b[real])
        torch.testing.assert_close(a[real], b[real], atol=2e-3, rtol=1e-3)
    for name, g in (("dk dense", g0[1]), ("dv dense", g0[2]), ("dk causal", g1[1]),
                    ("dv causal", g1[2])):
        assert torch.count_nonzero(g[~real]) == 0, name  # pad keys: exactly zero
    # dq on pad query rows is exactly zero too (dO = 0 there).
    assert torch.count_nonzero(g1[0][~real]) == 0 and torch.count_nonzero(g0[0][~real]) == 0


def test_causal_keeps_no_mask_for_backward(patched):
    seq = 200
    _, mask = hf_mask(lengths_for(seq, 7), seq)
    qkv = leaves(seq, 7)
    dout = torch.zeros(B, seq, HEADS, HEAD_DIM, device=DEV, dtype=torch.float16)
    seen = []

    def keep(t):  # the outer hook only sees what _SDPARecompute itself saves
        seen.append(tuple(t.shape))
        return t.detach()

    call(qkv, mask, dout, causal=True, hook=keep)
    q_shape = (B, HEADS, seq, HEAD_DIM)
    kv_shape = (B, KV_HEADS, seq, HEAD_DIM)
    assert sorted(seen) == sorted([q_shape, kv_shape, kv_shape]), seen


@pytest.mark.parametrize("case", ["flag_off", "no_scope", "equal_values_other_object",
                                  "sliding_window", "other_scope"])
def test_guards_keep_the_dense_call(patched, case):
    seq = 200
    _, mask = hf_mask(lengths_for(seq, 3), seq)
    qkv = leaves(seq, 3)
    dout = 0.1 * torch.ones(B, seq, HEADS, HEAD_DIM, device=DEV, dtype=torch.float16)
    ref_out, ref_g, n_ref = call(qkv, mask, dout, causal=False)
    assert n_ref == 0
    causal, scope, used, kw = True, "same", mask, {}
    if case == "flag_off":
        causal = False
    elif case == "no_scope":
        scope = None
    elif case == "equal_values_other_object":
        used, scope = mask.clone(), mask  # same values, other storage: not provably U's mask
    elif case == "sliding_window":
        kw = {"sliding_window": 64}
    elif case == "other_scope":
        scope = mask[:, :, :-1, :-1].clone()
    out, g, n = call(qkv, used, dout, causal=causal, scope_mask=scope, **kw)
    assert n == 0
    assert torch.equal(out, ref_out)  # the v6.1 dense call: same kernel, same inputs
    for a, b in zip(g, ref_g):
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-4)


def test_scope_restores_on_error():
    marker = torch.zeros(1)
    assert a1._CAUSAL_MASK is None
    with pytest.raises(RuntimeError):
        with a1.causal_mask_scope(marker):
            assert a1._CAUSAL_MASK is marker
            raise RuntimeError("boom")
    assert a1._CAUSAL_MASK is None


# ---------------------------------------------------------------------------------------------
# (b) the kernel
# ---------------------------------------------------------------------------------------------

def test_causal_runs_the_mem_efficient_kernel(patched):
    from torch.profiler import ProfilerActivity, profile

    seq = 512
    real, mask = hf_mask(lengths_for(seq, 11), seq)
    qkv = leaves(seq, 11)
    dout = 0.1 * torch.ones(B, seq, HEADS, HEAD_DIM, device=DEV, dtype=torch.float16)
    dout[~real] = 0
    call(qkv, mask, dout, causal=True)  # warm-up
    names = {}
    for causal in (False, True):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            call(qkv, mask, dout, causal=causal)
            torch.cuda.synchronize()
        names[causal] = {e.key: (e.count, e.device_time_total if hasattr(e, "device_time_total")
                                 else e.cuda_time_total) for e in prof.key_averages()}
    for causal, table in names.items():
        attn = {k: v for k, v in table.items() if "fmha" in k or "attention" in k.lower()}
        print("causal" if causal else "dense", attn)
    causal_names = names[True]
    assert any("fmha_cutlassF" in k for k in causal_names), sorted(causal_names)
    assert any("fmha_cutlassB" in k for k in causal_names), sorted(causal_names)
    assert not any("softmax" in k.lower() for k in causal_names), sorted(causal_names)  # no math


# ---------------------------------------------------------------------------------------------
# (c) real-width stack with every v6 patch under U
# ---------------------------------------------------------------------------------------------

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
pytest.importorskip("triton")

from solution.tests import test_unpad as tu  # noqa: E402  (also hides torchao from peft)
from solution.tests import test_v6_merged as tm  # noqa: E402

forwards = tu.forwards
v6_patches = tm.v6_patches


def _cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-300)).item()


def test_two_layer_stack_matches_v61(v6_patches):
    patched_forward = v6_patches
    pm = tm.build(layers=2)
    n_layers = pm.config.num_hidden_layers
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    real = batch["attention_mask"].bool()

    tm._set(True, causal=False)  # v6.1
    ref = tm._run(pm, batch, patched_forward, extra)
    c0 = a1.COUNTS["causal"]
    tm._set(True, causal=True)  # v7 I1
    new = tm._run(pm, batch, patched_forward, extra)
    assert a1.COUNTS["causal"] - c0 == 2 * n_layers  # one per layer forward + one per recompute

    assert torch.equal(new["rng_fwd"], ref["rng_fwd"]) and torch.equal(new["rng_bwd"], ref["rng_bwd"])
    loss_rel = abs(new["loss"].item() - ref["loss"].item()) / abs(ref["loss"].item())
    print(f"loss v6.1 {ref['loss'].item():.8f}  I1 {new['loss'].item():.8f}  rel {loss_rel:.3e}")
    assert loss_rel < 1e-3
    _report("hidden real rows", new["hidden"][real], ref["hidden"][real])
    assert tu.rel(new["hidden"][real], ref["hidden"][real]) < 1e-3
    assert torch.count_nonzero(new["hidden"][~real]) == 0  # U scatters zeros on padding
    assert new["grads"].keys() == ref["grads"].keys()
    worst = min(_cos(new["grads"][n], ref["grads"][n]) for n in ref["grads"])
    print(f"min cos(LoRA grads) {worst:.8f}")
    assert worst > 0.9999
    # A1 still keeps less than v5-style SDPA would; I1 additionally drops the saved mask.
    assert new["held"] <= ref["held"], (new["held"], ref["held"])


def test_foreign_mask_builder_keeps_the_dense_call(v6_patches, monkeypatch):
    """U enters the causal scope only for transformers' genuine create_causal_mask / sdpa_mask."""
    patched_forward = v6_patches
    genuine = mq.create_causal_mask

    def foreign(*args, **kwargs):  # same values, but not provably the causal+padding builder
        return genuine(*args, **kwargs)

    monkeypatch.setattr(mq, "create_causal_mask", foreign)
    pm = tm.build(layers=2)
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    tm._set(True, causal=True)
    c0, r0 = a1.COUNTS["causal"], a1.COUNTS["recompute"]
    tm._run(pm, batch, patched_forward, extra)
    assert a1.COUNTS["recompute"] - r0 == pm.config.num_hidden_layers  # A1 still runs
    assert a1.COUNTS["causal"] == c0  # ...on the dense mask
