"""M2 (solution/lora_recompute.py): the LoRA-A fp16 input is rebuilt in backward from r.

Checks, against the same code with the flag off (v8b):
- one pair of peft 4-bit LoRA layers (q and v style) fed x_bt = unpack_rows(K3 rmsnorm(r)) inside
  a source scope: outputs, LoRA gradients, r's gradient and the RNG stream bit for bit; autograd
  holds two fewer fp16 [P, 1536] tensors after the forward; one x_bt rebuild serves both layers;
  r without grad (layer 0) included; a second backward (retain_graph) gives the same gradients;
  x_bt is not kept alive; an in-place change of r is caught;
- the v6 merged real-width stack (K1 + K3 + K4 + S4 + U + B + A1 + A2 + D + I1, i.e. v8b's switches)
  with M2 on: loss, hidden states and RNG bit for bit, LoRA gradients within the derived
  GRAD_RTOL (the reference backward has a non-reproducible SDPA), every q/v call took M2's path,
  D packed nothing, and autograd holds 4 x N x 1536 bytes less per layer after the forward;
- every unexpected input runs the forward that was installed before M2.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import gc
import weakref

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
pytest.importorskip("triton")

from peft.tuners.lora import bnb as lora_bnb  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.layers as k3_layers  # noqa: E402
import solution.lora_recompute as m2  # noqa: E402
import solution.lora_replay as a2  # noqa: E402
import solution.saved_compact as sc  # noqa: E402
import solution.unpad as unpad  # noqa: E402
from solution.tests import test_lora_replay as tl  # noqa: E402
from solution.tests import test_peak_v6 as tp  # noqa: E402
from solution.tests import test_unpad as tu  # noqa: E402
from solution.tests import test_v6_merged as tm  # noqa: E402

forwards = tu.forwards  # fixtures used by tm.v6_patches
v6_patches = tm.v6_patches

DEV = "cuda"
H = tl.H
KV = tl.KV
BATCH, SEQ = 4, 128
LENGTHS = (128, 77, 1, 40)  # N = 246 of P = 512
EPS = 1e-6


def _mem():
    """Requested bytes of the live tensors when torch counts them (block rounding of the caching
    allocator can add up to 1 MiB per large tensor, see test_v6_merged._requested_bytes), else
    memory_allocated()."""
    req = tm._requested_bytes()
    return torch.cuda.memory_allocated() if req is None else req


def _unwrap(fn, *marks):
    while any(getattr(fn, m, False) for m in marks):
        fn = fn.__wrapped__
    return fn


@pytest.fixture
def patched(monkeypatch):
    """A2 then M2 on peft's 4-bit LoRA layer, K3 on Qwen2RMSNorm; all restored afterwards."""
    cls = lora_bnb.Linear4bit
    monkeypatch.setattr(cls, "forward", _unwrap(cls.forward, m2._MARK, a2._MARK))
    assert a2.patch_lora_dropout_replay()
    assert m2.patch_lora_input_recompute()
    installed = cls.forward
    assert m2.patch_lora_input_recompute() and cls.forward is installed  # idempotent
    assert getattr(installed.__wrapped__, a2._MARK, False)  # M2 outside, A2 inside
    monkeypatch.setattr(a2, "USE_LORA_DROPOUT_REPLAY", True)
    monkeypatch.setattr(m2, "USE_LORA_INPUT_RECOMPUTE", True)
    norm_cls = mq.Qwen2RMSNorm
    monkeypatch.setattr(norm_cls, "forward", _unwrap(norm_cls.forward, unpad._K3_MARK))
    k3_layers.patch_qwen2_rmsnorm()
    return cls


def _norm():
    norm = mq.Qwen2RMSNorm(H, eps=EPS).cuda()
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(H, device=DEV))
    norm.weight.requires_grad_(False)
    return norm


def _idx(lengths=LENGTHS, seq=SEQ):
    keep = torch.zeros(len(lengths), seq, dtype=torch.bool, device=DEV)
    for row, n in enumerate(lengths):
        keep[row, :n] = True
    return keep.reshape(-1).nonzero().view(-1)


def step(norm, layers, r0, idx, m2_on, r_grad, a2_on=True, seed=1234, mutate=False, twice=False):
    """x_bt = unpack_rows(norm(r)) -> q/v style LoRA layers (as unpad._layer_packed), backward."""
    m2.USE_LORA_INPUT_RECOMPUTE = m2_on
    a2.USE_LORA_DROPOUT_REPLAY = a2_on
    for layer in layers:
        layer.zero_grad(set_to_none=True)
    r = r0.clone().requires_grad_(r_grad)
    gc.collect()
    torch.cuda.synchronize()
    start = _mem()
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.float16):
        x = norm(r)
        x_bt = unpad.unpack_rows(x, idx, BATCH, SEQ)
        del x
        with m2.source_scope(norm, r, x_bt, idx, mq.Qwen2RMSNorm):
            outs = [layer(x_bt) for layer in layers]
        ref = weakref.ref(x_bt)
        del x_bt
        _ = torch.rand(7, device=DEV)  # an unrelated draw between forward and backward
    torch.cuda.synchronize()
    held = _mem() - start
    x_bt_alive = ref() is not None
    rng_fwd = torch.cuda.get_rng_state()
    gen = torch.Generator(device=DEV).manual_seed(99)
    loss = sum((o.float() * torch.randn(o.shape, device=DEV, generator=gen)).sum() for o in outs)
    if mutate:
        with torch.no_grad():
            r.add_(1.0)  # bumps r's version counter
    loss.backward(retain_graph=twice)
    rng_bwd = torch.cuda.get_rng_state()
    grads = {f"{i}.{n}": p.grad.clone() for i, layer in enumerate(layers)
             for n, p in layer.named_parameters() if p.grad is not None}
    if r_grad:
        grads["r"] = r.grad.clone()
    if twice:
        for layer in layers:
            layer.zero_grad(set_to_none=True)
        r.grad = None
        loss.backward()
        grads2 = {f"{i}.{n}": p.grad.clone() for i, layer in enumerate(layers)
                  for n, p in layer.named_parameters() if p.grad is not None}
        if r_grad:
            grads2["r"] = r.grad.clone()
        assert grads2.keys() == grads.keys()
        for k in grads:
            assert torch.equal(grads2[k], grads[k]), k
    return {"outs": [o.detach() for o in outs], "grads": grads, "rng_fwd": rng_fwd,
            "rng_bwd": rng_bwd, "held": held, "x_bt_alive": x_bt_alive}


def _layers():
    return [tl.make_layer(H, seed=0), tl.make_layer(KV, seed=1)]


@pytest.mark.parametrize("r_grad", [True, False])
@pytest.mark.parametrize("a2_on", [True, False])
def test_pair_matches_v8b_bit_for_bit_and_holds_less(patched, r_grad, a2_on):
    norm, layers, idx = _norm(), _layers(), _idx()
    r0 = torch.randn(idx.numel(), H, device=DEV)
    ref = step(norm, layers, r0, idx, m2_on=False, r_grad=r_grad, a2_on=a2_on)
    c = dict(m2.COUNTS)
    new = step(norm, layers, r0, idx, m2_on=True, r_grad=r_grad, a2_on=a2_on)
    d = {k: m2.COUNTS[k] - c[k] for k in c}
    assert d["scopes"] == 1 and d["recompute"] == 2 and d["original"] == 0
    assert d["packed"] == 2  # both LoRA-A inputs became recipes
    assert d["rebuilt"] == 1 and d["stashed"] == 1  # one x_bt rebuild served q and v

    assert torch.equal(new["rng_fwd"], ref["rng_fwd"])  # the same draws in the forward
    assert torch.equal(new["rng_bwd"], new["rng_fwd"])  # the replay put the generator back
    for a, b in zip(new["outs"], ref["outs"], strict=True):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert new["grads"].keys() == ref["grads"].keys()
    assert len(ref["grads"]) == 4 + int(r_grad)
    for k in ref["grads"]:
        assert torch.equal(new["grads"][k], ref["grads"][k]), k  # the same bits, not just close
    assert not new["x_bt_alive"]  # recipes keep r (alive anyway), never x_bt

    rows = BATCH * SEQ
    expect = 2 * rows * H * 2  # two fp16 [P, 1536] LoRA-A inputs (no D in this test)
    saved = ref["held"] - new["held"]
    assert saved >= 0.75 * expect, (ref["held"], new["held"], expect)


def test_retain_graph_second_backward(patched):
    norm, layers, idx = _norm(), _layers(), _idx()
    r0 = torch.randn(idx.numel(), H, device=DEV)
    c = dict(m2.COUNTS)
    step(norm, layers, r0, idx, m2_on=True, r_grad=True, twice=True)
    # First backward: one pair rebuild. Second: each recipe rebuilds alone.
    assert m2.COUNTS["rebuilt"] - c["rebuilt"] == 3
    assert m2.COUNTS["stashed"] - c["stashed"] == 1


def test_inplace_change_of_r_is_caught(patched):
    norm, layers, idx = _norm(), _layers(), _idx()
    r0 = torch.randn(idx.numel(), H, device=DEV)
    with pytest.raises(RuntimeError, match="inplace"):
        step(norm, layers, r0, idx, m2_on=True, r_grad=False, mutate=True)


def _call(layer, x, scope_x=None, norm=None, r=None, idx=None, **kwargs):
    """One LoRA call inside a scope built for scope_x (default: x itself)."""
    scope_x = x if scope_x is None else scope_x
    with torch.autocast("cuda", dtype=kwargs.pop("dtype", torch.float16),
                        enabled=kwargs.pop("autocast", True)):
        with m2.source_scope(norm, r, scope_x, idx, mq.Qwen2RMSNorm):
            return layer(x, **kwargs)


@pytest.mark.parametrize("case", ["flag_off", "no_scope", "other_tensor", "hooked_dropout",
                                  "autocast_off", "disabled", "adapter_names",
                                  "eval", "no_grad", "hooked_norm"])
def test_fallbacks_run_the_previous_forward(patched, case):
    norm, layer, idx = _norm(), tl.make_layer(H), _idx()
    r = torch.randn(idx.numel(), H, device=DEV, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        x_bt = unpad.unpack_rows(norm(r), idx, BATCH, SEQ)
    kwargs = {}
    scope_x = None
    if case == "flag_off":
        m2.USE_LORA_INPUT_RECOMPUTE = False
    elif case == "other_tensor":
        scope_x = x_bt.detach().clone()
    elif case == "hooked_dropout":
        layer.lora_dropout["default"].register_forward_hook(lambda m, a, o: None)
    elif case == "autocast_off":
        kwargs["autocast"] = False
    elif case == "disabled":
        layer.enable_adapters(False)
    elif case == "adapter_names":
        kwargs["adapter_names"] = ["default"] * BATCH
    elif case == "eval":
        layer.eval()
    elif case == "hooked_norm":
        norm.register_forward_pre_hook(lambda m, a: None)
    c = dict(m2.COUNTS)
    with torch.set_grad_enabled(case != "no_grad"):
        if case == "no_scope":
            with torch.autocast("cuda", dtype=torch.float16):
                layer(x_bt)
        else:
            _call(layer, x_bt, scope_x=scope_x, norm=norm, r=r, idx=idx, **kwargs)
    assert m2.COUNTS["recompute"] == c["recompute"]
    assert m2.COUNTS["packed"] == c["packed"]
    assert m2.COUNTS["original"] == c["original"] + 1


def test_scope_refuses_a_non_k3_norm(patched, monkeypatch):
    norm, idx = _norm(), _idx()
    r = torch.randn(idx.numel(), H, device=DEV)
    x_bt = unpad.unpack_rows(norm(r), idx, BATCH, SEQ)
    monkeypatch.setattr(mq.Qwen2RMSNorm, "forward", _unwrap(mq.Qwen2RMSNorm.forward, unpad._K3_MARK))
    c = m2.COUNTS["scopes"]
    with m2.source_scope(norm, r, x_bt, idx, mq.Qwen2RMSNorm):
        assert m2._SOURCE.get() is None
    assert m2.COUNTS["scopes"] == c


def test_recipes_hold_no_tensor_but_r_and_idx(patched):
    """A recipe's source keeps r (an alias: input_layernorm keeps it anyway) and idx (kept by
    U's row nodes); x_bt is dropped when the scope closes."""
    norm, idx = _norm(), _idx()
    r = torch.randn(idx.numel(), H, device=DEV)
    x_bt = unpad.unpack_rows(norm(r), idx, BATCH, SEQ)
    with m2.source_scope(norm, r, x_bt, idx, mq.Qwen2RMSNorm):
        src = m2._SOURCE.get()
        assert src is not None and src.x_bt is x_bt
    assert m2._SOURCE.get() is None and src.x_bt is None
    tensors = [getattr(src, s) for s in src.__slots__ if isinstance(getattr(src, s), torch.Tensor)]
    assert {t.data_ptr() for t in tensors} == {r.data_ptr(), idx.data_ptr()}


# ---------------------------------------------------------------------------------------------
# Whole stack: v8b's switches with and without M2
# ---------------------------------------------------------------------------------------------

def test_stack_with_m2_matches_v8b_and_holds_less(v6_patches, monkeypatch):
    patched_forward = v6_patches
    if tm._requested_bytes() is not None:  # tm._run uses memory_allocated(): count exact sizes
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: tm._requested_bytes())
    cls = lora_bnb.Linear4bit
    monkeypatch.setattr(cls, "forward", cls.forward)  # A2's, installed by the fixture
    assert m2.patch_lora_input_recompute()
    monkeypatch.setattr(m2, "USE_LORA_INPUT_RECOMPUTE", False)
    pm = tm.build()
    n_layers = pm.config.num_hidden_layers
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    tm._set(True, causal=True)  # B + A1 + A2 + D + I1, A off: what v8b's apply() switches on

    ref_a = tm._run(pm, batch, patched_forward, extra)
    ref_b = tm._run(pm, batch, patched_forward, extra)
    m2.USE_LORA_INPUT_RECOMPUTE = True
    c_m2, c_a2, c_sc, c_u = dict(m2.COUNTS), dict(a2.COUNTS), dict(sc.COUNTS), dict(unpad.COUNTS)
    new = tm._run(pm, batch, patched_forward, extra)
    d_m2 = {k: m2.COUNTS[k] - c_m2[k] for k in c_m2}
    assert unpad.COUNTS["packed"] - c_u["packed"] == 1
    assert d_m2["scopes"] == n_layers
    # Every q/v call (layer 0 too: its input needs no grad, its lora_A weights do).
    assert d_m2["recompute"] == 2 * n_layers and d_m2["original"] == 0
    assert d_m2["packed"] == 2 * n_layers
    assert d_m2["rebuilt"] == n_layers and d_m2["stashed"] == n_layers
    assert a2.COUNTS["replay"] == c_a2["replay"]  # M2 calls A2's dropout node itself
    assert sc.COUNTS["rows_packed"] == c_sc["rows_packed"]  # nothing left for D

    assert torch.equal(new["rng_fwd"], ref_a["rng_fwd"])
    assert torch.equal(new["rng_bwd"], new["rng_fwd"])
    tp._assert_matches(ref_a["hidden"], ref_b["hidden"], new["hidden"], "hidden")
    tp._assert_matches(ref_a["loss"], ref_b["loss"], new["loss"], "loss")
    assert new["grads"].keys() == ref_a["grads"].keys()
    for name in ref_a["grads"]:
        tp._assert_grad_close(ref_a["grads"][name], ref_b["grads"][name], new["grads"][name], name)

    n = int(batch["attention_mask"].sum())
    expect = n_layers * 2 * n * H * 2  # D's two fp16 [N, 1536] LoRA-A inputs per layer
    saved = ref_a["held"] - new["held"]
    assert saved >= 0.9 * expect, (ref_a["held"], new["held"], expect)
