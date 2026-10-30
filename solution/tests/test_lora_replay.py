"""A2 (solution/lora_replay.py): peft's 4-bit LoRA layer with the replayed dropout gives the same
output, the same RNG stream and the same gradients as peft's own forward, keeps no dropout mask,
and leaves anything unexpected on peft's forward.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")

from peft.tuners.lora import bnb as lora_bnb  # noqa: E402

import solution.lora_replay as a2  # noqa: E402

DEV = "cuda"
H = 1536
KV = 256  # v_proj width of Qwen2.5-1.5B
SHAPE = (4, 128, H)


def _orig(fn):
    return getattr(fn, "__wrapped__", fn)


@pytest.fixture
def patched(monkeypatch):
    cls = lora_bnb.Linear4bit
    monkeypatch.setattr(cls, "forward", _orig(cls.forward))
    assert a2.patch_lora_dropout_replay()
    installed = cls.forward
    assert a2.patch_lora_dropout_replay() and cls.forward is installed  # idempotent
    monkeypatch.setattr(a2, "USE_LORA_DROPOUT_REPLAY", True)
    return cls


def make_layer(out_features=H, dropout=0.05, seed=0):
    torch.manual_seed(seed)
    base = bnb.nn.Linear4bit(H, out_features, bias=True, compute_dtype=torch.float16,
                             compress_statistics=True, quant_type="nf4")
    # init_lora_weights=False: lora_B is not zero, so x and lora_A get a real gradient.
    cfg = peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout, init_lora_weights=False)
    layer = lora_bnb.Linear4bit(base, "default", config=cfg, r=16, lora_alpha=32)
    layer = layer.cuda().train()
    for name, p in layer.named_parameters():
        p.requires_grad_("lora_" in name)
    return layer


def step(layers, x0, enabled, seed=1234, autocast=True, hooks=()):
    """q and v style calls on the same input, backward, everything compared later."""
    a2.USE_LORA_DROPOUT_REPLAY = enabled
    for layer in layers:
        layer.zero_grad(set_to_none=True)
    x = x0.clone().requires_grad_(True)
    saved = {}

    def pack(t):
        st = t.untyped_storage()
        saved[st.data_ptr()] = (st.nbytes(), t.dtype)
        return t

    torch.manual_seed(seed)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            outs = [layer(x) for layer in layers]
            _ = torch.rand(7, device=DEV)  # an unrelated draw between forward and backward
    rng_fwd = torch.cuda.get_rng_state()
    gen = torch.Generator(device=DEV).manual_seed(99)
    loss = sum((o.float() * torch.randn(o.shape, device=DEV, generator=gen)).sum() for o in outs)
    loss.backward()
    rng_bwd = torch.cuda.get_rng_state()
    grads = {f"{i}.{n}": p.grad.clone() for i, layer in enumerate(layers)
             for n, p in layer.named_parameters() if p.grad is not None}
    return [o.detach() for o in outs], x.grad.clone(), grads, rng_fwd, rng_bwd, saved


@pytest.mark.parametrize("autocast", [True, False])
def test_matches_peft_and_drops_the_masks(patched, autocast):
    layers = [make_layer(H, seed=0), make_layer(KV, seed=1)]
    x0 = torch.randn(SHAPE, device=DEV)
    o0, dx0, g0, f0, b0, s0 = step(layers, x0, enabled=False, autocast=autocast)
    n = a2.COUNTS["replay"]
    o1, dx1, g1, f1, b1, s1 = step(layers, x0, enabled=True, autocast=autocast)
    assert a2.COUNTS["replay"] == n + 2

    assert torch.equal(f1, f0)  # the forward consumed the generator exactly like peft's
    assert torch.equal(b1, f1) and torch.equal(b0, f0)  # backward draws nothing (net)
    for a, b in zip(o1, o0):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert g1.keys() == g0.keys() and len(g0) == 4
    for k in g0:
        # Same masks and the same arithmetic: identical. rtol 1e-6 only allows a one-ulp difference
        # between dropout's forward and backward scale constants (both float(1/(1-p))).
        torch.testing.assert_close(g1[k], g0[k], rtol=1e-6, atol=0)
    torch.testing.assert_close(dx1, dx0, rtol=1e-6, atol=1e-12)

    # peft keeps one bool mask of x's shape per dropout call; we keep none.
    masks0 = [nb for nb, dt in s0.values() if dt == torch.bool]
    masks1 = [nb for nb, dt in s1.values() if dt == torch.bool]
    assert sorted(masks0) == [x0.numel()] * 2 and masks1 == []
    assert sum(nb for nb, _ in s0.values()) - sum(nb for nb, _ in s1.values()) >= 2 * x0.numel()


def test_replay_mask_is_the_forward_mask(patched):
    """The gradient through the dropout is g * keep / (1 - p) with the forward's own keep mask."""
    layer = make_layer(H)
    a2.USE_LORA_DROPOUT_REPLAY = True
    x = torch.randn(SHAPE, device=DEV, requires_grad=True)
    state = torch.cuda.get_rng_state()
    y = a2._ReplayDropout.apply(x, 0.05)
    keep = y.detach().ne(0)  # randn input: no exact zeros
    g = torch.randn_like(x)
    y.backward(g)
    with torch.random.fork_rng(devices=[x.device.index]):
        torch.cuda.set_rng_state(state, x.device)
        ref = torch.nn.functional.dropout(torch.ones_like(x), p=0.05, training=True)
    assert torch.equal(ref.ne(0), keep)
    assert torch.equal(x.grad, g * ref)
    del layer


@pytest.mark.parametrize("case", ["hooked_dropout", "eval", "no_dropout", "no_grad", "flag_off",
                                  "adapter_names", "disabled", "misaligned"])
def test_fallbacks_run_peft(patched, case):
    layer = make_layer(H, dropout=0.0 if case == "no_dropout" else 0.05)
    x = torch.randn(SHAPE, device=DEV, requires_grad=True)
    kwargs = {}
    if case == "hooked_dropout":
        layer.lora_dropout["default"].register_forward_hook(lambda m, a, o: None)
    elif case == "eval":
        layer.eval()
    elif case == "flag_off":
        a2.USE_LORA_DROPOUT_REPLAY = False
    elif case == "adapter_names":
        kwargs["adapter_names"] = ["default"] * SHAPE[0]
    elif case == "disabled":
        layer.enable_adapters(False)
    elif case == "misaligned":
        n = SHAPE[0] * SHAPE[1] * H
        x = torch.randn(n + 4, device=DEV).narrow(0, 1, n).view(SHAPE).detach().requires_grad_(True)
        assert x.data_ptr() % 16 != 0
    before = dict(a2.COUNTS)
    with torch.set_grad_enabled(case != "no_grad"), torch.autocast("cuda", dtype=torch.float16):
        layer(x, **kwargs)
    assert a2.COUNTS["replay"] == before["replay"]
    assert a2.COUNTS["original"] == before["original"] + 1
