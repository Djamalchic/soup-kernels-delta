"""K3: fused add + RMSNorm matches base Qwen2RMSNorm / Qwen2DecoderLayer math (fp32 residual,
frozen fp32 weights) in outputs, dtypes and gradients, and falls back on anything unexpected.
Every reference is computed on the unpatched classes before our patches are installed.

The real-attention decoder test and the spy-based fallback test are adapted from the GPT-6-Astra
K3 draft. Run from the repository root: python -m pytest solution -q
"""

import inspect

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from transformers.models.qwen2 import modeling_qwen2  # noqa: E402
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # noqa: E402

import solution.layers as layers  # noqa: E402
from solution.kernels import addnorm  # noqa: E402

H = addnorm.HIDDEN_SIZE
EPS = 1e-6
TOL = dict(rtol=1e-4, atol=1e-5)
NORM = modeling_qwen2.Qwen2RMSNorm
LAYER = modeling_qwen2.Qwen2DecoderLayer


def _orig(fn):
    return getattr(fn, "__wrapped__", fn)


@pytest.fixture
def install(monkeypatch):
    """Base forwards for the reference; calling the returned function installs our patches.
    monkeypatch puts back whatever was there before the test."""
    for cls in (NORM, LAYER):
        monkeypatch.setattr(cls, "forward", _orig(cls.forward))

    def _install():
        layers.patch_qwen2_rmsnorm()
        layers.patch_qwen2_decoder_layer()

    return _install


def _norm(seed=0, eps=EPS):
    torch.manual_seed(seed)
    m = NORM(H, eps=eps).cuda()
    with torch.no_grad():
        m.weight.copy_(1.0 + 0.1 * torch.randn(H))
    m.weight.requires_grad_(False)
    return m


def _frozen_norm_weights(layer):
    with torch.no_grad():
        for ln in (layer.input_layernorm, layer.post_attention_layernorm):
            ln.weight.copy_(1.0 + 0.1 * torch.randn(H, device=ln.weight.device))
            ln.weight.requires_grad_(False)


class _Attn(torch.nn.Module):
    """Stand-in for Qwen2Attention: same call contract (kwargs in, (out, weights) back)."""

    def __init__(self, out_dtype):
        super().__init__()
        self.proj = torch.nn.Linear(H, H, bias=False)
        self.out_dtype = out_dtype

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states).to(self.out_dtype), None


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(H, H, bias=False)

    def forward(self, x):
        return torch.tanh(self.fc(x))


def _stand_in_layer(delta_dtype=torch.float32):
    torch.manual_seed(1)
    cfg = Qwen2Config(hidden_size=H, intermediate_size=64, num_attention_heads=12,
                      num_key_value_heads=2, num_hidden_layers=1, vocab_size=32, rms_norm_eps=EPS)
    layer = LAYER(cfg, layer_idx=0)
    layer.self_attn = _Attn(delta_dtype)
    layer.mlp = _MLP()
    layer = layer.cuda()
    _frozen_norm_weights(layer)
    return layer


def _run(module, x, **kwargs):
    x = x.clone().requires_grad_(True)
    out = module(x, **kwargs)
    torch.manual_seed(7)
    out.backward(torch.randn_like(out))
    grads = [p.grad.clone() for p in module.parameters() if p.requires_grad]
    for p in module.parameters():
        p.grad = None
    return out.detach(), x.grad, grads


@pytest.mark.parametrize("shape", [(1, 7, H), (2, 33, H), (4, 512, H)])
def test_rmsnorm_matches_base(install, shape):
    m = _norm()
    x = 3.0 * torch.randn(shape, device="cuda")
    y0, dx0, _ = _run(m, x)
    install()
    y1, dx1, _ = _run(m, x)
    assert y1.dtype == y0.dtype == torch.float32
    torch.testing.assert_close(y1, y0, **TOL)
    torch.testing.assert_close(dx1, dx0, **TOL)
    assert m.weight.grad is None


@pytest.mark.parametrize("delta_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 512, H), (3, 77, H)])
def test_add_rmsnorm_outputs_and_grads(delta_dtype, shape):
    m = _norm()  # class not patched here: m(...) is the base norm
    torch.manual_seed(3)
    r0 = torch.randn(shape, device="cuda", requires_grad=True)
    d0 = torch.randn(shape, device="cuda", dtype=delta_dtype, requires_grad=True)
    r1 = r0.detach().clone().requires_grad_(True)
    d1 = d0.detach().clone().requires_grad_(True)

    ref_r = r0 + d0
    ref_y = _orig(NORM.forward)(m, ref_r)
    res, y = addnorm.add_rmsnorm(r1, d1, m.weight, EPS)
    assert res.dtype == y.dtype == torch.float32
    torch.testing.assert_close(res, ref_r, rtol=0, atol=0)  # the add is bit-exact
    torch.testing.assert_close(y, ref_y, **TOL)

    gr, gy = 0.2 * torch.randn_like(ref_r), 0.2 * torch.randn_like(ref_y)
    torch.autograd.backward((ref_r, ref_y), (gr, gy))
    torch.autograd.backward((res, y), (gr, gy))
    torch.testing.assert_close(r1.grad, r0.grad, **TOL)
    assert d1.grad.dtype == delta_dtype
    tol = dict(rtol=2e-3, atol=1e-3) if delta_dtype == torch.float16 else TOL
    torch.testing.assert_close(d1.grad, d0.grad, **tol)
    assert m.weight.grad is None


@pytest.mark.parametrize("mode", ["residual_only", "norm_only", "delta_frozen"])
def test_add_rmsnorm_partial_grads(mode):
    """dy or dr absent (set_materialize_grads(False)) and an input without grad."""
    m = _norm()
    torch.manual_seed(5)
    r0 = torch.randn(2, 5, H, device="cuda", requires_grad=True)
    d0 = torch.randn(2, 5, H, device="cuda", dtype=torch.float16,
                     requires_grad=(mode != "delta_frozen"))
    r1 = r0.detach().clone().requires_grad_(True)
    d1 = d0.detach().clone().requires_grad_(d0.requires_grad)

    ref_r = r0 + d0
    ref_y = _orig(NORM.forward)(m, ref_r)
    res, y = addnorm.add_rmsnorm(r1, d1, m.weight, EPS)
    ref_out, out = (ref_r, res) if mode == "residual_only" else (ref_y, y)
    seed = 0.2 * torch.randn_like(ref_out)
    ref_out.backward(seed)
    out.backward(seed)
    torch.testing.assert_close(r1.grad, r0.grad, **TOL)
    if mode == "delta_frozen":
        assert d1.grad is None
    else:
        torch.testing.assert_close(d1.grad, d0.grad, rtol=2e-3, atol=1e-3)


@pytest.mark.parametrize("delta_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1, 5, H), (2, 64, H)])
def test_stand_in_decoder_layer_matches_base(install, delta_dtype, shape):
    layer = _stand_in_layer(delta_dtype)
    x = torch.randn(shape, device="cuda")
    y0, dx0, g0 = _run(layer, x)
    install()
    y1, dx1, g1 = _run(layer, x)
    assert y1.dtype == y0.dtype == torch.float32
    torch.testing.assert_close(y1, y0, **TOL)
    torch.testing.assert_close(dx1, dx0, **TOL)
    assert len(g0) == len(g1) == 2  # attn proj + mlp fc; norm weights frozen
    for a, b in zip(g1, g0):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)


def _real_layer():
    torch.manual_seed(109)
    cfg = Qwen2Config(vocab_size=32, hidden_size=H, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                      rms_norm_eps=EPS)
    cfg._attn_implementation = "eager"  # 768-wide heads; setter configuration_utils.py:421
    layer = LAYER(cfg, layer_idx=0).cuda().eval()
    layer.requires_grad_(False)
    _frozen_norm_weights(layer)
    rotary = modeling_qwen2.Qwen2RotaryEmbedding(config=cfg).cuda()
    return layer, rotary


@pytest.mark.parametrize("autocast", [False, True])
def test_real_decoder_layer_matches_base(install, autocast):
    """Real Qwen2Attention: under fp16 autocast the attention output reaching the add is fp16."""
    layer, rotary = _real_layer()
    torch.manual_seed(131)
    x = torch.randn(2, 7, H, device="cuda")
    pos = torch.arange(7, device="cuda").unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        emb = rotary(x, pos)
    mask = torch.full((7, 7), torch.finfo(torch.float32).min, device="cuda").triu(1)[None, None]
    kw = dict(attention_mask=mask, position_ids=pos, past_key_values=None, use_cache=False,
              position_embeddings=emb)
    seed = 0.1 * torch.randn_like(x)

    seen = []
    layer.self_attn.register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append(dict(kwargs)), with_kwargs=True)
    deltas = []
    layer.self_attn.register_forward_hook(lambda mod, args, out: deltas.append(out[0].dtype))

    def run():
        xi = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            y = layer(xi, **kw)
        return y, torch.autograd.grad(y, xi, grad_outputs=seed)[0]

    y0, dx0 = run()
    install()
    sig0 = inspect.signature(_orig(LAYER.forward), follow_wrapped=False)
    sig1 = inspect.signature(LAYER.forward, follow_wrapped=False)
    assert [(p.name, p.kind, p.default) for p in sig1.parameters.values()] == \
        [(p.name, p.kind, p.default) for p in sig0.parameters.values()]
    installed = LAYER.forward
    install()  # idempotent
    assert LAYER.forward is installed
    y1, dx1 = run()

    assert deltas == [torch.float16 if autocast else torch.float32] * 2
    assert seen[0].keys() == seen[1].keys()  # the same arguments reach self_attn
    for k in ("attention_mask", "position_ids", "position_embeddings"):
        assert seen[1][k] is seen[0][k]
    assert isinstance(y1, torch.Tensor) and y1.dtype == y0.dtype == torch.float32
    tol = dict(rtol=3e-3, atol=3e-3) if autocast else dict(rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(y1, y0, **tol)
    torch.testing.assert_close(dx1, dx0, **(dict(rtol=4e-3, atol=4e-3) if autocast else tol))


def test_norm_fallbacks_run_original(install):
    m = _norm()
    base = _orig(NORM.forward)
    install()
    # CPU, fp16 input, trainable weight, other width: original path, bit-identical to base.
    mc = _norm().cpu()
    x = torch.randn(2, 3, H)
    torch.testing.assert_close(NORM.forward(mc, x), base(mc, x), rtol=0, atol=0)
    x = torch.randn(2, 3, H, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(NORM.forward(m, x), base(m, x), rtol=0, atol=0)
    m.weight.requires_grad_(True)
    x = torch.randn(4, H, device="cuda")
    torch.testing.assert_close(NORM.forward(m, x), base(m, x), rtol=0, atol=0)
    small = NORM(64).cuda().requires_grad_(False)
    xs = torch.randn(4, 64, device="cuda")
    torch.testing.assert_close(NORM.forward(small, xs), base(small, xs), rtol=0, atol=0)


def test_decoder_fallbacks_call_original_with_all_kwargs(install, monkeypatch):
    calls = []
    sentinel = object()

    def spy(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return sentinel

    monkeypatch.setattr(LAYER, "forward", spy)  # undone after the test by monkeypatch
    layers.patch_qwen2_decoder_layer()
    layer = _stand_in_layer()
    good = torch.randn(1, 2, H, device="cuda")
    kw = dict(attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
              position_embeddings=None, cache_position=None, soup_probe="preserve-me")

    def expect_original(inp):
        n = len(calls)
        assert layer(inp, **kw) is sentinel
        assert len(calls) == n + 1
        self_, args, kwargs = calls[-1]
        assert self_ is layer and args[0] is inp and kwargs == kw

    expect_original(good.cpu())
    expect_original(good.half())
    expect_original(good[..., : H - 1])

    ln2 = layer.post_attention_layernorm
    with monkeypatch.context() as ctx:
        ctx.setattr(layer, "post_attention_layernorm", torch.nn.Identity())
        expect_original(good)
    for size, dtype, trainable in ((H, torch.float32, True), (H, torch.float16, False),
                                   (H - 1, torch.float32, False)):
        with monkeypatch.context() as ctx:
            w = torch.nn.Parameter(torch.ones(size, device="cuda", dtype=dtype),
                                   requires_grad=trainable)
            ctx.setattr(ln2, "weight", w)
            expect_original(good)
    for register in (lambda: ln2.register_forward_hook(lambda *a: None),
                     lambda: ln2.register_forward_pre_hook(lambda *a: None),
                     lambda: ln2.register_full_backward_hook(lambda *a: None),
                     lambda: ln2.register_full_backward_pre_hook(lambda *a: None)):
        handle = register()
        expect_original(good)
        handle.remove()
    ln2.forward = lambda h: h  # instance-level override (e.g. accelerate's AlignDevicesHook)
    expect_original(good)
    del ln2.forward


def test_input_layernorm_hooks_fire_on_fused_path(install):
    layer = _stand_in_layer()
    x = torch.randn(1, 4, H, device="cuda")
    y0 = layer(x)
    install()
    seen = []
    layer.input_layernorm.register_forward_hook(lambda *a: seen.append(1))
    y1 = layer(x)
    assert seen == [1]
    torch.testing.assert_close(y1, y0, **TOL)


def test_patch_is_idempotent_and_named(install):
    install()
    f_norm, f_layer = NORM.forward, LAYER.forward
    install()
    assert NORM.forward is f_norm and LAYER.forward is f_layer
    for k in (addnorm.soup_addnorm_fwd_kernel, addnorm.soup_addnorm_bwd_kernel):
        assert k.fn.__name__.startswith("soup_")


def test_empty_rows(install):
    install()
    m = _norm()
    x = torch.empty(0, 0, H, device="cuda", requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert y.shape == x.shape and x.grad.shape == x.shape
