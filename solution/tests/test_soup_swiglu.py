"""v8 C2: soup_swiglu_kernel gives the example SwiGLU's bits, forward, backward and in place, and
with USE_SOUP_SWIGLU on K4 and S4 never launch the example's kernels.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import inspect

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
pytest.importorskip("triton")

import solution.swiglu as example  # noqa: E402
from solution.kernels import swiglu as ours  # noqa: E402

SHAPES = [(1, 8960), (77, 8960), (2, 300, 8960), (3, 77, 1001), (5,), (1,)]


def _same(a, b):
    """Same dtype, shape and bytes (-0.0 and NaN payloads included)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8))


def _inputs(shape, dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    g = torch.randn(shape, device="cuda", generator=gen) * 4
    flat = g.view(-1)
    extremes = torch.tensor([0.0, -0.0, 20.0, -20.0, 60000.0, -60000.0, 1e-6, -1e-6],
                            device="cuda")
    flat[: min(flat.numel(), extremes.numel())] = extremes[: flat.numel()]
    u = torch.randn(shape, device="cuda", generator=gen)
    dh = torch.randn(shape, device="cuda", generator=gen)
    return g.to(dtype), u.to(dtype), dh.to(dtype)


def _fwd_bwd(fn, g0, u0, dh):
    g = g0.clone().requires_grad_(True)
    u = u0.clone().requires_grad_(True)
    out = fn(g, u)
    out.backward(dh)
    return out.detach(), g.grad, u.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", SHAPES)
def test_same_bits_as_example(dtype, shape):
    g, u, dh = _inputs(shape, dtype, seed=sum(shape))
    ref = _fwd_bwd(example.swiglu, g, u, dh)
    got = _fwd_bwd(ours.swiglu, g, u, dh)
    for a, b in zip(ref, got):
        assert _same(a, b)


def test_mixed_dtypes_and_strides_as_example():
    """up in fp32 is cast to gate's fp16 and a transposed gate is made contiguous, as the example
    does (solution/swiglu.py:45-46, :56)."""
    g, _, dh = _inputs((8960, 40), torch.float16, seed=3)
    u32 = torch.randn(40, 8960, device="cuda")
    ref = _fwd_bwd(example.swiglu, g.t(), u32, dh.t())
    got = _fwd_bwd(ours.swiglu, g.t(), u32, dh.t())
    for a, b in zip(ref, got):
        assert _same(a, b)


@pytest.mark.parametrize("tokens", [1, 77, 2048])
def test_inplace_backward_same_bits(tokens):
    g, u, dh = _inputs((tokens, 8960), torch.float16, seed=tokens)
    n = g.numel()
    # The example kernel in place, launched as mlp_recompute._swiglu_backward_inplace does (v7).
    dh_e, g_e = dh.clone(), g.clone()
    example.example_swiglu_bwd_kernel[((n + example.BLOCK - 1) // example.BLOCK,)](
        dh_e, g_e, u, dh_e, g_e, n, BLOCK=example.BLOCK)
    dh_o, g_o, u_o = dh.clone(), g.clone(), u.clone()
    dg, du = ours.swiglu_backward_into(dh_o, g_o, u_o)
    assert dg.data_ptr() == dh_o.data_ptr() and du.data_ptr() == g_o.data_ptr()
    assert _same(dg, dh_e) and _same(du, g_e)
    assert _same(u_o, u)  # up is never written
    dg2, du2 = ours.swiglu_backward(dh, g, u)  # out of place: the same bits
    assert _same(dg2, dg) and _same(du2, du)


def _compiled_count(kernel):
    caches = getattr(kernel, "device_caches", None)
    if isinstance(caches, dict):
        total = 0
        for entry in caches.values():
            entry = entry[0] if isinstance(entry, tuple) else entry
            if not isinstance(entry, dict):
                pytest.skip("unknown Triton cache layout")
            total += len(entry)
        return total
    pytest.skip("unknown Triton cache layout")


def test_forward_backward_and_inplace_share_one_variant():
    """Training sizes (tokens * 8960) never add a compiled variant after the first call."""
    counts = []
    for tokens in (37, 300, 1, 1349):
        g, u, dh = _inputs((tokens, 8960), torch.float16, seed=tokens)
        _fwd_bwd(ours.swiglu, g, u, dh)
        ours.swiglu_backward_into(dh.clone(), g.clone(), u)
        counts.append(_compiled_count(ours.soup_swiglu_kernel))
    assert counts[0] >= 1 and counts == [counts[0]] * len(counts)


# ------------------------------------------------------------------------------------------------
# K4 / S4 with the switch: same bits as the example path, and the example is never launched
# ------------------------------------------------------------------------------------------------

HIDDEN, INTER = 256, 512


@pytest.fixture
def mlp_patched(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    from transformers.models.qwen2 import modeling_qwen2

    import solution.kernels.nf4 as nf4
    import solution.mlp as k4
    import solution.mlp_recompute as rc

    base = inspect.unwrap(modeling_qwen2.Qwen2MLP.forward)
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", base)
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", False)
    monkeypatch.setattr(rc, "RECOMPUTE_GATE", True)
    monkeypatch.setattr(rc, "SWIGLU_BACKWARD", "example")
    monkeypatch.setattr(rc, "INPLACE_SWIGLU_BACKWARD", True)
    monkeypatch.setattr(rc, "_warned", False)
    monkeypatch.setattr(k4, "USE_SOUP_SWIGLU", False)
    monkeypatch.setattr(bnb, "matmul_4bit", nf4.soup_matmul_4bit)  # K1, as apply() installs it
    rc.patch_qwen2_mlp_recompute()
    return bnb, modeling_qwen2, k4, rc


@pytest.fixture
def launches(monkeypatch):
    from triton.runtime.jit import JITFunction

    seen = []
    original = JITFunction.run

    def run(self, *args, **kwargs):
        seen.append(getattr(getattr(self, "fn", None), "__name__", "?"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", run)
    return seen


def _make_mlp(bnb, modeling_qwen2):
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    torch.manual_seed(0)
    mlp = modeling_qwen2.Qwen2MLP(Qwen2Config(hidden_size=HIDDEN, intermediate_size=INTER,
                                              hidden_act="silu"))
    for name, (n_in, n_out) in {"gate_proj": (HIDDEN, INTER), "up_proj": (HIDDEN, INTER),
                                "down_proj": (INTER, HIDDEN)}.items():
        setattr(mlp, name, bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                                             compress_statistics=True, quant_type="nf4"))
    return mlp.cuda()


@pytest.mark.parametrize("s4, reuse", [(False, "off"), (True, "off"), (True, "both")])
def test_k4_s4_same_bits_without_the_example(mlp_patched, monkeypatch, launches, s4, reuse):
    bnb, modeling_qwen2, k4, rc = mlp_patched
    monkeypatch.setattr(rc, "S4_DEQUANT_REUSE", reuse)
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", s4)
    inplace = []
    monkeypatch.setattr(ours, "swiglu_backward_into",
                        lambda *a, _f=ours.swiglu_backward_into: inplace.append(1) or _f(*a))
    mlp = _make_mlp(bnb, modeling_qwen2)
    x = torch.randn(300, HIDDEN, device="cuda")
    grad = torch.randn_like(x)

    def run(soup):
        monkeypatch.setattr(k4, "USE_SOUP_SWIGLU", soup)
        xi = x.clone().requires_grad_(True)
        out = mlp(xi)
        out.backward(grad)
        return out.detach(), xi.grad

    ref = run(False)
    assert any(name.startswith("example_swiglu") for name in launches)  # v7 path
    del launches[:]
    got = run(True)
    assert all(_same(a, b) for a, b in zip(ref, got))
    assert "soup_swiglu_kernel" in launches
    assert not any(name.startswith("example_") for name in launches)
    assert len(inplace) == (1 if s4 else 0)  # S4 recomputed gate: dgate over dh, dup over gate
