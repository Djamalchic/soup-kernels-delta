"""S4: the recompute MLP gives K4's outputs and input gradients bit for bit, saves less for backward,
and leaves anything unexpected on the K4 (or original) path.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import contextlib
import gc
import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("triton")

from transformers.models.qwen2 import modeling_qwen2  # noqa: E402
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # noqa: E402

import solution.kernels.nf4 as nf4  # noqa: E402
import solution.mlp as k4  # noqa: E402
import solution.mlp_recompute as rc  # noqa: E402
import solution.swiglu as example  # noqa: E402

HIDDEN, INTER = 256, 512
ORIGINAL_MATMUL_4BIT = nf4._get_original_matmul_4bit()
MiB = 1 << 20


def make_mlp(act="silu", four_bit=True, inter=INTER, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=HIDDEN, intermediate_size=inter, hidden_act=act)
    mlp = modeling_qwen2.Qwen2MLP(cfg)
    if four_bit:
        for name, (n_in, n_out) in {"gate_proj": (HIDDEN, inter), "up_proj": (HIDDEN, inter),
                                    "down_proj": (inter, HIDDEN)}.items():
            layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                                      compress_statistics=True, quant_type="nf4")
            setattr(mlp, name, layer)
    return mlp.cuda()


def uses_s4(tensor, depth=4):
    """True if SoupMLPRecompute's backward node is within `depth` steps of tensor.grad_fn."""
    frontier = [tensor.grad_fn]
    for _ in range(depth):
        nxt = []
        for node in frontier:
            if node is None:
                continue
            if type(node).__name__.startswith("SoupMLPRecompute"):
                return True
            nxt.extend(fn for fn, _ in node.next_functions)
        frontier = nxt
    return False


def set_k1(monkeypatch, on):
    monkeypatch.setattr(bnb, "matmul_4bit", nf4.soup_matmul_4bit if on else ORIGINAL_MATMUL_4BIT)


@pytest.fixture
def base_forward(monkeypatch):
    """Original Qwen2MLP.forward installed, then K4 + S4 on top; restored after each test."""
    base = inspect.unwrap(modeling_qwen2.Qwen2MLP.forward)
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", base)
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", False)
    monkeypatch.setattr(rc, "RECOMPUTE_GATE", True)
    monkeypatch.setattr(rc, "SWIGLU_BACKWARD", "example")
    rc.patch_qwen2_mlp_recompute()
    return base


def run(mlp, x, grad, monkeypatch, s4, autocast=False):
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", s4)
    xi = x.detach().clone().requires_grad_(True)
    ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else contextlib.nullcontext()
    with ctx:  # accelerate wraps only the forward in autocast; backward runs outside it
        out = mlp(xi)
    used = uses_s4(out)
    out.backward(grad)
    return out.detach(), xi.grad, used


@pytest.mark.parametrize("recompute_gate", [True, False])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("k1", [False, True])
@pytest.mark.parametrize("shape", [(1, 3, HIDDEN), (2, 37, HIDDEN), (4, 512, HIDDEN)])
def test_bit_exact_against_k4(base_forward, monkeypatch, shape, k1, autocast, recompute_gate):
    set_k1(monkeypatch, k1)
    monkeypatch.setattr(rc, "RECOMPUTE_GATE", recompute_gate)
    mlp = make_mlp()
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)

    ref_out, ref_grad, ref_used = run(mlp, x, grad, monkeypatch, s4=False, autocast=autocast)
    out, x_grad, used = run(mlp, x, grad, monkeypatch, s4=True, autocast=autocast)

    assert not ref_used and used
    assert out.dtype == ref_out.dtype == torch.float32
    assert torch.equal(out, ref_out)
    assert torch.equal(x_grad, ref_grad)


def test_torch_swiglu_backward_matches_example_kernel():
    torch.manual_seed(1)
    gate = (torch.randn(2048, 1024, device="cuda") * 3).half()
    up = torch.randn(2048, 1024, device="cuda").half()
    dh = torch.randn(2048, 1024, device="cuda").half()

    ref_dg, ref_du = example.SwiGLU.backward(rc._SavedPair(gate, up), dh)
    dg, du = rc._swiglu_backward_torch(dh, gate, up)

    assert dg.dtype == du.dtype == torch.float16
    for new, ref in ((dg, ref_dg), (du, ref_du)):
        torch.testing.assert_close(new, ref, rtol=2e-3, atol=1e-4)
        assert (new != ref).float().mean().item() < 0.01  # rare one-ulp fp16 differences only


def test_torch_swiglu_path_close_to_k4(base_forward, monkeypatch):
    set_k1(monkeypatch, True)
    monkeypatch.setattr(rc, "SWIGLU_BACKWARD", "torch")
    mlp = make_mlp()
    shape = (4, 512, HIDDEN)
    x = torch.randn(shape, device="cuda")
    grad = torch.randn(shape, device="cuda")

    ref_out, ref_grad, _ = run(mlp, x, grad, monkeypatch, s4=False)
    out, x_grad, used = run(mlp, x, grad, monkeypatch, s4=True)

    assert used
    assert torch.equal(out, ref_out)  # the forward does not depend on the backward choice
    torch.testing.assert_close(x_grad, ref_grad, rtol=2e-3,
                               atol=2e-3 * ref_grad.abs().max().item())


def _measure(mlps, x, grad, monkeypatch, s4):
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", s4)
    xi = x.detach().clone().requires_grad_(True)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    h = xi
    for mlp in mlps:  # a residual stack, like decoder layers: only the MLPs save activations
        h = h + mlp(h)
    torch.cuda.synchronize()
    saved = torch.cuda.memory_allocated() - start
    h.backward(grad)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - start
    del h, xi
    return saved, peak


@pytest.mark.parametrize("recompute_gate", [True, False])
def test_saved_and_peak_memory_drop(base_forward, monkeypatch, recompute_gate):
    set_k1(monkeypatch, True)
    monkeypatch.setattr(rc, "RECOMPUTE_GATE", recompute_gate)
    inter, layers = 1024, 6
    mlps = [make_mlp(inter=inter, seed=i) for i in range(layers)]
    shape = (4, 512, HIDDEN)
    tokens = shape[0] * shape[1]
    x = torch.randn(shape, device="cuda")
    grad = torch.randn(shape, device="cuda")
    for s4 in (False, True):  # warm-up: Triton compilation, cuBLAS workspace, K1's LUT
        _measure(mlps, x, grad, monkeypatch, s4)

    saved_k4, peak_k4 = _measure(mlps, x, grad, monkeypatch, s4=False)
    saved_s4, peak_s4 = _measure(mlps, x, grad, monkeypatch, s4=True)

    fp16 = 2
    k4_per_layer = 2 * tokens * inter * fp16  # gate and up outputs
    s4_per_layer = tokens * HIDDEN * fp16 + (0 if recompute_gate else tokens * inter * fp16)
    expected = layers * (k4_per_layer - s4_per_layer)
    print(f"saved K4 {saved_k4 / MiB:.1f} MiB, S4 {saved_s4 / MiB:.1f} MiB; "
          f"peak K4 {peak_k4 / MiB:.1f} MiB, S4 {peak_s4 / MiB:.1f} MiB")
    assert abs((saved_k4 - saved_s4) - expected) <= 0.15 * expected  # allocator block rounding
    assert peak_k4 - peak_s4 >= 0.5 * expected


def test_rng_untouched(base_forward, monkeypatch):
    mlp = make_mlp()
    x = torch.randn(2, 37, HIDDEN, device="cuda")
    grad = torch.randn_like(x)
    state = torch.cuda.get_rng_state()
    _, _, used = run(mlp, x, grad, monkeypatch, s4=True)
    assert used
    assert torch.equal(torch.cuda.get_rng_state(), state)


def test_no_grad_and_frozen_input_take_k4_path(base_forward, monkeypatch):
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
    mlp = make_mlp()
    x = torch.randn(2, 37, HIDDEN, device="cuda")

    with torch.no_grad():
        out = mlp(x)
    assert out.grad_fn is None

    out = mlp(x)  # x does not require grad: nothing to save, K4 path
    assert not uses_s4(out)

    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", False)
    with torch.no_grad():
        ref = mlp(x)
    assert torch.equal(out.detach(), ref)


def test_layer_hooks_take_k4_path(base_forward, monkeypatch):
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
    mlp = make_mlp()
    calls = []
    mlp.up_proj.register_forward_hook(lambda *args: calls.append(1))
    x = torch.randn(2, 37, HIDDEN, device="cuda", requires_grad=True)
    out = mlp(x)
    assert not uses_s4(out)
    out.sum().backward()
    assert calls == [1]  # the hook fired once: no recompute through the hooked layer


@pytest.mark.parametrize("case", ["gelu", "plain_linear", "cpu"])
def test_unexpected_inputs_take_the_original_path(monkeypatch, case):
    base = inspect.unwrap(modeling_qwen2.Qwen2MLP.forward)
    calls = []

    def spy(self, x):
        calls.append(case)
        return base(self, x)

    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", spy)
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
    rc.patch_qwen2_mlp_recompute()

    if case == "cpu":
        mlp = make_mlp(four_bit=False).cpu()
        x = torch.randn(1, 3, HIDDEN, requires_grad=True)
    else:
        mlp = make_mlp(act="gelu" if case == "gelu" else "silu", four_bit=case != "plain_linear")
        x = torch.randn(1, 3, HIDDEN, device="cuda", requires_grad=True)
    out = mlp(x)
    assert calls == [case]
    assert not uses_s4(out)


def test_patch_order_and_idempotence(monkeypatch):
    base = inspect.unwrap(modeling_qwen2.Qwen2MLP.forward)
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", base)

    k4.patch_qwen2_mlp_fp16()
    k4_forward = modeling_qwen2.Qwen2MLP.forward
    rc.patch_qwen2_mlp_recompute()
    s4_forward = modeling_qwen2.Qwen2MLP.forward
    assert s4_forward.__wrapped__ is k4_forward and k4_forward.__wrapped__ is base

    rc.patch_qwen2_mlp_recompute()
    k4.patch_qwen2_mlp_fp16()  # K4 after S4 must not wrap S4 inside K4
    assert modeling_qwen2.Qwen2MLP.forward is s4_forward

    # S4 first also installs K4 underneath.
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", base)
    rc.patch_qwen2_mlp_recompute()
    assert getattr(modeling_qwen2.Qwen2MLP.forward.__wrapped__, k4._PATCH_MARK, False)
    assert modeling_qwen2.Qwen2MLP.forward.__wrapped__.__wrapped__ is base


def test_import_touches_no_cuda():
    root = Path(__file__).resolve().parents[2]
    code = ("import torch, solution.mlp_recompute as m; "
            "assert m.USE_MLP_RECOMPUTE is False; "
            "assert not torch.cuda.is_initialized()")
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
