"""S4: the recompute MLP gives K4's outputs and input gradients bit for bit, saves less for backward,
and leaves anything unexpected on the K4 (or original) path.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import ast
import contextlib
import gc
import inspect
import subprocess
import sys
import warnings
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
    monkeypatch.setattr(rc, "_warned", False)
    rc.patch_qwen2_mlp_recompute()
    return base


def run(mlp, x, grad, monkeypatch, s4, autocast=False):
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", s4)
    xi = x.detach().clone().requires_grad_(True)
    ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else contextlib.nullcontext()
    with ctx:  # accelerate wraps only the forward in autocast (accelerator.py:1818-1829)
        out = mlp(xi)
    used = uses_s4(out)
    out.backward(grad)  # outside autocast, like accelerator.backward (accelerator.py:2846)
    return out.detach(), xi.grad, used


# [N, C] is the packed shape under unpadding (U); M = 3 / 74 / 300 / 2048 covers bnb's custom
# kernel, K1's m <= 1536 dispatch and K1's dequant + F.linear path (nf4.py:279-299).
SHAPES = [(1, 3, HIDDEN), (2, 37, HIDDEN), (300, HIDDEN), (4, 512, HIDDEN)]


@pytest.mark.parametrize("recompute_gate", [True, False])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("k1", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
def test_bit_exact_against_k4(base_forward, monkeypatch, shape, k1, autocast, recompute_gate):
    set_k1(monkeypatch, k1)
    monkeypatch.setattr(rc, "RECOMPUTE_GATE", recompute_gate)
    mlp = make_mlp()
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)

    ref_out, ref_grad, ref_used = run(mlp, x, grad, monkeypatch, s4=False, autocast=autocast)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="S4 MLP recompute")  # S4 must not be skipped
        out, x_grad, used = run(mlp, x, grad, monkeypatch, s4=True, autocast=autocast)

    assert not ref_used and used
    assert out.dtype == ref_out.dtype == torch.float32
    assert torch.equal(out, ref_out)
    assert torch.equal(x_grad, ref_grad)


def test_fp16_input_bit_exact(base_forward, monkeypatch):
    set_k1(monkeypatch, True)
    mlp = make_mlp()
    x = torch.randn(4, 512, HIDDEN, device="cuda", dtype=torch.float16)
    grad = torch.randn_like(x)
    ref_out, ref_grad, _ = run(mlp, x, grad, monkeypatch, s4=False)
    out, x_grad, used = run(mlp, x, grad, monkeypatch, s4=True)
    assert used and out.dtype == torch.float16
    assert torch.equal(out, ref_out) and torch.equal(x_grad, ref_grad)


def test_k1_dequant_used_only_when_installed(monkeypatch):
    set_k1(monkeypatch, True)
    assert rc._active_nf4() is nf4
    set_k1(monkeypatch, False)
    assert rc._active_nf4() is None


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

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="S4 MLP recompute")  # no training: no warning
        with torch.no_grad():
            out = mlp(x)
        assert out.grad_fn is None
        out = mlp(x)  # x does not require grad: nothing to save, K4 path
    assert not uses_s4(out)

    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", False)
    with torch.no_grad():
        ref = mlp(x)
    assert torch.equal(out.detach(), ref)


def test_autocast_guard():
    assert rc._autocast_ok("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        assert rc._autocast_ok("cuda")  # Soup's fp16 run must keep S4 alive
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if not torch.is_autocast_enabled("cuda"):
            pytest.skip("bf16 autocast is disabled on this GPU")
        assert not rc._autocast_ok("cuda")


def test_bf16_autocast_takes_k4_path(base_forward):
    mlp = make_mlp()
    x = torch.randn(2, 37, HIDDEN, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if not torch.is_autocast_enabled("cuda"):
            pytest.skip("bf16 autocast is disabled on this GPU")
        assert rc._reject_reason(mlp, x) is not None  # guard only: no bf16 GEMM on a T4


@pytest.mark.parametrize("kind", ["forward", "forward_pre", "full_backward", "full_backward_pre"])
def test_layer_hooks_take_k4_path(base_forward, monkeypatch, kind):
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
    mlp = make_mlp()
    calls = []
    register = getattr(mlp.up_proj, f"register_{kind}_hook")
    register(lambda *args: calls.append(1))
    x = torch.randn(2, 37, HIDDEN, device="cuda", requires_grad=True)
    with pytest.warns(RuntimeWarning, match="module hooks"):
        out = mlp(x)
    assert not uses_s4(out)
    out.sum().backward()
    assert calls == [1]  # the hook fired exactly once, as on the K4 path


def test_global_hook_takes_k4_path(base_forward):
    mlp = make_mlp()
    x = torch.randn(2, 37, HIDDEN, device="cuda", requires_grad=True)
    assert rc._reject_reason(mlp, x) is None
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        assert rc._reject_reason(mlp, x) is not None
    finally:
        handle.remove()
    assert rc._reject_reason(mlp, x) is None


def test_weight_guards(base_forward):
    mlp = make_mlp()
    x = torch.randn(2, 37, HIDDEN, device="cuda", requires_grad=True)
    assert rc._reject_reason(mlp, x) is None

    state = mlp.down_proj.weight.quant_state
    state.quant_type = "fp4"
    try:
        assert "nf4" in rc._reject_reason(mlp, x)
    finally:
        state.quant_type = "nf4"

    packed = mlp.gate_proj.weight
    plain = torch.nn.Parameter(packed.data.clone(), requires_grad=False)
    plain.quant_state = packed.quant_state
    mlp.gate_proj.weight = plain
    try:
        assert "Params4bit" in rc._reject_reason(mlp, x)
    finally:
        mlp.gate_proj.weight = packed
    assert rc._reject_reason(mlp, x) is None


@pytest.mark.parametrize("case", ["gelu", "plain_linear", "cpu"])
def test_unexpected_inputs_take_the_original_path(monkeypatch, case):
    base = inspect.unwrap(modeling_qwen2.Qwen2MLP.forward)
    calls = []

    def spy(self, x):
        calls.append(case)
        return base(self, x)

    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", spy)
    monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
    monkeypatch.setattr(rc, "_warned", True)  # warning behaviour is tested elsewhere
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


def test_module_has_no_kernels_and_no_banned_imports():
    tree = ast.parse(Path(rc.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "triton" not in imported  # S4 adds no kernel and no compile time
    assert not imported & {"unsloth", "liger_kernel", "xformers", "flash_attn", "torchao",
                           "cut_cross_entropy"}


def test_import_touches_no_cuda():
    root = Path(__file__).resolve().parents[2]
    code = ("import torch, solution.mlp_recompute as m; "
            "assert m.USE_MLP_RECOMPUTE is False; "
            "assert not torch.cuda.is_initialized()")
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


# ---------------------------------------------------------------------------------------------
# v7 I2: S4_DEQUANT_REUSE ("both" / "up") against v6.1 ("off"), at the model's real widths
# ---------------------------------------------------------------------------------------------

C, I = 1536, 8960  # Qwen2.5-1.5B hidden / intermediate
REUSE_MODES = ("off", "both", "up")
# Expected nf4_dequantize calls in steps 2-4 when K1 would run gate/up as dequant + F.linear.
DEQUANTS = {"off": 4, "both": 2, "up": 3}
_PROJS = ("gate_proj", "up_proj", "down_proj")


def _fp16_linear4bit(n_in, n_out):
    """Linear4bit quantized from an fp16 weight, so quant_state.dtype is fp16 and K1 takes it
    (as tests/test_peak_v6.py fp16_linear4bit; the real run loads the model in fp16)."""
    lin = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                            compress_statistics=True, quant_type="nf4")
    lin.weight.data = lin.weight.data.to(torch.float16)
    return lin


@pytest.fixture(scope="module")
def real_mlp():
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=C, intermediate_size=I, hidden_act="silu")
    mlp = modeling_qwen2.Qwen2MLP(cfg)
    for name, (n_in, n_out) in {"gate_proj": (C, I), "up_proj": (C, I),
                                "down_proj": (I, C)}.items():
        setattr(mlp, name, _fp16_linear4bit(n_in, n_out))
    mlp = mlp.cuda()
    assert all(getattr(mlp, n).weight.quant_state.dtype == torch.float16 for n in _PROJS)
    return mlp


@contextlib.contextmanager
def _count_dequants(monkeypatch):
    """Counts K1 dequant launches: SoupNF4Matmul.forward/backward and mlp_recompute both look
    nf4_dequantize up in the nf4 module at call time."""
    calls = []
    real = nf4.nf4_dequantize

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(nf4, "nf4_dequantize", counting)
    try:
        yield calls
    finally:
        monkeypatch.setattr(nf4, "nf4_dequantize", real)


def _dense(mlp, n):
    probe = torch.empty(n, C, device="cuda", dtype=torch.float16)
    w = mlp.up_proj.weight
    return nf4.k1_dense_path(probe, w, w.quant_state)


def _count_delta(before):
    return {k: rc.REUSE_COUNTS[k] - before[k] for k in before}


@pytest.mark.parametrize("n", [129, 320, 800, 1800])
def test_dequant_reuse_helper_bit_exact_and_fewer_dequants(real_mlp, base_forward, monkeypatch, n):
    set_k1(monkeypatch, True)
    torch.manual_seed(n)
    x16 = torch.randn(n, C, device="cuda", dtype=torch.float16)
    dh = torch.randn(n, I, device="cuda", dtype=torch.float16)
    dense = _dense(real_mlp, n)
    if n > 1536:
        assert dense  # bnb never asks its heuristic past M = 1536 (backends/cuda/ops.py:954-955)

    res = {}
    for mode in REUSE_MODES:
        monkeypatch.setattr(rc, "S4_DEQUANT_REUSE", mode)
        before = dict(rc.REUSE_COUNTS)
        with _count_dequants(monkeypatch) as calls, torch.no_grad():  # backward runs without grad
            dx = rc._gate_up_input_grad([dh.clone()], x16, real_mlp.gate_proj, real_mlp.up_proj)
        torch.cuda.synchronize()
        res[mode] = (dx, len(calls), _count_delta(before))

    for mode in REUSE_MODES:
        dx, n_deq, delta = res[mode]
        assert dx.dtype == torch.float16 and torch.equal(dx, res["off"][0]), mode
        if mode == "off":
            assert delta == {"both": 0, "up": 0, "fallback": 0}
            assert n_deq == (4 if dense else 2)  # not dense: bnb's own 4-bit GEMM recomputes g, u
        elif dense:
            assert delta == {"both": int(mode == "both"), "up": int(mode == "up"), "fallback": 0}
            assert n_deq == DEQUANTS[mode], (mode, n_deq)
        else:  # bnb's custom GEMM ran gate/up in the forward: v6.1 code, reported as a fallback
            assert delta == {"both": 0, "up": 0, "fallback": 1}
            assert n_deq == 2


def test_dequant_reuse_refuses_saved_gate_and_inactive_k1(real_mlp, base_forward, monkeypatch):
    x16 = torch.randn(1800, C, device="cuda", dtype=torch.float16)
    g = torch.empty(1800, I, device="cuda", dtype=torch.float16)
    set_k1(monkeypatch, True)
    assert rc._reuse_nf4("both", x16, real_mlp.gate_proj, real_mlp.up_proj, None) is nf4
    assert rc._reuse_nf4("up", x16, real_mlp.gate_proj, real_mlp.up_proj, g) is nf4
    assert rc._reuse_nf4("both", x16, real_mlp.gate_proj, real_mlp.up_proj, g) is None
    assert rc._reuse_nf4("both", x16.float(), real_mlp.gate_proj, real_mlp.up_proj, None) is None
    real_mlp.up_proj.forward = real_mlp.up_proj.forward  # an instance-level forward (accelerate hooks)
    try:
        assert rc._reuse_nf4("both", x16, real_mlp.gate_proj, real_mlp.up_proj, None) is None
    finally:
        del real_mlp.up_proj.__dict__["forward"]
    set_k1(monkeypatch, False)  # bnb's own matmul_4bit ran the forward: nothing to reuse
    assert rc._reuse_nf4("both", x16, real_mlp.gate_proj, real_mlp.up_proj, None) is None


class _Lora(torch.nn.Module):
    """x + B(A(x)) * 2: a trainable pair upstream of the MLP, so its grads go through dx16."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(C, 16, bias=False)
        self.b = torch.nn.Linear(16, C, bias=False)
        torch.nn.init.normal_(self.b.weight, std=0.02)

    def forward(self, x):
        return x + self.b(self.a(x)) * 2.0


@pytest.mark.parametrize("n", [129, 800, 1800])
def test_dequant_reuse_end_to_end_bit_exact(real_mlp, base_forward, monkeypatch, n):
    """Soup's setting: fp32 residual input, fp16 autocast forward, backward outside autocast."""
    set_k1(monkeypatch, True)
    torch.manual_seed(1)
    lora = _Lora().cuda()
    x = torch.randn(n, C, device="cuda")
    grad = torch.randn(n, C, device="cuda")
    dense = _dense(real_mlp, n)

    res = {}
    for mode in REUSE_MODES:
        monkeypatch.setattr(rc, "S4_DEQUANT_REUSE", mode)
        monkeypatch.setattr(rc, "USE_MLP_RECOMPUTE", True)
        lora.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        before = dict(rc.REUSE_COUNTS)
        with torch.autocast("cuda", dtype=torch.float16):
            out = real_mlp(lora(xi))
        assert uses_s4(out)
        out.backward(grad)
        delta = _count_delta(before)
        res[mode] = (out.detach(), xi.grad, lora.a.weight.grad.clone(), lora.b.weight.grad.clone())
        if mode != "off":
            assert delta[mode] + delta["fallback"] == 1
            assert delta[mode] == int(dense)

    for mode in ("both", "up"):
        for new, ref in zip(res[mode], res["off"]):
            assert torch.equal(new, ref), mode
