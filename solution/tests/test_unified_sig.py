"""v8 C1: the *_usig entries are the v7 kernels behind one launcher signature, bit for bit.

Static part (no GPU): every kernel the training path launches has the same twelve parameters
(slots 0-6 pointers, 7-10 ints, 11 float) and no constexpr parameter; every int / float slot and
every spare pointer slot is in do_not_specialize except the int slots v7 already specialised
(head_ce n_cols, the SwiGLU count); every @triton.jit function in solution/kernels is named soup_*;
every jit function in solution/ stays below check_rules' 0.8 example likeness.
GPU part (T4): with USE_UNIFIED_SIG on, K3, K2 and K1 return the same bits as with it off, and
only the *_usig entries are launched.

Run from the repository root: python -m pytest solution/tests/test_unified_sig.py -q
"""

import ast
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from solution.kernels import addnorm, head_ce, nf4  # noqa: E402

KERNELS = Path(nf4.__file__).resolve().parent
SOLUTION = KERNELS.parent
REPO = SOLUTION.parent

# Launched entries: file, name, int slots that stay specialised (as in v7).
LAUNCHED = [
    ("addnorm.py", "soup_addnorm_fwd_usig", set()),
    ("addnorm.py", "soup_addnorm_bwd_usig", set()),
    ("head_ce.py", "soup_head_ce_usig", {"n_cols"}),
    ("nf4.py", "soup_nf4_dequant_usig", set()),
    ("swiglu.py", "soup_swiglu_kernel", {"n"}),
]


def _jit_functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "jit":
                found[node.name] = (node, dec)
    return found


def _do_not_specialize(dec):
    if not isinstance(dec, ast.Call):
        return set()
    for kw in dec.keywords:
        if kw.arg == "do_not_specialize":
            return {elt.value for elt in kw.value.elts}
    return set()


@pytest.mark.parametrize("file, name, specialised", LAUNCHED)
def test_entry_has_the_shared_layout(file, name, specialised):
    node, dec = _jit_functions(KERNELS / file)[name]
    args = node.args
    params = [a.arg for a in args.args]
    assert len(params) == 12 and not args.posonlyargs and not args.kwonlyargs
    assert not args.defaults and args.vararg is None and args.kwarg is None
    assert all(a.annotation is None for a in args.args)  # no tl.constexpr parameter
    dns = _do_not_specialize(dec)
    assert dns <= set(params)
    for slot, param in enumerate(params):
        if slot >= 7:  # int and float slots
            assert (param in dns) != (param in specialised), param
        elif param.startswith("spare_"):  # unused pointer slots never split a variant
            assert param in dns
        else:  # real pointers keep v7's alignment specialisation
            assert param not in dns
    assert params[11] in dns  # the float slot


def test_every_kernel_is_ours_by_name():
    for path in sorted(KERNELS.glob("*.py")):
        for name in _jit_functions(path):
            assert name.startswith("soup_"), f"{path.name}: {name}"


def test_no_jit_function_close_to_the_example(monkeypatch):
    rules_path = REPO / "harness" / "check_rules.py"
    if not rules_path.exists():
        pytest.skip("harness/check_rules.py not found")
    monkeypatch.setattr(sys, "path", list(sys.path))
    had_common = "common" in sys.modules
    spec = importlib.util.spec_from_file_location("_soup_check_rules_c1", rules_path)
    rules = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(rules)
        own, copies, lookalikes, banned, broken = rules.source_kernels(SOLUTION)
        likeness = {}
        for path in sorted(SOLUTION.rglob("*.py")):
            for node in rules.triton_kernels(path)[1]:
                if not node.name.startswith(rules.EXAMPLE_PREFIX):
                    likeness[node.name] = round(rules.example_likeness(node), 3)
        print(f"\nexample likeness: {likeness}")
        for _, name, _ in LAUNCHED:
            assert name in own
        assert not lookalikes and not banned and not broken
        assert set(copies) == {"example_swiglu_fwd_kernel", "example_swiglu_bwd_kernel"}
        assert max(likeness.values()) < rules.EXAMPLE_REVIEW
    finally:
        if not had_common:
            sys.modules.pop("common", None)


def test_module_defaults_are_v7_and_import_touches_no_cuda():
    code = ("import torch; import solution.kernels.addnorm as a, solution.kernels.head_ce as h, "
            "solution.kernels.nf4 as n, solution.kernels.swiglu as s, solution.mlp as m; "
            "assert a.USE_UNIFIED_SIG is h.USE_UNIFIED_SIG is n.USE_UNIFIED_SIG is False; "
            "assert m.USE_SOUP_SWIGLU is False and m.soup_swiglu_module() is None; "
            "assert not torch.cuda.is_initialized()")
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


# ------------------------------------------------------------------------------------------------
# GPU: same bits with the flag on, and only the entries run
# ------------------------------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture
def launches(monkeypatch):
    """Names of the @triton.jit functions launched (the harness counts them the same way)."""
    from triton.runtime.jit import JITFunction

    seen = []
    original = JITFunction.run

    def run(self, *args, **kwargs):
        seen.append(getattr(getattr(self, "fn", None), "__name__", "?"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", run)
    return seen


def _both(monkeypatch, module, fn, launches):
    """fn() with the module's USE_UNIFIED_SIG off, then on; (off, on, names launched when on)."""
    monkeypatch.setattr(module, "USE_UNIFIED_SIG", False)
    ref = fn()
    torch.cuda.synchronize()
    monkeypatch.setattr(module, "USE_UNIFIED_SIG", True)
    del launches[:]
    got = fn()
    torch.cuda.synchronize()
    return ref, got, set(launches)


def _equal(a, b):
    """Same dtype, shape and bytes (torch.equal alone takes -0.0 for 0.0 and fails on NaN)."""
    if a is None or b is None:
        return a is None and b is None
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8))


@cuda
@pytest.mark.parametrize("rows", [1, 77, 300, 2048])
@pytest.mark.parametrize("with_delta, store_r", [(False, True), (True, True), (True, False)])
def test_addnorm_forward_same_bits(monkeypatch, launches, rows, with_delta, store_r):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, addnorm.HIDDEN_SIZE, device="cuda", generator=gen) * 3
    delta = (torch.randn(rows, addnorm.HIDDEN_SIZE, device="cuda", generator=gen)
             .to(torch.float16) if with_delta else None)
    w = torch.rand(addnorm.HIDDEN_SIZE, device="cuda", generator=gen) + 0.5

    ref, got, names = _both(monkeypatch, addnorm,
                            lambda: addnorm._launch_fwd(x, delta, w, 1e-6, store_r=store_r),
                            launches)
    assert names == {"soup_addnorm_fwd_usig"}
    for a, b in zip(ref, got):
        assert _equal(a, b)


@cuda
@pytest.mark.parametrize("rows", [1, 77, 2048])
@pytest.mark.parametrize("with_dr, with_delta", [(False, False), (True, False), (True, True)])
def test_addnorm_backward_same_bits(monkeypatch, launches, rows, with_dr, with_delta):
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    shape = (rows, addnorm.HIDDEN_SIZE)
    x = torch.randn(shape, device="cuda", generator=gen)
    delta = torch.randn(shape, device="cuda", generator=gen).to(torch.float16) if with_delta else None
    w = torch.rand(addnorm.HIDDEN_SIZE, device="cuda", generator=gen) + 0.5
    _, _, rstd = addnorm._launch_fwd(x if delta is None else x + delta.float(), None, w, 1e-6)
    dy = torch.randn(shape, device="cuda", generator=gen)
    dr = torch.randn(shape, device="cuda", generator=gen) if with_dr else None

    ref, got, names = _both(monkeypatch, addnorm,
                            lambda: addnorm._launch_bwd(dy, dr, x, w, rstd, delta=delta), launches)
    assert names == {"soup_addnorm_bwd_usig"}
    assert _equal(ref, got)


@cuda
@pytest.mark.parametrize("num_items", [None, 300])
def test_head_ce_same_bits(monkeypatch, launches, num_items):
    """Modes 0, 1 and 2 over five vocabulary tiles (the last one 896 wide), forward and backward."""
    monkeypatch.setattr(head_ce, "VOCAB_TILE", 1024)
    gen = torch.Generator(device="cuda").manual_seed(7)
    V, H, B, T = 4992, 256, 3, 77
    weight = torch.randn(V, H, device="cuda", generator=gen) * 0.05
    hidden0 = torch.randn(B, T, H, device="cuda", generator=gen)
    labels = torch.randint(0, V, (B, T), device="cuda", generator=gen)
    labels[:, :5] = -100
    mask = torch.ones(B, T, device="cuda", dtype=torch.long)
    mask[2, -20:] = 0
    labels[2, -20:] = -100

    def run():
        hidden = hidden0.clone().requires_grad_(True)
        outs = head_ce.SoupHeadCE.apply(hidden, weight, labels, mask, num_items)
        outs[0].backward(torch.tensor(1024.0, device="cuda"))
        return [t.detach() for t in outs] + [hidden.grad]

    ref, got, names = _both(monkeypatch, head_ce, run, launches)
    assert names == {"soup_head_ce_usig"}
    for a, b in zip(ref, got):
        assert _equal(a, b)


def _nf4_state(n_out, k_in, seed):
    count = n_out * k_in
    blocks = count // 64
    gen = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (count // 2,), dtype=torch.uint8, generator=gen).cuda()
    state2 = types.SimpleNamespace(absmax=torch.rand((blocks + 255) // 256, generator=gen).cuda(),
                                   code=torch.randn(256, generator=gen).cuda())
    state = types.SimpleNamespace(
        absmax=torch.randint(0, 256, (blocks,), dtype=torch.uint8, generator=gen).cuda(),
        state2=state2, offset=torch.rand(1, generator=gen).cuda())
    return packed, state, count


@cuda
@pytest.mark.parametrize("fast_host", [False, True])
@pytest.mark.parametrize("nk", [(256, 1536), (1536, 1536), (8960, 1536), (1536, 8960), (3, 64)])
def test_nf4_same_bits(monkeypatch, launches, nk, fast_host):
    monkeypatch.setattr(nf4, "USE_K1_FAST_HOST", fast_host)
    packed, state, count = _nf4_state(*nk, seed=sum(nk))

    def run():
        out = torch.empty(nk, device="cuda", dtype=torch.float16)
        nf4._launch(packed, state, count, out)
        return out

    ref, got, names = _both(monkeypatch, nf4, run, launches)
    assert names == {"soup_nf4_dequant_usig"}
    assert _equal(ref, got)  # bits, NaN included


@cuda
def test_nf4_real_weight_same_bits(monkeypatch, launches):
    bnb = pytest.importorskip("bitsandbytes")
    w = (torch.randn(8960, 1536, device="cuda") * 0.02).to(torch.float16)
    packed, state = bnb.functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
    ref, got, names = _both(monkeypatch, nf4, lambda: nf4.nf4_dequantize(packed, state), launches)
    assert names == {"soup_nf4_dequant_usig"}
    assert _equal(ref, got)
    assert _equal(got, bnb.functional.dequantize_4bit(packed, quant_state=state))
