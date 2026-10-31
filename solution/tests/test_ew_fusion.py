"""v10 P2 (USE_ELEMENTWISE_FUSION): element-wise casts / copies / adds inside our own launches.

What must hold (asserted):
- the two K3 *_v10 entries keep the shared twelve-slot launcher layout, every int / float slot at
  runtime, all seven pointer slots real (so one compiled variant each);
- within v10: K3 mode 1's fp16 y is torch's `.to(fp16)` of mode 0's y; the fp16-dy backward equals
  the fp32-dy backward on dy16.float(), and its in-place da16 is `.to(fp16)` of the result; B with
  the flag on equals the unfused K3 add_rmsnorm + S4 MLP + add (both on the v10 entries), bit
  for bit;
- the LoRA epilogue (modes 2 / 3) equals peft's `.to(fp32)` + clone + `* scaling` + add and their
  autograd backward bit for bit, overflow to inf included, for q (1536) and v (256) widths;
- M2 with the epilogue / without the clone equals the v8b LoRA pair bit for bit (outputs, LoRA and
  residual gradients, RNG), and runs the epilogue on both calls;
- the real-width stack with every v10 P2 flag launches only the v10 K3 entries (no v8 K3 cubin)
  and matches v9b's flags: bit for bit when the v10 entries keep v8's register layout (checked
  inline), else within fp32-order noise of the forward;
- apply() wiring of the four flags.
Informative (a failure means fp32 ulps against v9b, not a wrong value; every launch of a run then
uses the v10 entries, so forward, recompute and replay agree):
- test_v10_fwd_bits_match_v9b / test_v10_bwd_bits_match_v9b: the v10 entries sum the row in v8's
  order. If they fail, keep EW_K3_FP16 only if the checks 5 / 6 margins allow (they compare to base
  Soup, which K3 already differs from by such ulps).

CUDA only (Kaggle T4). Run from the repository root:
    python -m pytest solution/tests/test_ew_fusion.py -q -rs
"""

import json
import os
import subprocess
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
pytest.importorskip("triton")

from peft.tuners.lora import bnb as lora_bnb  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution  # noqa: E402
import solution.fused_block as fb  # noqa: E402
import solution.lora_recompute as m2  # noqa: E402
import solution.unpad as unpad  # noqa: E402
from solution.kernels import addnorm  # noqa: E402
from solution.tests import test_compile_probe as tcp  # noqa: E402
from solution.tests import test_lora_recompute as tlr  # noqa: E402
from solution.tests import test_peak_v6 as tp  # noqa: E402
from solution.tests import test_unified_sig as tus  # noqa: E402
from solution.tests import test_unpad as tu  # noqa: E402
from solution.tests import test_v6_apply as tva  # noqa: E402
from solution.tests import test_v6_merged as tm  # noqa: E402

# Fixtures reused from the other files (pytest finds them through these names).
forwards = tu.forwards
s4 = tp.s4
patched = tlr.patched
v6_patches = tm.v6_patches
recorder = tva.recorder

DEV = "cuda"
H = addnorm.HIDDEN_SIZE
EPS = 1e-6
NEW = ("soup_addnorm_fwd_v10", "soup_addnorm_bwd_v10")
OLD_K3 = ("soup_addnorm_fwd_kernel", "soup_addnorm_bwd_kernel", "soup_addnorm_fwd_usig",
          "soup_addnorm_bwd_usig")


@pytest.fixture
def k3_flags(monkeypatch):
    """Tests below write addnorm's switches directly; restore them afterwards."""
    monkeypatch.setattr(addnorm, "USE_EW_K3", addnorm.USE_EW_K3)
    monkeypatch.setattr(addnorm, "USE_UNIFIED_SIG", addnorm.USE_UNIFIED_SIG)


def _w(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (1.0 + 0.1 * torch.randn(H, device=DEV, generator=g)).requires_grad_(False)


def _rows(rows, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = 2.0 * torch.randn(rows, H, device=DEV, generator=g)
    a16 = torch.randn(rows, H, device=DEV, generator=g).to(torch.float16)
    return r, a16


def _v9b(fn, *args, **kwargs):
    addnorm.USE_EW_K3, addnorm.USE_UNIFIED_SIG = False, True
    return fn(*args, **kwargs)


def _v10(fn, *args, **kwargs):
    addnorm.USE_EW_K3 = True
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# Static: launcher layout and flags
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NEW)
def test_v10_entry_has_the_shared_layout(name):
    node, dec = tus._jit_functions(tus.KERNELS / "addnorm.py")[name]
    args = node.args
    params = [a.arg for a in args.args]
    assert len(params) == 12 and not args.posonlyargs and not args.kwonlyargs
    assert not args.defaults and args.vararg is None and args.kwarg is None
    assert all(a.annotation is None for a in args.args)  # no tl.constexpr parameter
    # Every int / float slot runtime (no value-1 or %16 split); all seven pointers real and kept
    # aligned (fresh tensors or the v8 operands), so their alignment never splits a variant.
    assert tus._do_not_specialize(dec) == set(params[7:])
    assert name.startswith("soup_")


def test_module_defaults_are_v9b_and_solution_defaults_on():
    assert addnorm.USE_EW_K3 is False
    assert m2.USE_LORA_EPILOGUE is False and m2.USE_LORA_NO_CLONE is False
    assert solution.USE_ELEMENTWISE_FUSION is True
    assert solution.EW_K3_FP16 is True and solution.EW_LORA_EPILOGUE is True
    assert solution.EW_LORA_NO_CLONE is True


@pytest.mark.parametrize("master, k3, epi, clone", [
    (True, True, True, True), (False, True, True, True), (True, False, True, True),
    (True, True, False, False), (True, False, False, True)])
def test_apply_wires_the_p2_flags(recorder, monkeypatch, master, k3, epi, clone):
    for flag, value in (("USE_ELEMENTWISE_FUSION", master), ("EW_K3_FP16", k3),
                        ("EW_LORA_EPILOGUE", epi), ("EW_LORA_NO_CLONE", clone)):
        monkeypatch.setattr(solution, flag, value)
    solution.apply()
    assert addnorm.USE_EW_K3 is (master and k3)
    assert m2.USE_LORA_EPILOGUE is (master and k3 and epi)  # never without the v10 K3 entries
    assert m2.USE_LORA_NO_CLONE is (master and clone)


# Every v10 K3 call form, added to test_compile_probe's round (inside one_round, after K3's calls).
_P2_CALLS = '''
    v16 = d16[:, :256].contiguous()  # a v_proj-wide LoRA output
    timed("addnorm_v10", lambda: (
        addnorm.launch_fwd16(x, d16, w, 1e-6),
        addnorm.launch_fwd16(x, d16, w, 1e-6, store_r=False),
        addnorm.launch_bwd16(d16.clone(), dy, x, w, rstd, delta=d16, store_da=True),
        addnorm.launch_bwd16(d16.clone(), None, x, w, rstd),
        addnorm._launch_bwd(dy, dy, x, w, rstd, delta=d16),
        addnorm.lora_epilogue_bwd(addnorm.lora_epilogue_fwd(d16, d16.clone(), 2.0), 2.0),
        addnorm.lora_epilogue_bwd(addnorm.lora_epilogue_fwd(v16, v16.clone(), 2.0), 2.0)))
'''


def test_compile_probe_with_p2_still_five_cubins_one_launcher(tmp_path):
    """test_compile_probe's training-default round with USE_EW_K3 on and every v10 call form:
    the v10 entries replace the v8 K3 pair (5 cubins, 1 launcher) and a second round with other
    sizes compiles nothing."""
    flags = "for mod in (addnorm, head_ce, nf4):\n    mod.USE_UNIFIED_SIG = unified\n"
    head = "    hid = torch.randn(2, tokens, hidden, device=dev, requires_grad=True)\n"
    src = tcp.PROBE
    assert flags in src and head in src
    src = src.replace(flags, flags + "addnorm.USE_EW_K3 = True\n").replace(head, _P2_CALLS + head)
    cache = tmp_path / "triton_cache_p2"
    cache.mkdir()
    env = dict(os.environ, TRITON_CACHE_DIR=str(cache))
    proc = subprocess.run([sys.executable, "-c", src, str(tcp.REPO), "1", "1"], cwd=tcp.REPO,
                          env=env, capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [s for s in proc.stdout.splitlines() if s.startswith("PROBE ")][-1]
    result = json.loads(line[len("PROBE "):])
    first, second = result["after_first"], result["after_second"]
    print(f"\nP2: {first}; first round {result['first']}")
    assert first["cubin"] == 5 and first["launcher"] == 1
    assert second == first


def test_fp16_exact_scale():
    assert m2._fp16_exact_scale(2.0) == 2.0 and m2._fp16_exact_scale(2) == 2.0
    assert m2._fp16_exact_scale(0.5) == 0.5 and m2._fp16_exact_scale(1.5) == 1.5
    for bad in (1 / 3, 0.1, 70000.0, float("nan"), float("inf"), True, None, "2",
                torch.tensor(2.0)):
        assert m2._fp16_exact_scale(bad) is None, bad


# ---------------------------------------------------------------------------------------------
# K3 v10 entries
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rows", [1, 77, 1350])
@pytest.mark.parametrize("with_delta", [True, False])
def test_fwd16_is_the_fp16_cast_of_fwd(k3_flags, rows, with_delta):
    r, a16 = _rows(rows)
    w = _w()
    delta = a16 if with_delta else None
    r32, y32, s32 = _v10(addnorm._launch_fwd, r, delta, w, EPS)
    r16, y16, s16 = addnorm.launch_fwd16(r, delta, w, EPS)
    assert y32.dtype == torch.float32 and y16.dtype == torch.float16
    assert torch.equal(y16, y32.to(torch.float16))  # torch's cast, round to nearest even
    assert torch.equal(s16, s32)
    if with_delta:
        assert torch.equal(r16, r32) and torch.equal(r32, r + a16.float())  # one fp32 add
    else:
        assert r16 is None and r32 is None
    _, y16b, _ = addnorm.launch_fwd16(r, delta, w, EPS, store_r=False)  # B's backward recompute
    assert torch.equal(y16b, y16)


@pytest.mark.parametrize("rows", [1, 77, 1350])
@pytest.mark.parametrize("with_dr", [True, False])
@pytest.mark.parametrize("with_delta", [True, False])
def test_bwd16_equals_fp32_dy_and_writes_da(k3_flags, rows, with_dr, with_delta):
    r, a16 = _rows(rows)
    w = _w()
    delta = a16 if with_delta else None
    g = torch.Generator(device=DEV).manual_seed(5)
    _, _, rstd = _v10(addnorm._launch_fwd, r, delta, w, EPS)
    dy16 = torch.randn(rows, H, device=DEV, generator=g).to(torch.float16)
    dr = torch.randn(rows, H, device=DEV, generator=g) if with_dr else None
    # dy_mode 1 (fp32 dy; a residual rides in DX, has_dr 2): the rmsnorm / _AddRMSNorm entry.
    ref = addnorm._launch_bwd(dy16.float(), dr, r, w, rstd, delta=delta)
    buf = dy16.clone()
    new = addnorm.launch_bwd16(buf, dr, r, w, rstd, delta=delta, store_da=True)
    assert torch.equal(new, ref)
    assert torch.equal(buf, ref.to(torch.float16))  # B's d_mid.to(fp16), in place
    buf = dy16.clone()
    new = addnorm.launch_bwd16(buf, dr, r, w, rstd, delta=delta, store_da=False)
    assert torch.equal(new, ref) and torch.equal(buf, dy16)  # dy untouched without store_da


def _k3_bits_equal(rows=1350):
    """True when the v10 entries give v9b's (usig) bits for one training-shaped row block."""
    r, a16 = _rows(rows, seed=3)
    w = _w(1)
    old = _v9b(addnorm._launch_fwd, r, a16, w, EPS)
    new = _v10(addnorm._launch_fwd, r, a16, w, EPS)
    dy = torch.randn(rows, H, device=DEV)
    b_old = _v9b(addnorm._launch_bwd, dy, None, r, w, old[2])
    b_new = _v10(addnorm._launch_bwd, dy, None, r, w, old[2])
    return all(torch.equal(a, b) for a, b in zip(old, new)) and torch.equal(b_old, b_new)


@pytest.mark.parametrize("rows", [1, 77, 1350, 2048])
@pytest.mark.parametrize("with_delta", [True, False])
def test_v10_fwd_close_to_v9b(k3_flags, rows, with_delta):
    r, a16 = _rows(rows)
    w = _w()
    delta = a16 if with_delta else None
    old = _v9b(addnorm._launch_fwd, r, delta, w, EPS)
    new = _v10(addnorm._launch_fwd, r, delta, w, EPS)
    if with_delta:
        assert torch.equal(new[0], old[0])  # the residual add has no order to change
    for a, b in zip(new[1:], old[1:]):
        torch.testing.assert_close(a, b, rtol=tp.RTOL_ORDER, atol=0.0)


@pytest.mark.parametrize("rows", [1, 77, 1350, 2048])
@pytest.mark.parametrize("with_delta", [True, False])
def test_v10_fwd_bits_match_v9b(k3_flags, rows, with_delta):
    """Informative: v10's forward sums the row in v8's order (see the module docstring)."""
    r, a16 = _rows(rows)
    w = _w()
    delta = a16 if with_delta else None
    old = _v9b(addnorm._launch_fwd, r, delta, w, EPS)
    new = _v10(addnorm._launch_fwd, r, delta, w, EPS)
    assert torch.equal(new[1], old[1]) and torch.equal(new[2], old[2]), \
        "v10 K3 forward: another fp32 summation order than v9b (ulp-level, consistent in-run)"


def _bwd_pair(rows, with_dr, with_delta):
    r, a16 = _rows(rows)
    w = _w()
    delta = a16 if with_delta else None
    _, _, rstd = _v9b(addnorm._launch_fwd, r, delta, w, EPS)
    g = torch.Generator(device=DEV).manual_seed(9)
    dy = torch.randn(rows, H, device=DEV, generator=g)
    dr = torch.randn(rows, H, device=DEV, generator=g) if with_dr else None
    old = _v9b(addnorm._launch_bwd, dy, dr, r, w, rstd, delta=delta)
    new = _v10(addnorm._launch_bwd, dy, dr, r, w, rstd, delta=delta)
    return old, new


@pytest.mark.parametrize("rows", [1, 77, 1350])
@pytest.mark.parametrize("with_dr", [True, False])
@pytest.mark.parametrize("with_delta", [True, False])
def test_v10_bwd_close_to_v9b(k3_flags, rows, with_dr, with_delta):
    old, new = _bwd_pair(rows, with_dr, with_delta)
    torch.testing.assert_close(new, old, rtol=tp.RTOL_ORDER, atol=1e-6)


@pytest.mark.parametrize("rows", [1, 77, 1350])
@pytest.mark.parametrize("with_dr", [True, False])
@pytest.mark.parametrize("with_delta", [True, False])
def test_v10_bwd_bits_match_v9b(k3_flags, rows, with_dr, with_delta):
    """Informative: v10's backward sums the row in v8's order (see the module docstring)."""
    old, new = _bwd_pair(rows, with_dr, with_delta)
    assert torch.equal(new, old), \
        "v10 K3 backward: another fp32 summation order than v9b (ulp-level, consistent in-run)"


# ---------------------------------------------------------------------------------------------
# B with the v10 entries
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rows", [300, 1350])
@pytest.mark.parametrize("r_grad", [True, False])  # False: layer 0 (frozen embeddings)
def test_fused_block_ew_bit_exact_vs_unfused(s4, k3_flags, monkeypatch, rows, r_grad):
    monkeypatch.setattr(fb, "USE_FUSED_BLOCK", True)
    layer = tp._Layer()
    torch.manual_seed(7)
    r0 = 2.0 * torch.randn(rows, H, device=DEV)
    a0 = torch.randn(rows, H, device=DEV, dtype=torch.float16)
    g = torch.randn(rows, H, device=DEV)
    out = {}
    for case in ("unfused", "fused", "fused_v9b"):
        addnorm.USE_EW_K3 = case != "fused_v9b"
        r = r0.clone().requires_grad_(r_grad)
        a = a0.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            if case == "unfused":
                nxt = tp._v5_ops(layer, r, a)  # K3 add_rmsnorm (v10, mode 0) + S4 + add
            else:
                assert fb.fused_ok(layer, r, a, mq.Qwen2MLP, mq.Qwen2RMSNorm)
                nxt = fb.add_norm_mlp(r, a, layer)
        nxt.backward(g)
        out[case] = (nxt.detach(), r.grad, a.grad)
    (n0, dr0, da0), (n1, dr1, da1) = out["unfused"], out["fused"]
    assert torch.equal(n1, n0)
    assert da1.dtype == torch.float16 and torch.equal(da1, da0)
    if r_grad:
        assert torch.equal(dr1, dr0)
    else:
        assert dr0 is None and dr1 is None
    # Against v9b's B: bit for bit when the v10 entries keep v8's order, else fp32-order noise
    # carried through one fp16 rounding of x16 (a flipped x16 element moves the MLP output by an
    # fp16 ulp of that product).
    n2, dr2, da2 = out["fused_v9b"]
    if _k3_bits_equal():
        assert torch.equal(n1, n2) and torch.equal(da1, da2)
    else:
        assert tu.rel(n1, n2) <= 1e-4 and tu.rel(da1.float(), da2.float()) <= 1e-2


# ---------------------------------------------------------------------------------------------
# LoRA epilogue (K3 v10 modes 2 / 3)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("width", [H, 256])  # q_proj / v_proj outputs of Qwen2.5-1.5B
@pytest.mark.parametrize("scale", [2.0, 0.5, 1.5])
@pytest.mark.parametrize("strided_grad", [False, True])
def test_lora_epilogue_matches_peft_ops(width, scale, strided_grad):
    g = torch.Generator(device=DEV).manual_seed(11)
    base16 = (3.0 * torch.randn(4, 77, width, device=DEV, generator=g)).to(torch.float16)
    lb16 = torch.randn(4, 77, width, device=DEV, generator=g).to(torch.float16)
    lb16.view(-1)[:16] = 60000.0  # the fp16 product overflows to inf (scale 2.0 / 1.5)
    b_ref = base16.clone().requires_grad_(True)
    l_ref = lb16.clone().requires_grad_(True)
    ref = b_ref.to(torch.float32).clone() + l_ref * scale  # bnb .to(fp32), peft clone, mul, add
    b_new = base16.clone().requires_grad_(True)
    l_new = lb16.clone().requires_grad_(True)
    assert addnorm.lora_epilogue_ok(b_new, l_new)
    new = m2._LoraEpilogue.apply(b_new, l_new, scale)
    assert new.dtype == torch.float32 and new.shape == ref.shape
    assert torch.equal(new, ref)
    shape = (width, 77, 4) if strided_grad else (4, 77, width)
    grad = 300.0 * torch.randn(shape, device=DEV, generator=g)
    grad.view(-1)[:8] = 1.0e6  # fp16 overflow of the incoming gradient
    if strided_grad:
        grad = grad.permute(2, 1, 0)  # [4, 77, width], not contiguous
    ref.backward(grad)
    new.backward(grad)
    assert b_new.grad.dtype == torch.float16 and torch.equal(b_new.grad, b_ref.grad)
    assert l_new.grad.dtype == torch.float16 and torch.equal(l_new.grad, l_ref.grad)


def test_lora_epilogue_refuses_odd_operands():
    a = torch.zeros(4, 8, 256, device=DEV, dtype=torch.float16)
    assert addnorm.lora_epilogue_ok(a, a.clone())
    assert not addnorm.lora_epilogue_ok(a, a.float())
    assert not addnorm.lora_epilogue_ok(a, torch.zeros(4, 8, 128, device=DEV, dtype=torch.float16))
    assert not addnorm.lora_epilogue_ok(a.transpose(0, 1), a.transpose(0, 1).clone())
    flat = torch.zeros(4 * 8 * 256 + 4, device=DEV, dtype=torch.float16)
    assert flat[4:].data_ptr() % 16 == 8
    assert not addnorm.lora_epilogue_ok(flat[4:], flat[4:].clone())  # 16-byte misaligned
    odd = torch.zeros(3, 5, device=DEV, dtype=torch.float16)
    assert not addnorm.lora_epilogue_ok(odd, odd.clone())  # numel % 16
    assert not addnorm.lora_epilogue_ok(a.cpu(), a.cpu())


# ---------------------------------------------------------------------------------------------
# M2 with the epilogue / without the clone
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("epilogue, ew_k3", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("r_grad", [True, False])
def test_m2_pair_with_p2_matches_v8b_bit_for_bit(patched, monkeypatch, k3_flags, epilogue, ew_k3,
                                                 r_grad):
    """test_lora_recompute.py's pair (q and v widths, scaling 32 / 16 = 2.0): M2 with the
    v10 P2 flags against the v8b LoRA path. Both runs use the same K3 entries (ew_k3), so the norm
    in front is the same; (False, False) is the clone skip alone."""
    addnorm.USE_EW_K3 = ew_k3
    monkeypatch.setattr(m2, "USE_LORA_EPILOGUE", epilogue)
    monkeypatch.setattr(m2, "USE_LORA_NO_CLONE", True)
    norm, layers, idx = tlr._norm(), tlr._layers(), tlr._idx()
    r0 = torch.randn(idx.numel(), H, device=DEV)
    ref = tlr.step(norm, layers, r0, idx, m2_on=False, r_grad=r_grad)
    c = dict(m2.COUNTS)
    new = tlr.step(norm, layers, r0, idx, m2_on=True, r_grad=r_grad)
    d = {k: m2.COUNTS[k] - c[k] for k in c}
    assert d["recompute"] == 2 and d["original"] == 0
    assert d["epilogue"] == (2 if epilogue else 0)
    assert torch.equal(new["rng_fwd"], ref["rng_fwd"])
    assert torch.equal(new["rng_bwd"], new["rng_fwd"])
    for a, b in zip(new["outs"], ref["outs"], strict=True):
        assert a.dtype == b.dtype == torch.float32 and torch.equal(a, b)
    assert new["grads"].keys() == ref["grads"].keys()
    assert len(ref["grads"]) == 4 + int(r_grad)
    for k in ref["grads"]:
        assert torch.equal(new["grads"][k], ref["grads"][k]), k


def test_m2_epilogue_off_for_inexact_scaling(patched, monkeypatch):
    """A scaling fp16 cannot hold (here 1/3) keeps v9b's ops: no epilogue, same output."""
    monkeypatch.setattr(m2, "USE_LORA_EPILOGUE", True)
    norm, layers, idx = tlr._norm(), tlr._layers(), tlr._idx()
    for layer in layers:
        layer.scaling["default"] = 1.0 / 3.0
    r0 = torch.randn(idx.numel(), H, device=DEV)
    ref = tlr.step(norm, layers, r0, idx, m2_on=False, r_grad=True)
    c = m2.COUNTS["epilogue"]
    new = tlr.step(norm, layers, r0, idx, m2_on=True, r_grad=True)
    assert m2.COUNTS["epilogue"] == c
    for a, b in zip(new["outs"], ref["outs"], strict=True):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------------------------
# Real-width stack: v9b's switches with and without P2
# ---------------------------------------------------------------------------------------------

def test_stack_with_p2_matches_v9b_and_launches_only_v10_k3(v6_patches, monkeypatch, k3_flags):
    from triton.runtime.jit import JITFunction

    patched_forward = v6_patches
    cls = lora_bnb.Linear4bit
    monkeypatch.setattr(cls, "forward", cls.forward)  # A2's, installed by the fixture
    assert m2.patch_lora_input_recompute()
    for flag in ("USE_LORA_INPUT_RECOMPUTE", "USE_LORA_EPILOGUE", "USE_LORA_NO_CLONE"):
        monkeypatch.setattr(m2, flag, False)
    bits = _k3_bits_equal()
    pm = tm.build()
    n_layers = pm.config.num_hidden_layers
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    tm._set(True, causal=True)  # B + A1 + A2 + D + I1 (A off), as test_lora_recompute's stack
    m2.USE_LORA_INPUT_RECOMPUTE = True  # v9 M2

    addnorm.USE_EW_K3 = False
    ref_a = tm._run(pm, batch, patched_forward, extra)
    ref_b = tm._run(pm, batch, patched_forward, extra)

    names = []
    original_run = JITFunction.run

    def run(jit, *args, **kwargs):
        names.append(getattr(getattr(jit, "fn", None), "__name__", "?"))
        return original_run(jit, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", run)
    addnorm.USE_EW_K3 = True
    m2.USE_LORA_EPILOGUE = True
    m2.USE_LORA_NO_CLONE = True
    c_m2, c_u = dict(m2.COUNTS), dict(unpad.COUNTS)
    new = tm._run(pm, batch, patched_forward, extra)
    assert unpad.COUNTS["packed"] - c_u["packed"] == 1
    assert m2.COUNTS["recompute"] - c_m2["recompute"] == 2 * n_layers
    assert m2.COUNTS["epilogue"] - c_m2["epilogue"] == 2 * n_layers
    assert set(NEW) <= set(names)
    assert not set(OLD_K3) & set(names), sorted(set(names))  # still two K3 cubins per run

    assert torch.equal(new["rng_fwd"], ref_a["rng_fwd"])
    assert torch.equal(new["rng_bwd"], new["rng_fwd"])
    if bits:
        tp._assert_matches(ref_a["hidden"], ref_b["hidden"], new["hidden"], "hidden")
        tp._assert_matches(ref_a["loss"], ref_b["loss"], new["loss"], "loss")
    else:  # the v10 K3 entries sum in another order: ulp-level drift through 3 layers
        assert tu.rel(new["hidden"], ref_a["hidden"]) <= 2e-3
        assert tu.rel(new["loss"], ref_a["loss"]) <= 1e-4
    assert new["grads"].keys() == ref_a["grads"].keys()
    for name in ref_a["grads"]:
        tp._assert_grad_close(ref_a["grads"][name], ref_b["grads"][name], new["grads"][name], name)
    # P2 keeps nothing extra for backward (the epilogue saves no tensor; B keeps (r, a, rstd)).
    assert new["held"] <= ref_a["held"] + (1 << 20), (new["held"], ref_a["held"])
