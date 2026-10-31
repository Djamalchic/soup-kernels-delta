"""v11 Q (solution/merged_qkv.py): q/k/v base projections as one fp16 GEMM on U's packed path.

What must hold (asserted):
- W_qkv's rows are bit for bit each projection's K1 dequant and bitsandbytes' own dequant, built
  per call (q by K1, k / v from I3's cache or K1) or kept (cache mode);
- the merged node against three F.linear calls on the same operands: every output within one
  fp16 rounding of the separate result (different cuBLAS kernel for N = 2048), k exactly fp16
  values widened to fp32; dX within fp16 rounding of the fp32 reference, as close to it as the
  three-GEMM sum is;
- on the real-width stack with v10d's switches (U, K1 + I3, K3 v10, B, S4, A1 + I1 + M3, A2, D,
  M2 + epilogue + no clone), merged against v10d: the same RNG stream bit for bit (forward and
  after backward), loss within 1e-4, hidden states within 5e-3, LoRA gradients within 2e-2
  (relative norms; the tolerances of test_unpad / test_v6_merged); the attention runs merged in
  every layer, M2 and its epilogue still run on q and v; autograd holds no more than v10d; K1
  launches per step unchanged in build mode, q / k / v dequants gone in cache mode, whose I3
  k / v entries are views of W_qkv (k / v not stored twice);
- bnb's first-call bias cast is reproduced (biases become fp16 with the same bits);
- every guard miss (hooks on q_proj / k_proj / a base layer, a trainable bias, K1 not installed,
  flag off) runs the module call, which is v10d bit for bit.

CUDA only (Kaggle T4). Run from the repository root:
    python -m pytest solution/tests/test_merged_qkv.py -q -rs
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
pytest.importorskip("triton")

import torch.nn.functional as F  # noqa: E402
from peft.tuners.lora import bnb as lora_bnb  # noqa: E402

import solution.lora_recompute as m2  # noqa: E402
import solution.merged_qkv as mqkv  # noqa: E402
import solution.sdpa_recompute as a1  # noqa: E402
import solution.unpad as unpad  # noqa: E402
from solution.kernels import addnorm, nf4  # noqa: E402
from solution.tests import test_peak_v6 as tp  # noqa: E402
from solution.tests import test_unpad as tu  # noqa: E402  (also hides torchao from peft)
from solution.tests import test_v6_merged as tm  # noqa: E402
from solution.tests import test_v6_no_leak as tnl  # noqa: E402

# Fixtures reused from the other files (pytest finds them through these names).
forwards = tu.forwards
v6_patches = tm.v6_patches

DEV = "cuda"
H = tm.H
KV = tm.KV_HEADS * tm.HEAD_DIM
LAYERS = 3
# Tolerances (relative norms) of test_unpad.py and test_v6_merged.py.
HIDDEN_RTOL, LOSS_RTOL = 5e-3, 1e-4


@pytest.fixture
def q_flags(monkeypatch):
    """Every switch these tests write, restored afterwards; both weight caches emptied."""
    for mod, flag in ((mqkv, "USE_MERGED_QKV"), (mqkv, "MERGED_QKV_CACHE"),
                      (mqkv, "MERGED_CACHE_MIB"), (nf4, "USE_K1_FAST_HOST"),
                      (nf4, "USE_DENSE_CACHE"), (nf4, "DENSE_CACHE_QO"), (nf4, "DENSE_CACHE_MIB"),
                      (addnorm, "USE_EW_K3"), (addnorm, "USE_UNIFIED_SIG"),
                      (m2, "USE_LORA_INPUT_RECOMPUTE"), (m2, "USE_LORA_EPILOGUE"),
                      (m2, "USE_LORA_NO_CLONE")):
        monkeypatch.setattr(mod, flag, getattr(mod, flag))
    nf4.USE_K1_FAST_HOST, nf4.USE_DENSE_CACHE = True, True  # v7 I3, as apply() sets them
    nf4.DENSE_CACHE_QO, nf4.DENSE_CACHE_MIB = False, 64
    mqkv.USE_MERGED_QKV, mqkv.MERGED_QKV_CACHE, mqkv.MERGED_CACHE_MIB = False, False, 192
    mqkv.cache_clear()
    nf4.dense_cache_clear()
    assert mqkv.install() and mqkv.install()  # idempotent, patches nothing
    yield
    mqkv.cache_clear()
    nf4.dense_cache_clear()


# ---------------------------------------------------------------------------------------------
# W_qkv and the merged node on one layer's projections
# ---------------------------------------------------------------------------------------------

def _projections(seed=0):
    """q / k / v as the real run has them: NF4 from fp16 weights (K1 takes them), fp16 biases."""
    torch.manual_seed(seed)
    mods = [tp.fp16_linear4bit(H, n, bias=True) for n in (H, KV, KV)]
    mods = [m.cuda() for m in mods]
    for m in mods:
        m.bias.data = (0.05 * torch.randn(m.bias.shape, device=DEV)).to(torch.float16)
        m.bias.requires_grad_(False)
        assert m.weight.quant_state.dtype == torch.float16
    plan = mqkv._Plan()
    plan.bases = tuple(mods)
    plan.weights = tuple(m.weight for m in mods)
    plan.states = tuple(w.quant_state for w in plan.weights)
    plan.entries = tuple(nf4._checked_entry(w, s)[1] for w, s in zip(plan.weights, plan.states))
    plan.rows = (H, KV, KV)
    plan.biases = tuple(m.bias for m in mods)
    plan.hidden = H
    plan.q_layer = plan.v_layer = plan.q_name = plan.v_name = plan.src = None
    return mods, plan


def _bnb_dense(mod):
    return bnb.functional.dequantize_4bit(mod.weight.data, quant_state=mod.weight.quant_state)


@pytest.mark.parametrize("use_i3", [True, False])
def test_weight_rows_are_the_per_projection_dequant(q_flags, use_i3):
    mods, plan = _projections()
    w = mqkv._build_weight(plan, use_i3=use_i3)
    assert w.shape == (H + 2 * KV, H) and w.dtype == torch.float16 and w.is_contiguous()
    lo = 0
    for mod, n in zip(mods, plan.rows):
        k1 = nf4.nf4_dequantize(mod.weight, mod.weight.quant_state)
        assert torch.equal(w[lo:lo + n], k1)
        assert torch.equal(w[lo:lo + n], _bnb_dense(mod))  # bitsandbytes' own dequant
        lo += n
    # I3 holds exactly k and v (q is never kept in build mode).
    assert nf4._DENSE_BYTES[0] == 2 * KV * H * 2


@pytest.mark.parametrize("rows", [77, 600, 2048])
def test_merged_node_against_three_linears(q_flags, rows):
    mods, plan = _projections(seed=1)
    g = torch.Generator(device=DEV).manual_seed(rows)
    x16 = torch.randn(1, rows, H, device=DEV, generator=g).to(torch.float16).requires_grad_(True)
    q16, k32, v16 = mqkv._MergedQKV.apply(x16, plan)
    assert q16.dtype == v16.dtype == torch.float16 and k32.dtype == torch.float32
    assert q16.is_contiguous() and k32.is_contiguous() and v16.is_contiguous()
    assert torch.equal(k32, k32.to(torch.float16).float())  # exact widening of fp16 values
    weights = [_bnb_dense(m) for m in mods]
    seps = [F.linear(x16.detach(), w, m.bias) for w, m in zip(weights, mods)]
    for new, sep, w, m in zip((q16, k32, v16), seps, weights, mods):
        ref32 = F.linear(x16.detach().float(), w.float(), m.bias.float())
        # One fp16 rounding of an fp32 sum whose order may differ: within 2 fp16 ulps of the
        # exact value and of the separate GEMM (rtol 2**-10), absolute slack for values near 0.
        slack = 1e-3 * ref32.abs().max().item()
        torch.testing.assert_close(new.float(), ref32, rtol=2 ** -10, atol=slack)
        torch.testing.assert_close(new.float(), sep.float(), rtol=2 ** -10, atol=slack)
        same = (new.float() == sep.float()).float().mean().item()
        print(f"\nrows {rows}, n {w.shape[0]}: {same:.4f} of the outputs bit-identical")
    gq = torch.randn(q16.shape, device=DEV, generator=g).to(torch.float16)
    gk = 100.0 * torch.randn(k32.shape, device=DEV, generator=g)
    gv = torch.randn(v16.shape, device=DEV, generator=g).to(torch.float16)
    torch.autograd.backward((q16, k32, v16), (gq, gk, gv))
    dx = x16.grad
    assert dx.dtype == torch.float16 and dx.shape == x16.shape
    gk16 = gk.to(torch.float16)  # ToCopyBackward of Linear4bit's `.to(fp32)`
    sep_dx = sum(torch.matmul(gp, w).float() for gp, w in zip((gq, gk16, gv), weights))
    ref_dx = sum(torch.matmul(gp.float(), w.float()) for gp, w in zip((gq, gk16, gv), weights))
    err_new, err_sep = tu.rel(dx, ref_dx), tu.rel(sep_dx, ref_dx)
    assert err_new <= 2e-3 and err_new <= 1.5 * err_sep + 1e-6, (err_new, err_sep)


# ---------------------------------------------------------------------------------------------
# Real-width stack with v10d's switches
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def stack(v6_patches, q_flags, monkeypatch):
    """v10d's training path on test_v6_no_leak's fp16-NF4 stack (K1 runs on every projection)."""
    patched_forward = v6_patches
    cls = lora_bnb.Linear4bit
    monkeypatch.setattr(cls, "forward", cls.forward)  # A2's, installed by v6_patches
    assert m2.patch_lora_input_recompute()
    tm._set(True, causal=True)  # B + A1 + A2 + D + I1 (A off)
    a1.USE_ROW_COMPACT = True  # v9 M3 (restored by v6_patches)
    m2.USE_LORA_INPUT_RECOMPUTE = True  # v9 M2
    addnorm.USE_EW_K3 = True  # v10 P2
    m2.USE_LORA_EPILOGUE = m2.USE_LORA_NO_CLONE = True
    assert unpad._REFS.get("merged_qkv") is mqkv  # captured by unpad's patch (forwards fixture)
    pm = tnl._build_fp16_nf4(layers=LAYERS)
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    return patched_forward, pm, batch, extra


def _attn(pm):
    return [layer.self_attn for layer in tu.inner_model(pm).layers]


def _deltas(before):
    now = {"q": dict(mqkv.COUNTS), "m2": dict(m2.COUNTS), "u": dict(unpad.COUNTS)}
    return {k: {n: now[k][n] - before[k][n] for n in now[k]} for k in now}


def _snap():
    return {"q": dict(mqkv.COUNTS), "m2": dict(m2.COUNTS), "u": dict(unpad.COUNTS)}


def _k1_launch_counter(monkeypatch, pm):
    """Counts K1 launches per projection kind (q / k / v weights of the stack, other)."""
    kinds = {}
    for att in _attn(pm):
        kinds[att.q_proj.base_layer.weight.data_ptr()] = "q"
        kinds[att.k_proj.weight.data_ptr()] = "k"
        kinds[att.v_proj.base_layer.weight.data_ptr()] = "v"
    seen = {"q": 0, "k": 0, "v": 0, "other": 0}
    real = nf4._launch_here

    def counting(packed, quant_state, n_groups, out32):
        seen[kinds.get(packed.data_ptr(), "other")] += 1
        return real(packed, quant_state, n_groups, out32)

    monkeypatch.setattr(nf4, "_launch_here", counting)
    return seen


def _check_close(new, ref_a, ref_b, what):
    assert torch.equal(new["rng_fwd"], ref_a["rng_fwd"]), what  # same draws, same order
    assert torch.equal(new["rng_bwd"], new["rng_fwd"]), what  # A2's replays restore it
    assert tu.rel(new["loss"], ref_a["loss"]) <= LOSS_RTOL, (what, new["loss"], ref_a["loss"])
    err = tu.rel(new["hidden"], ref_a["hidden"])
    assert err <= HIDDEN_RTOL, (what, err)
    assert new["grads"].keys() == ref_a["grads"].keys()
    for name in ref_a["grads"]:
        tp._assert_grad_close(ref_a["grads"][name], ref_b["grads"][name], new["grads"][name],
                              f"{what}: {name}")


def test_stack_merged_matches_v10d(stack, monkeypatch):
    patched_forward, pm, batch, extra = stack
    n_layers = pm.config.num_hidden_layers
    ref_a = tm._run(pm, batch, patched_forward, extra)  # v10d (merged off)
    ref_b = tm._run(pm, batch, patched_forward, extra)
    for att in _attn(pm):  # bnb's first call cast every q/k/v bias to fp16
        assert att.k_proj.bias.dtype == torch.float16
    seen = _k1_launch_counter(monkeypatch, pm)
    c = _snap()
    ref_c = tm._run(pm, batch, patched_forward, extra)
    ref_launches = dict(seen)
    assert _deltas(c)["q"]["merged"] == 0
    # Steady state of v10d: q dequantized in forward and in backward (not layer 0: its input has
    # no grad); k / v come from I3's cache (expected 0 launches, printed, not asserted).
    assert ref_launches["q"] == 2 * n_layers - 1
    print(f"\nv10d K1 launches per step: {ref_launches}")

    mqkv.USE_MERGED_QKV = True  # build mode (default)
    for key in seen:
        seen[key] = 0
    c = _snap()
    new = tm._run(pm, batch, patched_forward, extra)
    d = _deltas(c)
    assert d["u"]["packed"] == 1
    assert d["q"]["merged"] == n_layers and d["q"]["original"] == 0
    assert d["q"]["built"] == 2 * n_layers - 1 and d["q"]["cached"] == 0
    assert d["m2"]["recompute"] == 2 * n_layers and d["m2"]["original"] == 0
    assert d["m2"]["epilogue"] == 2 * n_layers
    # Same K1 work as v10d: q by K1 in forward and backward, k / v copied from I3's cache.
    assert seen["q"] == ref_launches["q"] and seen["other"] == ref_launches["other"]
    assert seen["k"] == seen["v"] == 0
    _check_close(new, ref_a, ref_b, "build")
    # Nothing extra kept for backward (the node saves no tensor; ctx holds module references).
    assert new["held"] <= ref_c["held"] + (1 << 20), (new["held"], ref_c["held"])

    # Cache mode: W_qkv built once per layer (K1 for q, k and v), then no dequant at all.
    mqkv.MERGED_QKV_CACHE = True
    for key in seen:
        seen[key] = 0
    c = _snap()
    new_c1 = tm._run(pm, batch, patched_forward, extra)
    d = _deltas(c)
    assert d["q"]["merged"] == n_layers and d["q"]["built"] == n_layers
    assert d["q"]["cached"] == n_layers - 1  # the backward of layers 1..n-1
    assert seen["q"] == seen["k"] == seen["v"] == n_layers
    assert mqkv._WCACHE_BYTES[0] == n_layers * (H + 2 * KV) * H * 2
    _check_close(new_c1, ref_a, ref_b, "cache, first step")
    for key in seen:
        seen[key] = 0
    new_c2 = tm._run(pm, batch, patched_forward, extra)
    assert seen["q"] == seen["k"] == seen["v"] == 0
    assert seen["other"] == ref_launches["other"]
    _check_close(new_c2, ref_a, ref_b, "cache")
    # The kept weights give the build mode's bits (same rows, same GEMM).
    assert torch.equal(new_c2["loss"], new["loss"]) and torch.equal(new_c2["hidden"], new["hidden"])
    assert new_c2["held"] <= ref_c["held"] + (1 << 20), (new_c2["held"], ref_c["held"])
    # k / v are not stored twice: I3's entries are views of W_qkv's rows.
    assert nf4._DENSE_BYTES[0] == n_layers * 2 * KV * H * 2
    for att in _attn(pm):
        w_qkv = mqkv._WCACHE[id(att.q_proj.base_layer.weight.quant_state)][2]
        for proj, lo in ((att.k_proj, H), (att.v_proj.base_layer, H + KV)):
            dense = nf4.nf4_dequantize(proj.weight, proj.weight.quant_state)
            assert dense.untyped_storage().data_ptr() == w_qkv.untyped_storage().data_ptr()
            assert torch.equal(dense, w_qkv[lo:lo + KV])


def test_first_call_casts_the_biases_as_bnb(v6_patches, q_flags, monkeypatch):
    """A fresh model whose first forward is merged: the fp32 biases (what peft's 4-bit preparation
    leaves) become fp16 with the bits bnb's own first call gives them (nn/modules.py:630-635)."""
    patched_forward = v6_patches
    cls = lora_bnb.Linear4bit
    monkeypatch.setattr(cls, "forward", cls.forward)
    assert m2.patch_lora_input_recompute()
    tm._set(True, causal=True)
    m2.USE_LORA_INPUT_RECOMPUTE = True
    addnorm.USE_EW_K3 = True
    m2.USE_LORA_EPILOGUE = m2.USE_LORA_NO_CLONE = True
    out = {}
    for merged in (False, True):
        mqkv.USE_MERGED_QKV = merged
        nf4.dense_cache_clear()
        pm = tnl._build_fp16_nf4(layers=2)
        before = [(a.q_proj.base_layer.bias.detach().clone(), a.k_proj.bias.detach().clone(),
                   a.v_proj.base_layer.bias.detach().clone()) for a in _attn(pm)]
        assert all(b.dtype == torch.float32 for trio in before for b in trio)
        batch = tu.make_batch(pm.config.vocab_size)
        c = _snap()
        res = tm._run(pm, batch, patched_forward, {"num_items_in_batch": tu.num_items(batch)})
        assert _deltas(c)["q"]["merged"] == (2 if merged else 0)
        after = [(a.q_proj.base_layer.bias, a.k_proj.bias, a.v_proj.base_layer.bias)
                 for a in _attn(pm)]
        for trio_before, trio_after in zip(before, after):
            for b0, b1 in zip(trio_before, trio_after):
                assert b1.dtype == torch.float16 and torch.equal(b1, b0.to(torch.float16))
        out[merged] = (res, [b.detach().clone() for trio in after for b in trio])
        del pm
    (ref, b_ref), (new, b_new) = out[False], out[True]
    assert all(torch.equal(a, b) for a, b in zip(b_ref, b_new))
    assert torch.equal(new["rng_fwd"], ref["rng_fwd"])
    assert tu.rel(new["loss"], ref["loss"]) <= LOSS_RTOL
    assert tu.rel(new["hidden"], ref["hidden"]) <= HIDDEN_RTOL


@pytest.mark.parametrize("case", ["q_hook", "k_hook", "v_base_hook", "k_bias_trainable",
                                  "k1_off", "flag_off", "attn_hook"])
def test_guard_misses_run_the_module_call(stack, monkeypatch, case):
    """One layer (or all) refuses: those layers make the module call, which is v10d bit for bit
    (the forward is deterministic); the other layers stay merged."""
    patched_forward, pm, batch, extra = stack
    n_layers = pm.config.num_hidden_layers
    ref = tm._run(pm, batch, patched_forward, extra)  # v10d
    att = _attn(pm)[1]
    fired = []
    hook = None
    refused = 1
    if case == "q_hook":
        hook = att.q_proj.register_forward_hook(lambda *a: fired.append(1))
    elif case == "k_hook":
        hook = att.k_proj.register_forward_pre_hook(lambda *a: fired.append(1))
    elif case == "v_base_hook":
        hook = att.v_proj.base_layer.register_forward_hook(lambda *a: fired.append(1))
    elif case == "attn_hook":
        hook = att.register_forward_hook(lambda *a: fired.append(1))
    elif case == "k_bias_trainable":
        att.k_proj.bias.requires_grad_(True)
    elif case == "k1_off":
        monkeypatch.setattr(bnb, "matmul_4bit", nf4._get_original_matmul_4bit())
        refused = n_layers
    mqkv.USE_MERGED_QKV = case != "flag_off"
    if case == "flag_off":
        refused = n_layers
    c = _snap()
    try:
        new = tm._run(pm, batch, patched_forward, extra)
    finally:
        if hook is not None:
            hook.remove()
        att.k_proj.bias.requires_grad_(False)
    d = _deltas(c)
    assert d["q"]["merged"] == n_layers - refused
    if case == "flag_off":
        assert d["q"]["original"] == 0  # unpad never asks
    elif case == "attn_hook":
        assert d["q"]["original"] == 0  # unpad's hook check refuses before the module is asked
    else:
        assert d["q"]["original"] == refused
    if hook is not None:
        assert fired == [1]  # the module call ran, hook included
    if case == "k_bias_trainable":
        return  # one more trainable tensor: another gradient set, compared nowhere
    if case == "flag_off":  # v10d everywhere: deterministic forward, bit for bit
        assert torch.equal(new["loss"], ref["loss"]) and torch.equal(new["hidden"], ref["hidden"])
    else:  # k1_off: bnb's own dequant + F.linear (K1 is bit-exact with it, not asserted here)
        assert tu.rel(new["loss"], ref["loss"]) <= LOSS_RTOL
        assert tu.rel(new["hidden"], ref["hidden"]) <= HIDDEN_RTOL
    assert torch.equal(new["rng_fwd"], ref["rng_fwd"])


def test_other_shapes_keep_the_module_call(forwards, q_flags):
    """test_unpad's small model (hidden 128, plain nn.Linear q/v): never merged, never raises."""
    base, patched = forwards
    mqkv.USE_MERGED_QKV = True
    pm = tu.build()
    batch = tu.make_batch(pm.config.vocab_size)
    c = _snap()
    tu.run(pm, batch, patched)
    d = _deltas(c)
    assert d["q"]["merged"] == 0
