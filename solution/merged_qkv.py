"""Q (v11): q_proj / k_proj / v_proj base projections as ONE fp16 GEMM on U's packed path.

What it replaces. transformers 5.16.1 Qwen2Attention.forward (models/qwen2/modeling_qwen2.py:
195-234) calls q_proj, k_proj and v_proj on the same hidden states (:206-208). Each base layer is a
bitsandbytes Linear4bit (nn/modules.py:609-637): `x.to(fp16)` (:626-628), the bias cast to fp16
once (:631-635), `bnb.matmul_4bit` (:637) -> K1's SoupNF4Matmul = F.linear(x16, dequant(W), b)
(solution/kernels/nf4.py), then `.to(fp32)` (:637). On the T4 that is three GEMMs per layer in
forward (N = 1536, 256, 256 against K = 1536) and three dX GEMMs in backward, three fp32 -> fp16
casts of x, three fp16 -> fp32 casts of their input gradients and five gradient contributions
summed into x's gradient (q base, q LoRA, k, v base, v LoRA). N = 256 fills a T4 poorly
(2 x 16 output tiles of 128 x 128 at P = 2048 rows for 40 SMs [estimate: cuBLAS's tile choice is
not known here]; profile v8b/F2: GEMMs ~300 of ~460 ms GPU per step).

What runs instead (only on U's packed path, inside unpad._layer_packed's attention scopes):
- x16 = x.to(fp16) once (the same cast, now one autograd node instead of three);
- _MergedQKV: one F.linear(x16, W_qkv, b_qkv) with W_qkv = [W_q; W_k; W_v] ([2048, 1536] fp16)
  and b_qkv = [b_q; b_k; b_v]; inside the Function the [.., 2048] output is split into a
  contiguous fp16 q, an fp32 k (Linear4bit's `.to(fp32)`, exact) and a contiguous fp16 v, so no
  view of a temporary leaves the Function. Backward: the three incoming gradients go into one
  [.., 2048] fp16 buffer (k's fp32 gradient cast to fp16 by copy_, round-to-nearest-even, as
  ToCopyBackward's `.to(fp16)`), then ONE GEMM dX16 = g @ W_qkv;
- q and v: solution/lora_recompute.py's LoRA half (_lora_tail through forward_from_base16) on
  their fp16 slices: the same RNG-state read, the same dropout draw on x, the same lora_A /
  lora_B module calls, the same epilogue (or `.to(fp32)` + clone + add), in the same order
  (q's draw, then v's), so the CUDA generator stream and the LoRA numerics are unchanged;
- k: the fp32 slice as is (k_proj has no LoRA in this task);
- the rest of Qwen2Attention.forward line for line (view / transpose, apply_rotary_pos_emb,
  ALL_ATTENTION_FUNCTIONS lookup and call with the same kwargs, reshape, o_proj as a module).

W_qkv rows are bit-identical to the per-projection weights: the q rows come from K1's dequant
(nf4_dequantize with out= into the buffer, bit-exact with bitsandbytes), the k / v rows from the
same K1 dequant (I3's dense cache when it holds them, else K1 into the buffer).
- MERGED_QKV_CACHE False (default): W_qkv is built per call (forward and backward): one K1 launch
  for q (as today), two 0.75 MiB copies from I3's k / v cache. Static memory unchanged; the
  transient weight buffer is 6 MiB instead of q's 4.5 MiB.
- MERGED_QKV_CACHE True: W_qkv is kept for the run (28 x 6 MiB = 168 MiB on Qwen2.5-1.5B), and
  I3's k / v entries become views into it (k / v are not stored twice): +126 MiB static over
  I3's 42 MiB [arithmetic: 28 x 1536 x 1536 x 2 B = 126 MiB], two K1 launches per layer per step
  fewer. Capped at MERGED_CACHE_MIB; above it the call builds per call.

Numerics (NOT bit-identical; stated plainly):
- forward: every q / k / v element is still round_fp16(sum_k x16 * W + b) of the same fp16
  operands, but cuBLAS picks another kernel / tiling for N = 2048 than for N = 1536 / 256, so an
  element may land on the neighbouring fp16 value (1 ulp, 2**-11 relative) when the fp32
  accumulation order differs. Everything after it runs the same ops.
- backward: dX16 = round_fp16(gq @ Wq + gk @ Wk + gv @ Wv) accumulated in fp32 inside one GEMM and
  rounded once, where today each term is rounded to fp16 on its own and the three are summed in
  fp32 by autograd; x's gradient then sums 3 contributions instead of 5 (another fp32 order).
  Overflow: the merged dX16 is inf where the SUM exceeds 65504; today a term is inf only where it
  alone does (the fp32 sum of three finite terms may exceed it). The sum is dominated by the q
  term, so the window is narrow, but a GradScaler skip could come one scale step earlier.
- RNG: identical stream (same draws, same order, nothing new draws).
Tolerances in tests/test_merged_qkv.py are those of tests/test_unpad.py / test_v6_merged.py: hidden
5e-3, loss 1e-4, LoRA gradients 2e-2 (relative norms); the RNG state bit for bit.

Guards (all read-only; any miss -> None and unpad makes the usual module call): flag on and
install() succeeded; exact Qwen2Attention whose class forward is the genuine one and that carries
no hooks except transformers' idle capture hooks (checked by the caller, unpad._bypassed_layer_ok);
no cache; grad enabled, fp16 CUDA autocast, not tracing / compiling, no global module hooks; x a
contiguous fp32 CUDA [B, T, H] tensor; q_proj / v_proj exact peft bnb Linear4bit on M2's path for
this very x (lora_recompute.merged_adapter); k_proj, q_proj.base_layer, v_proj.base_layer exact
hook-free bitsandbytes Linear4bit with the genuine class forward, compute dtype fp16 already set,
an exact Params4bit weight that is frozen, carries its quant_state and passes K1's NF4 validation,
biases all present (frozen, 1-D) or all absent; bnb.matmul_4bit is K1's soup_matmul_4bit and K1's
dense path (not bnb's 4-bit GEMM) would take each projection (nf4.py:643-685 on the same
M = B*T); (n_q, n_k, n_v, hidden) in _SHAPES (Qwen2.5-1.5B: 1536, 256, 256, 1536). Side effects
only once committed: bnb's own first-call bias cast to fp16 (nn/modules.py:631-635), done here
because Linear4bit.forward is not called.

No Triton kernel here (K1 is launched through nf4_dequantize); nothing touches CUDA at import or in
install(). Idea credit: fused QKV projection as in Megatron-LM's query_key_value linear and
Unsloth's LoRA QKV path (idea only); code written from scratch.
"""

import weakref

import torch
import torch.nn.functional as F

from .lora_recompute import _compiling, _fp16_autocast, _hookless, _no_global_hooks

USE_MERGED_QKV = False  # read on every call; solution.apply() sets it
MERGED_QKV_CACHE = False  # True: keep W_qkv for the run (+126 MiB on Qwen2.5-1.5B)
MERGED_CACHE_MIB = 192  # cap of the W_qkv cache (28 x 6 MiB = 168 MiB)

# (n_q, n_k, n_v, hidden): Qwen2.5-1.5B (12 x 128, 2 x 128, 2 x 128, 1536)
_SHAPES = frozenset({(1536, 256, 256, 1536)})
_BNB_CUSTOM_MAX_M = 1536  # = nf4._BNB_CUSTOM_MAX_M (bitsandbytes backends/cuda/ops.py:926)

# merged: attention calls on this path; original: calls refused by a guard; built: W_qkv built for
# one call; cached: W_qkv cache hits.
COUNTS = {"merged": 0, "original": 0, "built": 0, "cached": 0}
_REFS = {}


def _is_genuine(fn, module_name, qualname):
    return (getattr(fn, "__module__", None) == module_name
            and getattr(fn, "__qualname__", None) == qualname)


def install():
    """Capture the classes and functions the guards compare against. Patches nothing, no CUDA
    work, idempotent. True when every reference is there (Soup's tests may pass fake modules)."""
    if _REFS.get("ok"):
        return True
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit, Params4bit
        from peft.tuners.lora import bnb as lora_bnb
        from transformers.models.qwen2 import modeling_qwen2 as mq

        from . import lora_recompute as m2
        from .kernels import nf4

        lora_cls = lora_bnb.Linear4bit
        attn_cls = mq.Qwen2Attention
        for name in ("apply_rotary_pos_emb", "ALL_ATTENTION_FUNCTIONS", "eager_attention_forward"):
            getattr(mq, name)  # the globals the genuine forward resolves (must exist)
    except Exception:  # noqa: BLE001 - a missing or fake library: the path stays off
        return False
    if not _is_genuine(attn_cls.forward, mq.__name__, "Qwen2Attention.forward"):
        return False  # someone else's attention forward: never bypassed
    if not _is_genuine(Linear4bit.forward, "bitsandbytes.nn.modules", "Linear4bit.forward"):
        return False
    _REFS.update(bnb=bnb, linear4bit=Linear4bit, linear4bit_forward=Linear4bit.forward,
                 params4bit=Params4bit, lora_cls=lora_cls, modeling=mq, attn_cls=attn_cls,
                 attn_forward=attn_cls.forward, m2=m2, nf4=nf4, ok=True)
    return True


# ---------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------

class _Plan:
    __slots__ = ("bases", "weights", "states", "entries", "rows", "biases", "q_layer", "v_layer",
                 "q_name", "v_name", "src", "hidden")


def _linear4bit_ok(base):
    """An exact, hook-free bnb Linear4bit whose forward would reach K1 with fp16 compute."""
    refs = _REFS
    if type(base) is not refs["linear4bit"] or not _hookless(base):
        return False
    if type(base).forward is not refs["linear4bit_forward"]:
        return False
    # compute dtype fixed at construction (nn/modules.py:569-570), never re-derived (:622-624)
    if getattr(base, "compute_type_is_set", False) is not True:
        return False
    if getattr(base, "compute_dtype", None) is not torch.float16:
        return False
    weight = base.weight
    if type(weight) is not refs["params4bit"] or weight.requires_grad:
        return False
    # Linear4bit.forward first restores a lost quant_state (nn/modules.py:487-501, :610).
    if getattr(weight, "quant_state", None) is None:
        return False
    bias = base.bias
    if bias is not None:
        if (not isinstance(bias, torch.Tensor) or bias.requires_grad or bias.dim() != 1
                or not bias.is_cuda or bias.dtype not in (torch.float16, torch.float32)):
            return False
    return True


def _k1_dense(x, weight, state, nf4):
    """(n, k, entry) when K1's soup_matmul_4bit takes its dense path for x.to(fp16) against this
    weight, as nf4.k1_dense_path (nf4.py:643-685) decides it; None otherwise. entry is the K1
    validation record (None: no cache may be keyed on it)."""
    try:
        (n, k), entry = nf4._checked_entry(weight, state)
    except (TypeError, ValueError):
        return None
    # absmax sits on the packed weight's device (nf4._validate_inputs); a plain tensor, so no
    # trip through Params4bit's __torch_function__ (as nf4._launch, nf4.py:466-472).
    if state.absmax.device != x.device or x.shape[-1] != k:
        return None
    m = x.numel() // k
    if m <= 4:
        return None
    if m <= _BNB_CUSTOM_MAX_M:
        try:
            if bool(nf4._bnb_cuda_ops()._gemm_4bit_use_custom_fn(
                    x.device.index, torch.float16, m, n, k)):
                return None
        except Exception:  # noqa: BLE001 - as nf4.k1_dense_path: bnb's path then
            return None
    return n, k, entry


def _plan(attn, x):
    refs = _REFS
    if not USE_MERGED_QKV or not refs.get("ok"):
        return None
    if type(attn) is not refs["attn_cls"] or type(attn).forward is not refs["attn_forward"]:
        return None
    if not torch.is_grad_enabled() or _compiling() or not _fp16_autocast():
        return None
    if not _no_global_hooks():
        return None
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float32
            and x.dim() == 3 and x.is_contiguous() and x.numel() > 0):
        return None
    m2 = refs["m2"]
    q_layer, k_proj, v_layer = attn.q_proj, attn.k_proj, attn.v_proj
    q_name = m2.merged_adapter(q_layer, x)
    if q_name is None:
        return None
    v_name = m2.merged_adapter(v_layer, x)
    if v_name is None:
        return None
    bases = (q_layer.base_layer, k_proj, v_layer.base_layer)
    if not all(_linear4bit_ok(b) for b in bases):
        return None
    nf4 = refs["nf4"]
    if refs["bnb"].matmul_4bit is not nf4.soup_matmul_4bit:
        return None  # Linear4bit would not reach K1
    biases = tuple(b.bias for b in bases)
    if any(b is None for b in biases) and not all(b is None for b in biases):
        return None
    weights = tuple(b.weight for b in bases)
    states = tuple(w.quant_state for w in weights)
    found = [_k1_dense(x, w, s, nf4) for w, s in zip(weights, states)]
    if any(f is None for f in found):
        return None
    (nq, k, eq), (nk, k2, ek), (nv, k3, ev) = found
    if (nq, nk, nv, k) not in _SHAPES or k2 != k or k3 != k:
        return None
    for b, n in zip(biases, (nq, nk, nv)):
        if b is not None and (b.numel() != n or b.device != x.device):
            return None
    plan = _Plan()
    plan.bases, plan.weights, plan.states = bases, weights, states
    plan.entries = (eq, ek, ev)
    plan.rows = (nq, nk, nv)
    plan.biases = biases
    plan.q_layer, plan.v_layer, plan.q_name, plan.v_name = q_layer, v_layer, q_name, v_name
    plan.src = m2.current_source()
    plan.hidden = k
    return plan


# ---------------------------------------------------------------------------------------------
# W_qkv: built per call (default) or kept for the run (MERGED_QKV_CACHE)
# ---------------------------------------------------------------------------------------------

_WCACHE = {}  # id(q quant_state) -> (quant_state weakrefs, K1 records, W_qkv)
_WCACHE_BYTES = [0]


def _wcache_drop(key, ref=None, cache=_WCACHE, counter=_WCACHE_BYTES):
    # Bound defaults: callable from a weakref callback late in shutdown (as nf4._dense_drop).
    item = cache.get(key)
    if item is not None and (ref is None or item[0][0] is ref):
        del cache[key]
        counter[0] -= item[2].numel() * item[2].element_size()


def cache_clear():
    """Drop every kept W_qkv (tests; not used by the solution)."""
    _WCACHE.clear()
    _WCACHE_BYTES[0] = 0


def _i3_holds(nf4, nk):
    return bool(nf4.USE_DENSE_CACHE) and nf4._dense_wanted(nk)


def _build_weight(plan, use_i3=True):
    """[n_q + n_k + n_v, K] fp16, rows bit-identical to each projection's K1 dequant."""
    nf4 = _REFS["nf4"]
    nq, nk, nv = plan.rows
    k = plan.hidden
    w = torch.empty((nq + nk + nv, k), device=plan.states[0].absmax.device, dtype=torch.float16)
    bounds = ((0, nq), (nq, nq + nk), (nq + nk, nq + nk + nv))
    for (lo, hi), weight, state in zip(bounds, plan.weights, plan.states):
        if use_i3 and _i3_holds(nf4, (hi - lo, k)):
            # I3's kept k / v dequant (built by K1 on first use): a copy instead of a K1 launch.
            w[lo:hi].copy_(nf4.nf4_dequantize(weight, state))
        else:
            # K1 straight into the rows: 16-byte aligned starts (row offsets x 3072 B), so the
            # same Triton specialisation as a fresh tensor (nf4.py:511-579).
            nf4.nf4_dequantize(weight, state, out=w[lo:hi])
    COUNTS["built"] += 1
    return w


def _cached_weight(plan):
    """W_qkv kept for the run; None when this plan cannot be cached (then build per call)."""
    if any(e is None or e[1]() is None for e in plan.entries):
        return None  # no live K1 record for one of the weights: nothing may be keyed on it
    key = id(plan.states[0])
    item = _WCACHE.get(key)
    if item is not None:
        refs, entries, w = item
        if (all(r() is s for r, s in zip(refs, plan.states))
                and all(a is b for a, b in zip(entries, plan.entries))):
            COUNTS["cached"] += 1
            return w
        _wcache_drop(key)  # made under older records: stale
    nq, nk, nv = plan.rows
    nbytes = (nq + nk + nv) * plan.hidden * 2
    if _WCACHE_BYTES[0] + nbytes > MERGED_CACHE_MIB * 2**20:
        return None
    try:
        refs = (weakref.ref(plan.states[0], lambda r, key=key, drop=_wcache_drop: drop(key, r)),
                weakref.ref(plan.states[1]), weakref.ref(plan.states[2]))
    except TypeError:
        return None
    w = _build_weight(plan, use_i3=False)  # once per layer per run: K1 for all three
    _WCACHE[key] = (refs, plan.entries, w)
    _WCACHE_BYTES[0] += nbytes
    _share_kv_with_i3(plan, w)
    return w


def _share_kv_with_i3(plan, w):
    """I3's k / v entries become views of W_qkv's rows (k / v not stored twice). A later
    nf4_dequantize(k) (e.g. a batch outside U's packed path) then returns the view; callers only
    read it (nf4.py:528-531)."""
    nf4 = _REFS["nf4"]
    nq, nk, nv = plan.rows
    k = plan.hidden
    for (lo, hi), state, entry in (((nq, nq + nk), plan.states[1], plan.entries[1]),
                                   ((nq + nk, nq + nk + nv), plan.states[2], plan.entries[2])):
        if not _i3_holds(nf4, (hi - lo, k)) or torch.is_inference_mode_enabled():
            continue
        key = id(state)
        nf4._dense_drop(key)
        nf4._dense_store(key, state, entry, w[lo:hi])


def _weight(plan):
    if MERGED_QKV_CACHE:
        w = _cached_weight(plan)
        if w is not None:
            return w
    return _build_weight(plan)


def _bias(plan):
    """b_qkv fp16, or None. bnb casts a non-fp16 bias in place on its first call
    (nn/modules.py:631-635: `bias.data = bias.data.to(x.dtype)`, x already fp16); the merged path
    replaces that call, so it makes the same cast once."""
    if plan.biases[0] is None:
        return None
    for b in plan.biases:
        if b.dtype != torch.float16:
            b.data = b.data.to(torch.float16)
    return torch.cat(plan.biases)


class _MergedQKV(torch.autograd.Function):
    """(q16, k32, v16) = split(F.linear(x16, W_qkv, b_qkv)); backward: one dX GEMM."""

    @staticmethod
    def forward(ctx, x16, plan):
        nq, nk, _ = plan.rows
        out = F.linear(x16, _weight(plan), _bias(plan))  # one GEMM, fp16 (K1: nf4.py:587)
        q16 = out[..., :nq].contiguous()
        k32 = out[..., nq:nq + nk].to(torch.float32, memory_format=torch.contiguous_format)
        v16 = out[..., nq + nk:].contiguous()
        del out
        ctx.plan = plan
        return q16, k32, v16

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gq, gk, gv):
        if not ctx.needs_input_grad[0]:
            return None, None
        plan = ctx.plan
        nq, nk, nv = plan.rows
        lead = gq.shape[:-1]
        g = torch.empty((*lead, nq + nk + nv), device=gq.device, dtype=torch.float16)
        g[..., :nq].copy_(gq)
        g[..., nq:nq + nk].copy_(gk)  # fp32 -> fp16, round to nearest even (= `.to(fp16)`)
        g[..., nq + nk:].copy_(gv)
        del gq, gk, gv
        dx16 = torch.matmul(g, _weight(plan))  # as SoupNF4Matmul.backward (nf4.py:597)
        return dx16, None


# ---------------------------------------------------------------------------------------------
# The attention call
# ---------------------------------------------------------------------------------------------

def _projections(plan, x):
    """Qwen2Attention's q_proj(x), k_proj(x), v_proj(x) with one base GEMM."""
    m2 = _REFS["m2"]
    x16 = x.to(torch.float16)  # Linear4bit's `x.to(self.compute_dtype)` (nn/modules.py:626-628)
    q16, k, v16 = _MergedQKV.apply(x16, plan)
    del x16
    q = m2.forward_from_base16(plan.q_layer, x, plan.q_name, plan.src, q16)  # q's dropout draw
    del q16
    v = m2.forward_from_base16(plan.v_layer, x, plan.v_name, plan.src, v16)  # then v's
    plan.src = None  # the node's ctx keeps the plan until backward; M2's recipes keep the source
    return q, k, v


def attention_forward(attn, hidden_states, attention_mask, position_embeddings, kwargs):
    """`attn(hidden_states=..., attention_mask=..., past_key_values=None,
    position_embeddings=..., **kwargs)[0]` with merged q/k/v, or None when a guard refuses (then
    nothing has run). The caller (unpad._layer_packed) checked the module's hooks."""
    try:
        plan = _plan(attn, hidden_states)
    except Exception:  # noqa: BLE001 - an unexpected object is never ours to handle
        plan = None
    if plan is None:
        COUNTS["original"] += 1
        return None
    COUNTS["merged"] += 1
    # From here on no fallback: the dropout draws below must happen exactly once.
    mq = _REFS["modeling"]
    # modeling_qwen2.py:204-234, line for line, past_key_values None.
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    q, k, v = _projections(plan, hidden_states)
    query_states = q.view(hidden_shape).transpose(1, 2)
    key_states = k.view(hidden_shape).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)
    del q, k, v
    cos, sin = position_embeddings
    query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
    attention_interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, mq.eager_attention_forward)
    attn_output, _ = attention_interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not attn.training else attn.attention_dropout,
        scaling=attn.scaling,
        sliding_window=attn.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return attn.o_proj(attn_output)
