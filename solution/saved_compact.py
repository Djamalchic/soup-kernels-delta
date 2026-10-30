"""A + D (v6 peak-timing): smaller saved tensors inside the attention of a packed decoder layer.

Where the v5 peak comes from [estimate, from the v4 -> v5 difference and the shapes below]: at the
start of backward autograd still holds, per decoder layer, about 62 MiB at B*T = P = 2048 rows
with N = 1350 real ones:
  - residual stream (K3 / S4 saves; see solution/fused_block.py)          ~20 MiB
  - LoRA q/v: dropout keep-mask (bool) + fp16 dropout output, [P, 1536]x2 ~18 MiB
  - SDPA: q16, k16, v16 (k/v after repeat_kv to 12 heads), out [P, 1536]  ~24 MiB
x 28 layers = ~1.75 GiB on top of ~1.57 GiB of weights and ~0.35 GiB of CUDA context.

This module shrinks two of those buckets without changing a single bit of what backward computes.
Both use torch.autograd.graph.saved_tensors_hooks (public API, the mechanism behind TRL's and Soup's
activation offloading: trl/models/activation_offloading.py:79, soup_cli/utils/activation_offload.py
:120-126; here it compresses, nothing leaves the GPU and no Soup switch is touched). The pack hook
returns a small handle, the unpack hook rebuilds a tensor equal to the original element for
element, with the original shape, dtype and contiguous strides. Only the innermost hooks run
(trl/models/activation_offloading.py:562-566), so the SDPA hooks below override the LoRA ones for
the duration of the SDPA call.

A. SDPA key/value after repeat_kv (patch of the "sdpa" entry of ALL_ATTENTION_FUNCTIONS).
   With a padding mask, transformers cannot use SDPA's GQA mode (integrations/sdpa_attention.py:
   27-36) and expands key/value from 2 to 12 heads (lines 97-100, repeat_kv lines 15-24, an
   expand+reshape copy); fp16 autocast then casts the copies, and SDPA saves the fp16 [B, 12, T,
   128] copies for backward. Our wrapper does the same ops in the same order (repeat_kv on the fp32
   tensors, then the fp16 casts autocast would do, then F.scaled_dot_product_attention with the
   same arguments, lines 104-170 for the case "mask given": is_causal False, no position_bias),
   and its pack hook recognises those two tensors by storage pointer, shape, strides and dtype and
   keeps one head of each group: [B, 2, T, 128]. Unpack repeats them back. Saves 2 x 5/6 of
   [P, 1536] fp16 per layer: 10 MiB at P = 2048 -> ~280 MiB over 28 layers [estimate].

D. LoRA dropout masks and LoRA-A inputs on the padding rows (context around self_attn).
   U feeds attention a [B, T, 1536] tensor that is exactly +0 on padding rows (unpad.unpack_rows).
   PEFT's LoRA (peft/tuners/lora/bnb.py:527-552) saves, per q_proj and v_proj, the dropout
   keep-mask (bool, [B, T, 1536]) and the fp16 cast of dropout(x) that the lora_A matmul keeps for
   its weight gradient ([P, 1536]). The pack hook keeps only the N real rows of every fp16 or bool
   tensor of shape [B, T, 1536] or [B*T, 1536] saved inside the attention call; unpack scatters
   them into zeros. Exact because:
   - the fp16 dropout output is +0 on padding rows (0 * mask * scale, then cast), so zeros are its
     true values there and lora_A's weight gradient (a sum over all P rows) is unchanged;
   - a mask row only scales the gradient of its own row, and every gradient on a padding row of
     the attention input is dropped by unpack_rows' backward (index_select of the real rows), so
     the mask's padding rows never reach anything that is kept;
   - every other tensor saved in there has another shape (SDPA 4-D [B, 12, T, 128], LoRA-B's
     [P, 16], RoPE's cos/sin, K1's packed uint8 weights) and is kept as a detached alias of
     itself (same storage; never the tensor itself, see _row_hooks: v6 fix). bitsandbytes'
     MatMul4Bit and K1 keep no activation (`_functions.py:356-359`, `kernels/nf4.py:408`).
   Saves 6 (P - N) x 1536 bytes per layer: ~6.1 MiB at P = 2048, N = 1350 -> ~172 MiB [estimate].

Cost [estimate]: A two 1 MiB slices in forward and two 6 MiB repeats in backward per layer; D four
index_select in forward and four zero+index_copy in backward per layer. Together ~8-11 ms per
step, ~0.6 s per run. No Triton kernel lives here; nothing touches CUDA at import or patch time.

Guards: A runs only inside a packed-layer scope (unpad sets it), for an exact Qwen2Attention with
GQA, fp32 q/k/v under fp16 autocast, a bool mask, dropout 0, no position_bias, not tracing, not on
NPU/XPU, grad enabled; anything else calls the original sdpa_attention_forward. D runs only on U's
packed path, where the zero padding rows are guaranteed by construction. A failure to enter the
hook context leaves the call unhooked (same values, just no saving).
"""

import contextlib
import contextvars
import functools

import torch

USE_SDPA_KV_COMPACT = False  # A; solution.apply() switches it on
USE_LORA_ROW_COMPACT = False  # D; solution.apply() switches it on

_MARK = "_soup_delta_sdpa_kv_compact"
_ACTIVE = contextvars.ContextVar("soup_delta_attention_scope", default=None)
_REFS = {}
COUNTS = {"sdpa_compact": 0, "sdpa_original": 0, "rows_packed": 0, "kv_packed": 0}


# ---------------------------------------------------------------------------------------------
# Hook context helper
# ---------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _hooks(pack, unpack):
    """saved_tensors_hooks(pack, unpack), or nothing if they cannot be entered here."""
    ctx = None
    graph = getattr(torch.autograd, "graph", None)
    factory = getattr(graph, "saved_tensors_hooks", None)
    if factory is not None:
        try:
            ctx = factory(pack, unpack)
            ctx.__enter__()
        except RuntimeError:  # e.g. hooks disabled by torch.autograd.graph.disable_saved_tensors_hooks
            ctx = None
    try:
        yield
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------------------------
# D: real rows only
# ---------------------------------------------------------------------------------------------

class _Rows:
    __slots__ = ("small", "shape")

    def __init__(self, small, shape):
        self.small = small
        self.shape = shape


def _row_hooks(idx, batch, seq, width):
    rows = batch * seq
    shapes = ((batch, seq, width), (rows, width))

    def pack(t):
        if (t.dtype in (torch.float16, torch.bool) and tuple(t.shape) in shapes
                and t.is_contiguous() and t.device == idx.device):
            with torch.no_grad():
                small = t.detach().reshape(rows, width).index_select(0, idx)
            COUNTS["rows_packed"] += 1
            return _Rows(small, t.shape)
        # Everything else is kept as a detached alias (same storage, same values), never as `t`
        # itself: for a saved OUTPUT, `t` carries the grad_fn of the very node that owns the saved
        # variable, a reference cycle that only that node's backward breaks. A1's throw-away
        # forward graph (never back-propagated) leaked ~0.5 GiB per step this way before
        # sdpa_recompute.py got its private hooks; detaching makes D safe for any such graph.
        return t.detach()

    def unpack(obj):
        if isinstance(obj, _Rows):
            small = obj.small
            with torch.no_grad():
                full = small.new_zeros((rows, width))
                full.index_put_((idx,), small)  # unique indices: a plain scatter, bool included
            return full.view(obj.shape)
        return obj

    return pack, unpack


class _Scope:
    __slots__ = ("idx", "batch", "seq")

    def __init__(self, idx, batch, seq):
        self.idx = idx
        self.batch = batch
        self.seq = seq


@contextlib.contextmanager
def attention_scope(idx, batch, seq, width):
    """Around one packed layer's self_attn call (unpad._layer_packed). Enables A for the SDPA
    calls inside and, if USE_LORA_ROW_COMPACT, D's row hooks for everything saved inside."""
    token = _ACTIVE.set(_Scope(idx, batch, seq))
    try:
        if USE_LORA_ROW_COMPACT:
            pack, unpack = _row_hooks(idx, batch, seq, width)
            with _hooks(pack, unpack):
                yield
        else:
            yield
    finally:
        _ACTIVE.reset(token)


# ---------------------------------------------------------------------------------------------
# A: SDPA key/value, one head per group
# ---------------------------------------------------------------------------------------------

class _Rep:
    __slots__ = ("small", "n_rep")

    def __init__(self, small, n_rep):
        self.small = small
        self.n_rep = n_rep


def _tensor_key(t):
    return (t.data_ptr(), t.dtype, tuple(t.shape), t.stride(), t.device)


def _kv_hooks(k16, v16, n_rep):
    # The hooks live as long as the SavedVariables they packed (until backward), so they must not
    # close over k16 / v16: that kept both full 12-head copies alive next to the small ones (A held
    # MORE than base, test_sdpa_kv_compact on the T4). They keep plain identity keys instead; k16
    # and v16 are alive for the whole SDPA call the hooks are entered for, so a key cannot be
    # reused by another tensor while it matters.
    keys = (_tensor_key(k16), _tensor_key(v16))
    dtype, shape = k16.dtype, tuple(k16.shape)  # cheap checks first, as before

    def pack(t):
        if t.dtype == dtype and tuple(t.shape) == shape and _tensor_key(t) in keys:
            with torch.no_grad():
                # repeat_kv's layout: head h = kv * n_rep + j, every j a copy of j = 0.
                small = t.detach()[:, ::n_rep].contiguous()
            COUNTS["kv_packed"] += 1
            return _Rep(small, n_rep)
        return t.detach()  # never `t` itself (see _row_hooks)

    def unpack(obj):
        if isinstance(obj, _Rep):
            small, n_rep_ = obj.small, obj.n_rep
            b, h_kv, s, d = small.shape
            with torch.no_grad():
                # repeat_kv (sdpa_attention.py:23-24): a contiguous copy, as the saved original.
                return small[:, :, None, :, :].expand(b, h_kv, n_rep_, s, d).reshape(
                    b, h_kv * n_rep_, s, d)
        return obj

    return pack, unpack


def _sdpa_ok(module, query, key, value, attention_mask, dropout, kwargs):
    refs = _REFS
    if _ACTIVE.get() is None or not USE_SDPA_KV_COMPACT:
        return False
    if type(module) is not refs["attn_cls"]:
        return False
    n_rep = getattr(module, "num_key_value_groups", None)
    if type(n_rep) is not int or n_rep <= 1:
        return False
    sa = refs["sdpa_module"]
    if getattr(sa, "_is_torch_npu_available", True) or getattr(sa, "_is_torch_xpu_available", True):
        return False
    if torch.jit.is_tracing() or not torch.is_grad_enabled():
        return False
    if kwargs.get("position_bias") is not None or kwargs.get("output_attentions", False):
        return False
    if dropout != 0.0:
        return False
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool:
        return False
    # fp32 (q/k/v from bitsandbytes' Linear4bit on the fp32 stream, then RoPE) is what training
    # sees; fp16 also matches base, since autocast leaves fp16 inputs as they are and .to() below
    # is then a no-op without a graph node.
    for t in (query, key, value):
        if (not isinstance(t, torch.Tensor) or not t.is_cuda or t.dim() != 4
                or t.dtype not in (torch.float32, torch.float16)):
            return False
    if not (key.requires_grad or value.requires_grad or query.requires_grad):
        return False
    if key.shape != value.shape or key.shape[1] * n_rep != query.shape[1]:
        return False
    if key.shape[0] != query.shape[0] or key.shape[2] != query.shape[2]:
        return False
    if key.shape[-1] != query.shape[-1]:
        return False
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
        return False
    return True


def _sdpa_compact(module, query, key, value, attention_mask, dropout, scaling):
    """sdpa_attention_forward (sdpa_attention.py:79-170) for a bool mask under fp16 autocast."""
    repeat_kv = _REFS["repeat_kv"]
    n_rep = module.num_key_value_groups
    key = repeat_kv(key, n_rep)  # line 99, before the fp16 cast as in base
    value = repeat_kv(value, n_rep)  # line 100
    # What fp16 autocast does to SDPA's floating inputs (fp32 -> fp16, fp16 untouched); the same
    # ToCopy nodes in the graph.
    q16 = query.to(torch.float16)
    k16 = key.to(torch.float16)
    v16 = value.to(torch.float16)
    del key, value
    pack, unpack = _kv_hooks(k16, v16, n_rep)  # holds keys, not k16 / v16
    with _hooks(pack, unpack):
        # Lines 108 + 124: with a mask is_causal is False; lines 158-167, no sdpa_kwargs (no GQA).
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q16, k16, v16, attn_mask=attention_mask, dropout_p=dropout, scale=scaling,
            is_causal=False)
    attn_output = attn_output.transpose(1, 2).contiguous()  # line 168
    return attn_output, None


def patch_sdpa_kv_compact():
    """Wrap ALL_ATTENTION_FUNCTIONS["sdpa"]. Idempotent; no CUDA work. True when in place."""
    try:
        from transformers.integrations import sdpa_attention as sa
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen2 import modeling_qwen2 as mq

        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
        attn_cls = mq.Qwen2Attention
    except Exception:  # noqa: BLE001 - a missing or fake library means nothing to patch
        return False
    if getattr(current, _MARK, False):
        return True
    if current is not sa.sdpa_attention_forward:
        return False  # someone else's attention function: leave it alone
    _REFS.update(attn_cls=attn_cls, sdpa_module=sa, repeat_kv=sa.repeat_kv, original=current)
    original = current

    @functools.wraps(original)
    def sdpa_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                               scaling=None, is_causal=None, **kwargs):
        try:
            ok = _sdpa_ok(module, query, key, value, attention_mask, dropout, kwargs)
        except Exception:  # noqa: BLE001 - an unexpected object is never ours to handle
            ok = False
        if ok:
            COUNTS["sdpa_compact"] += 1
            return _sdpa_compact(module, query, key, value, attention_mask, dropout, scaling)
        COUNTS["sdpa_original"] += 1
        return original(module, query, key, value, attention_mask, dropout=dropout,
                        scaling=scaling, is_causal=is_causal, **kwargs)

    setattr(sdpa_attention_forward, _MARK, True)
    # GeneralInterface.__setitem__ writes the instance's local mapping (utils/generic.py:1116-1118),
    # which __getitem__ reads first (lines 1110-1114); Qwen2 imports this very instance
    # (modeling_qwen2.py:26) and looks the function up on every call (modeling_qwen2.py:216-218).
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward
    return True
