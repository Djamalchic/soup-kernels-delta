"""A1: SDPA keeps only fp16 q and the un-repeated k, v for backward and recomputes the rest there.

What base keeps per decoder layer (transformers 5.16.1, integrations/sdpa_attention.py):
with a padding mask `use_gqa_in_sdpa` is False (:27-36), so k and v are repeated to all 12 heads
(`repeat_kv`, :15-24 and :96-102) before `F.scaled_dot_product_attention` (:158-167). Under fp16
autocast the call runs on fp16 copies, and its autograd node keeps q, the repeated k and v, the
output, the logsumexp and the float bias SDPA builds from the bool mask. Per padded token that is
4 x [12 x 128] fp16 = 12 KiB, plus a [B, 1, T, T] fp16 bias (2 MiB at B=4, T=512), in each of the
28 layers.

`_SDPARecompute` keeps q (fp16, 3 KiB/token), k and v before the repeat (fp16, 2 x 0.5 KiB/token)
and a reference to the mask (one tensor shared by all layers, created once per forward by
create_causal_mask). In backward it repeats k and v again, runs the same SDPA call once more with
grad enabled and lets torch's own SDPA backward produce the gradients. Saving per layer:
~8 KiB per padded token + the bias = ~18 MiB at B*T = 2048 [estimate]; cost: one extra SDPA
forward per layer in backward (~5 % of the base step is attention in total, tz.md s.2).

Why the numbers are the base numbers:
- forward: under fp16 autocast base SDPA sees fp16(q), fp16(repeat_kv(k)), fp16(repeat_kv(v)).
  repeat_kv only copies, so fp16(repeat_kv(k)) == repeat_kv(fp16(k)) bit for bit; `.to()` keeps
  q's strides (preserve_format) like autocast's cast; same mask, same scale, dropout 0, is_causal
  False (:96-102 with a mask, :124 and :158-167). Same inputs -> same kernel -> same output.
- backward: the recompute feeds the same tensors to the same kernel, so out and logsumexp are the
  same bits; SDPA's backward then gets the same inputs as in base. Its gradients go through the
  same casts and the same head sum as base's autograd (ToCopy back to the input dtype, then
  repeat_kv's expand backward = sum over the 6 repeated heads).
- no RNG: dropout_p is 0 (guarded), so neither call touches the generator.

Hook: the guarded wrapper replaces the "sdpa" entry of transformers' AttentionInterface through its
public `register` classmethod (utils/generic.py:1130-1132, modeling_utils.py:5148-5159); Qwen2Attention
looks the function up on every call (models/qwen2/modeling_qwen2.py:216-229). Anything the guard
does not recognise goes to the genuine sdpa_attention_forward with the untouched arguments.

Guards: flag on; grad enabled and an input needing grad; not tracing/compiling; dropout == 0; no
position_bias / is_causal / output_attentions; a 4D bool mask on the device (a float mask would be
cast by autocast; None means base already uses enable_gqa without a repeat); CUDA 4D q/k/v of one
dtype, q_len == kv_len, heads = kv_heads * num_key_value_groups; transformers' own
use_gqa_in_sdpa(...) says "repeat"; compute dtype fp16 (fp16 inputs with autocast off or fp16, or
fp32 inputs under fp16 autocast). No Triton kernel and no CUDA work at import or patch time.

Private saves (v6 fix, the step-22 OOM on Kaggle): the forward builds a throw-away SDPA graph
(grad-requiring leaves, so SDPA takes its training variant) that is never back-propagated. Default
saved-tensor hooks are thread-local and also apply inside a custom Function's forward, so any outer
hook (solution/saved_compact.py's D scope around self_attn, Soup/TRL activation offloading, a test's
counter) used to see that graph's saves. SDPA's node saves its own OUTPUTS (out, logsumexp); an
outer pack hook that returns the tensor itself stores the output with its grad_fn, i.e. the node
that owns the saved variable: a C++ reference cycle that only backward's release breaks and
Python's GC cannot see. This graph never gets a backward, so q, the 12-head k/v, out, lse and the
bias of every layer stayed alive forever (~12 KiB per padded token per layer, ~0.5 GiB per
step). The inner calls now run under our own innermost hooks (`_private_saves`), which keep a
detached alias (no grad_fn, no cycle; SavedVariable rebuilds the autograd metadata on unpack), so
outer hooks only ever see what `_SDPARecompute` itself saves (q16, k16, v16, mask) and the throw-away graph is freed with
`res`. Same values: hooks only decide how a saved tensor is stored.

Idea credit: activation recomputation (Chen et al. 2016, arXiv:1604.06174) applied to the attention
core only, as in Korthikanti et al. 2022 "Reducing Activation Recomputation in Large Transformer
Models" (arXiv:2205.05198, selective recomputation). Code written from scratch.

v7 I1, causal-only SDPA on U's packed path (USE_CAUSAL_PADDED, off unless solution.apply() sets
it). solution/unpad.py builds the [B, 1, T, T] bool mask with transformers' genuine
create_causal_mask from a 2D mask it has validated on the device as binary, right-padded and with a
real first token in every row (unpad.py _row_indices), and enters `causal_mask_scope(mask)` around
its layer loop. With no or/and mask function, no block ids and a 2D mask (so no packed-sequence
mask, masking_utils.py:851-857), that mask is causal AND padding (masking_utils.py:496-518):
mask[b, 0, i, j] = (j <= i) and (j < L_b). For a REAL query row i < L_b every pad key j >= L_b > i
is already excluded by j <= i, so SDPA(is_causal=True, attn_mask=None) gives the same attention.
Pad query rows (i >= L_b) now also see pad keys L_b..i: their output differs, but U drops those rows
(pack_rows after self_attn, unpad.py _layer_packed), so their dO is exactly 0 and their share of
dQ / dK / dV is exactly 0 (dV += P^T dO = 0; dP = dO V^T = 0, D = rowsum(dO * O) = 0, dS = 0).
No row is fully masked under causal (key 0 is always visible), so no NaN. The cutlass
mem-efficient kernel then skips the key blocks above the diagonal (forward) and the query blocks
below it (backward), and torch no longer turns the bool mask into a float bias per call (where /
fill kernels). Real rows are expected equal up to fp16 ulp-level drift [not proven bit-exact].
Only the exact mask object entered by U qualifies (identity / same storage, shape, stride); any
other mask, a sliding window, or the flag off runs the v6.1 dense-mask call. GQA stays on
repeat_kv (no enable_gqa: the sm75 mem-efficient path must not fall back to math).

v9 M3, real rows only (USE_ROW_COMPACT, off unless solution.apply() sets it). On U's packed path
(solution/unpad.py _row_scope enters `row_compact_scope(mask, idx, batch, seq)` around its layer
loop, for the same provenance-checked mask as I1) `_SDPARecompute` saves q16 / k16 / v16 as
[N, heads, 128] gathers of the real rows (idx = U's flat row index, N real tokens) instead of the
padded [B, heads, T, 128] tensors, plus the two [N] int64 vectors (b, t) = divmod(idx, T), made
once per forward on the device (no host sync) and shared by all layers. Backward scatters them into
zero-filled tensors of the original layout (BTHD-strided view or contiguous BHTD) and runs the
unchanged recompute. The forward SDPA call still sees the full tensors, so `out` is v8b's bit for
bit. Saving per layer: 4 KiB per PADDING row (3 KiB q + 2 x 0.5 KiB k/v); 28 x (2048 - 1350) x
4 KiB = ~76 MiB at P = 2048, N = 1350 [estimate].

Why backward is unchanged. Notation per batch row b with L_b real tokens: rebuilt q'_i = q_i,
k'_j = k_j, v'_j = v_j on real rows (i, j < L_b) and 0 on pad rows. Write S = scale * Q K^T,
P = softmax(S + bias), O = P V, D_i = sum_d dO_i,d O_i,d, dP = dO V^T, dS = P * (dP - D),
dQ = scale * dS K, dK = scale * dS^T Q, dV = P^T dO (what torch's mem-efficient backward computes).
(1) dO_i = 0 exactly on every pad query row i: the layer output's pad rows are dropped by U's
    pack_rows (backward = zeros + index_copy, unpad.py _PackRows), and o_proj (+ LoRA if any) and
    the transpose/reshape are row-wise, so 0 @ W = 0 reaches dO (all values finite).
(2) Real query row i never sees a pad key j >= L_b: dense path, mask[b,0,i,j] = (j <= i) and
    (j < L_b) (masking_utils.py:496-518); I1 causal path, j >= L_b > i is above the diagonal
    (right padding, validated by unpad._row_indices). Its masked scores are -inf (bool mask -> -inf
    bias, or the kernel's causal fill) whatever k'_j is, so P_ij = 0 and P_ij * v'_j = 0 for finite
    v'_j. Every real row i therefore has the same S_ij on visible j, the same running max, the same
    O_i and logsumexp as v8b, in the same tile order.
(3) Pad query rows change (q'_i = 0, and under causal they see k'_j = 0, v'_j = 0 for pad keys), but
    their O_i stays finite (key 0 is visible to every row, so no fully masked row) and with
    dO_i = 0: D_i = 0, dP_ij = 0, dS_ij = P_ij * 0 = 0. So their share of dK / dV is 0 * q'_i = 0
    and P_ij * 0 = 0, as in v8b, and their dQ_i = 0, as in v8b.
(4) Real rows: dS_ij uses only real-row values (2), with dS_ij = 0 on masked j; dQ_i = sum_j dS_ij k_j
    gets 0 * k'_j = 0 terms on masked j instead of 0 * k_j = 0; dK_j / dV_j for real keys get
    their real-query terms unchanged and zero terms from pad queries (3). Pad keys get dK = dV = 0
    in both versions (no visible real query, pad queries contribute 0).
So every gradient that leaves the layer is v8b's; the only possible difference is the sign of an
exactly-zero sum (+0 vs -0), which compares equal. q's / k's / v's pad-row gradients feed only
pad rows of q_proj / k_proj / v_proj inputs (dropped by U's unpack_rows backward) and LoRA weight
gradients through x = 0 rows (0 * finite = 0), so nothing downstream changes either.
Guards: flag on and A1 qualified; inside the scope; the exact scoped mask object; q/k [B, ., T, .]
with (B, T) of the scope; no sliding window; idx a 1-D int64 tensor on q's device; q16/k16/v16 in
BTHD-strided or contiguous layout. Anything else saves the full tensors (v8b). No Triton kernel; the
gathers are torch index ops, no RNG.
"""

import contextlib

import torch

USE_SDPA_RECOMPUTE = False  # read on every call; solution.apply() sets it
USE_CAUSAL_PADDED = False  # v7 I1; read on every call; solution.apply() sets it
USE_ROW_COMPACT = False  # v9 M3; read on every call; solution.apply() sets it

_MARK = "_soup_delta_sdpa_recompute"
# "rows": forward calls that saved real rows only (M3); "rows_rebuilt": their backward rebuilds.
COUNTS = {"recompute": 0, "original": 0, "causal": 0, "rows": 0, "rows_rebuilt": 0}

# v7 I1: the mask object U entered with causal_mask_scope(), else None.
_CAUSAL_MASK = None
# v9 M3: the _RowScope U entered with row_compact_scope(), else None.
_ROW_SCOPE = None


class _RowScope:
    __slots__ = ("mask", "idx", "batch", "seq", "bt")

    def __init__(self, mask, idx, batch, seq):
        self.mask, self.idx, self.batch, self.seq = mask, idx, batch, seq
        self.bt = None

    def rows(self):
        """(b, t) = divmod(idx, seq) as two [N] int64 tensors, made once per scope (no sync)."""
        if self.bt is None:
            b = torch.div(self.idx, self.seq, rounding_mode="floor")
            self.bt = (b, self.idx - b * self.seq)
        return self.bt


@contextlib.contextmanager
def row_compact_scope(mask, idx, batch, seq):
    """While active, A1 calls with exactly `mask` save q/k/v of the rows in idx only (M3)."""
    global _ROW_SCOPE
    previous = _ROW_SCOPE
    _ROW_SCOPE = _RowScope(mask, idx, batch, seq)
    try:
        yield
    finally:
        _ROW_SCOPE = previous


@contextlib.contextmanager
def causal_mask_scope(mask):
    """While active, SDPA calls whose mask is exactly `mask` may run as is_causal=True (I1)."""
    global _CAUSAL_MASK
    previous = _CAUSAL_MASK
    _CAUSAL_MASK = mask
    try:
        yield
    finally:
        _CAUSAL_MASK = previous


def _same_mask(a, b):
    if a is b:
        return True
    return (a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride()
            and a.dtype == b.dtype and a.device == b.device)


def _causal_ok(query, attention_mask, kwargs):
    """True when this (already A1-qualified) call may drop the mask for is_causal=True. Reads only."""
    if not USE_CAUSAL_PADDED:
        return False
    scoped = _CAUSAL_MASK
    if scoped is None or not isinstance(scoped, torch.Tensor):
        return False
    if kwargs.get("sliding_window") is not None:
        return False
    # q_len == kv_len is already guarded by _fast_n_rep; the scoped mask is [B, 1, T, T].
    b, _, t, _ = query.shape
    if tuple(attention_mask.shape) != (b, 1, t, t):
        return False
    return _same_mask(attention_mask, scoped)


def _rows_ok(query, key, attention_mask, kwargs):
    """M3: (b, t) index vectors when this (already A1-qualified) call may save real rows only,
    else None. Reads only, apart from making the scope's two [N] vectors on first use."""
    if not USE_ROW_COMPACT:
        return None
    scope = _ROW_SCOPE
    if scope is None or not isinstance(scope.idx, torch.Tensor):
        return None
    if not isinstance(scope.mask, torch.Tensor):
        return None
    if kwargs.get("sliding_window") is not None:
        return None
    b, _, t, _ = query.shape
    if (b, t) != (scope.batch, scope.seq) or key.shape[0] != b or key.shape[2] != t:
        return None
    if tuple(attention_mask.shape) != (b, 1, t, t) or not _same_mask(attention_mask, scope.mask):
        return None
    idx = scope.idx
    if idx.ndim != 1 or idx.dtype != torch.int64 or idx.device != query.device or idx.numel() == 0:
        return None
    return scope.rows()


def _pack_rows(x, bi, ti):
    """[B, H, T, D] -> ([N, H, D] rows (bi, ti), layout), or None for a layout we do not rebuild.
    Runs inside Function.forward (grad off)."""
    rows = x.transpose(1, 2)  # [B, T, H, D]
    if rows.is_contiguous():
        layout = "bthd"  # q/k after RoPE, v as v_proj's view (modeling_qwen2.py:206-211)
    elif x.is_contiguous():
        layout = "bhtd"
    else:
        return None
    return rows[bi, ti].contiguous(), layout  # advanced-index gather, a fresh [N, H, D] tensor


def _unpack_rows(small, bi, ti, batch, seq, layout):
    """Inverse of _pack_rows: zeros on the rows not listed, the original layout and strides."""
    n, h, d = small.shape
    if layout == "bthd":
        full = small.new_zeros((batch, seq, h, d))
        full.index_put_((bi, ti), small)  # unique (b, t) pairs: a plain scatter
        return full.transpose(1, 2)
    full = small.new_zeros((batch, h, seq, d))
    full.transpose(1, 2).index_put_((bi, ti), small)
    return full


# Filled by patch_sdpa_recompute(): the genuine sdpa_attention_forward, repeat_kv, use_gqa_in_sdpa
# and torch.nn.functional.scaled_dot_product_attention. Soup's tests may swap sys.modules["torch"]
# later (tests/test_callback.py:165-174), so nothing is re-imported per call.
_REFS = {}


def _autocast_fp16_on(device_type):
    """(enabled, dtype_is_fp16) for device_type; (None, None) if the torch API is unknown."""
    try:
        try:
            enabled = torch.is_autocast_enabled(device_type)
        except TypeError:
            enabled = torch.is_autocast_enabled()
        if not enabled:
            return False, False
        try:
            dtype = torch.get_autocast_dtype(device_type)
        except (AttributeError, TypeError):
            dtype = torch.get_autocast_gpu_dtype()
        return True, dtype == torch.float16
    except Exception:  # noqa: BLE001
        return None, None


def _compiling():
    try:
        if torch.jit.is_tracing():
            return True
        compiler = getattr(torch, "compiler", None)
        return bool(compiler is not None and compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return True


def _detached(t):
    # A pack hook must not return its input: for a saved output that would store the grad_fn of
    # the node that owns the saved variable (a reference cycle). A detached alias shares storage
    # and version counter; unpack restores grad_fn / requires_grad from the saved metadata.
    return t.detach()


def _same(t):
    return t


@contextlib.contextmanager
def _private_saves():
    """Innermost saved-tensor hooks around our inner SDPA calls, so no outer hook sees (or keeps)
    the tensors those calls save. If hooks cannot be pushed here (e.g. inside
    torch.autograd.graph.disable_saved_tensors_hooks, where no outer hook can be active either),
    run without them."""
    cm = None
    try:
        cm = torch.autograd.graph.saved_tensors_hooks(_detached, _same)
        cm.__enter__()
    except Exception:  # noqa: BLE001
        cm = None
    try:
        yield
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)


def _sdpa(q, k, v, mask, scale, causal=False):
    # The call of sdpa_attention_forward (integrations/sdpa_attention.py:158-167) with a mask:
    # dropout 0 (guarded), is_causal False (:124), no enable_gqa (k, v already repeated).
    # I1 (causal): no mask and is_causal=True, the call base itself makes when the mask is None
    # (:124); top-left alignment equals bottom-right here because q_len == kv_len (guarded).
    if causal:
        return _REFS["sdpa"](q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=True)
    return _REFS["sdpa"](q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=False)


class _SDPARecompute(torch.autograd.Function):
    """SDPA(q, repeat_kv(k), repeat_kv(v), mask) in fp16 that saves fp16 q, k, v and the mask.
    With causal=True (I1) the mask is neither used nor saved. With bi / ti (M3, [N] int64 row
    vectors) q, k, v are saved for those rows only and rebuilt with zero pad rows in backward."""

    @staticmethod
    def forward(ctx, query, key, value, mask, n_rep, scale, causal=False, bi=None, ti=None):
        causal = bool(causal)
        repeat_kv = _REFS["repeat_kv"]
        f16 = torch.float16
        # .to() keeps the strides of query (a transposed view), as autocast's cast does.
        q16 = query if query.dtype == f16 else query.to(f16)
        k16 = key if key.dtype == f16 else key.to(f16)
        v16 = value if value.dtype == f16 else value.to(f16)
        # Run with grad-requiring leaves, as base does: SDPA then takes the same training variant
        # (it also computes the logsumexp it needs for backward), so out has base's bits. The
        # little graph built here is dropped with `res`; only the detached output leaves.
        # _private_saves: outer hooks (D's scope, offloading) must not see or keep this graph.
        device_type = query.device.type
        with torch.enable_grad(), torch.autocast(device_type, enabled=False), _private_saves():
            q = q16.detach().requires_grad_(True)
            k = repeat_kv(k16, n_rep).detach().requires_grad_(True)
            v = repeat_kv(v16, n_rep).detach().requires_grad_(True)
            res = _sdpa(q, k, v, mask, scale, causal)
        out = res.detach()
        del res, q, k, v
        ctx.rows = None
        if bi is not None and ti is not None:  # M3, after the call above (full tensors)
            packed = [_pack_rows(t, bi, ti) for t in (q16, k16, v16)]
            if all(p is not None for p in packed):
                ctx.rows = (q16.shape[0], q16.shape[2], tuple(p[1] for p in packed))
                q16, k16, v16 = (p[0] for p in packed)
                COUNTS["rows"] += 1
            del packed
        if ctx.rows is None:
            bi = ti = None
        ctx.save_for_backward(q16, k16, v16, None if causal else mask, bi, ti)
        ctx.causal = causal
        ctx.n_rep = n_rep
        ctx.scale = scale
        ctx.in_dtypes = (query.dtype, key.dtype, value.dtype)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        need = ctx.needs_input_grad[:3]
        if dout is None or not any(need):
            return None, None, None, None, None, None, None, None, None
        q16, k16, v16, mask, bi, ti = ctx.saved_tensors
        if ctx.rows is not None:  # M3: zero pad rows, original layout (proof in the docstring)
            batch, seq, layouts = ctx.rows
            with torch.no_grad():
                q16, k16, v16 = (_unpack_rows(t, bi, ti, batch, seq, lay)
                                 for t, lay in zip((q16, k16, v16), layouts))
            COUNTS["rows_rebuilt"] += 1
        del bi, ti
        n_rep, scale, causal = ctx.n_rep, ctx.scale, ctx.causal
        repeat_kv = _REFS["repeat_kv"]
        with torch.enable_grad(), torch.autocast(q16.device.type, enabled=False), _private_saves():
            q = q16.detach().requires_grad_(need[0])
            k = repeat_kv(k16, n_rep).detach().requires_grad_(need[1])
            v = repeat_kv(v16, n_rep).detach().requires_grad_(need[2])
            out = _sdpa(q, k, v, mask, scale, causal)
            if causal:
                COUNTS["causal"] += 1
            leaves = [t for t, n in zip((q, k, v), need) if n]
            grads = list(torch.autograd.grad(out, leaves, dout))
        del out, q, k, v

        result = []
        for i, n in enumerate(need):
            if not n:
                result.append(None)
                continue
            g = grads.pop(0)
            g = g.to(ctx.in_dtypes[i])  # base: ToCopyBackward of autocast's cast
            if i > 0 and n_rep > 1:
                # base: repeat_kv's reshape + expand backward (sdpa_attention.py:23-24) = sum over
                # the n_rep copies of each kv head, in the input dtype.
                b, h_kv, t, d = (k16 if i == 1 else v16).shape
                g = g.reshape(b, h_kv, n_rep, t, d).sum(2)
            result.append(g)
        return result[0], result[1], result[2], None, None, None, None, None, None


def _fast_n_rep(module, query, key, value, attention_mask, args, kwargs):
    """num_key_value_groups when the recompute reproduces base, else None. Reads only."""
    if args or not torch.is_grad_enabled() or _compiling():
        return None
    if kwargs.get("dropout", 0.0) != 0.0:
        return None
    if kwargs.get("position_bias") is not None or kwargs.get("is_causal") is not None:
        return None
    if kwargs.get("output_attentions", False):
        return None
    scale = kwargs.get("scaling")
    if scale is not None and not isinstance(scale, float):
        return None
    tensors = (query, key, value)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return None
    if not any(t.requires_grad for t in tensors):
        return None
    if not all(t.is_cuda and t.ndim == 4 for t in tensors):
        return None
    if any(t.dtype not in (torch.float16, torch.float32) for t in tensors):
        return None
    if key.shape != value.shape or query.shape[0] != key.shape[0]:
        return None
    if query.shape[2] != key.shape[2] or query.shape[3] != key.shape[3] or query.numel() == 0:
        return None
    n_rep = getattr(module, "num_key_value_groups", 1)
    if not isinstance(n_rep, int) or n_rep < 1 or query.shape[1] != key.shape[1] * n_rep:
        return None
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool:
        return None
    if attention_mask.ndim != 4 or attention_mask.device != query.device:
        return None
    if n_rep > 1 and _REFS["use_gqa_in_sdpa"](attention_mask, key, value):
        return None  # base would pass enable_gqa=True instead of repeating
    enabled, fp16 = _autocast_fp16_on(query.device.type)
    if enabled is None:
        return None
    if enabled:
        if not fp16:
            return None  # e.g. bf16 autocast: base would not compute in fp16
    elif any(t.dtype != torch.float16 for t in tensors):
        return None  # no autocast and an fp32 input: base SDPA would not run in fp16
    # Under fp16 autocast every fp32 input is cast to fp16 by SDPA's autocast rule; mixed fp32 / fp16
    # inputs (e.g. nn.Linear v under autocast, fp32 q after RoPE) are cast one by one, as we do.
    return n_rep


def sdpa_forward(module, query, key, value, attention_mask, *args, **kwargs):
    """Drop-in for transformers' sdpa_attention_forward (same signature and return value)."""
    original = _REFS["original"]
    if USE_SDPA_RECOMPUTE:
        try:
            n_rep = _fast_n_rep(module, query, key, value, attention_mask, args, kwargs)
        except Exception:  # noqa: BLE001 - an unexpected object is never ours to handle
            n_rep = None
        if n_rep is not None:
            try:  # I1: reads only; any surprise keeps the v6.1 dense-mask call
                causal = _causal_ok(query, attention_mask, kwargs)
            except Exception:  # noqa: BLE001
                causal = False
            try:  # M3: any surprise keeps v8b's full saves
                rows = _rows_ok(query, key, attention_mask, kwargs)
            except Exception:  # noqa: BLE001
                rows = None
            bi, ti = rows if rows is not None else (None, None)
            # All nine arguments always (backward returns nine gradients).
            out = _SDPARecompute.apply(query, key, value, attention_mask, n_rep,
                                       kwargs.get("scaling"), causal, bi, ti)
            COUNTS["recompute"] += 1
            if causal:
                COUNTS["causal"] += 1
            # sdpa_attention_forward returns (attn_output.transpose(1, 2).contiguous(), None) (:168-170)
            return out.transpose(1, 2).contiguous(), None
    COUNTS["original"] += 1
    return original(module, query, key, value, attention_mask, *args, **kwargs)


def patch_sdpa_recompute():
    """Register the wrapper as "sdpa". Idempotent; no CUDA work. True when in place."""
    try:
        import torch.nn.functional as F
        from transformers.integrations import sdpa_attention as sa
        from transformers.modeling_utils import AttentionInterface
    except Exception:  # noqa: BLE001 - a missing or fake library means nothing to patch
        return False
    mapping = getattr(AttentionInterface, "_global_mapping", None)
    if not isinstance(mapping, dict):
        return False
    current = mapping.get("sdpa")
    if getattr(current, _MARK, False):
        return True
    if current is not sa.sdpa_attention_forward:
        return False  # someone else's "sdpa": leave it alone
    _REFS.clear()
    _REFS.update(original=current, repeat_kv=sa.repeat_kv, use_gqa_in_sdpa=sa.use_gqa_in_sdpa,
                 sdpa=F.scaled_dot_product_attention)
    setattr(sdpa_forward, _MARK, True)
    sdpa_forward.__wrapped__ = current
    AttentionInterface.register("sdpa", sdpa_forward)
    return True


def unpatch_sdpa_recompute():
    """Put the genuine function back (tests)."""
    try:
        from transformers.modeling_utils import AttentionInterface
    except Exception:  # noqa: BLE001
        return
    current = AttentionInterface._global_mapping.get("sdpa")
    if getattr(current, _MARK, False):
        AttentionInterface.register("sdpa", current.__wrapped__)
