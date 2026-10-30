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
"""

import contextlib

import torch

USE_SDPA_RECOMPUTE = False  # read on every call; solution.apply() sets it

_MARK = "_soup_delta_sdpa_recompute"
COUNTS = {"recompute": 0, "original": 0}

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


def _sdpa(q, k, v, mask, scale):
    # The call of sdpa_attention_forward (integrations/sdpa_attention.py:158-167) with a mask:
    # dropout 0 (guarded), is_causal False (:124), no enable_gqa (k, v already repeated).
    return _REFS["sdpa"](q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=False)


class _SDPARecompute(torch.autograd.Function):
    """SDPA(q, repeat_kv(k), repeat_kv(v), mask) in fp16 that saves fp16 q, k, v and the mask."""

    @staticmethod
    def forward(ctx, query, key, value, mask, n_rep, scale):
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
            res = _sdpa(q, k, v, mask, scale)
        out = res.detach()
        del res, q, k, v
        ctx.save_for_backward(q16, k16, v16, mask)
        ctx.n_rep = n_rep
        ctx.scale = scale
        ctx.in_dtypes = (query.dtype, key.dtype, value.dtype)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        need = ctx.needs_input_grad[:3]
        if dout is None or not any(need):
            return None, None, None, None, None, None
        q16, k16, v16, mask = ctx.saved_tensors
        n_rep, scale = ctx.n_rep, ctx.scale
        repeat_kv = _REFS["repeat_kv"]
        with torch.enable_grad(), torch.autocast(q16.device.type, enabled=False), _private_saves():
            q = q16.detach().requires_grad_(need[0])
            k = repeat_kv(k16, n_rep).detach().requires_grad_(need[1])
            v = repeat_kv(v16, n_rep).detach().requires_grad_(need[2])
            out = _sdpa(q, k, v, mask, scale)
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
        return result[0], result[1], result[2], None, None, None


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
            out = _SDPARecompute.apply(query, key, value, attention_mask, n_rep, kwargs.get("scaling"))
            COUNTS["recompute"] += 1
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
