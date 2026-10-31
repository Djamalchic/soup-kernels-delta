"""U: run the token-wise half of every Qwen2 decoder layer on the real (unpadded) rows only.

TRL's collator right-pads every batch to its longest row (trl/trainer/sft_trainer.py:182, 215-217),
so at batch 4 roughly 40-47 % of the [B, T] positions are padding [estimate].
RMSNorm, the residual adds and the MLP work row by row, so on the fast path they run on the N real
rows ([N, C]). Attention keeps the exact base shapes ([B, T, C] in and out): q/k/v/o, LoRA dropout,
RoPE, the causal+padding mask and SDPA see tensors of the base shape in the base call order, so
the LoRA dropout masks and the CUDA Philox offsets are identical to base.

One guarded class-level patch: `Qwen2Model.forward` (transformers 5.16.1,
models/qwen2/modeling_qwen2.py:339-402). The fast path rebuilds lines 352-402 for the training case
and assembles each layer from its submodules (the body of Qwen2DecoderLayer.forward, lines
279-298), all called as modules, so their own class patches (K3 rmsnorm, K4 MLP, S4 inside the
MLP) and hooks still apply:
- input_layernorm / post_attention_layernorm / model.norm on [N, C] (K3's rmsnorm when patched);
- self_attn on [B, T, C] (zeros on the padding rows), its output gathered back to [N, C];
- residual add + post_attention_layernorm -> K3's fused add_rmsnorm when K3's decoder-layer patch is
  installed (same conditions K3 checks), otherwise `r + a` and the norm module (lines 291, 295);
- mlp on [N, C] (K4 -> K1 when their guards pass; N >= 129 keeps bitsandbytes off its small-M GEMM);
- last_hidden_state scattered back to [B, T, C] with zeros on padding, so the base head and K2
  (which takes [B, T, C] and the mask) work unchanged.

What the fast path reproduces of the two decorators (lines 339-340):
- merge_with_config_defaults (utils/generic.py:1000-1077): use_cache is resolved the same way
  (lines 1019-1033) and must be falsy; `is_causal` (lines 1049-1057) and `debug_io` (line 1060) must
  be absent.
- capture_outputs (utils/output_capturing.py:215-286): `return_dict` is popped (line 234) and must not
  be False (line 281); output_hidden_states / output_attentions are read as at lines 238-241 and must
  be falsy; the output collector is set to an empty dict around the body exactly as at lines 253-265,
  so transformers' installed-but-idle capture hooks (lines 105-110) behave as in base.
Everything else goes to the untouched original, which keeps both decorators.

Guards (types, shapes and values only, never the environment): training with grad enabled; exact
Qwen2Model / Qwen2DecoderLayer / Qwen2RMSNorm types; the decoder-layer forward is the genuine one
or K3's patch of it; no gradient checkpointing; SDPA; attention_dropout == 0; full attention only;
CUDA int input_ids [B, T] on the embedding's device; a 2D bool/int attention_mask of the same shape
that is binary, right-padded and has no empty row (checked on the device and folded into the one
`nonzero`, idea from a GPT-6-Astra draft); no inputs_embeds / position_ids / past_key_values /
cache; no hooks or instance-level forward on the decoder layers we bypass except transformers' own
idle output-capture hooks; no global module hooks; no active dropout inside the MLP (LoRA on MLP
projections would draw RNG on N rows); N >= 129 and at least 1/32 of the rows padding. Any exception
while guards run means the original (Soup's tests pass fake modules); nothing is retried once
attention may have consumed RNG.

v6 peak-timing additions, both bit-identical to the ops they replace (module defaults off; apply()
turns D and B on, while A, A1's fallback, stays off): the attention call runs inside
solution/saved_compact.py's attention_scope (A: SDPA keeps one head per GQA group of its repeated
key/value; D: LoRA keep-masks and LoRA-A inputs keep only the real rows), and with K3 the add +
norm + S4 MLP + residual add run as fused_block.py's single node, saving (r, a), not (r_mid, x16).

v7 I4 (S4_OFF_FRACTION; 0.0 = v6.1, as submitted): the first int(n_layers * fraction) layers
skip B and S4 and run K3's add_rmsnorm + K4's MLP, which keeps fp16 gate/up for backward instead
of recomputing them (same bits; see _mlp_no_recompute). First layers, because their backward runs
last, when most saved activations are already freed, so K4's larger backward transient (fresh
dgate/dup next to the kept gate/up) stays off the step's peak.

No Triton kernel here (gather/scatter are torch index ops); nothing runs on CUDA at import or patch
time. Idea credit: removing padding for token-wise ops as in ByteDance's Effective Transformer and
the "unpad" path of FlashAttention's varlen BERT; code written from scratch.
"""

import contextlib
import functools
import importlib

import torch

USE_UNPAD = False  # a record set by solution.apply(), which installs the patch only when U is on

# v7 I4: the first int(num_hidden_layers * S4_OFF_FRACTION) packed layers skip S4 + B and run the
# K3 add_rmsnorm + K4 MLP (fp16 gate/up kept for backward, no recompute). Same values bit for bit
# (see _mlp_no_recompute); it spends memory (~38 KiB per real token per layer, 32 KiB on layer 0)
# to save the gate/up recompute GEMMs and dequants. A fraction, not a count, so 2-3-layer test
# models get 0 off layers. solution.apply() sets it; 0.0 is exactly v6.1.
S4_OFF_FRACTION = 0.0

_MARK = "_soup_delta_unpad"
_K3_MARK = "_soup_delta_addnorm"  # = solution/layers.py _MARK (K3)
_S4_MARK = "_soup_delta_mlp_recompute"  # = solution/mlp_recompute.py _PATCH_MARK

# At M <= 128 bitsandbytes on a T4 (sm75) runs gate/up through its own small-M GEMM, and down at
# M <= 64 (bitsandbytes/backends/cuda/ops.py:703-709; M = A.numel() // K at :947-948), i.e. not
# through K1's dequant + F.linear. Such batches keep the original forward.
_MIN_N = 129
# Fewer than rows / 32 padding rows (~3 %): the scatter/gather traffic is not paid back by the MLP
# work saved [estimate], so keep the exact original.
_MIN_PAD_DIV = 32

_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")

# Calls that took each path. Python ints only; tests read them.
# "s4_off": packed layers whose MLP ran K4 directly (I4); "s4_off_fallback": I4 wanted but the MLP
# was called as a module (guards failed), i.e. v6.1's S4 path.
COUNTS = {"packed": 0, "original": 0, "s4_off": 0, "s4_off_fallback": 0}

# Real classes and functions captured by patch_qwen2_model_unpad(); Soup's tests may replace
# sys.modules["torch"] later (tests/test_callback.py:165-174), so nothing is re-imported per call.
_REFS = {}


# ---------------------------------------------------------------------------------------------
# Row gather / scatter. Both save only the int64 index; neither touches the RNG or uses atomics.
# ---------------------------------------------------------------------------------------------

class _PackRows(torch.autograd.Function):
    """[rows, C] -> [N, C] = x[idx]. Backward copies the gradient into zeros (idx is unique)."""

    @staticmethod
    def forward(ctx, x2, idx):
        ctx.save_for_backward(idx)
        ctx.rows = x2.shape[0]
        return x2.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        out = grad.new_zeros((ctx.rows, grad.shape[-1]))
        out.index_copy_(0, idx, grad)
        return out, None


class _UnpackRows(torch.autograd.Function):
    """[N, C] -> [rows, C] with x at idx and zeros elsewhere. Backward is grad[idx]."""

    @staticmethod
    def forward(ctx, x, idx, rows):
        ctx.save_for_backward(idx)
        out = x.new_zeros((rows, x.shape[-1]))
        out.index_copy_(0, idx, x)
        return out

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        return grad.index_select(0, idx), None, None


def pack_rows(x_bt, idx):
    """[B, T, C] -> [N, C] (rows of the flattened batch listed in idx)."""
    return _PackRows.apply(x_bt.reshape(-1, x_bt.shape[-1]), idx)


def unpack_rows(x, idx, batch, seq):
    """[N, C] -> [B, T, C], zeros on the rows not in idx."""
    return _UnpackRows.apply(x, idx, batch * seq).view(batch, seq, x.shape[-1])


def _row_indices(attention_mask):
    """Flat indices of the real rows, or None when the batch does not qualify.

    The mask must be binary, right-padded (TRL's collator) and have a real first token in every
    row. The check stays on the device and gates the mask, so a bad mask makes `nonzero` return
    nothing instead of costing a second host sync (idea from a GPT-6-Astra draft). Reads only.
    """
    keep = attention_mask.eq(1)
    valid = (
        (keep | attention_mask.eq(0)).all()          # binary
        & keep[:, 0].all()                           # no empty row
        & (keep[:, :-1] | ~keep[:, 1:]).all()        # right-padded: keep[t] implies keep[t-1]
    )
    idx = (keep & valid).reshape(-1).nonzero().view(-1)
    rows = keep.numel()
    n_real = idx.numel()  # the only host sync U adds: N is a shape
    if n_real < _MIN_N or (rows - n_real) * _MIN_PAD_DIV < rows:
        return None
    return idx


# ---------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------

def _hookless(module):
    for name in _HOOK_DICTS:
        if getattr(module, name, None):
            return False
    return "forward" not in module.__dict__  # instance-level forward (accelerate's add_hook_to_module)


def _is_capture_hook(fn):
    """transformers' output-capturing hook (utils/output_capturing.py:100-119). It only appends to
    the active collector and returns None; with an empty collector it returns at once (:105-110)."""
    qualname = _REFS.get("capture_qualname")
    return (qualname is not None
            and getattr(fn, "__module__", None) == _REFS.get("capture_module")
            and getattr(fn, "__qualname__", None) == qualname)


def _bypassed_layer_ok(layer):
    """We never call the decoder layer itself, so it must carry nothing that would run around it,
    except transformers' idle capture hooks (output_* is required falsy, see _fast_path_ok)."""
    if "forward" in layer.__dict__:
        return False
    for name in ("_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
        if getattr(layer, name, None):
            return False
    hooks = getattr(layer, "_forward_hooks", None)
    if hooks:
        if getattr(layer, "_forward_hooks_with_kwargs", None) or getattr(
                layer, "_forward_hooks_always_called", None):
            return False
        if not all(_is_capture_hook(h) for h in hooks.values()):
            return False
    return True


def _no_global_hooks():
    glob = torch.nn.modules.module
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                 "_global_backward_pre_hooks"):
        if getattr(glob, name, None):
            return False
    return True


def _rowwise_parts_ok(layer, norm_cls):
    """The parts we run on [N, C] must be row-wise and draw no random numbers."""
    if type(layer.input_layernorm) is not norm_cls or type(layer.post_attention_layernorm) is not norm_cls:
        return False
    for m in layer.mlp.modules():
        if isinstance(m, torch.nn.Dropout) and m.training and m.p > 0:
            return False  # e.g. LoRA on gate/up/down: its dropout would draw on N rows, not B*T
    return True


def _layer_mode(layer_forward):
    """'plain' for the genuine Qwen2DecoderLayer.forward, 'k3' for K3's patch of it, else None."""
    genuine = _REFS.get("layer_forward")
    if genuine is None:
        return None
    if layer_forward is genuine:
        return "plain"
    if getattr(layer_forward, _K3_MARK, False) and getattr(layer_forward, "__wrapped__", None) is genuine:
        return "k3"
    return None


def _resolve_use_cache(model, use_cache):
    # merge_with_config_defaults, utils/generic.py:1019-1033: an explicit value wins, else the config;
    # gradient checkpointing in training turns a truthy value off.
    value = use_cache if use_cache is not None else getattr(model.config, "use_cache", None)
    if value and getattr(model, "gradient_checkpointing", False) and model.training:
        value = False
    return value


def _fast_path_ok(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
                  use_cache, kwargs):
    """The layer mode ('plain' or 'k3') when the fast path reproduces base, else None.
    `use_cache` is the already resolved value. Reads only; never touches CUDA memory."""
    if type(model) is not _REFS["model_cls"]:
        return None
    if not model.training or not torch.is_grad_enabled():
        return None
    if getattr(model, "gradient_checkpointing", True):
        return None
    if inputs_embeds is not None or position_ids is not None or past_key_values is not None:
        return None
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        return None  # a dict mask (line 367) or no mask goes to the original
    if input_ids.ndim != 2 or not input_ids.is_cuda or input_ids.dtype not in (torch.int64, torch.int32):
        return None
    if attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device:
        return None
    if attention_mask.dtype not in (torch.bool, torch.int64, torch.int32):
        return None
    if input_ids.numel() == 0:
        return None
    if use_cache:
        return None  # base would build a DynamicCache (lines 358-359) and return it
    if model.embed_tokens.weight.device != input_ids.device:
        return None

    config = model.config
    if "is_causal" in kwargs or hasattr(config, "is_causal"):
        return None  # merge_with_config_defaults would rewrite the config (generic.py:1049-1057)
    if kwargs.get("debug_io", False):
        return None
    # capture_outputs reads these exactly so (output_capturing.py:238-241).
    for key in ("output_hidden_states", "output_attentions"):
        if kwargs.get(key, getattr(config, key, False)):
            return None
    if kwargs.get("return_dict", getattr(config, "return_dict", True)) is False:
        return None  # capture_outputs would return a tuple (output_capturing.py:281-282)

    if getattr(config, "_attn_implementation", None) != "sdpa":
        return None
    if getattr(config, "attention_dropout", 0.0) != 0.0:
        return None  # SDPA dropout (modeling_qwen2.py:226); Qwen2Config default 0.0 (configuration_qwen2.py:78)
    if getattr(model, "has_sliding_layers", True):
        return None
    n_layers = config.num_hidden_layers
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None or len(layer_types) < n_layers:
        return None
    if any(t != "full_attention" for t in layer_types[:n_layers]):
        return None
    layers = model.layers[:n_layers]
    if len(layers) != n_layers or not _no_global_hooks():
        return None
    norm_cls = _REFS["norm_cls"]
    if type(model.norm) is not norm_cls:
        return None
    layer_cls = _REFS["layer_cls"]
    for layer in layers:
        if type(layer) is not layer_cls or getattr(layer, "gradient_checkpointing", True):
            return None
        if not _bypassed_layer_ok(layer) or not _rowwise_parts_ok(layer, norm_cls):
            return None
    return _layer_mode(layer_cls.forward)


# ---------------------------------------------------------------------------------------------
# K3 hand-off: the public kernel entry points, with the contract its docstring states
# (solution/kernels/addnorm.py add_rmsnorm: caller checks norm_ok and that delta is a CUDA
# fp16/fp32 tensor of residual's shape and device; solution/layers.py also requires a plain norm).
# ---------------------------------------------------------------------------------------------

def _k3_ops():
    """(add_rmsnorm, norm_ok) from K3, or None if K3 is not merged."""
    if "k3" not in _REFS:
        try:
            from .kernels.addnorm import add_rmsnorm, norm_ok

            _REFS["k3"] = (add_rmsnorm, norm_ok)
        except Exception:  # noqa: BLE001 - without K3 the unfused reference ops run
            _REFS["k3"] = None
    return _REFS["k3"]


def _plain_norm(module):
    # We call the kernel instead of module(...), so nothing may run around the module call.
    return (type(module) is _REFS["norm_cls"] and _hookless(module)
            and isinstance(getattr(module, "variance_epsilon", None), float))


def _delta_ok(residual, delta):
    return (delta.is_cuda and delta.device == residual.device
            and delta.dtype in (torch.float16, torch.float32) and delta.shape == residual.shape)


# ---------------------------------------------------------------------------------------------
# v7 I4: S4 off on the first layers
# ---------------------------------------------------------------------------------------------

def s4_off_count(n_layers):
    """int(n_layers * S4_OFF_FRACTION): 28 -> 7 and 3 -> 0 at 0.25, 2 -> 0. 0 for odd values."""
    frac = S4_OFF_FRACTION
    if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not 0.0 <= frac <= 1.0:
        return 0  # NaN fails the range check too
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers <= 0:
        return 0
    return int(n_layers * frac)


def _k4_forward(mlp):
    """The forward S4 itself falls back to, when it may be called directly on this module.

    S4's class patch keeps the forward it wrapped (K4) in `__wrapped__` and calls exactly that
    when it does not recompute (mlp_recompute.py: `return previous(self, x)`). Calling it directly
    skips only Module.__call__, which with no hooks on the module (checked here) and no global
    hooks (checked by _fast_path_ok) does nothing else. Qwen2MLP has no transformers capture hooks
    (modeling_qwen2.py:314-317 records decoder layers and attention only). None otherwise. Reads only.
    """
    if type(mlp) is not _REFS.get("mlp_cls") or not _hookless(mlp):
        return None
    fwd = type(mlp).forward
    if not getattr(fwd, _S4_MARK, False):
        return None  # S4 not installed: the module call already is the no-recompute path
    inner = getattr(fwd, "__wrapped__", None)
    return inner if callable(inner) else None


def _mlp_no_recompute(mlp, y):
    """mlp(y) without S4's recompute: K4 keeps the fp16 gate/up for backward instead.

    Bit-identical to S4 (solution/tests: test_mlp_recompute.py::test_bit_exact_against_k4) and so
    to B (test_peak_v6.py::test_fused_block_bit_exact_and_saves_less checks B == K3 + S4).
    Any guard failure calls the module, which is v6.1's unfused S4 path (same values again).
    """
    try:
        inner = _k4_forward(mlp)
    except Exception:  # noqa: BLE001 - a read-only guard; nothing ran yet
        inner = None
    if inner is None:
        COUNTS["s4_off_fallback"] += 1
        return mlp(y)
    COUNTS["s4_off"] += 1
    return inner(mlp, y)


# ---------------------------------------------------------------------------------------------
# The packed forward
# ---------------------------------------------------------------------------------------------

def _attention_scope(idx, batch, seq, width):
    """v6: saved_compact.attention_scope when A or D is on, else a no-op context."""
    sc = _REFS.get("saved_compact")
    if sc is not None and (sc.USE_SDPA_KV_COMPACT or sc.USE_LORA_ROW_COMPACT):
        return sc.attention_scope(idx, batch, seq, width)
    return contextlib.nullcontext()


def _lora_source_scope(norm, r, x_bt, idx):
    """v9 M2: solution/lora_recompute.py source_scope when its flag is on, else a no-op context.
    The scope runs its own guards (reads only) and is a no-op when they fail."""
    m2 = _REFS.get("lora_recompute")
    if m2 is None or not m2.USE_LORA_INPUT_RECOMPUTE:
        return contextlib.nullcontext()
    return m2.source_scope(norm, r, x_bt, idx, _REFS["norm_cls"])


def _causal_scope(mask, batch, seq):
    """v7 I1: solution/sdpa_recompute.py causal_mask_scope(mask) when A1's USE_CAUSAL_PADDED is on
    and `mask` provably is causal AND (validated right-) padding, else a no-op context. Reads only.

    Provenance (transformers 5.16.1): mask came from the module-level create_causal_mask of
    modeling_qwen2 (imported from masking_utils, modeling_qwen2.py:16), called by us with a 2D mask,
    position_ids and no or/and mask function or block ids (_packed_forward), so with sdpa's mask
    interface it is causal_mask_function AND padding_mask_function (masking_utils.py:946-996,
    496-518; no packed-sequence mask with a 2D mask, :851-857). Both functions must be the genuine
    ones, and the result a [B, 1, T, T] bool tensor (no skip to None)."""
    a1 = _REFS.get("sdpa_recompute")
    if a1 is None or not getattr(a1, "USE_CAUSAL_PADDED", False):
        return contextlib.nullcontext()
    if not _causal_padding_mask(mask, batch, seq):
        return contextlib.nullcontext()
    return a1.causal_mask_scope(mask)


def _causal_padding_mask(mask, batch, seq):
    """True when `mask` provably is causal AND padding (provenance in _causal_scope). Reads only."""
    mu = _REFS.get("masking")
    if mu is None:
        return False
    try:
        return bool(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
                    and tuple(mask.shape) == (batch, 1, seq, seq)
                    and _REFS["modeling"].create_causal_mask is mu.create_causal_mask
                    and mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] is mu.sdpa_mask
                    and mu.causal_mask_function is _REFS.get("causal_mask_function"))
    except Exception:  # noqa: BLE001
        return False


def _row_scope(mask, idx, batch, seq):
    """v9 M3: solution/sdpa_recompute.py row_compact_scope(mask, idx, batch, seq) when A1's
    USE_ROW_COMPACT is on and `mask` provably is causal AND padding, else a no-op context.

    A1 may then keep q/k/v of the real rows only: on this path every pad query row's output is
    dropped by pack_rows after self_attn (its dO is exactly 0) and, with this mask or I1's causal
    call on right-padded rows, no real query sees a pad key (proof in sdpa_recompute.py). Reads
    only; the scope makes its two [N] index vectors lazily, inside A1's guarded call."""
    a1 = _REFS.get("sdpa_recompute")
    if a1 is None or not getattr(a1, "USE_ROW_COMPACT", False):
        return contextlib.nullcontext()
    if not _causal_padding_mask(mask, batch, seq):
        return contextlib.nullcontext()
    return a1.row_compact_scope(mask, idx, batch, seq)


def _merged_attention(attn, x_bt, mask, pos_ids, use_cache, pos_emb, kwargs):
    """v11 Q: solution/merged_qkv.py attention_forward (the attention output, o_proj included)
    when its flag is on and the module may be called without Module.__call__ (no hooks but
    transformers' idle capture hooks, as for the bypassed decoder layers: the collector is {} on
    this path, output_capturing.py:105-110), else None. A None means nothing ran."""
    q = _REFS.get("merged_qkv")
    if q is None or not q.USE_MERGED_QKV:
        return None
    try:
        ok = _bypassed_layer_ok(attn)
    except Exception:  # noqa: BLE001 - a read-only guard
        ok = False
    if not ok:
        return None
    attn_kwargs = dict(kwargs)
    attn_kwargs.update(position_ids=pos_ids, use_cache=use_cache)  # the call's other kwargs
    return q.attention_forward(attn, x_bt, mask, pos_emb, attn_kwargs)


def _layer_packed(layer, r, idx, batch, seq, mask, pos_emb, pos_ids, use_cache, kwargs, k3,
                  s4_off=False):
    """One decoder layer, modeling_qwen2.py:279-298, with the token-wise ops on [N, C].

    s4_off (v7 I4): no fused block B and no S4 recompute in this layer; K3's add_rmsnorm (or the
    plain ops) and K4's MLP that keeps gate/up. Same values; more memory, less backward work.
    """
    x = layer.input_layernorm(r)  # line 280, row-wise
    x_bt = unpack_rows(x, idx, batch, seq)  # exact zeros on the padding rows
    del x
    # v6 A + D (solution/saved_compact.py): smaller saved tensors inside attention, same values.
    # v9 M2 (solution/lora_recompute.py): LoRA layers fed exactly x_bt keep no LoRA-A input and
    # rebuild it from r in backward (entered second, so its lora_A hooks are the innermost).
    with _attention_scope(idx, batch, seq, x_bt.shape[-1]), \
            _lora_source_scope(layer.input_layernorm, r, x_bt, idx):
        # v11 Q (solution/merged_qkv.py): q/k/v base projections as one GEMM when its guards
        # pass (inside both scopes: it needs M2's source); None means nothing ran.
        a_bt = _merged_attention(layer.self_attn, x_bt, mask, pos_ids, use_cache, pos_emb,
                                 kwargs)
        if a_bt is None:
            a_bt, _ = layer.self_attn(  # lines 282-290: same arguments and kwargs, base shapes
                hidden_states=x_bt,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=None,
                use_cache=use_cache,
                position_embeddings=pos_emb,
                **kwargs,
            )
    del x_bt
    a = pack_rows(a_bt, idx)
    del a_bt
    ln2 = layer.post_attention_layernorm
    if k3 is not None:
        if not s4_off:
            # v6 B (solution/fused_block.py): add + norm + S4 MLP + residual add as one node that
            # saves (r, a) instead of (r_mid, x16); bit-identical to the ops below.
            fb = _REFS.get("fused_block")
            try:  # a guard reads only; evaluating it consumes no RNG, so falling through is safe
                fused = fb is not None and fb.fused_ok(layer, r, a, _REFS["mlp_cls"], _REFS["norm_cls"])
            except Exception:  # noqa: BLE001
                fused = False
            if fused:
                return fb.add_norm_mlp(r, a, layer)
        add_rmsnorm, norm_ok = k3
        weight = getattr(ln2, "weight", None)
        # The conditions of K3's own layer patch (solution/layers.py), on [N, C].
        if _plain_norm(ln2) and norm_ok(r, weight) and _delta_ok(r, a):
            r, y = add_rmsnorm(r, a, weight, ln2.variance_epsilon)
            return r + (_mlp_no_recompute(layer.mlp, y) if s4_off else layer.mlp(y))  # lines 296-297
    r = r + a  # line 291: fp32 + fp16 -> fp32, the same elementwise values as base
    y = ln2(r)  # line 295
    return r + (_mlp_no_recompute(layer.mlp, y) if s4_off else layer.mlp(y))  # lines 296-297


def _packed_forward(model, input_ids, attention_mask, idx, use_cache, kwargs, mode):
    """modeling_qwen2.py:352-402 for the guarded training case."""
    batch, seq = input_ids.shape
    mq = _REFS["modeling"]
    collector = _REFS.get("collector")
    # capture_outputs sets the collector to {} when nothing is requested (output_capturing.py:253-258)
    token = collector.set({}) if collector is not None else None
    try:
        embeds = model.embed_tokens(input_ids)  # line 356
        pos_ids = torch.arange(seq, device=embeds.device).unsqueeze(0)  # lines 361-364, no cache
        mask = mq.create_causal_mask(  # lines 369-378, the same module-level function as base
            config=model.config,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=pos_ids,
        )
        pos_emb = model.rotary_emb(embeds, pos_ids)  # line 385 (cos/sin in the embeddings' dtype)
        r = pack_rows(embeds, idx)
        del embeds

        k3 = _k3_ops() if mode == "k3" else None
        n_layers = model.config.num_hidden_layers
        n_off = s4_off_count(n_layers)  # v7 I4: the first n_off layers keep gate/up (no S4)
        # v7 I1: SDPA may take is_causal=True for this mask; v9 M3: A1 may save real rows only.
        with _causal_scope(mask, batch, seq), _row_scope(mask, idx, batch, seq):
            for i, layer in enumerate(model.layers[:n_layers]):  # line 387
                r = _layer_packed(layer, r, idx, batch, seq, mask, pos_emb, pos_ids, use_cache,
                                  kwargs, k3, s4_off=i < n_off)
        hidden = model.norm(r)  # line 398, row-wise
    finally:
        if collector is not None:
            collector.reset(token)  # output_capturing.py:265
    return mq.BaseModelOutputWithPast(  # lines 399-402; use_cache is falsy here
        last_hidden_state=unpack_rows(hidden, idx, batch, seq),
        past_key_values=None,
    )


def forward_impl(original, model, input_ids=None, attention_mask=None, position_ids=None,
                 past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    idx = None
    try:
        resolved = _resolve_use_cache(model, use_cache)
        mode = _fast_path_ok(model, input_ids, attention_mask, position_ids, past_key_values,
                             inputs_embeds, resolved, kwargs)
        if mode is not None:
            idx = _row_indices(attention_mask)  # reads the mask only; no module ran yet
    except Exception:  # noqa: BLE001 - an unexpected object is never ours to handle
        idx = None
    if idx is None:
        COUNTS["original"] += 1
        return original(model, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
    kw = dict(kwargs)
    kw.pop("return_dict", None)  # capture_outputs pops it before the body (output_capturing.py:234)
    # No try from here on: once attention ran, a retry would draw the dropout masks twice.
    out = _packed_forward(model, input_ids, attention_mask, idx, resolved, kw, mode)
    COUNTS["packed"] += 1
    return out


# ---------------------------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------------------------

def _is_genuine(fn, module_name, qualname):
    return getattr(fn, "__module__", None) == module_name and getattr(fn, "__qualname__", None) == qualname


def _innermost_layer_forward(fn):
    # K3 keeps the original in __wrapped__ (solution/layers.py); unwrap only K3's marked patches.
    while getattr(fn, _K3_MARK, False) and getattr(fn, "__wrapped__", None) is not None:
        fn = fn.__wrapped__
    return fn


def patch_qwen2_model_unpad():
    """Install the Qwen2Model.forward patch. Idempotent; no CUDA work. True when in place."""
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq

        model_cls, layer_cls, norm_cls = mq.Qwen2Model, mq.Qwen2DecoderLayer, mq.Qwen2RMSNorm
        mlp_cls = mq.Qwen2MLP
        mq.create_causal_mask, mq.BaseModelOutputWithPast  # noqa: B018 - both must exist
    except Exception:  # noqa: BLE001 - a missing or fake library means nothing to patch
        return False

    current = model_cls.forward
    if getattr(current, _MARK, False):
        return True
    # Wrap only the genuine (decorated) forward: functools.wraps in both decorators keeps this
    # module and qualname (generic.py:1006, output_capturing.py:231). Someone else's replacement
    # would be bypassed by the fast path, so then we do not install at all.
    if not _is_genuine(current, mq.__name__, "Qwen2Model.forward"):
        return False

    layer_forward = _innermost_layer_forward(layer_cls.forward)
    refs = dict(modeling=mq, model_cls=model_cls, layer_cls=layer_cls, norm_cls=norm_cls,
                mlp_cls=mlp_cls,
                layer_forward=(layer_forward if _is_genuine(layer_forward, mq.__name__,
                                                            "Qwen2DecoderLayer.forward") else None))
    # Helpers from v6 on (pure Python, no CUDA at import); a missing one keeps the path without it.
    for key, name in (("fused_block", "fused_block"), ("saved_compact", "saved_compact"),
                      ("sdpa_recompute", "sdpa_recompute"), ("lora_recompute", "lora_recompute"),
                      ("merged_qkv", "merged_qkv")):  # v11 Q
        try:
            refs[key] = importlib.import_module(f"{__package__}.{name}")
        except Exception:  # noqa: BLE001
            refs[key] = None
    try:  # v7 I1: the genuine mask builders, captured now (Soup's tests may swap modules later)
        from transformers import masking_utils as mu

        mu.create_causal_mask, mu.sdpa_mask, mu.ALL_MASK_ATTENTION_FUNCTIONS  # noqa: B018
        refs["masking"] = mu
        refs["causal_mask_function"] = mu.causal_mask_function
    except Exception:  # noqa: BLE001 - without them I1 stays off
        refs["masking"] = None
        refs["causal_mask_function"] = None
    try:
        from transformers.utils import output_capturing as oc

        refs["collector"] = oc._active_collector
        refs["capture_module"] = oc.__name__
        refs["capture_qualname"] = (oc.install_output_capuring_hook.__qualname__
                                    + ".<locals>.output_capturing_hook")
    except Exception:  # noqa: BLE001 - without them capture hooks keep the original path
        for key in ("collector", "capture_module", "capture_qualname"):
            refs.pop(key, None)
    _REFS.clear()
    _REFS.update(refs)
    original = current

    # functools.wraps keeps __wrapped__ = original, so inspect.signature still shows the original
    # parameters (Trainer reads them, trainer.py:499-505); the original keeps its decorators for
    # every call we do not take.
    @functools.wraps(original)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        return forward_impl(original, self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

    setattr(forward, _MARK, True)
    model_cls.forward = forward
    return True
