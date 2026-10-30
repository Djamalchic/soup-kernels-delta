"""U: unpadding inside each batch for the token-wise half of every Qwen2 decoder layer.

TRL's collator right-pads every batch to its longest row, so at batch 4 roughly 40-47 % of the
[B, T] positions are padding [estimate, docs/UNPAD_DESIGN.md section 1]. RMSNorm, the residual
adds and the MLP work row by row, so on the fast path they run on the N real rows only ([N, C]).
Attention keeps the exact base shapes ([B, T, C] in, [B, T, C] out): q/k/v/o, LoRA dropout, RoPE,
the causal+padding mask and SDPA see the same tensors shapes in the same call order as base, so
the LoRA dropout masks and the CUDA RNG offsets are identical to base (see NOTES.md).

One guarded class-level patch: `Qwen2Model.forward` (transformers 5.16.1,
models/qwen2/modeling_qwen2.py:339-402). The fast path rebuilds lines 352-402 for the training
case and assembles each layer from its submodules (the body of Qwen2DecoderLayer.forward,
lines 269-298), called as modules so their own patches and hooks still apply:
- input_layernorm / post_attention_layernorm / model.norm -> Qwen2RMSNorm (K3 rmsnorm if patched);
- self_attn on [B, T, C] (zeros on padding rows), output gathered back to [N, C];
- first residual add + post_attention_layernorm -> K3's fused add_rmsnorm when K3's decoder-layer
  patch is installed, otherwise `r + a` and the norm module, exactly like base line 291 + 295;
- mlp on [N, C] -> K4 (fp16 MLP through K1) when its guard passes;
- last_hidden_state scattered back to [B, T, C] with zeros on padding, so the base head and K2
  (which packs the real rows itself) work unchanged.

What the fast path skips of the decorators (lines 339-340), and why that is safe:
- merge_with_config_defaults (utils/generic.py:1000-1077): only resolves `use_cache` from the config
  (lines 1007-1046; we resolve it the same way and require it to be falsy) and injects `is_causal`
  (lines 1049-1057; we require it absent); `debug_io` (line 1060) must be absent.
- capture_outputs (utils/output_capturing.py:215-286): pops `return_dict` (line 234; we pop it too
  and require it truthy) and records hidden states / attentions only when output_* is set
  (lines 238-256; we require both falsy, read the same way: kwargs first, then the config).
Everything else goes to the untouched original, which keeps both decorators.

Guards (types and shapes only, never the environment): training with grad enabled, exact
Qwen2Model / Qwen2DecoderLayer types, no gradient checkpointing, SDPA, full attention only, no
sliding layers, CUDA int input_ids [B, T] with a 2D int/bool attention_mask of the same shape, no
inputs_embeds / position_ids / past_key_values / cache, no module hooks on the decoder layers we
bypass (accelerate puts an instance-level `forward` there only for multi-device maps,
accelerate/big_modeling.py:363-375), no global module hooks, and enough padding to pay for the
scatter/gather. The decoder-layer forward must be the original or K3's patch of it (marker
`_soup_delta_addnorm`); any other patch of that class keeps the original model forward. Guard
evaluation never raises (Soup's tests pass fake modules): any exception means the original.

No Triton kernel here (the gather/scatter are torch index ops); nothing runs on CUDA at import or
patch time. Idea credit: removing padding for token-wise ops as in ByteDance's Effective
Transformer and the "unpad" path of FlashAttention's varlen BERT; code written from scratch.
"""

import functools

import torch

USE_UNPAD = False  # read by solution.apply(); False keeps the base Qwen2Model.forward untouched

_MARK = "_soup_delta_unpad"
_K3_MARK = "_soup_delta_addnorm"  # = solution/layers.py _MARK (K3)

# Below 129 real rows bitsandbytes on a T4 runs gate/up (and down) through its own small-M GEMM
# instead of dequant + F.linear (bitsandbytes/backends/cuda/ops.py:703-708; M = A.numel() // K at
# line 948), so the MLP would leave K1's path. Such batches keep the original forward.
_MIN_N = 129
# Fewer than rows / 32 padding rows (~3 %): the scatter/gather traffic (~40 MB per layer at 2048
# rows) is not paid back by the MLP work saved [estimate], so keep the exact original.
_MIN_PAD_DIV = 32

# Calls that took each path. Python ints only; tests and a debug notebook read them.
COUNTS = {"packed": 0, "original": 0}

# Real classes and functions captured by patch_qwen2_model_unpad(); Soup's tests may replace
# sys.modules["torch"] later (tests/test_callback.py:165-174), so nothing is re-imported per call.
_REFS = {}


# ---------------------------------------------------------------------------------------------
# Row gather / scatter. Both save only the int64 index; neither touches the RNG or uses atomics.
# ---------------------------------------------------------------------------------------------

class _PackRows(torch.autograd.Function):
    """[rows, C] -> [N, C] = x[idx]. Backward scatters the gradient into zeros (idx is unique)."""

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


# ---------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------

def _no_hooks(module):
    for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
        if getattr(module, name, None):
            return False
    return "forward" not in module.__dict__  # instance-level forward (accelerate's add_hook_to_module)


def _no_global_hooks():
    glob = torch.nn.modules.module
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                 "_global_backward_pre_hooks"):
        if getattr(glob, name, None):
            return False
    return True


def _layer_mode(layer_forward):
    """'plain' for the original Qwen2DecoderLayer.forward, 'k3' for K3's patch of it, else None."""
    original = _REFS["layer_forward"]
    if layer_forward is original:
        return "plain"
    if getattr(layer_forward, _K3_MARK, False) and getattr(layer_forward, "__wrapped__", None) is original:
        return "k3"
    return None


def _resolve_use_cache(model, use_cache):
    # merge_with_config_defaults, utils/generic.py:1019-1046: an explicit value wins, else the config.
    value = use_cache if use_cache is not None else getattr(model.config, "use_cache", None)
    if value and getattr(model, "gradient_checkpointing", False) and model.training:
        value = False
    return value


def _fast_path_ok(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
                  use_cache, kwargs):
    """Returns the layer mode ('plain' or 'k3') when the fast path reproduces base, else None."""
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
    if input_ids.ndim != 2 or not input_ids.is_cuda or input_ids.is_floating_point():
        return None
    if attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device:
        return None
    if attention_mask.dtype not in (torch.bool, torch.int64, torch.int32):
        return None
    if input_ids.numel() == 0:
        return None
    if _resolve_use_cache(model, use_cache):
        return None  # base would build a DynamicCache (line 358-359) and return it

    config = model.config
    if "is_causal" in kwargs or hasattr(config, "is_causal"):
        return None  # merge_with_config_defaults would rewrite the config (generic.py:1049-1057)
    if kwargs.get("debug_io", False):
        return None
    # capture_outputs reads these exactly so (output_capturing.py:238-242).
    for key in ("output_hidden_states", "output_attentions"):
        if kwargs.get(key, getattr(config, key, False)):
            return None
    if kwargs.get("return_dict", getattr(config, "return_dict", True)) is False:
        return None  # capture_outputs would return a tuple (output_capturing.py:281-282)

    if getattr(config, "_attn_implementation", None) != "sdpa":
        return None
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
    layer_cls = _REFS["layer_cls"]
    for layer in layers:
        if type(layer) is not layer_cls or getattr(layer, "gradient_checkpointing", True):
            return None
        if not _no_hooks(layer):
            return None
    return _layer_mode(layer_cls.forward)


def _k3_ops():
    """(add_rmsnorm, norm_ok, plain_norm_module, delta_ok) from K3, or None if K3 is not merged."""
    if "k3" not in _REFS:
        try:
            from .kernels.addnorm import add_rmsnorm, norm_ok
            from .layers import _delta_ok, _plain_norm_module

            _REFS["k3"] = (add_rmsnorm, norm_ok, _plain_norm_module, _delta_ok)
        except Exception:  # noqa: BLE001 - without K3 the unfused reference ops run
            _REFS["k3"] = None
    return _REFS["k3"]


# ---------------------------------------------------------------------------------------------
# The packed forward
# ---------------------------------------------------------------------------------------------

def _layer_packed(layer, r, idx, batch, seq, mask, pos_emb, pos_ids, use_cache, kwargs, k3):
    """One decoder layer, modeling_qwen2.py:279-298, with the token-wise ops on [N, C]."""
    x = layer.input_layernorm(r)  # line 280, row-wise
    a_bt, _ = layer.self_attn(  # lines 282-290: same arguments, base shapes
        hidden_states=unpack_rows(x, idx, batch, seq),
        attention_mask=mask,
        position_ids=pos_ids,
        past_key_values=None,
        use_cache=use_cache,
        position_embeddings=pos_emb,
        **kwargs,
    )
    a = pack_rows(a_bt, idx)
    ln2 = layer.post_attention_layernorm
    if k3 is not None:
        add_rmsnorm, norm_ok, plain_norm_module, delta_ok = k3
        # The same conditions as K3's own layer patch (solution/layers.py), on [N, C].
        if (plain_norm_module(ln2, _REFS["norm_cls"]) and norm_ok(r, getattr(ln2, "weight", None))
                and delta_ok(r, a)):
            r, y = add_rmsnorm(r, a, ln2.weight, ln2.variance_epsilon)
            return r + layer.mlp(y)  # lines 296-298
    r = r + a  # line 291: fp32 + fp16 -> fp32, the same elementwise values as base
    return r + layer.mlp(ln2(r))  # lines 294-298


def _packed_forward(model, input_ids, attention_mask, use_cache, kwargs, mode):
    """modeling_qwen2.py:352-402 for the guarded training case. Returns None (nothing computed
    yet) when the batch does not qualify."""
    batch, seq = input_ids.shape
    rows = batch * seq
    # The only host sync U adds: N is a shape.
    idx = attention_mask.reshape(-1).nonzero().view(-1)
    n_real = idx.numel()
    if n_real < _MIN_N or (rows - n_real) * _MIN_PAD_DIV < rows:
        return None

    mq = _REFS["modeling"]
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
    for layer in model.layers[: model.config.num_hidden_layers]:  # line 387
        r = _layer_packed(layer, r, idx, batch, seq, mask, pos_emb, pos_ids, use_cache, kwargs, k3)
    hidden = model.norm(r)  # line 398, row-wise
    return mq.BaseModelOutputWithPast(  # lines 399-402; use_cache is falsy here
        last_hidden_state=unpack_rows(hidden, idx, batch, seq),
        past_key_values=None,
    )


def forward_impl(original, model, input_ids=None, attention_mask=None, position_ids=None,
                 past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    try:
        mode = _fast_path_ok(model, input_ids, attention_mask, position_ids, past_key_values,
                             inputs_embeds, use_cache, kwargs)
    except Exception:  # noqa: BLE001 - an unexpected object is never ours to handle
        mode = None
    out = None
    if mode is not None:
        kw = dict(kwargs)
        kw.pop("return_dict", None)  # capture_outputs pops it before the body (line 234)
        # merge_with_config_defaults hands the body the resolved value (generic.py:1043-1046).
        resolved = use_cache if use_cache is not None else getattr(model.config, "use_cache", None)
        out = _packed_forward(model, input_ids, attention_mask, resolved, kw, mode)
    if out is not None:
        COUNTS["packed"] += 1
        return out
    COUNTS["original"] += 1
    return original(model, input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)


def _innermost_layer_forward(fn):
    # K3 keeps the original in __wrapped__ (solution/layers.py); unwrap our own kind of patches only.
    while getattr(fn, _K3_MARK, False) and getattr(fn, "__wrapped__", None) is not None:
        fn = fn.__wrapped__
    return fn


def patch_qwen2_model_unpad():
    """Install the Qwen2Model.forward patch. Idempotent; no CUDA work. True when in place."""
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq

        model_cls, layer_cls, norm_cls = mq.Qwen2Model, mq.Qwen2DecoderLayer, mq.Qwen2RMSNorm
        mq.create_causal_mask, mq.BaseModelOutputWithPast  # noqa: B018 - both must exist
    except Exception:  # noqa: BLE001 - a missing or fake library means nothing to patch
        return False

    current = model_cls.forward
    if getattr(current, _MARK, False):
        return True
    _REFS.update(modeling=mq, model_cls=model_cls, layer_cls=layer_cls, norm_cls=norm_cls,
                 layer_forward=_innermost_layer_forward(layer_cls.forward))
    original = current

    # functools.wraps keeps __wrapped__ = original, so inspect.signature still shows the original
    # parameters; the original keeps its decorators for every call we do not take.
    @functools.wraps(original)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        return forward_impl(original, self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

    setattr(forward, _MARK, True)
    model_cls.forward = forward
    return True
