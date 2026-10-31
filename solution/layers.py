"""K3 wiring: class-level patches of Qwen2RMSNorm.forward and Qwen2DecoderLayer.forward.

transformers 5.16.1 models/qwen2/modeling_qwen2.py:
- Qwen2RMSNorm.forward, lines 247-252 -> our rmsnorm. Covers input_layernorm (still called as a
  module at line 280, so its hooks keep working) and the final model.norm (line 398).
- Qwen2DecoderLayer.forward, lines 269-298 -> the same steps, but lines 291 + 294-295 (first
  residual add and post_attention_layernorm) run as one add_rmsnorm kernel. The second residual
  add (line 297) stays a torch add: fusing it into the next layer's input_layernorm would need
  state carried between layers, which breaks under gradient-checkpointing recompute.
  The signature below is the original one (lines 269-278); every argument and **kwargs reaches
  self_attn exactly as at lines 282-290, and the return value is the same single tensor.

Guards: exact module types; no hooks or instance-level forward on the norm we bypass (the hook
checks follow a GPT-6-Astra K3 draft); CUDA fp32 residual of width 1536; frozen fp32 weights;
delta of the same shape in fp16/fp32. Otherwise the original forward runs. Nothing touches CUDA at
import or at patch time. Both patches are idempotent and keep the original in __wrapped__ (same
convention as solution/mlp.py).
"""

import torch

_MARK = "_soup_delta_addnorm"


def _delta_ok(residual, delta):
    return (
        isinstance(delta, torch.Tensor)
        and delta.is_cuda
        and delta.device == residual.device
        and delta.dtype in (torch.float16, torch.float32)
        and delta.shape == residual.shape
    )


def _plain_norm_module(m, cls):
    # We call the kernel instead of m(...), so anything Module.__call__ or an instance-level
    # forward would do (hooks, accelerate's AlignDevicesHook) must be absent.
    return (
        type(m) is cls
        and not m._forward_hooks
        and not m._forward_pre_hooks
        and not m._backward_hooks
        and not m._backward_pre_hooks
        and "forward" not in m.__dict__
        and isinstance(getattr(m, "variance_epsilon", None), float)
    )


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    from .kernels.addnorm import norm_ok, rmsnorm

    current = modeling_qwen2.Qwen2RMSNorm.forward
    if getattr(current, _MARK, False):
        return
    original = current

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        eps = getattr(self, "variance_epsilon", None)
        if not isinstance(eps, float) or not norm_ok(hidden_states, weight):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, weight, eps)

    setattr(forward, _MARK, True)
    forward.__wrapped__ = original
    modeling_qwen2.Qwen2RMSNorm.forward = forward


def patch_qwen2_decoder_layer():
    from transformers.models.qwen2 import modeling_qwen2

    from .kernels.addnorm import add_rmsnorm, norm_ok

    cls = modeling_qwen2.Qwen2DecoderLayer
    current = cls.forward
    if getattr(current, _MARK, False):
        return
    original = current
    norm_cls = modeling_qwen2.Qwen2RMSNorm

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        ln2 = getattr(self, "post_attention_layernorm", None)
        if not (
            _plain_norm_module(ln2, norm_cls)
            and norm_ok(hidden_states, getattr(ln2, "weight", None))
        ):
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)

        # Lines 279-290 unchanged (input_layernorm goes through the patched Qwen2RMSNorm).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Lines 291 + 294-295 fused. Attention already ran, so an unexpected output continues on
        # the reference ops instead of re-running attention (RNG / cache state).
        if _delta_ok(residual, hidden_states):
            residual, hidden_states = add_rmsnorm(
                residual, hidden_states, ln2.weight, ln2.variance_epsilon)
        else:
            residual = residual + hidden_states
            hidden_states = ln2(residual)
        # Lines 296-298 unchanged (mlp goes through K4 when its guard passes).
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    setattr(forward, _MARK, True)
    forward.__wrapped__ = original
    cls.forward = forward
