"""A2: the LoRA dropout on q_proj / v_proj keeps no mask; backward replays it from the RNG state.

peft 0.20 (tuners/lora/bnb.py:513-565, Linear4bit.forward) computes
`lora_B(lora_A(dropout(x))) * scaling` with an nn.Dropout(p=0.05) (tuners/lora/layer.py:251-256;
Soup's default lora.dropout, soup_cli/config/schema.py:91). torch's dropout keeps a bool mask of x's
shape for its backward: [B, T, 1536] = 1.5 KiB per padded token, for q and for v, in each of the
28 layers (6 MiB per layer at B*T = 2048). lora_A's own saved input (the fp16 cast of the dropout
output) is still needed for lora_A's weight gradient and stays.

`_ReplayDropout` runs the same `F.dropout(x, p, training=True)` call in the forward (the same
kernel, the same generator draw, so the same output and the same RNG state afterwards) and keeps
only the CUDA generator state it started from (a 16-byte CPU tensor). In backward it sets that
state, runs the same dropout on the incoming gradient and restores the generator:
F.dropout(g) = g * mask * 1/(1-p) with the replayed mask, which is what dropout's backward computes
from the saved mask. The replay draws the same mask because it runs the same kernel on a tensor of
the same numel, dtype and memory layout (contiguous, 16-byte aligned: both guarded, the gradient is
cloned if needed) from the same generator state. solution/tests/test_unpad.py (run() /
after_dropout) already checks on the T4 that a replay on ones from the recorded state equals the
real LoRA dropout mask for exactly these inputs; torch.cuda.get_rng_state / set_rng_state are the
calls used there (and by transformers, trainer.py:3210, trainer_pt_utils.py:1315).

The class-level patch of peft's lora bnb Linear4bit.forward replays lines 513-565 for the single
vanilla adapter case, with the dropout module call replaced by `_ReplayDropout`. Everything else
(base_layer, the defensive clone, lora_A/lora_B as module calls, the autocast-dependent dtype
conversion, the add) is the same code in the same order. Guards: flag on; exact class; no extra
args/kwargs (adapter_names, variant kwargs); adapters enabled and not merged; exactly one active
adapter that has lora_A/lora_B and no variant (DoRA, aLoRA, ...); its dropout is a hook-free
nn.Dropout in training mode with 0 < p < 1, not inplace; no global module hooks; grad enabled and x
requires grad; x a contiguous, 16-byte aligned CUDA float tensor; not tracing/compiling.
Otherwise the original forward runs. No Triton kernel, no CUDA work at import or patch time.

Idea credit: RNG-state replay of dropout masks as in torch.utils.checkpoint (preserve_rng_state)
and Megatron-LM's CUDA RNG tracker; code written from scratch.
"""

import torch

USE_LORA_DROPOUT_REPLAY = False  # read on every call; solution.apply() sets it

_MARK = "_soup_delta_lora_replay"
COUNTS = {"replay": 0, "original": 0}
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_GLOBAL_HOOKS = ("_global_forward_hooks", "_global_forward_pre_hooks",
                 "_global_backward_hooks", "_global_backward_pre_hooks")
_REFS = {}  # dropout, cls; captured at patch time (Soup's tests may swap sys.modules["torch"])


def _aligned(t):
    return t.is_contiguous() and t.data_ptr() % 16 == 0


class _ReplayDropout(torch.autograd.Function):
    """F.dropout(x, p, training=True) that saves the generator state instead of the mask."""

    @staticmethod
    def forward(ctx, x, p):
        device = x.device
        ctx.state = torch.cuda.get_rng_state(device)  # before the draw: the offset this call uses
        ctx.p = p
        ctx.meta = (x.shape, x.dtype, device)
        return _REFS["dropout"](x, p=p, training=True)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return None, None
        shape, dtype, device = ctx.meta
        if grad.dtype != dtype or grad.shape != shape:
            grad = grad.to(dtype).reshape(shape)
        if not _aligned(grad):
            grad = grad.clone(memory_format=torch.contiguous_format)  # fresh, allocator-aligned
        current = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(ctx.state, device)
        try:
            out = _REFS["dropout"](grad, p=ctx.p, training=True)
        finally:
            torch.cuda.set_rng_state(current, device)  # the forward stream of draws is unchanged
        return out, None


def _hookless(module):
    if any(getattr(module, name, None) for name in _HOOK_DICTS):
        return False
    return "forward" not in module.__dict__


def _no_global_hooks():
    glob = torch.nn.modules.module
    return not any(getattr(glob, name, None) for name in _GLOBAL_HOOKS)


def _compiling():
    try:
        if torch.jit.is_tracing():
            return True
        compiler = getattr(torch, "compiler", None)
        return bool(compiler is not None and compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return True


def _fast_adapter(layer, x, args, kwargs):
    """The active adapter name when the replay path reproduces peft, else None. Reads only."""
    if args or kwargs or type(layer) is not _REFS["cls"]:
        return None
    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.dtype not in (torch.float16, torch.float32):
        return None
    if not torch.is_grad_enabled() or not x.requires_grad or x.numel() == 0 or not _aligned(x):
        return None
    if _compiling() or layer.disable_adapters or layer.merged:
        return None
    active = layer.active_adapters
    if len(active) != 1:
        return None
    name = active[0]
    if name not in layer.lora_A or name not in layer.lora_B or name not in layer.lora_dropout:
        return None
    if name in getattr(layer, "lora_variant", {}) or name not in layer.scaling:
        return None
    drop = layer.lora_dropout[name]
    if type(drop) is not torch.nn.Dropout or not drop.training or drop.inplace:
        return None
    if not (0.0 < drop.p < 1.0) or not _hookless(drop) or not _no_global_hooks():
        return None
    return name


def _forward_replay(layer, x, name):
    """peft tuners/lora/bnb.py:513-565 for one vanilla adapter, dropout replaced."""
    layer._check_forward_args(x)  # :514 (returns at once: no adapter_names)
    result = layer.base_layer(x)  # :527
    result = result.clone()  # :533, peft's defensive clone for 4-bit
    lora_A = layer.lora_A[name]
    lora_B = layer.lora_B[name]
    dropout = layer.lora_dropout[name]
    scaling = layer.scaling[name]
    requires_conversion = not torch.is_autocast_enabled()  # :543
    if requires_conversion:
        expected_dtype = result.dtype
        x = layer._cast_input_dtype(x, lora_A.weight.dtype)
    if _aligned(x):
        dropped = _ReplayDropout.apply(x, float(dropout.p))
    else:  # a cast produced an odd layout (never seen): keep peft's module call
        dropped = dropout(x)
    output = lora_B(lora_A(dropped)) * scaling  # :549
    if requires_conversion:
        output = output.to(expected_dtype)
    return result + output  # :552, :565


def patch_lora_dropout_replay():
    """Install on peft's lora bnb Linear4bit. Idempotent; no CUDA work. True when in place."""
    try:
        import torch.nn.functional as F
        from peft.tuners.lora import bnb as lora_bnb

        cls = lora_bnb.Linear4bit
    except Exception:  # noqa: BLE001 - no peft / no bitsandbytes 4-bit: nothing to patch
        return False
    current = cls.forward
    if getattr(current, _MARK, False):
        return True
    _REFS.update(dropout=F.dropout, cls=cls)
    original = current

    def forward(self, x, *args, **kwargs):
        if USE_LORA_DROPOUT_REPLAY:
            try:
                name = _fast_adapter(self, x, args, kwargs)
            except Exception:  # noqa: BLE001 - an unexpected object is never ours to handle
                name = None
            if name is not None:
                COUNTS["replay"] += 1
                return _forward_replay(self, x, name)
        COUNTS["original"] += 1
        return original(self, x, *args, **kwargs)

    setattr(forward, _MARK, True)
    forward.__wrapped__ = original
    cls.forward = forward
    return True
