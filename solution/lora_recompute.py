"""M2 (v9): LoRA-A's fp16 input is rebuilt in backward instead of being kept.

What it replaces. peft 0.20's 4-bit LoRA layer (tuners/lora/bnb.py:513-565) computes
`lora_B(lora_A(dropout(x))) * scaling`; under fp16 autocast (`requires_conversion` False, :543)
lora_A's nn.Linear casts the fp32 dropout output to fp16 and keeps that fp16 copy for its weight
gradient. On U's packed path q_proj and v_proj both get the same x_bt ([B, T, 1536] fp32,
modeling_qwen2.py:206/208), so every layer keeps two such tensors: 6 KiB per real row after D
(saved_compact.py), 221 MiB at N = 1350 over 28 layers [estimate: 2 x 1350 x 1536 x 2 B x 28].

What it keeps instead. x_bt = unpack_rows(input_layernorm(r)) (unpad._layer_packed), and r (fp32
[N, 1536]) is already kept by that very norm (kernels/addnorm.py:200, _RMSNorm saves x) and by B
(fused_block.py, layer 0 included). The LoRA-A input is therefore a pure function of tensors that
are alive anyway plus the dropout's generator state:
    d16 = fp16(dropout_state(unpack_rows(norm(r))))
So for the lora_A call only, a saved_tensors_hooks pack (innermost, so D's hooks do not see it;
trl/models/activation_offloading.py:562-566) swaps that one tensor for a small recipe (16-byte CPU
state, p, a handle to the layer's source). The unpack replays the norm with the same module forward
on the same r (same Triton kernel, same operands), the same zero-fill + index_copy as
_UnpackRows.forward (unpad.py), F.dropout from the saved state on a fresh contiguous tensor of the
same shape and dtype (A2's replay argument, lora_replay.py docstring, checked on the T4 by
test_unpad run() / after_dropout), and the same fp32 -> fp16 cast. Every element is the same bit
as in the forward, so mm's backward (lora_A's weight gradient) computes exactly what it computes
from the kept tensor; nothing else in the graph changes.

The fp16 cast is done by us (`dropped.to(torch.float16)`) instead of inside autocast: autocast's
cast of an fp16 input is a no-op and of an fp32 non-leaf is `arg.to(fp16)`, the same ToCopy node
[from memory, aten/src/ATen/autocast_mode.cpp cached_cast; torch source is not on disk]. The test
checks forward and gradients bit for bit against peft on one layer.

Pair rebuild: the first unpack in a layer rebuilds x_bt once and draws every pending recipe of that
layer (q and v), keeping the others in a per-layer stash until their own unpack (one fp16
[B, T, 1536] for a few kernels of attention backward). Backward transient per layer, P = B*T:
x_bt fp32 + one dropout output fp32 + its bool mask + two fp16 = 13 B x P x 1536 [~39 MiB at
P = 2048; estimate], below S4's MLP backward transient (~107 MiB) that precedes it in the layer.

Guards (forward): flag on; a source scope is active (unpad sets it only on the packed path, with an
exact, hook-free Qwen2RMSNorm whose class forward is K3's patch, K3-eligible r, x_bt contiguous
fp32); the LoRA call's input IS that x_bt; exact peft bnb Linear4bit, no extra args/kwargs,
adapters on and unmerged, one vanilla active adapter, exact hook-free nn.Dropout in training mode
with 0 < p < 1, lora_A an exact nn.Linear; no global hooks; grad enabled; CUDA autocast on with
fp16; not tracing/compiling. Anything else runs the forward that was installed before us (A2's or
peft's). Backward: r's version counter must not have moved (as torch's own saved-tensor check).
No Triton kernel lives here; nothing touches CUDA at import or patch time.

v10 P2 (USE_LORA_NO_CLONE / USE_LORA_EPILOGUE, set by solution.apply()): on this path the base
output goes straight into an out-of-place add, so peft's defensive clone (bnb.py:528-533) is an
identity and is skipped. With the epilogue the base matmul runs on fp16(x) (the cast Linear4bit
makes itself, nn/modules.py:626-628) and `f32(base16) + f32(fp16(lora_B(h) * s))` plus its backward
(fp16(g) for the base, fp16(f32(fp16(g)) * s) for lora_B) are one K3-v10 launch each way
(_LoraEpilogue, kernels/addnorm.py modes 2 / 3) instead of `.to(fp32)`, clone, mul, add and two
casts + a mul in backward. Taken only for an exact fp16-compute bnb Linear4bit base and an
fp16-exact scaling (the task's 32 / 16 = 2.0); anything else keeps the ops above. Same bits
(tests/test_ew_fusion.py: the LoRA pair against v8b, torch.equal).

Idea credit: selective activation recomputation (Korthikanti et al. 2022, arXiv:2205.05198) and
torch.utils.checkpoint's RNG-state replay; code written from scratch. Fusing a LoRA output
epilogue (scale + residual add) into one pass is a common trick (e.g. Unsloth's LoRA kernels, idea
only); code written from scratch.
"""

import contextlib
import contextvars
import struct

import torch

from .saved_compact import _hooks

USE_LORA_INPUT_RECOMPUTE = False  # read on every call; solution.apply() sets it
# v10 P2 (read on every call; solution.apply() sets them, module defaults = v9b):
# USE_LORA_NO_CLONE: no copy of the base output before the out-of-place add (peft's defensive
#   clone, bnb.py:528-533, is an identity there: same values, same gradients).
# USE_LORA_EPILOGUE: `.to(fp32)` + clone + `* scaling` + add, and their backward casts and mul, in
#   one K3-v10 launch each way (_LoraEpilogue). Needs kernels/addnorm.py USE_EW_K3 (apply() ties
#   them), so the K3 forward entry it launches is the one the run compiles anyway.
USE_LORA_NO_CLONE = False
USE_LORA_EPILOGUE = False

_MARK = "_soup_delta_lora_recompute"
_K3_MARK = "_soup_delta_addnorm"  # = solution/layers.py _MARK
_SOURCE = contextvars.ContextVar("soup_delta_lora_source", default=None)
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_GLOBAL_HOOKS = ("_global_forward_hooks", "_global_forward_pre_hooks",
                 "_global_backward_hooks", "_global_backward_pre_hooks")
# recompute: LoRA calls on our path; original: calls handed to the previous forward;
# packed: tensors swapped for a recipe; rebuilt: x_bt rebuilds in backward; stashed: pair hits;
# epilogue (v10 P2): calls whose output came from _LoraEpilogue.
COUNTS = {"recompute": 0, "original": 0, "packed": 0, "rebuilt": 0, "stashed": 0, "scopes": 0,
          "epilogue": 0}
_REFS = {}  # captured at patch time (Soup's tests may swap sys.modules["torch"] later)


# ---------------------------------------------------------------------------------------------
# Per-layer source and per-call recipe. The source never references a recipe or a hook closure,
# so there is no reference cycle: it dies with the last recipe (its SavedVariable), at once.
# ---------------------------------------------------------------------------------------------

class _Source:
    __slots__ = ("x_bt", "r", "version", "norm", "norm_forward", "idx", "rows", "shape",
                 "entries", "stash")

    def __init__(self, x_bt, r, norm, norm_forward, idx):
        self.x_bt = x_bt  # identity check during the forward only; cleared on scope exit
        self.r = r.detach()  # same storage and version counter as r: no new memory
        self.version = r._version
        self.norm = norm
        self.norm_forward = norm_forward
        self.idx = idx
        self.rows = x_bt.shape[0] * x_bt.shape[1]
        self.shape = tuple(x_bt.shape)
        self.entries = []  # [rng state (CPU), p, done]
        self.stash = {}  # entry index -> rebuilt fp16 tensor waiting for its own unpack


class _Recipe:
    __slots__ = ("src", "i", "shape")

    def __init__(self, src, i, shape):
        self.src = src
        self.i = i
        self.shape = shape


def _replay(src, order):
    """{i: fp16 LoRA-A input of entry i} for the entries in `order`, bit for bit the forward's."""
    if src.r._version != src.version:
        raise RuntimeError("solution.lora_recompute: the residual a LoRA-A input is rebuilt from "
                           "was modified by an inplace operation after the forward")
    out = {}
    with torch.no_grad():
        # input_layernorm(r): the class forward the module call ran (hook-free, checked).
        x = src.norm_forward(src.norm, src.r)
        width = x.shape[-1]
        # _UnpackRows.forward (unpad.py): zeros, then the real rows at idx.
        x_bt = x.new_zeros((src.rows, width))
        x_bt.index_copy_(0, src.idx, x)
        del x
        x_bt = x_bt.view(src.shape)
        device = x_bt.device
        dropout = _REFS["dropout"]
        current = torch.cuda.get_rng_state(device)
        try:
            for i in order:
                state, p, _ = src.entries[i]
                torch.cuda.set_rng_state(state, device)
                dropped = dropout(x_bt, p=p, training=True)
                out[i] = dropped.to(torch.float16)
                del dropped
        finally:
            torch.cuda.set_rng_state(current, device)  # the forward stream of draws is unchanged
    COUNTS["rebuilt"] += 1
    return out


def _rebuild(rec):
    src = rec.src
    t = src.stash.pop(rec.i, None)
    if t is not None:
        COUNTS["stashed"] += 1
    else:
        # This entry first, then every other entry of the layer not yet unpacked (pair rebuild).
        order = [rec.i] + [j for j, e in enumerate(src.entries)
                           if j != rec.i and not e[2] and j not in src.stash]
        built = _replay(src, order)
        t = built.pop(rec.i)
        src.stash.update(built)
    src.entries[rec.i][2] = True  # a second backward (retain_graph) rebuilds this one alone
    return t.view(rec.shape)


def _input_hooks(src, i, d16):
    # Plain identity keys, never d16 itself (it is alive for the whole lora_A call these hooks are
    # entered for, so no other tensor can share its pointer meanwhile).
    ptr, numel, device = d16.data_ptr(), d16.numel(), d16.device

    def pack(t):
        if (t.dtype == torch.float16 and t.data_ptr() == ptr and t.numel() == numel
                and t.device == device and t.is_contiguous()):
            COUNTS["packed"] += 1
            return _Recipe(src, i, tuple(t.shape))  # mm keeps a 2-D view of d16: same bits
        return t.detach()  # never `t` itself (saved_compact._row_hooks: v6 fix)

    def unpack(obj):
        if isinstance(obj, _Recipe):
            return _rebuild(obj)
        return obj

    return pack, unpack


# ---------------------------------------------------------------------------------------------
# Source scope (unpad._layer_packed, around self_attn)
# ---------------------------------------------------------------------------------------------

def _hookless(module):
    if any(getattr(module, name, None) for name in _HOOK_DICTS):
        return False
    return "forward" not in module.__dict__


def _no_global_hooks():
    glob = torch.nn.modules.module
    return not any(getattr(glob, name, None) for name in _GLOBAL_HOOKS)


def _aligned(t):
    return t.is_contiguous() and t.data_ptr() % 16 == 0


def _source_ok(norm, r, x_bt, idx, norm_cls):
    if not USE_LORA_INPUT_RECOMPUTE or "cls" not in _REFS:
        return False
    if type(norm) is not norm_cls or not _hookless(norm) or not _no_global_hooks():
        return False
    norm_forward = type(norm).forward
    if not getattr(norm_forward, _K3_MARK, False):  # K3: a Triton kernel, autocast-independent
        return False
    eps = getattr(norm, "variance_epsilon", None)
    if not isinstance(eps, float):
        return False
    from .kernels.addnorm import norm_ok

    if not norm_ok(r, getattr(norm, "weight", None)):
        return False
    if r.dim() != 2 or not r.is_contiguous() or not torch.is_grad_enabled():
        return False
    if not (isinstance(idx, torch.Tensor) and idx.dtype == torch.int64 and idx.dim() == 1
            and idx.numel() == r.shape[0] and idx.device == r.device):
        return False
    return (isinstance(x_bt, torch.Tensor) and x_bt.dim() == 3 and x_bt.dtype == torch.float32
            and x_bt.device == r.device and x_bt.shape[-1] == r.shape[-1] and _aligned(x_bt))


@contextlib.contextmanager
def source_scope(norm, r, x_bt, idx, norm_cls):
    """Around one packed layer's self_attn call: LoRA layers fed exactly x_bt rebuild their LoRA-A
    input from r in backward. A no-op when the guards fail. Reads only."""
    try:
        ok = _source_ok(norm, r, x_bt, idx, norm_cls)
    except Exception:  # noqa: BLE001 - an unexpected object is never ours to handle
        ok = False
    if not ok:
        yield
        return
    src = _Source(x_bt, r, norm, type(norm).forward, idx)
    token = _SOURCE.set(src)
    COUNTS["scopes"] += 1
    try:
        yield
    finally:
        _SOURCE.reset(token)
        src.x_bt = None  # recipes must not keep x_bt alive


# ---------------------------------------------------------------------------------------------
# The LoRA layer
# ---------------------------------------------------------------------------------------------

def _compiling():
    try:
        if torch.jit.is_tracing():
            return True
        compiler = getattr(torch, "compiler", None)
        return bool(compiler is not None and compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return True


def _fp16_autocast():
    try:
        return (torch.is_autocast_enabled() and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.float16)
    except Exception:  # noqa: BLE001 - an older torch without the device-type API
        return False


def _adapter(layer, x):
    """The active adapter name when our path reproduces peft for this call, else None. Reads only."""
    if type(layer) is not _REFS["cls"] or not torch.is_grad_enabled() or _compiling():
        return None
    if not x.is_cuda or x.dtype != torch.float32 or not _aligned(x) or not _fp16_autocast():
        return None
    if layer.disable_adapters or layer.merged:
        return None
    active = layer.active_adapters
    if len(active) != 1:
        return None
    name = active[0]
    if name not in layer.lora_A or name not in layer.lora_B or name not in layer.lora_dropout:
        return None
    if name in getattr(layer, "lora_variant", {}) or name not in layer.scaling:
        return None
    if type(layer.lora_A[name]) is not torch.nn.Linear:
        return None
    drop = layer.lora_dropout[name]
    if type(drop) is not torch.nn.Dropout or not drop.training or drop.inplace:
        return None
    if not (0.0 < drop.p < 1.0) or not _hookless(drop) or not _no_global_hooks():
        return None
    return name


def _fp16_exact_scale(scaling):
    """float(scaling) when it is a finite Python number that fp16 holds exactly, else None. Then
    fp16(s) == fp32(s) == s, so the scalar operand of `fp16 tensor * scaling` has one value
    however torch narrows it (P2 epilogue). Soup's task: alpha 32 / r 16 = 2.0."""
    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        return None
    try:
        s = float(scaling)
        if s != s or abs(s) == float("inf"):
            return None
        if struct.unpack("<e", struct.pack("<e", s))[0] != s:
            return None
    except (OverflowError, struct.error, ValueError):
        return None
    return s


def _epilogue_base(layer):
    """v10 P2: True when `layer.base_layer(x.to(fp16))` returns exactly the fp16 matmul output that
    `layer.base_layer(x)` casts to fp32 for an fp32 x: an exact, hook-free bitsandbytes
    Linear4bit whose compute dtype is fp16 and already fixed (bitsandbytes nn/modules.py:570,
    622-628, 630-635: with compute_type_is_set it never re-derives the compute dtype from the
    input; the input cast `x.to(self.compute_dtype)` is a no-op on fp16, the bias is cast to x's
    (= the compute) dtype either way, and the final `.to(inp_dtype)` (:637) is a no-op on the
    fp16 result). Reads only."""
    base = getattr(layer, "base_layer", None)
    cls = _REFS.get("bnb_linear4bit")
    return (cls is not None and type(base) is cls and _hookless(base)
            and getattr(base, "compute_type_is_set", False) is True
            and getattr(base, "compute_dtype", None) is torch.float16)


class _LoraEpilogue(torch.autograd.Function):
    """v10 P2: peft's `result + lora_B(h) * scaling` (bnb.py:549-552) with result = f32(base16) and
    the defensive clone (:533), in one K3-v10 launch each way (kernels/addnorm.py modes 2 / 3).

    Forward: f32(base16) + f32(fp16(f32(lb16) * s)), the values of bnb's `.to(fp32)` (exact),
    the clone (identity), torch's fp16 * scalar (product in fp32, one rounding to fp16) and the
    type-promoting fp32 add. Backward from the fp32 g: base16 gets fp16(g) (ToCopyBackward of
    `.to(fp32)` through CloneBackward / AddBackward), lb16 gets fp16(f32(fp16(g)) * s) (the add's
    cast of g to its fp16 input, then MulBackward)."""

    @staticmethod
    def forward(ctx, base16, lb16, scale):
        from .kernels import addnorm as k3

        ctx.set_materialize_grads(False)
        ctx.scale = scale
        return k3.lora_epilogue_fwd(base16, lb16, scale)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        from .kernels import addnorm as k3

        need_b, need_l = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if g is None or not (need_b or need_l):
            return None, None, None
        if g.dtype != torch.float32:
            g = g.float()  # never in training: the output is fp32
        g16, gl16 = k3.lora_epilogue_bwd(g, ctx.scale)
        return (g16 if need_b else None), (gl16 if need_l else None), None


def _forward_recompute(layer, x, name, src):
    """peft tuners/lora/bnb.py:513-565 for one vanilla adapter under fp16 autocast."""
    layer._check_forward_args(x)  # :514 (returns at once: no adapter_names)
    scaling = layer.scaling[name]
    # v10 P2 epilogue: the base matmul is called on fp16(x), which is the cast Linear4bit makes
    # itself (bnb nn/modules.py:626-628: the same `x.to(fp16)` op, the same autograd node), and
    # its fp16 output goes to the fused epilogue instead of `.to(fp32)` + clone + mul + add.
    scale = _fp16_exact_scale(scaling) if USE_LORA_EPILOGUE else None
    if scale is not None and _epilogue_base(layer):
        base16 = layer.base_layer(x.to(torch.float16))  # :527, fp16 out
        result = None
    else:
        base16 = None
        result = layer.base_layer(x)  # :527
        if not USE_LORA_NO_CLONE:
            result = result.clone()  # :533 (P2 skips it: an identity before an out-of-place add)
    lora_A = layer.lora_A[name]
    lora_B = layer.lora_B[name]
    p = float(layer.lora_dropout[name].p)
    # :543 requires_conversion is False (autocast on): x goes to dropout as it is.
    state = torch.cuda.get_rng_state(x.device)  # before the draw: the offset this call uses
    a2 = _REFS["a2"]
    if x.requires_grad and a2 is not None and a2.USE_LORA_DROPOUT_REPLAY and "dropout" in a2._REFS:
        dropped = a2._ReplayDropout.apply(x, p)  # A2: the same F.dropout call, no mask kept
    else:
        dropped = _REFS["dropout"](x, p=p, training=True)  # nn.Dropout.forward's call
    d16 = dropped.to(torch.float16)  # autocast's cast inside lora_A, done here (same ToCopy)
    del dropped
    i = len(src.entries)
    src.entries.append([state, p, False])
    pack, unpack = _input_hooks(src, i, d16)
    with _hooks(pack, unpack):
        h = lora_A(d16)  # module call: fp16 input untouched by autocast, fp16 weight cast cached
    del d16
    if base16 is not None:
        lb16 = lora_B(h)  # :549 without `* scaling` (the epilogue applies it)
        from .kernels import addnorm as k3

        if k3.lora_epilogue_ok(base16, lb16):
            COUNTS["epilogue"] += 1
            return _LoraEpilogue.apply(base16, lb16, scale)
        result = base16.to(x.dtype)  # bnb's `.to(inp_dtype)` (:637); peft's ops from here on
        output = lb16 * scaling  # :549
        return result + output  # :552
    output = lora_B(h) * scaling  # :549
    return result + output  # :552, :565


def patch_lora_input_recompute():
    """Wrap peft's lora bnb Linear4bit.forward (after A2, so ours is the outer one). Idempotent;
    no CUDA work. True when in place."""
    try:
        import torch.nn.functional as F
        from peft.tuners.lora import bnb as lora_bnb

        cls = lora_bnb.Linear4bit
    except Exception:  # noqa: BLE001 - no peft / no bitsandbytes 4-bit: nothing to patch
        return False
    current = cls.forward
    if getattr(current, _MARK, False):
        return True
    try:
        from . import lora_replay as a2
    except Exception:  # noqa: BLE001
        a2 = None
    try:  # v10 P2 epilogue: the exact base-layer class it accepts (peft imported it already)
        from bitsandbytes.nn import Linear4bit as bnb_linear4bit
    except Exception:  # noqa: BLE001 - without it the epilogue stays off
        bnb_linear4bit = None
    _REFS.update(dropout=F.dropout, cls=cls, a2=a2, bnb_linear4bit=bnb_linear4bit)
    original = current

    def forward(self, x, *args, **kwargs):
        if USE_LORA_INPUT_RECOMPUTE and not args and not kwargs:
            src = _SOURCE.get()
            if src is not None and src.x_bt is x:
                try:
                    name = _adapter(self, x)
                except Exception:  # noqa: BLE001
                    name = None
                if name is not None:
                    COUNTS["recompute"] += 1
                    return _forward_recompute(self, x, name, src)
        COUNTS["original"] += 1
        return original(self, x, *args, **kwargs)

    setattr(forward, _MARK, True)
    forward.__wrapped__ = original
    cls.forward = forward
    return True
