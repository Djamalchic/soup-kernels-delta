"""NF4 dequantization and a guarded bitsandbytes matmul replacement.

Importing this module performs no CUDA queries or allocations.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# BLOCK counts packed bytes: 1024 bytes -> 2048 fp16 outputs.
# Four warps nominally handle eight bytes / sixteen outputs per thread.
# Each CTA covers 32 NF4 blocks, all within one 256-block nested-scale group.
# This halves the CTA count of the previous 1024-output configuration.
# It is a fixed, unbenchmarked T4 configuration, not an autotuned result.
_BLOCK = 1024
_NUM_WARPS = 4

_NF4_VALUES = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

_ORIGINAL_BNB_MATMUL_4BIT = None

# Bounded, insertion-ordered cache. Strong quant_state references prevent
# id reuse while an entry exists. No decoded values or packed tensors are
# retained. In-place changes to tensor contents are read by every launch.
_VALIDATION_CACHE = {}
_VALIDATION_CACHE_LIMIT = 512


@triton.jit(do_not_specialize=["n_elements"])
def soup_nf4_dequant_kernel(
    Packed,
    Abs8,
    Abs2,
    Code2,
    Offset,
    Lut,
    Output,
    n_elements,
    BLOCK: tl.constexpr,
):
    tl.static_assert(BLOCK >= 32)
    tl.static_assert(BLOCK % 32 == 0)
    tl.static_assert(8192 % BLOCK == 0)

    tile = tl.program_id(0)
    byte_index = tile * BLOCK + tl.arange(0, BLOCK)
    byte_valid = byte_index < tl.cdiv(n_elements, 2)

    # Credit: the user's packed-byte, paired-store, and grouped-scale ideas.
    # Streaming packed bytes bypass L1, leaving room for the small codebooks.
    byte = tl.load(
        Packed + byte_index,
        mask=byte_valid,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)
    byte = tl.reshape(byte, (BLOCK // 32, 32))
    byte_valid = tl.reshape(byte_valid, (BLOCK // 32, 32))

    first_group = tile * (BLOCK // 32)
    group = first_group + tl.arange(0, BLOCK // 32)
    group_valid = group < tl.cdiv(n_elements, 64)

    abs_code = tl.load(
        Abs8 + group,
        mask=group_valid,
        other=0,
        cache_modifier=".ca",
    ).to(tl.int32)
    code = tl.load(
        Code2 + abs_code,
        mask=group_valid,
        other=0,
        cache_modifier=".ca",
    )

    # BLOCK divides 8192 packed bytes, so this CTA cannot cross an Abs2 group.
    scale = tl.load(
        Abs2 + first_group // 256,
        mask=first_group * 64 < n_elements,
        other=0,
        cache_modifier=".ca",
    )
    offset = tl.load(Offset)

    # Two separate fp32 roundings; every launcher disables FP fusion.
    scaled_code = code * scale
    absmax = scaled_code + offset

    hi = tl.load(
        Lut + (byte >> 4),
        mask=byte_valid,
        other=0,
        cache_modifier=".ca",
    )
    lo = tl.load(
        Lut + (byte & 15),
        mask=byte_valid,
        other=0,
        cache_modifier=".ca",
    )
    hi = (hi * absmax[:, None]).to(
        tl.float16, fp_downcast_rounding="rtne"
    )
    lo = (lo * absmax[:, None]).to(
        tl.float16, fp_downcast_rounding="rtne"
    )

    # join appends a two-element dimension: high nibble, then low nibble.
    result = tl.reshape(tl.join(hi, lo), (2 * BLOCK,))
    element = tile * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(Output + element, result, mask=element < n_elements)


@lru_cache(maxsize=None)
def _device_lut(device_index):
    # Lazy, once per device. No CUDA work happens at module import.
    return torch.tensor(
        _NF4_VALUES,
        dtype=torch.float32,
        device=torch.device("cuda", device_index),
    )


def _check_tensor(tensor, name, dtype, device, count, *, vector=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.numel() != count:
        raise ValueError(
            f"{name} must contain {count} elements, got {tensor.numel()}"
        )
    if vector and tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")


def _validate_inputs(packed, quant_state):
    if not isinstance(packed, torch.Tensor):
        raise TypeError("packed must be a torch.Tensor")
    if not packed.is_cuda:
        raise ValueError("packed must be a CUDA tensor")
    if packed.dtype != torch.uint8:
        raise TypeError("packed must have dtype torch.uint8")
    if not packed.is_contiguous():
        raise ValueError("packed must be contiguous")

    if getattr(quant_state, "quant_type", None) != "nf4":
        raise ValueError("quant_state.quant_type must be 'nf4'")
    if getattr(quant_state, "blocksize", None) != 64:
        raise ValueError("quant_state.blocksize must be 64")
    if not getattr(quant_state, "nested", False):
        raise ValueError("quant_state must use nested/double quantization")

    # A bf16 reference would introduce different intermediate rounding.
    if getattr(quant_state, "dtype", None) != torch.float16:
        raise TypeError("quant_state.dtype must be torch.float16")

    shape = getattr(quant_state, "shape", None)
    if not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2:
        raise ValueError("quant_state.shape must be a two-dimensional shape")
    if any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("quant_state.shape must contain two positive integers")
    n, k = shape
    count = n * k

    if packed.numel() != (count + 1) // 2:
        raise ValueError("packed size does not match quant_state.shape")

    state2 = getattr(quant_state, "state2", None)
    if getattr(state2, "blocksize", None) != 256:
        raise ValueError("quant_state.state2.blocksize must be 256")

    blocks = (count + 63) // 64
    device = packed.device

    _check_tensor(
        getattr(quant_state, "absmax", None),
        "quant_state.absmax",
        torch.uint8,
        device,
        blocks,
        vector=True,
    )
    _check_tensor(
        getattr(state2, "absmax", None),
        "quant_state.state2.absmax",
        torch.float32,
        device,
        (blocks + 255) // 256,
        vector=True,
    )
    _check_tensor(
        getattr(state2, "code", None),
        "quant_state.state2.code",
        torch.float32,
        device,
        256,
        vector=True,
    )
    _check_tensor(
        getattr(quant_state, "offset", None),
        "quant_state.offset",
        torch.float32,
        device,
        1,
    )
    return n, k


def _tensor_stamp(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    # Shape covers numel and rank; contiguity covers the layout guard.
    # data_ptr includes storage_offset. Unlike _version alone, these checks
    # also detect relevant changes made through tensor.data assignments.
    return (
        tensor.data_ptr(),
        tensor.dtype,
        tensor.device,
        tensor.shape,
        tensor.is_contiguous(),
    )


def _validation_stamp(packed, quant_state):
    shape = getattr(quant_state, "shape", None)
    if not isinstance(shape, (tuple, list, torch.Size)):
        return None

    # Snapshot mutable lists and distinguish bool/float dimensions from ints.
    shape_stamp = (
        type(shape),
        tuple((type(d), d) for d in shape),
    )
    state2 = getattr(quant_state, "state2", None)
    return (
        getattr(quant_state, "quant_type", None),
        getattr(quant_state, "blocksize", None),
        bool(getattr(quant_state, "nested", False)),
        getattr(quant_state, "dtype", None),
        shape_stamp,
        getattr(state2, "blocksize", None),
        _tensor_stamp(packed),
        _tensor_stamp(getattr(quant_state, "absmax", None)),
        _tensor_stamp(getattr(state2, "absmax", None)),
        _tensor_stamp(getattr(state2, "code", None)),
        _tensor_stamp(getattr(quant_state, "offset", None)),
    )


def _cached_validate_inputs(packed, quant_state):
    key = id(quant_state)
    entry = _VALIDATION_CACHE.get(key)
    stamp = None

    if entry is not None and entry[0] is quant_state:
        try:
            stamp = _validation_stamp(packed, quant_state)
            if stamp is not None and stamp == entry[1]:
                return entry[2]
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # Let the original validator determine the public failure.
            stamp = None
        _VALIDATION_CACHE.pop(key, None)

    shape = _validate_inputs(packed, quant_state)

    if stamp is None:
        try:
            stamp = _validation_stamp(packed, quant_state)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # An otherwise valid tensor need not be cacheable.
            return shape

    if stamp is not None:
        if len(_VALIDATION_CACHE) >= _VALIDATION_CACHE_LIMIT:
            _VALIDATION_CACHE.pop(next(iter(_VALIDATION_CACHE)))
        _VALIDATION_CACHE[key] = (quant_state, stamp, shape)

    return shape


def _launch_one(packed, quant_state, out, count):
    lut = _device_lut(packed.device.index)
    soup_nf4_dequant_kernel[(triton.cdiv(count, 2 * _BLOCK),)](
        packed,
        quant_state.absmax,
        quant_state.state2.absmax,
        quant_state.state2.code,
        quant_state.offset,
        lut,
        out,
        count,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )


def nf4_dequantize(packed, quant_state, out=None):
    """Return a contiguous [N, K] fp16 tensor, optionally writing into out.

    Contiguous slices with nonzero storage offsets are supported.
    Output storage must be disjoint from packed and all input metadata.
    """
    n, k = _cached_validate_inputs(packed, quant_state)

    if out is None:
        out = torch.empty((n, k), device=packed.device, dtype=torch.float16)
    else:
        _check_tensor(out, "out", torch.float16, packed.device, n * k)
        if tuple(out.shape) != (n, k):
            raise ValueError(f"out must have shape {(n, k)}")
        if out.requires_grad:
            raise ValueError("out must not require gradients")

    if torch.cuda.current_device() == packed.device.index:
        _launch_one(packed, quant_state, out, n * k)
    else:
        with torch.cuda.device(packed.device):
            _launch_one(packed, quant_state, out, n * k)
    return out


class SoupNF4Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, packed, quant_state, bias):
        ctx.save_for_backward(packed)
        ctx.quant_state = quant_state
        return F.linear(A, nf4_dequantize(packed, quant_state), bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_A = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            (packed,) = ctx.saved_tensors
            weight = nf4_dequantize(packed, ctx.quant_state)
            grad_A = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[3]:
            if grad_output.ndim == 1:
                grad_bias = grad_output
            else:
                # Match the reference's sum(0), then explicitly perform the
                # broadcast reduction needed for a vector bias and batched A.
                grad_bias = grad_output.sum(0)
                grad_bias = grad_bias.sum_to_size((grad_output.shape[-1],))

        return grad_A, None, None, grad_bias


def _get_original_matmul_4bit():
    global _ORIGINAL_BNB_MATMUL_4BIT
    if _ORIGINAL_BNB_MATMUL_4BIT is None:
        import bitsandbytes as bnb

        _ORIGINAL_BNB_MATMUL_4BIT = bnb.matmul_4bit
    return _ORIGINAL_BNB_MATMUL_4BIT


def soup_matmul_4bit(A, B, quant_state, out=None, bias=None):
    original = _get_original_matmul_4bit()

    if (
        not isinstance(A, torch.Tensor)
        or not A.is_cuda
        or A.dtype != torch.float16
        or A.ndim == 0
        or out is not None
    ):
        return original(A, B, quant_state=quant_state, out=out, bias=bias)

    try:
        n, k = _cached_validate_inputs(B, quant_state)
    except (TypeError, ValueError):
        return original(A, B, quant_state=quant_state, out=out, bias=bias)

    if A.device != B.device or A.shape[-1] != k:
        return original(A, B, quant_state=quant_state, out=out, bias=bias)

    m = A.numel() // k
    if m <= 4:
        return original(A, B, quant_state=quant_state, out=out, bias=bias)

    if m <= 1536:
        try:
            from bitsandbytes.backends.cuda.ops import (
                _gemm_4bit_use_custom_fn,
            )

            use_custom = bool(
                _gemm_4bit_use_custom_fn(A.device.index, A.dtype, m, n, k)
            )
        except Exception:
            # An unavailable or failing private API must preserve bnb's path.
            return original(A, B, quant_state=quant_state, out=out, bias=bias)

        if use_custom:
            return original(A, B, quant_state=quant_state, out=out, bias=bias)

    return SoupNF4Matmul.apply(A, B, quant_state, bias)


def patch_bnb_matmul_4bit():
    """Install the module-level replacement, idempotently."""
    import bitsandbytes as bnb

    if bnb.matmul_4bit is soup_matmul_4bit:
        return
    _get_original_matmul_4bit()
    bnb.matmul_4bit = soup_matmul_4bit
