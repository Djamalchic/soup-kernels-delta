"""NF4 dequantization and a guarded bitsandbytes matmul replacement.

Importing this module performs no CUDA queries or allocations.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# One launch configuration. Four warps process eight output elements per
# thread: enough contiguous work to amortize metadata loads while keeping
# register pressure modest on T4. This is a bandwidth-oriented starting point,
# not an empirically tuned configuration.
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
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = e < n_elements

    byte = tl.load(Packed + e // 2, mask=valid, other=0).to(tl.int32)
    nibble = tl.where((e & 1) == 0, byte >> 4, byte & 15)

    block = e // 64
    abs_code = tl.load(Abs8 + block, mask=valid, other=0).to(tl.int32)
    code = tl.load(Code2 + abs_code, mask=valid, other=0)
    scale = tl.load(Abs2 + block // 256, mask=valid, other=0)
    offset = tl.load(Offset)

    # Both operations are fp32. The launcher MUST disable FP fusion so the
    # multiplication rounds before the addition.
    scaled_code = code * scale
    absmax = scaled_code + offset

    value = tl.load(Lut + nibble, mask=valid, other=0)
    result = (value * absmax).to(tl.float16, fp_downcast_rounding="rtne")
    tl.store(Output + e, result, mask=valid)


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

    # Restrict the reference dtype to fp16. In particular, a bf16 reference
    # would introduce a different intermediate rounding before conversion.
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


def nf4_dequantize(packed, quant_state, out=None):
    """Return a contiguous [N, K] fp16 tensor, optionally writing into out.

    Contiguous slices with nonzero storage offsets are supported. For example:

        buffer = torch.empty((17920, 1536), device="cuda", dtype=torch.float16)
        nf4_dequantize(gate, gate_state, out=buffer[:8960])
        nf4_dequantize(up, up_state, out=buffer[8960:])

    These are two launches of the same kernel; each launch uses its weight's
    own quantization metadata.
    """
    n, k = _validate_inputs(packed, quant_state)

    if out is None:
        out = torch.empty((n, k), device=packed.device, dtype=torch.float16)
    else:
        _check_tensor(out, "out", torch.float16, packed.device, n * k)
        if tuple(out.shape) != (n, k):
            raise ValueError(f"out must have shape {(n, k)}")
        if out.requires_grad:
            raise ValueError("out must not require gradients")

    # The caller must provide output storage disjoint from the input metadata.
    # This context selects the device without creating a stream.
    with torch.cuda.device(packed.device):
        lut = _device_lut(packed.device.index)
        soup_nf4_dequant_kernel[(triton.cdiv(n * k, _BLOCK),)](
            packed,
            quant_state.absmax,
            quant_state.state2.absmax,
            quant_state.state2.code,
            quant_state.offset,
            lut,
            out,
            n * k,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
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
        n, k = _validate_inputs(B, quant_state)
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
