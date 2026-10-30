"""NF4 dequantization (K1 v3s) and a guarded bitsandbytes matmul replacement.

Importing this module performs no CUDA queries or allocations.

Kernel layout. One program owns ROWS consecutive 4-byte groups of the packed
weight (ROWS * 4 bytes -> ROWS * 8 fp16 outputs). Tiles are [ROWS, 4]: dim 1 is
the 4 bytes a thread reads with one 32-bit load, dim 0 is a "row" vector that
carries the per-group metadata (one absmax per 4 bytes). Each byte b gives the
fp16 pair (nf4[b >> 4] * absmax, nf4[b & 15] * absmax); the pair is packed into
one int32 (even element in the low half, little-endian) and stored through an
int32 view of the output, so each thread stores 16 contiguous bytes.

The bounds mask is per row and broadcast along dim 1. The size argument is in
do_not_specialize, so Triton cannot assume it is a multiple of 16; a per-element
mask would then defeat vectorisation. A row mask is constant along dim 1 for
any size, so the load and store stay vectorised.

The 16-entry NF4 table is a 4-level select tree on the nibble bits (15 selects
per value, no table memory traffic). Its constants are the fp32 values of
bitsandbytes' `nf4_dequantization_lut` (csrc/kernels.cu:26-43).

Idea credit: the tile (one absmax per thread, one packed byte -> two outputs,
1-2K outputs per CTA) follows bitsandbytes kDequantizeBlockwise
(csrc/kernels.cu:465-528, launch csrc/ops.cu:78-95). The code is our own.

Measured on a Kaggle T4 (results/measurements.md, 30.10, median of 7 x 200
calls): 0.166 / 0.030 / 0.029 ms on [8960,1536] / [1536,1536] / [256,1536],
bit-exact with bitsandbytes.functional.dequantize_4bit; K1 v1 was 0.289 /
0.050 / 0.051, bnb's full dequantize_4bit 0.300 / 0.083 / 0.081.
"""

import weakref

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Fixed launch configuration for sm_75 (T4), the one that was measured.
# 4 warps = 128 threads; ROWS=256 groups -> 2 groups per thread, 1024 packed
# bytes and 2048 fp16 outputs per program. 2048 % ROWS == 0 keeps each program
# inside one second-level absmax group (256 blocks * 64 = 2048 groups).
_ROWS = 256
_NUM_WARPS = 4

# The kernel indexes bytes and output words in int32.
_MAX_KERNEL_COUNT = 2**31

# Used only by the torch fallback (numel % 8 != 0 or huge weights). Same fp32
# values as bitsandbytes csrc/kernels.cu:26-43 and as the kernel's select tree.
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


@triton.jit(do_not_specialize=["n_groups"])
def soup_nf4_dequant_kernel(
    Packed,  # uint8 [n_groups * 4]
    Abs8,  # uint8 [blocks], one code per 64 outputs (= 8 groups)
    Abs2,  # fp32 [ceil(blocks / 256)]
    Code2,  # fp32 [256]
    Offset,  # fp32 [1]
    Out32,  # int32 view of the fp16 output, [n_groups * 4]
    n_groups,  # numel // 8
    ROWS: tl.constexpr,
):
    tl.static_assert(2048 % ROWS == 0)

    start = tl.program_id(0) * ROWS
    row = start + tl.arange(0, ROWS)
    row_ok = row < n_groups
    idx = row[:, None] * 4 + tl.arange(0, 4)[None, :]  # byte == output word
    ok = tl.broadcast_to(row_ok[:, None], (ROWS, 4))

    # absmax of the row's 64-element block, double-dequantized. Two separate
    # fp32 roundings as in bnb (dequantize_blockwise, then `absmax += offset`,
    # bitsandbytes/functional.py:1052-1054); the launcher passes
    # enable_fp_fusion=False so this is not contracted into an FMA.
    abs_code = tl.load(Abs8 + (row >> 3), mask=row_ok, other=0).to(tl.int32)
    code = tl.load(Code2 + abs_code, mask=row_ok, other=0.0)
    scale = tl.load(Abs2 + (start >> 11))
    offset = tl.load(Offset)
    a = (code * scale + offset)[:, None]

    q = tl.load(Packed + idx, mask=ok, other=0).to(tl.int32)

    # High nibble (element 2i): bits 4..7. Leaf order follows the table index.
    h1 = (q & 0x10) != 0
    h2 = (q & 0x20) != 0
    h4 = (q & 0x40) != 0
    h8 = (q & 0x80) != 0
    vh = tl.where(
        h8,
        tl.where(
            h4,
            tl.where(h2, tl.where(h1, 1.0, 0.7229568362236023),
                     tl.where(h1, 0.5626170039176941, 0.44070982933044434)),
            tl.where(h2, tl.where(h1, 0.33791524171829224, 0.24611230194568634),
                     tl.where(h1, 0.16093020141124725, 0.07958029955625534)),
        ),
        tl.where(
            h4,
            tl.where(h2, tl.where(h1, 0.0, -0.09105003625154495),
                     tl.where(h1, -0.18477343022823334, -0.28444138169288635)),
            tl.where(h2, tl.where(h1, -0.39491748809814453, -0.5250730514526367),
                     tl.where(h1, -0.6961928009986877, -1.0)),
        ),
    ).to(tl.float32)

    # Low nibble (element 2i + 1): bits 0..3.
    l1 = (q & 0x1) != 0
    l2 = (q & 0x2) != 0
    l4 = (q & 0x4) != 0
    l8 = (q & 0x8) != 0
    vl = tl.where(
        l8,
        tl.where(
            l4,
            tl.where(l2, tl.where(l1, 1.0, 0.7229568362236023),
                     tl.where(l1, 0.5626170039176941, 0.44070982933044434)),
            tl.where(l2, tl.where(l1, 0.33791524171829224, 0.24611230194568634),
                     tl.where(l1, 0.16093020141124725, 0.07958029955625534)),
        ),
        tl.where(
            l4,
            tl.where(l2, tl.where(l1, 0.0, -0.09105003625154495),
                     tl.where(l1, -0.18477343022823334, -0.28444138169288635)),
            tl.where(l2, tl.where(l1, -0.39491748809814453, -0.5250730514526367),
                     tl.where(l1, -0.6961928009986877, -1.0)),
        ),
    ).to(tl.float32)

    # fp32 multiply, one rtne rounding to fp16, as bnb's
    # `vals[j*2] = dDequantizeNF4(q >> 4) * local_abs_max` (csrc/kernels.cu:517-522).
    h = (vh * a).to(tl.float16, fp_downcast_rounding="rtne")
    l = (vl * a).to(tl.float16, fp_downcast_rounding="rtne")
    hb = h.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    lb = l.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    tl.store(Out32 + idx, hb | (lb << 16), mask=ok)


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


# Validation cache: id(quant_state) -> what was validated. A hit requires the
# same live quant_state and packed objects, the same packed storage pointer,
# the same metadata tensor objects and the same scalar fields; anything else
# re-runs the full validation. Entries drop out when their quant_state dies
# (weakref callback). packed is held weakly; the small metadata tensors are held
# strongly only while their quant_state is alive (it holds them too).
# Not detected: in-place resize_/set_ of a metadata tensor (never done to bnb
# quant states during training).
_VALID = {}
_VALID_LIMIT = 4096


def _signature(quant_state, state2):
    return (
        getattr(quant_state, "quant_type", None),
        getattr(quant_state, "blocksize", None),
        getattr(quant_state, "nested", None),
        getattr(quant_state, "dtype", None),
        getattr(quant_state, "shape", None),
        getattr(state2, "blocksize", None),
    )


def _checked_shape(packed, quant_state):
    """(n, k) of a valid NF4 weight; raises TypeError/ValueError otherwise."""
    entry = _VALID.get(id(quant_state))
    if entry is not None:
        qs_ref, packed_ref, ptr, absmax, state2, abs2, code2, offset, sig, nk = entry
        s2 = getattr(quant_state, "state2", None)
        if (
            qs_ref() is quant_state
            and packed_ref() is packed
            and s2 is state2
            and getattr(quant_state, "absmax", None) is absmax
            and getattr(quant_state, "offset", None) is offset
            and getattr(s2, "absmax", None) is abs2
            and getattr(s2, "code", None) is code2
            and packed.data_ptr() == ptr
            and _signature(quant_state, s2) == sig
        ):
            return nk

    n, k = _validate_inputs(packed, quant_state)

    key = id(quant_state)
    try:
        qs_ref = weakref.ref(
            quant_state, lambda _ref, key=key, cache=_VALID: cache.pop(key, None)
        )
        packed_ref = weakref.ref(packed)
    except TypeError:
        return n, k  # not weak-referenceable: validate every call
    if len(_VALID) >= _VALID_LIMIT:
        _VALID.clear()
    state2 = quant_state.state2
    sig = _signature(quant_state, state2)
    sig = sig[:4] + (tuple(sig[4]),) + sig[5:]  # snapshot of a mutable shape
    _VALID[key] = (
        qs_ref,
        packed_ref,
        packed.data_ptr(),
        quant_state.absmax,
        state2,
        state2.absmax,
        state2.code,
        quant_state.offset,
        sig,
        (n, k),
    )
    return n, k


def _torch_dequant(packed, quant_state, n, k):
    # Bit-exact torch path for sizes the kernel does not take (numel % 8 != 0
    # or >= 2^31 elements); no Qwen2.5 projection lands here. Same arithmetic
    # as bnb: code2[abs8] * abs2 then + offset in fp32 (two roundings), then
    # nf4 * absmax in fp32 rounded to fp16.
    count = n * k
    state2 = quant_state.state2
    flat = packed.reshape(-1)
    nib = torch.stack((flat >> 4, flat & 15), dim=1).reshape(-1)[:count].long()
    blocks = (count + 63) // 64
    absmax = state2.code[quant_state.absmax.long()]
    absmax = absmax * state2.absmax.repeat_interleave(256)[:blocks]
    absmax = absmax + quant_state.offset.reshape(())
    lut = torch.tensor(_NF4_VALUES, dtype=torch.float32, device=packed.device)
    values = lut[nib] * absmax.repeat_interleave(64)[:count]
    return values.to(torch.float16).view(n, k)


def _launch(packed, quant_state, count, out):
    out32 = out.view(-1).view(torch.int32)
    n_groups = count // 8
    # The caller must provide output storage disjoint from the input metadata.
    # This context selects the device without creating a stream.
    with torch.cuda.device(packed.device):
        soup_nf4_dequant_kernel[(triton.cdiv(n_groups, _ROWS),)](
            packed,
            quant_state.absmax,
            quant_state.state2.absmax,
            quant_state.state2.code,
            quant_state.offset,
            out32,
            n_groups,
            ROWS=_ROWS,
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )


def nf4_dequantize(packed, quant_state, out=None):
    """Return a contiguous [N, K] fp16 tensor, optionally writing into out.

    Contiguous slices with nonzero storage offsets are supported. For example:

        buffer = torch.empty((17920, 1536), device="cuda", dtype=torch.float16)
        nf4_dequantize(gate, gate_state, out=buffer[:8960])
        nf4_dequantize(up, up_state, out=buffer[8960:])

    These are two launches of the same kernel; each launch uses its weight's
    own quantization metadata. An `out` whose start is not 16-byte aligned
    works, but Triton compiles a second specialization for it; an odd element
    offset goes through a temporary.
    """
    n, k = _checked_shape(packed, quant_state)
    count = n * k

    if out is not None:
        _check_tensor(out, "out", torch.float16, packed.device, count)
        if tuple(out.shape) != (n, k):
            raise ValueError(f"out must have shape {(n, k)}")
        if out.requires_grad:
            raise ValueError("out must not require gradients")

    if count % 8 != 0 or count >= _MAX_KERNEL_COUNT:
        result = _torch_dequant(packed, quant_state, n, k)
        if out is None:
            return result
        out.copy_(result)
        return out

    if out is None:
        out = torch.empty((n, k), device=packed.device, dtype=torch.float16)
    elif out.storage_offset() % 2 != 0:
        # An int32 view needs a 4-byte aligned start.
        tmp = torch.empty((n, k), device=packed.device, dtype=torch.float16)
        _launch(packed, quant_state, count, tmp)
        out.copy_(tmp)
        return out

    _launch(packed, quant_state, count, out)
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
    # Same signature as bitsandbytes/autograd/_functions.py:407 matmul_4bit,
    # which Linear4bit.forward reaches as `bnb.matmul_4bit`
    # (bitsandbytes/nn/modules.py:637).
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
        n, k = _checked_shape(B, quant_state)
    except (TypeError, ValueError):
        return original(A, B, quant_state=quant_state, out=out, bias=bias)

    if A.device != B.device or A.shape[-1] != k:
        return original(A, B, quant_state=quant_state, out=out, bias=bias)

    m = A.numel() // k
    if m <= 4:
        return original(A, B, quant_state=quant_state, out=out, bias=bias)

    # bnb takes dequant + F.linear for M > 1536 and otherwise asks its private
    # heuristic (bitsandbytes/backends/cuda/ops.py:921-926, 954-964); we only
    # replace the dequant + F.linear path.
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
