"""K3: fused residual add + Qwen2 RMSNorm, forward and backward (the run launches the *_v10 pair).

Reference math, transformers 5.16.1 models/qwen2/modeling_qwen2.py:
- Qwen2RMSNorm.forward (lines 247-252): u = f32(x); r = rsqrt(mean(u^2) + eps); y = w * cast_in(u * r)
- Qwen2DecoderLayer.forward (lines 291, 294-295): residual = residual + attn_out;
  post_attention_layernorm(residual)

Scope (checked by the callers in solution/layers.py, solution/unpad.py and solution/fused_block.py;
anything else stays on the original path): the residual stream and the norm weights are fp32 (base
Soup keeps the residual stream in fp32, see solution/mlp.py docstring), so cast_in() is the identity
and y = w * (u * r) in fp32, the base output dtype promote(fp32, fp32). Under accelerate's fp16
autocast the attention output (delta) arrives in fp16: bitsandbytes Linear4bit returns its input
dtype (bitsandbytes/nn/modules.py:626, 637) and peft's LoRA adds an fp16 update to it
(peft/tuners/lora/bnb.py:527-552).

v6 (peak-timing variant): the delta pointer is fp16. The kernels load it and widen it to fp32
(exact), which is the value base's type-promoting add uses, so r = residual + f32(delta) is the
same fp32 add as before (bit for bit); the Python-side fp32 copy of the delta (alive during the
launch) is gone. y and rstd are NOT bit-identical to v5's fp32-pointer launch: the fp16 load is
vectorised differently, so Triton compiles another register layout and tl.sum adds the 1536
squares in another order (a few fp32 ulp on rstd, Kaggle T4: test_peak_v6 rows 77 / 2048). This is
the same kind of difference v5 itself has against torch's RMSNorm; within v6 every forward and
recompute uses this one variant, so forward and backward stay consistent.
The rmsnorm-only paths pass a one-element fp16 dummy for the delta, so in training every pointer
keeps one dtype and each kernel still compiles once (an fp32 delta, which no training path
produces, would add a second variant). New runtime flags, all in do_not_specialize:
- fwd `store_r`: 0 skips the store of r (solution/fused_block.py recomputes y in backward without
  materialising r);
- bwd `has_delta`: the kernel rebuilds r = X + f32(D) with the forward's exact expression instead of
  reading a saved r, so a caller can keep (residual, fp16 delta) for backward instead of the fp32
  r (solution/fused_block.py). With has_delta = 0 the X row is used unchanged (tl.where, no + 0.0).

Kernels:
- soup_addnorm_fwd_kernel: optional add (runtime flag has_delta), stores r (fp32) only when it
  added and store_r is set, y, rstd.
- soup_addnorm_bwd_kernel: dx = rstd * (g - x * rstd^2 * mean(g * x)) with g = w * dy, plus the
  incoming residual gradient dr (runtime flag has_dr). No dW: the norm weights are frozen.
- soup_addnorm_fwd_usig / soup_addnorm_bwd_usig (v8 C1, USE_UNIFIED_SIG): the same two kernels
  behind the launcher signature all our kernels share (solution/kernels/__init__.py).

v10 P2 (USE_EW_K3, solution/__init__.py EW_K3_FP16): two more entries with the same twelve-slot
signature, `soup_addnorm_fwd_v10` and `soup_addnorm_bwd_v10`. While the flag is on EVERY K3 launch
goes to them (the v8 pair is not launched), so a run still compiles two K3 cubins. They carry the
v8 bodies plus casts that used to be separate torch launches:
- fwd mode 0: the v8 forward (fp32 y). mode 1: the same computation, y stored as fp16 into Y16
  (B's `y.to(torch.float16)` in forward and in the backward recompute, solution/fused_block.py).
  modes 2 / 3: the q/v LoRA epilogue of solution/lora_recompute.py (no norm; flat elementwise over
  rows of 1536 elements): out = f32(base16) + f32(fp16(f32(lora16) * s)) and its backward
  g16 = fp16(g), gl16 = fp16(f32(g16) * s), the bits of peft's `.to(fp32)` / clone / `* scaling` /
  `+` and of their autograd backward (casts, MulBackward), see lora_recompute.py.
- bwd: dy is read either as fp16 from slot 0 (dy_mode 0, B: dx16 straight from the S4 GEMMs,
  no `dx16.float()`) or as fp32 from the DR slot (dy_mode 1, the rmsnorm-only backward); the
  residual gradient comes from DR (has_dr 1) or is pre-loaded in DX (has_dr 2, _AddRMSNorm);
  store_da writes fp16(dx) back over the fp16 dy row (B's `d_mid.to(fp16)`, in place over a
  temporary only B owns). Every fp32 -> fp16 cast is an explicit round-to-nearest-even, as torch's.
The reductions keep the v8 expressions; whether Triton keeps the v8 register layout (and so the
v8 summation order of rstd / mean(g x)) with the extra fp16 loads and stores is checked on the T4
(solution/tests/test_ew_fusion.py); if it does not, values move by fp32 ulps against v9b and stay
consistent inside a run (forward, recompute and replay all use these entries).
"""

import torch
import triton
import triton.language as tl

# v10d: precompile.wait() joins a pending background compile (a module-global None check after it)
from .. import precompile as _precompile

HIDDEN_SIZE = 1536  # Qwen2.5-1.5B hidden size; the only width this file accepts
BLOCK = 2048  # next power of two >= HIDDEN_SIZE
NUM_WARPS = 4  # 128 threads x 16 elements per row
# v8 C1: launch the *_usig entries below instead of the two kernels directly. solution.apply()
# sets it; the module default is the v7 launch.
USE_UNIFIED_SIG = False
# v10 P2: every K3 launch goes to the *_v10 entries (see the docstring). solution.apply() sets it;
# the module default is v9b.
USE_EW_K3 = False
_INT32_END = 2**31


@triton.jit(do_not_specialize=["has_delta", "store_r", "eps"])
def soup_addnorm_fwd_kernel(
    X, D, W, Y, R, RSTD,
    has_delta, store_r, eps,
    H: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)  # grid = (rows,), one row per program
    col = tl.arange(0, BLOCK)
    cmask = col < H
    offs = row * H + col
    add = cmask & (has_delta != 0)

    x = tl.load(X + offs, mask=cmask, other=0.0).to(tl.float32)
    d = tl.load(D + offs, mask=add, other=0.0).to(tl.float32)
    x = x + d  # residual + f32(delta): the same single fp32 add as base (line 291)
    tl.store(R + offs, x, mask=add & (store_r != 0))  # otherwise nothing stored, R may be a dummy

    w = tl.load(W + col, mask=cmask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)  # torch.rsqrt in base (line 251)
    y = w * (x * rstd)  # base order: hs * rsqrt, then weight * hs (lines 251-252)
    tl.store(Y + offs, y, mask=cmask)
    tl.store(RSTD + row, rstd)


@triton.jit(do_not_specialize=["has_dr", "has_delta"])
def soup_addnorm_bwd_kernel(
    DY, DR, X, D, W, RSTD, DX,
    has_dr, has_delta,
    H: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    cmask = col < H
    offs = row * H + col

    x = tl.load(X + offs, mask=cmask, other=0.0).to(tl.float32)
    d = tl.load(D + offs, mask=cmask & (has_delta != 0), other=0.0).to(tl.float32)
    # The forward's r = x + d, bit for bit (same fp32 operands, one IEEE add); has_delta = 0 keeps
    # the saved row as it is.
    x = tl.where(has_delta != 0, x + d, x)
    dy = tl.load(DY + offs, mask=cmask, other=0.0).to(tl.float32)
    w = tl.load(W + col, mask=cmask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)

    # y_j = w_j x_j r, dr/dx_i = -r^3 x_i / H  =>  dx_i = r (g_i - x_i r^2 mean(g x)), g = w dy.
    # fp32 input: the cast boundary of line 252 is the identity, no rounding of g.
    g = dy * w
    mean_gx = tl.sum(g * x, axis=0) / H
    dx = rstd * (g - x * (rstd * rstd * mean_gx))
    dr = tl.load(DR + offs, mask=cmask & (has_dr != 0), other=0.0).to(tl.float32)
    tl.store(DX + offs, dx + dr, mask=cmask)


# v8 C1 entries: slots 0-6 pointers, 7-10 int32, 11 fp32, no constexpr parameter
# (solution/kernels/__init__.py). The bodies are the kernels above, inlined, with the same
# arguments and constexpr values; the slots the v7 kernels left specialised stay specialised.
_H = tl.constexpr(HIDDEN_SIZE)
_BLOCK = tl.constexpr(BLOCK)


@triton.jit(do_not_specialize=["spare_p6", "has_delta", "store_r", "spare_i2", "spare_i3", "eps"])
def soup_addnorm_fwd_usig(
    X, D, W, Y, R, RSTD, spare_p6,
    has_delta, store_r, spare_i2, spare_i3,
    eps,
):
    soup_addnorm_fwd_kernel(X, D, W, Y, R, RSTD, has_delta, store_r, eps, _H, _BLOCK)


@triton.jit(do_not_specialize=["has_dr", "has_delta", "spare_i2", "spare_i3", "spare_f"])
def soup_addnorm_bwd_usig(
    DY, DR, X, D, W, RSTD, DX,
    has_dr, has_delta, spare_i2, spare_i3,
    spare_f,
):
    soup_addnorm_bwd_kernel(DY, DR, X, D, W, RSTD, DX, has_dr, has_delta, _H, _BLOCK)


# v10 P2 entries (USE_EW_K3). Same twelve-slot layout as the *_usig entries (one shared launcher);
# every pointer slot is a real, 16-byte aligned tensor of a fixed dtype on every call (dummies are
# fresh allocations), so each entry compiles once. The norm code is the v8 body line for line.
@triton.jit(do_not_specialize=["has_delta", "store_r", "mode", "n16", "eps"])
def soup_addnorm_fwd_v10(
    X, D, W, Y, R, RSTD, Y16,
    has_delta, store_r, mode, n16,
    eps,
):
    """mode 0: the v8 forward, fp32 y -> Y. mode 1: the same computation, fp16 y -> Y16.
    mode 2 (LoRA epilogue): Y = f32(D) + f32(fp16(f32(Y16) * eps)) over the first n16 * 16 elements.
    mode 3 (its backward): D = fp16(X), Y16 = fp16(f32(D) * eps) over the first n16 * 16 elements.
    One program per row of _H elements; modes 2 / 3 read the tensors as flat rows of _H."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, _BLOCK)
    cmask = col < _H
    offs = row * _H + col
    if mode < 2:
        add = cmask & (has_delta != 0)
        x = tl.load(X + offs, mask=cmask, other=0.0).to(tl.float32)
        d = tl.load(D + offs, mask=add, other=0.0).to(tl.float32)
        x = x + d  # residual + f32(delta), as soup_addnorm_fwd_kernel
        tl.store(R + offs, x, mask=add & (store_r != 0))
        w = tl.load(W + col, mask=cmask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / _H
        rstd = tl.rsqrt(var + eps)
        y = w * (x * rstd)
        tl.store(Y + offs, y, mask=cmask & (mode == 0))
        # torch's fp32 -> fp16 `.to()` rounds to nearest even; so does this cast.
        tl.store(Y16 + offs, y.to(tl.float16, fp_downcast_rounding="rtne"),
                 mask=cmask & (mode != 0))
        tl.store(RSTD + row, rstd)
    else:
        # n16 * 16 is a multiple of 16 for the compiler, so the bound keeps 16-byte vectors.
        emask = cmask & (offs < n16 * 16)
        if mode == 2:
            base = tl.load(D + offs, mask=emask, other=0.0).to(tl.float32)  # exact widening
            lb = tl.load(Y16 + offs, mask=emask, other=0.0).to(tl.float32)
            # fp16 tensor * python scalar: the product in fp32, rounded once to fp16 (torch's
            # mul on an fp16 tensor), then the type-promoting fp32 add of two exact widenings.
            lora = (lb * eps).to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
            tl.store(Y + offs, base + lora, mask=emask)
        else:
            g = tl.load(X + offs, mask=emask, other=0.0)
            g16 = g.to(tl.float16, fp_downcast_rounding="rtne")  # the grad cast to fp16
            tl.store(D + offs, g16, mask=emask)
            gl = (g16.to(tl.float32) * eps).to(tl.float16, fp_downcast_rounding="rtne")
            tl.store(Y16 + offs, gl, mask=emask)


@triton.jit(do_not_specialize=["has_dr", "has_delta", "dy_mode", "store_da", "spare_f"])
def soup_addnorm_bwd_v10(
    DY16, DR, X, D, W, RSTD, DX,
    has_dr, has_delta, dy_mode, store_da,
    spare_f,
):
    """The v8 backward with dy read as fp16 from DY16 (dy_mode 0) or as fp32 from DR (dy_mode 1);
    the residual gradient from DR (has_dr 1) or already in DX (has_dr 2); store_da 1 also writes
    fp16(dx) over the DY16 row it read (every load of the row precedes the stores through the
    row reduction)."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, _BLOCK)
    cmask = col < _H
    offs = row * _H + col

    x = tl.load(X + offs, mask=cmask, other=0.0).to(tl.float32)
    d = tl.load(D + offs, mask=cmask & (has_delta != 0), other=0.0).to(tl.float32)
    x = tl.where(has_delta != 0, x + d, x)
    dy_h = tl.load(DY16 + offs, mask=cmask & (dy_mode == 0), other=0.0).to(tl.float32)
    dy_f = tl.load(DR + offs, mask=cmask & (dy_mode != 0), other=0.0).to(tl.float32)
    dy = tl.where(dy_mode != 0, dy_f, dy_h)  # f32(dx16) is exact: torch's `.float()` values
    w = tl.load(W + col, mask=cmask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)

    g = dy * w
    mean_gx = tl.sum(g * x, axis=0) / _H
    dx = rstd * (g - x * (rstd * rstd * mean_gx))
    dr_a = tl.load(DR + offs, mask=cmask & (has_dr == 1), other=0.0).to(tl.float32)
    dr_b = tl.load(DX + offs, mask=cmask & (has_dr == 2), other=0.0).to(tl.float32)
    dr = tl.where(has_dr == 2, dr_b, dr_a)
    out = dx + dr
    tl.store(DX + offs, out, mask=cmask)
    tl.store(DY16 + offs, out.to(tl.float16, fp_downcast_rounding="rtne"),
             mask=cmask & (store_da != 0))


def _dummy16(x):
    # Never read (its loads are masked off); fp16 so the training paths keep one variant.
    return torch.empty((1,), device=x.device, dtype=torch.float16)


def _launch_fwd(x, delta, weight, eps, store_r=True):
    """(r or None, y, rstd). r is returned only when delta is given and store_r is True."""
    if USE_EW_K3:  # v10 P2: the v10 entry, mode 0 (same outputs)
        return _launch_fwd_v10(x, delta, weight, eps, store_r, False)
    _precompile.wait()  # v10d
    rows = x.shape[0]
    y = torch.empty_like(x)
    keep_r = delta is not None and store_r
    r = torch.empty_like(x) if keep_r else y  # y doubles as a dummy pointer, its store is masked off
    rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
    if rows:
        with torch.cuda.device(x.device):
            if USE_UNIFIED_SIG:  # C1: y fills the spare pointer slot (never read)
                soup_addnorm_fwd_usig[(rows,)](
                    x, delta if delta is not None else _dummy16(x), weight, y, r, rstd, y,
                    1 if delta is not None else 0, 1 if keep_r else 0, 0, 0,
                    float(eps),
                    num_warps=NUM_WARPS,
                )
            else:
                soup_addnorm_fwd_kernel[(rows,)](
                    x, delta if delta is not None else _dummy16(x), weight, y, r, rstd,
                    1 if delta is not None else 0, 1 if keep_r else 0, eps,
                    H=HIDDEN_SIZE, BLOCK=BLOCK, num_warps=NUM_WARPS,
                )
    return (r if keep_r else None), y, rstd


def _launch_bwd(dy, dr, x, weight, rstd, delta=None):
    """dx (+ dr). With delta, x is the residual before the add and the kernel rebuilds x + delta."""
    if USE_EW_K3:  # v10 P2: the v10 entry with fp32 dy (dy_mode 1); same outputs
        return _launch_bwd_v10(None, dy, dr, x, weight, rstd, delta, False)
    _precompile.wait()  # v10d
    rows = x.shape[0]
    dx = torch.empty_like(x)
    if rows:
        with torch.cuda.device(x.device):
            if USE_UNIFIED_SIG:  # C1: seven pointers already; spare ints 0, spare float 0.0
                soup_addnorm_bwd_usig[(rows,)](
                    dy, dr if dr is not None else dy, x,
                    delta if delta is not None else _dummy16(x), weight, rstd, dx,
                    1 if dr is not None else 0, 1 if delta is not None else 0, 0, 0,
                    0.0,
                    num_warps=NUM_WARPS,
                )
            else:
                soup_addnorm_bwd_kernel[(rows,)](
                    dy, dr if dr is not None else dy, x,
                    delta if delta is not None else _dummy16(x), weight, rstd, dx,
                    1 if dr is not None else 0, 1 if delta is not None else 0,
                    H=HIDDEN_SIZE, BLOCK=BLOCK, num_warps=NUM_WARPS,
                )
    return dx


# ---------------------------------------------------------------------------------------------
# v10 P2 launchers (USE_EW_K3)
# ---------------------------------------------------------------------------------------------

def _launch_fwd_v10(x, delta, weight, eps, store_r, fp16_out):
    """(r or None, y, rstd) as _launch_fwd; with fp16_out, y is fp16 (mode 1) and no fp32 y is
    allocated. Unused pointer slots get fresh tensors of the slot's dtype (masked off)."""
    _precompile.wait()  # v10d
    rows = x.shape[0]
    keep_r = delta is not None and store_r
    rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
    dummy = _dummy16(x)
    if fp16_out:
        y = torch.empty(x.shape, device=x.device, dtype=torch.float16)
        y32, y16 = rstd, y  # Y's store is masked off in mode 1
    else:
        y = torch.empty_like(x)
        y32, y16 = y, dummy  # Y16's store is masked off in mode 0
    r = torch.empty_like(x) if keep_r else y32  # R's store is masked off unless keep_r
    if rows:
        with torch.cuda.device(x.device):
            soup_addnorm_fwd_v10[(rows,)](
                x, delta if delta is not None else dummy, weight, y32, r, rstd, y16,
                1 if delta is not None else 0, 1 if keep_r else 0, 1 if fp16_out else 0, 0,
                float(eps),
                num_warps=NUM_WARPS,
            )
    return (r if keep_r else None), y, rstd


def _launch_bwd_v10(dy16, dy, dr, x, weight, rstd, delta, store_da):
    """dx (+ dr) as _launch_bwd. dy16 (fp16, dy_mode 0; with store_da, fp16(dx + dr) is written
    over it) or dy (fp32, dy_mode 1; a residual gradient then rides in DX, has_dr 2)."""
    _precompile.wait()  # v10d
    rows = x.shape[0]
    dummy = _dummy16(x)
    if dy16 is not None:
        dx = torch.empty_like(x)
        dy_slot, dr_slot, dy_mode = dy16, (dr if dr is not None else dx), 0
        has_dr = 1 if dr is not None else 0
    else:
        if dr is not None:
            dx = dr.clone(memory_format=torch.contiguous_format)  # fresh: read, then overwritten
            has_dr = 2
        else:
            dx = torch.empty_like(x)
            has_dr = 0
        dy_slot, dr_slot, dy_mode = dummy, dy, 1
        store_da = False
    if rows:
        with torch.cuda.device(x.device):
            soup_addnorm_bwd_v10[(rows,)](
                dy_slot, dr_slot, x, delta if delta is not None else dummy, weight, rstd, dx,
                has_dr, 1 if delta is not None else 0, dy_mode, 1 if store_da else 0,
                0.0,
                num_warps=NUM_WARPS,
            )
    return dx


def launch_fwd16(x, delta, weight, eps, store_r=True):
    """(r or None, fp16 y, rstd): the K3 forward with y stored as fp16 (mode 1), i.e. the bits
    of `_launch_fwd(...)[1].to(torch.float16)` without the fp32 y. Caller: B (fused_block.py)
    while USE_EW_K3 is on, on its own checked operands."""
    return _launch_fwd_v10(x, delta, weight, eps, store_r, True)


def launch_bwd16(dy16, dr, x, weight, rstd, delta=None, store_da=False):
    """The K3 backward from an fp16 dy: the bits of `_launch_bwd(dy16.float(), dr, ...)`; with
    store_da, dy16 then holds fp16 of the result (`.to(torch.float16)`). dy16 must be a
    contiguous, 16-byte aligned fp16 tensor of x's shape that nothing else reads afterwards."""
    return _launch_bwd_v10(dy16, None, dr, x, weight, rstd, delta, store_da)


def lora_epilogue_ok(base16, lora16):
    """True when lora_epilogue_fwd takes these operands: CUDA fp16, same shape and device,
    contiguous, 16-byte aligned, numel a nonzero multiple of 16 below 2**31."""
    if not (isinstance(base16, torch.Tensor) and isinstance(lora16, torch.Tensor)):
        return False
    if base16.dtype != torch.float16 or lora16.dtype != torch.float16:
        return False
    if not base16.is_cuda or base16.device != lora16.device or base16.shape != lora16.shape:
        return False
    n = base16.numel()
    if n == 0 or n % 16 or n >= _INT32_END:
        return False
    return (base16.is_contiguous() and lora16.is_contiguous()
            and base16.data_ptr() % 16 == 0 and lora16.data_ptr() % 16 == 0)


def lora_epilogue_fwd(base16, lora16, scale):
    """f32(base16) + f32(fp16(f32(lora16) * scale)) as a new fp32 tensor (mode 2). The caller
    checked lora_epilogue_ok and that scale is a float."""
    _precompile.wait()  # v10d
    out = torch.empty(base16.shape, device=base16.device, dtype=torch.float32)
    n = base16.numel()
    with torch.cuda.device(base16.device):
        soup_addnorm_fwd_v10[(triton.cdiv(n, HIDDEN_SIZE),)](
            out, base16, out, out, out, out, lora16,
            0, 0, 2, n // 16,
            float(scale),
            num_warps=NUM_WARPS,
        )
    return out


def lora_epilogue_bwd(g, scale):
    """(fp16(g), fp16(f32(fp16(g)) * scale)) for an fp32 g (mode 3). g is made contiguous and
    16-byte aligned here if it is not."""
    _precompile.wait()  # v10d
    n = g.numel()
    if n % 16 or n >= _INT32_END:  # never for an output lora_epilogue_fwd made; torch's two ops
        g16 = g.to(torch.float16)
        return g16, g16 * scale
    if not g.is_contiguous() or g.data_ptr() % 16:
        g = g.clone(memory_format=torch.contiguous_format)
    g16 = torch.empty(g.shape, device=g.device, dtype=torch.float16)
    gl16 = torch.empty(g.shape, device=g.device, dtype=torch.float16)
    if n:
        with torch.cuda.device(g.device):
            soup_addnorm_fwd_v10[(triton.cdiv(n, HIDDEN_SIZE),)](
                g, g16, g, g, g, g, gl16,
                0, 0, 3, n // 16,
                float(scale),
                num_warps=NUM_WARPS,
            )
    return g16, gl16


def _grad(t):
    # Our outputs are fp32, so autograd hands fp32 grads; only contiguity can differ (e.g. a
    # grad produced by expand/transpose downstream).
    return t if t is None or t.is_contiguous() else t.contiguous()


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        _, y, rstd = _launch_fwd(x, None, weight, eps)
        ctx.save_for_backward(x, rstd, weight)
        return y

    @staticmethod
    def backward(ctx, dy):
        # Only reached when x requires grad: the weight is frozen and eps is a float.
        x, rstd, weight = ctx.saved_tensors
        return _launch_bwd(_grad(dy), None, x, weight, rstd), None, None


class _AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, weight, eps):
        r, y, rstd = _launch_fwd(x, delta, weight, eps)
        ctx.save_for_backward(r, rstd, weight)
        ctx.set_materialize_grads(False)
        ctx.delta_dtype = delta.dtype
        return r, y

    @staticmethod
    def backward(ctx, dr, dy):
        r, rstd, weight = ctx.saved_tensors
        if dy is None:
            d = _grad(dr)
        else:
            d = _launch_bwd(_grad(dy), _grad(dr), r, weight, rstd)
        # d(residual + delta) = (1, 1): base AddBackward also hands the same grad to both inputs.
        # An fp16 delta gets the fp16 rounding of it, which is what base's ToCopyBackward of the
        # fp32 promotion (and v5's explicit delta.to(fp32)) produced.
        need_x, need_delta = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        d_delta = None
        if need_delta and d is not None:
            d_delta = d if d.dtype == ctx.delta_dtype else d.to(ctx.delta_dtype)
        return (d if need_x else None), d_delta, None, None


def norm_ok(x, weight):
    """True when (x, weight) fit the kernels: CUDA fp32 [..., 1536] input, frozen fp32 weight."""
    return (
        isinstance(x, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.float32
        and x.ndim >= 1
        and x.shape[-1] == HIDDEN_SIZE
        and weight.dtype == torch.float32
        and weight.ndim == 1
        and weight.numel() == HIDDEN_SIZE
        and weight.device == x.device
        and not weight.requires_grad
        and weight.is_contiguous()
    )


def rmsnorm(x, weight, eps):
    """Qwen2RMSNorm on an fp32 residual stream. Caller checks norm_ok(x, weight)."""
    shape = x.shape
    x2 = (x if x.is_contiguous() else x.contiguous()).view(-1, HIDDEN_SIZE)
    return _RMSNorm.apply(x2, weight, float(eps)).view(shape)


def add_rmsnorm(residual, delta, weight, eps):
    """(residual + delta, RMSNorm(residual + delta)) in one kernel. Caller checks norm_ok and that
    delta is a CUDA fp16/fp32 tensor of the same shape and device as residual."""
    shape = residual.shape
    x2 = (residual if residual.is_contiguous() else residual.contiguous()).view(-1, HIDDEN_SIZE)
    d2 = (delta if delta.is_contiguous() else delta.contiguous()).view(-1, HIDDEN_SIZE)
    r, y = _AddRMSNorm.apply(x2, d2, weight, float(eps))
    return r.view(shape), y.view(shape)
