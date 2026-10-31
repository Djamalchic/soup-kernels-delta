"""S4: the Qwen2 MLP keeps only its fp16 input for backward and recomputes gate/up there.

K4 (`solution/mlp.py`) runs the MLP on fp16 activations; autograd then keeps the fp16 gate and up
outputs for the SwiGLU backward: 2 x [tokens, 8960] fp16 = 70 MiB per layer at 2048 tokens.
Nothing else in the MLP saves activations: bitsandbytes' MatMul4Bit keeps only the packed weight
(`bitsandbytes/autograd/_functions.py:358`, `ctx.tensors = (None, B)`), and so does K1's
`SoupNF4Matmul.forward` (`solution/kernels/nf4.py`, `ctx.save_for_backward(packed)`).

`SoupMLPRecompute` wraps the whole MLP in one autograd node that saves only x16 (6 MiB per layer at
2048 tokens) and, in backward, replays the ops autograd would have run for K4:

  1. dh   = grad_out @ dequant(W_down)                 (MatMul4Bit.backward, `_functions.py:384`;
                                                        K1's `SoupNF4Matmul.backward`)
  2. g, u = gate_proj(x16), up_proj(x16)              (the same Linear4bit calls as the forward)
  3. dg, du = K4's SwiGLU backward(dh, g, u)           (v8 C2: `kernels/swiglu.py`; flag off: the
                                                        example's, `swiglu.py:53-61`, called)
  4. dx16 = dg @ dequant(W_gate) + du @ dequant(W_up)  (summed in fp16, as autograd sums the two
                                                        gradients that flow into x16)

Why the recompute gives the forward's bits: inside Function.forward and backward grad mode is off,
so bitsandbytes' matmul_4bit takes its no-grad branch (`_functions.py:453-489`), which calls the
same `gemm_4bit.default` op with the same arguments as `MatMul4Bit.forward` (`:323-345`); B is
normalised to `view(-1, 1)` before either branch (`:431`), and the CUDA dispatch picks its kernel
from M, N, K only (`backends/cuda/ops.py:935-985`). K1's `soup_matmul_4bit` (`kernels/nf4.py`)
does not look at grad mode either. So with the default settings outputs and input gradients are
bit-identical to K4 (the tests check `torch.equal`).

The pass is guarded: unless USE_MLP_RECOMPUTE is on, grad is being recorded, x requires grad,
K4's guard passes (`mlp._fp16_path_ok`: CUDA input, exact bnb Linear4bit with fp16 compute dtype,
config.hidden_act == "silu"), autocast is off or fp16, no global module hooks exist, and every
projection has no bias, a frozen Params4bit nf4 weight on x's device and no module hooks, the call
goes to K4 (which itself falls back to the original Qwen2MLP.forward). A guard that rejects a
training call while the flag is on prints one RuntimeWarning, so a silently dead S4 shows in the
run log. Importing this module does no CUDA work and imports neither Triton nor bitsandbytes.

v8 C2: while K4's switch USE_SOUP_SWIGLU is on (`mlp.soup_swiglu_module()`), step 3 and the
forward's SwiGLU run our soup_swiglu_kernel (solution/kernels/swiglu.py: forward, backward and the
same in-place backward) instead of the example's kernels. Same values; the example code below stays
as the flag-off path.

v9 M5 (BWD_ROW_CHUNKS > 1 from USE_BWD_CHUNKING, off since v9b): on the "both" path the backward
keeps one dense weight at a time and recomputes u in row chunks (_gate_up_input_grad_rows). Per
MLP backward at N rows (U = N x 8960 fp16, W = one dense [8960, 1536] fp16 weight, 26.25 MiB):
"both" holds dh, g, u, W_gate, W_up = 3U + 2W at the recompute of u; "rows" holds dh, g, W_up and
one u chunk = 2U + W + U/k. At N = 1350, k = 4 (384-row chunks): 121.7 -> 78.9 MiB of transients.
Cost: W_gate is dequantized twice (3 dequants instead of 2) and the u GEMM runs in k pieces.

Idea credit: activation recomputation, Chen et al. 2016, "Training Deep Nets with Sublinear Memory
Cost" (arXiv:1604.06174); row-chunked MLP passes are a common trick (e.g. Liger-Kernel's chunked
fused-linear losses, Unsloth's chunked MLP); no code taken. The code is our own. No Triton kernel
lives in this file.
"""

import sys
import warnings

import torch
import torch.nn.functional as F

from . import mlp as _k4

# Module default off (tests import this module alone); solution.apply() sets it (on since v5f).
USE_MLP_RECOMPUTE = False
# True: recompute gate and up (saves 64 MiB per layer at 2048 tokens).
# False: keep the fp16 gate output, recompute only up (saves 29 MiB per layer, half the extra GEMMs).
RECOMPUTE_GATE = True
# "example" (a historical value name): the SwiGLU backward kernel K4 uses (bit-exact with K4),
# ours with v8 C2's USE_SOUP_SWIGLU on (as submitted), else the starter kit's. "torch": fp32 math.
SWIGLU_BACKWARD = "example"
# C (v6): K4's SwiGLU backward kernel (ours with C2, the example's without) writes its outputs
# over its own inputs (dgate over grad, dup over gate) when gate was recomputed here. Same kernel,
# same arguments otherwise, so the same compiled variant and the same bits; two [tokens, 8960]
# fp16 buffers fewer at the moment of the MLP backward (see _swiglu_backward_inplace).
INPLACE_SWIGLU_BACKWARD = True
# v7 I2: dense gate/up weights in the backward. "off" (v6.1): gate_proj(x16) and up_proj(x16)
# dequantize W_gate and W_up inside K1's SoupNF4Matmul.forward, then step 4 dequantizes both
# again (4 [8960, 1536] dequants per layer). "both": dequantize each once, recompute g = x16 W_g^T
# and u = x16 W_u^T from those dense weights and keep them for dx (2 dequants; +1 dense weight,
# 26.25 MiB, alive at the recompute of u). "up": keep only W_up (3 dequants; no extra weight at
# the recompute of u, +1 weight at step 4 only, where far fewer buffers are alive). Taken only
# when K1 would have run the same F.linear (kernels/nf4.py k1_dense_path); otherwise the v6.1
# code runs and REUSE_COUNTS["fallback"] counts it. apply() sets this (solution/__init__.py).
S4_DEQUANT_REUSE = "off"
REUSE_COUNTS = {"both": 0, "up": 0, "fallback": 0}
# v9 M5: row chunks for the recompute of u on the "both" path (_gate_up_input_grad_rows). 1 (the
# module default) is v7 I2's "both" unchanged; apply() writes it. Chunks start at multiples of
# BWD_ROW_ALIGN rows (cuBLAS's 128-row tiles: the k pieces cover as many tiles as the full GEMM),
# and batches under BWD_CHUNK_MIN_ROWS rows keep "both" (they are not the step's memory peak).
# A rows call is also counted as "both" in REUSE_COUNTS (it is the same dequant-reuse path).
BWD_ROW_CHUNKS = 1
BWD_CHUNK_MIN_ROWS = 512
BWD_ROW_ALIGN = 128
ROWS_COUNTS = {"rows": 0}

_PATCH_MARK = "_soup_delta_mlp_recompute"
_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")
_LAYER_HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
# torch.nn.modules.module keeps the register_module_*_hook registries in these dicts. They are
# private names, read with getattr; a missing name simply reads as "no hooks".
_GLOBAL_HOOKS = ("_global_forward_hooks", "_global_forward_pre_hooks",
                 "_global_backward_hooks", "_global_backward_pre_hooks")
_NF4_MODULE = f"{__package__}.kernels.nf4"
_NOT_TRAINING = "not a training call"

_swiglu_forward = None  # set by the patch: solution.swiglu.swiglu
_swiglu_example_backward = None  # set by the patch: solution.swiglu.SwiGLU.backward, if present
# set by the patch: (solution.swiglu.example_swiglu_bwd_kernel, solution.swiglu.BLOCK), if present
_swiglu_example_kernel = None
_warned = False


class _SavedPair:
    """The one attribute the example's SwiGLU.backward reads from its ctx (`swiglu.py:55`)."""

    __slots__ = ("saved_tensors",)

    def __init__(self, gate, up):
        self.saved_tensors = (gate, up)


def _swiglu_backward_torch(dh, gate, up):
    """dgate, dup of silu(gate) * up in fp32 math, rounded to the input dtype once at the end.

    Same expressions and multiplication order as the example kernel (`swiglu.py:33-35`):
      dup   = (grad * gate) * sig
      dgate = ((grad * up) * sig) * (1 + gate * (1 - sig))
    Mixed fp32 x fp16 ops promote to fp32, so the fp16 operands enter exactly. At most three fp32
    [tokens, intermediate] buffers are alive at once. Slow (~11 passes): a fallback only.
    """
    out_dtype = gate.dtype
    d = dh.float()
    sig = torch.sigmoid(gate.float())
    du = torch.mul(d, gate)
    du.mul_(sig)
    du_out = du.to(out_dtype)
    del du
    t = torch.rsub(sig, 1.0)  # 1 - sig
    t.mul_(gate).add_(1.0)  # 1 + gate * (1 - sig)
    d.mul_(up).mul_(sig).mul_(t)
    del sig, t
    return d.to(out_dtype), du_out


def _swiglu_fwd(g, u):
    """silu(g) * u through K4's SwiGLU: ours (v8 C2) or the example's (v7)."""
    soup = _k4.soup_swiglu_module()
    if soup is not None:
        return soup.swiglu(g, u)
    return _swiglu_forward(g, u)


def _swiglu_backward(dh, gate, up):
    if SWIGLU_BACKWARD == "example":
        soup = _k4.soup_swiglu_module()
        if soup is not None:
            # v8 C2: the operands the example's forward would have saved (no-ops for our
            # contiguous fp16 g and u), then our kernel's backward into fresh buffers.
            return soup.swiglu_backward(dh, gate.contiguous(), up.to(gate.dtype).contiguous())
    if SWIGLU_BACKWARD == "example" and _swiglu_example_backward is not None:
        try:
            # The example's forward saves gate.contiguous() and up.to(gate.dtype).contiguous()
            # (`swiglu.py:45-50`); both are no-ops for our fp16 contiguous g and u.
            return _swiglu_example_backward(
                _SavedPair(gate.contiguous(), up.to(gate.dtype).contiguous()), dh
            )
        except (AttributeError, TypeError, ValueError):
            pass  # the example's API changed: same math in torch below
    return _swiglu_backward_torch(dh, gate, up)


def _inplace_ok(dh, gate, up):
    if not INPLACE_SWIGLU_BACKWARD or SWIGLU_BACKWARD != "example":
        return False
    if _k4.soup_swiglu_module() is None and _swiglu_example_kernel is None:
        return False
    return (dh.dtype == gate.dtype == up.dtype == torch.float16
            and dh.shape == gate.shape == up.shape
            and dh.is_cuda and gate.device == dh.device and up.device == dh.device
            and dh.is_contiguous() and gate.is_contiguous() and up.is_contiguous()
            and dh.data_ptr() != gate.data_ptr() and dh.data_ptr() != up.data_ptr()
            and gate.data_ptr() != up.data_ptr())


def _swiglu_backward_inplace(dh, gate, up):
    """(dgate, dup) = the example's SwiGLU backward, written into dh (dgate) and gate (dup).

    The example kernel (`swiglu.py:25-37`) loads grad, gate and up at its offsets once, then stores
    dgate = f(grad, up, gate) and dup = f(grad, gate). Every store depends on the load of the
    buffer it overwrites (dgate needs grad, dup needs gate), so no compiler reordering can store
    before that load; programs touch disjoint offsets. `up` is not written: dup does not depend on
    up, so writing it there could be reordered before up's load. The launch is the example
    wrapper's own (`swiglu.py:57-60`: grid cdiv(n, BLOCK), the same BLOCK, fp16 pointers, the
    same n), so Triton reuses its compiled variant. The caller passes fresh dh and gate buffers
    that nothing else references.

    v8 C2: with USE_SOUP_SWIGLU on, our kernel's backward does the same in place
    (kernels/swiglu.py swiglu_backward_into: same argument, same outputs, same buffers).
    """
    soup = _k4.soup_swiglu_module()
    if soup is not None:
        return soup.swiglu_backward_into(dh, gate, up)
    kernel, block = _swiglu_example_kernel
    n = gate.numel()
    kernel[((n + block - 1) // block,)](dh, gate, up, dh, gate, n, BLOCK=block)
    return dh, gate


def _down_input_grad(grad_out, down_proj):
    """dh = grad_out @ dequant(W_down) (MatMul4Bit.backward, `_functions.py:384`)."""
    weight = _dequantize_weight(down_proj, grad_out.dtype)
    dh = torch.matmul(grad_out, weight)
    del weight
    return dh


def _reuse_nf4(mode, x16, gate_proj, up_proj, g_saved):
    """K1's module when the dequant-reuse path `mode` may replace the layer calls, else None.

    The replaced call is gate_proj(x16) / up_proj(x16): Linear4bit.forward (bitsandbytes
    nn/modules.py:609-637) with an fp16 compute dtype casts nothing, passes the Params4bit itself
    and bias=None to bnb.matmul_4bit (:637), which is K1's soup_matmul_4bit while _active_nf4()
    says so; when k1_dense_path holds, that is SoupNF4Matmul.apply -> F.linear(A,
    nf4_dequantize(B, quant_state), None) (kernels/nf4.py SoupNF4Matmul.forward), and the
    trailing .to(inp_dtype) is a no-op on fp16. Any doubt returns None (the v6.1 code runs).
    """
    if mode == "both" and g_saved is not None:
        return None
    nf4 = _active_nf4()
    dense_path = getattr(nf4, "k1_dense_path", None)
    if dense_path is None:
        return None
    try:
        from bitsandbytes.nn import Linear4bit

        for proj in ((gate_proj, up_proj) if mode == "both" else (up_proj,)):
            # K4's guard already demands exactly this (mlp._fp16_path_ok) and _reject_reason a
            # missing bias and no hooks; rechecked here because the fused block reaches this
            # helper from its own backward. An instance-level forward (e.g. an accelerate hook)
            # would be bypassed, so it keeps the layer call.
            if type(proj) is not Linear4bit or "forward" in proj.__dict__:
                return None
            if getattr(proj, "compute_dtype", None) not in (None, torch.float16):
                return None
            if getattr(proj, "bias", None) is not None:
                return None
            weight = proj.weight
            if not dense_path(x16, weight, weight.quant_state):
                return None
    except Exception:  # noqa: BLE001 - unexpected objects: the layer calls decide, as in v6.1
        return None
    return nf4


def _swiglu_grads(dh, g, u, may_inplace):
    if may_inplace and _inplace_ok(dh, g, u):
        return _swiglu_backward_inplace(dh, g, u)
    return _swiglu_backward(dh, g, u)


def _gate_up_input_grad_both(nf4, dh_box, x16, gate_proj, up_proj):
    """"both": steps 2-4 with each of W_gate, W_up dequantized once.

    The same operations on the same operands as the v6.1 code, in the same order: F.linear(x16, W)
    with bias None is what SoupNF4Matmul.forward runs inside gate_proj(x16); W is K1's dequant of
    the same Params4bit (bit-identical to the view(-1, 1) form _dequantize_weight used); dx is
    dg @ W_gate, then += du @ W_up. So g, u, dg, du and dx are bit-identical to "off".
    """
    dh = dh_box.pop()
    w_gate_q, w_up_q = gate_proj.weight, up_proj.weight
    with torch.autocast(x16.device.type, enabled=False):  # as in _gate_up_input_grad
        w_gate = nf4.nf4_dequantize(w_gate_q, w_gate_q.quant_state)
        g = F.linear(x16, w_gate)
        w_up = nf4.nf4_dequantize(w_up_q, w_up_q.quant_state)
        u = F.linear(x16, w_up)
    dg, du = _swiglu_grads(dh, g, u, True)
    del dh, g, u
    dx = torch.matmul(dg, w_gate)
    del w_gate, dg
    dx.add_(torch.matmul(du, w_up))
    del w_up, du
    return dx


def _rows_bounds(dh, x16):
    """Chunk boundaries [0, b1, ..., N] for _gate_up_input_grad_rows, or None ("both" runs).

    Needs dh and x16 contiguous fp16 with the same leading shape (so both flatten to [N, .] views
    without a copy) and N >= BWD_CHUNK_MIN_ROWS. Step: ceil(N / k) rounded up to BWD_ROW_ALIGN,
    so every chunk but the last starts and ends on a cuBLAS tile boundary; a last piece shorter
    than BWD_ROW_ALIGN is merged into the one before (no tiny-M GEMM, which cuBLAS may run with
    another kernel). One chunk left means no chunking: None.
    """
    k = BWD_ROW_CHUNKS
    if type(k) is not int or k < 2:
        return None
    if not (isinstance(dh, torch.Tensor) and isinstance(x16, torch.Tensor)):
        return None
    if dh.dtype != torch.float16 or x16.dtype != torch.float16 or x16.dim() < 2:
        return None
    if dh.dim() != x16.dim() or dh.shape[:-1] != x16.shape[:-1]:
        return None
    if not (dh.is_contiguous() and x16.is_contiguous()) or dh.device != x16.device:
        return None
    n = x16.numel() // max(x16.shape[-1], 1)
    min_rows = BWD_CHUNK_MIN_ROWS if type(BWD_CHUNK_MIN_ROWS) is int else 512
    if n < max(min_rows, k):
        return None
    align = BWD_ROW_ALIGN if type(BWD_ROW_ALIGN) is int and BWD_ROW_ALIGN > 0 else 1
    step = -(-n // k)
    step = -(-step // align) * align
    bounds = list(range(0, n, step)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < align:
        del bounds[-2]
    return bounds if len(bounds) > 2 else None


def _gate_up_input_grad_rows(nf4, dh_box, x16, gate_proj, up_proj, bounds):
    """v9 M5: "both" with one dense weight alive at a time and u recomputed in row chunks.

    Order (N rows; U = an [N, 8960] fp16 buffer, W = a dense [8960, 1536] fp16 weight):
      1. W_gate; g = F.linear(x16, W_gate) over all N rows (the forward's GEMM, M = N); free W_gate.
      2. W_up; per chunk c: u_c = F.linear(x16[c], W_up); SwiGLU backward in place:
         dg_c over dh[c], du_c over g[c] (the same kernel and arguments as "both", on row slices
         that start 16-byte aligned: 8960 * 2 and 1536 * 2 bytes per row); free u_c.
      3. dx = du @ W_up over all N rows (M = N, as "both"); free g (= du) and W_up.
      4. W_gate again; dx += dg @ W_gate (M = N, as "both").
    Peak: dh + g + W_up + u_c = 2U + W + U/k (+ dx at step 3: 2U + W + N x 1536 x 2), instead of
    "both"'s dh + g + u + W_gate + W_up = 3U + 2W. One more W_gate dequant per call.

    Values. g, dx's two GEMMs and the element-wise SwiGLU backward are the operations "both" runs
    on the same operands with the same shapes; dx = fp16(du W_up) + fp16(dg W_gate) is the same
    rounded IEEE sum as "both"'s fp16(dg W_gate) + fp16(du W_up) (addition commutes). The only
    shape change is the u GEMM (M = rows of the chunk instead of N). cuBLAS picks its kernel per
    (m, n, k), so the order in which an element of u is accumulated over K = 1536 can change with M.
    So "rows" is NOT bit-identical to "both" in general: Kaggle T4 (31.10) gave equal bits at
    N = 600, 1350, 1351, 1400, 2000, 2048 but not at N = 513 (chunks 384 + 129 and 256 + 257).
    The difference is ulp-level in u, hence in dgate (dup does not read u) and dx; checks 5/6 use
    tolerances. solution/tests/test_bwd_chunking.py checks "rows" bit for bit against "both"'s
    operations with u from the same per-chunk GEMMs (the chunk indexing), and against "both"
    itself within the S4 input-gradient bound.
    """
    dh = dh_box.pop()
    x2 = x16.view(-1, x16.shape[-1])
    d2 = dh.view(-1, dh.shape[-1])
    w_gate_q, w_up_q = gate_proj.weight, up_proj.weight
    with torch.autocast(x16.device.type, enabled=False):  # as in _gate_up_input_grad_both
        w = nf4.nf4_dequantize(w_gate_q, w_gate_q.quant_state)
        g = F.linear(x2, w)
        del w
        w = nf4.nf4_dequantize(w_up_q, w_up_q.quant_state)
        for r0, r1 in zip(bounds[:-1], bounds[1:]):
            u = F.linear(x2[r0:r1], w)
            dh_c, g_c = d2[r0:r1], g[r0:r1]
            dg, du = _swiglu_grads(dh_c, g_c, u, True)
            del u
            if dg is not dh_c:  # out-of-place fallback (torch SwiGLU or a refused in-place call)
                dh_c.copy_(dg)
            if du is not g_c:
                g_c.copy_(du)
            del dg, du, dh_c, g_c
    dx = torch.matmul(g, w)  # du @ W_up: g's buffer holds du now
    del g, w
    w = nf4.nf4_dequantize(w_gate_q, w_gate_q.quant_state)
    dx.add_(torch.matmul(d2, w))  # + dg @ W_gate: dh's buffer holds dg
    del w, d2, dh
    return dx.view(*x16.shape[:-1], dx.shape[-1])


def _gate_up_input_grad_up(nf4, dh_box, x16, gate_proj, up_proj, g_saved):
    """"up": W_up dequantized once and kept for dx; gate exactly as in v6.1. Same bits."""
    dh = dh_box.pop()
    w_up_q = up_proj.weight
    with torch.autocast(x16.device.type, enabled=False):
        g = g_saved if g_saved is not None else gate_proj(x16)
        w_up = nf4.nf4_dequantize(w_up_q, w_up_q.quant_state)
        u = F.linear(x16, w_up)
    dg, du = _swiglu_grads(dh, g, u, g_saved is None)
    del dh, g, u
    weight = _dequantize_weight(gate_proj, dg.dtype)
    dx = torch.matmul(dg, weight)
    del weight, dg
    dx.add_(torch.matmul(du, w_up))
    del w_up, du
    return dx


def _gate_up_input_grad(dh_box, x16, gate_proj, up_proj, g_saved=None):
    """Steps 2-4 of the S4 backward from dh: recompute g/u, SwiGLU backward, dx16.

    dh comes in a one-element list that is emptied here, so no caller frame keeps dh alive and
    its buffer (or, in place, its reuse as dgate) is the only one of its size.
    """
    mode = S4_DEQUANT_REUSE
    if mode == "both" or mode == "up":
        nf4 = _reuse_nf4(mode, x16, gate_proj, up_proj, g_saved)
        if nf4 is not None:
            REUSE_COUNTS[mode] += 1
            if mode == "both":
                bounds = _rows_bounds(dh_box[0], x16)  # v9 M5
                if bounds is not None:
                    ROWS_COUNTS["rows"] += 1
                    return _gate_up_input_grad_rows(nf4, dh_box, x16, gate_proj, up_proj, bounds)
                return _gate_up_input_grad_both(nf4, dh_box, x16, gate_proj, up_proj)
            return _gate_up_input_grad_up(nf4, dh_box, x16, gate_proj, up_proj, g_saved)
        REUSE_COUNTS["fallback"] += 1

    dh = dh_box.pop()
    # 2. Recompute through the layers. The forward ran with fp16 operands, where fp16 autocast
    #    casts nothing, so autocast is switched off here: if a caller ever runs backward inside
    #    an autocast region, the recompute still sees the forward's dtypes.
    with torch.autocast(x16.device.type, enabled=False):
        g = g_saved if g_saved is not None else gate_proj(x16)
        u = up_proj(x16)

    # 3. SwiGLU backward.
    if g_saved is None and _inplace_ok(dh, g, u):
        dg, du = _swiglu_backward_inplace(dh, g, u)
    else:
        dg, du = _swiglu_backward(dh, g, u)
    del dh, g, u

    # 4. gate_proj and up_proj input gradients, summed in fp16 like autograd's accumulation of
    #    two gradients into x16 (one rounded fp16 add; IEEE addition is commutative).
    weight = _dequantize_weight(gate_proj, dg.dtype)
    dx = torch.matmul(dg, weight)
    del weight, dg
    weight = _dequantize_weight(up_proj, du.dtype)
    dx.add_(torch.matmul(du, weight))
    return dx


def mlp_forward_nograd(x16, gate_proj, up_proj, down_proj):
    """down_proj(silu(gate_proj(x16)) * up_proj(x16)) with nothing kept; call with grad off."""
    g = gate_proj(x16)
    u = up_proj(x16)
    h = _swiglu_fwd(g, u)
    del g, u
    out = down_proj(h)
    del h
    return out


def _active_nf4():
    """K1's module when K1's bnb.matmul_4bit replacement is installed, else None.

    Reads sys.modules only: no import (and so no Triton import) happens in backward.
    """
    bnb = sys.modules.get("bitsandbytes")
    nf4 = sys.modules.get(_NF4_MODULE)
    if bnb is None or nf4 is None:
        return None
    fn = getattr(nf4, "soup_matmul_4bit", None)
    if fn is not None and getattr(bnb, "matmul_4bit", None) is fn:
        return nf4
    return None


def _dequantize_weight(layer, dtype):
    """dequantize(W).to(dtype) as MatMul4Bit.backward builds it (`_functions.py:384`).

    The packed weight goes in as `view(-1, 1)`, the canonical form MatMul4Bit stores (`:321`), so
    dequantize_4bit returns [N, K] without its transposed-input shim (`functional.py:1075`).
    With K1 installed, K1's kernel is used: it returns the same fp16 bits as bnb's dequantize_4bit
    (K1's tests and the Kaggle micro-benchmark check torch.equal) and keeps the dequant work in
    our kernel (check 4). K1 refuses non-nf4 / non-nested / non-fp16 states with TypeError or
    ValueError (`nf4._validate_inputs`); bitsandbytes handles those.

    v7 I2 (S4_DEQUANT_REUSE != "off"): K1 gets the Params4bit itself, as K1's forward does. K1
    reads only its storage (same pointer, numel, contiguity: the same bits), but its validation
    cache (`kernels/nf4.py` _checked_shape) keys on the packed object: a fresh view each call
    missed it and evicted the forward's entry, so every forward and backward dequant of the
    weight re-ran the full validation. bnb's fallback below keeps the view(-1, 1) form.
    """
    weight = layer.weight
    quant_state = weight.quant_state
    packed = weight.view(-1, 1)
    nf4 = _active_nf4()
    if nf4 is not None:
        try:
            dense = nf4.nf4_dequantize(weight if S4_DEQUANT_REUSE != "off" else packed,
                                       quant_state)
        except (TypeError, ValueError):
            dense = None
        if dense is not None:
            return dense if dense.dtype == dtype else dense.to(dtype)
    from bitsandbytes import functional as bnb_functional

    return bnb_functional.dequantize_4bit(packed, quant_state).to(dtype)


class SoupMLPRecompute(torch.autograd.Function):
    """down_proj(silu(gate_proj(x16)) * up_proj(x16)) that saves only x16 (and gate, if asked)."""

    @staticmethod
    def forward(ctx, x16, gate_proj, up_proj, down_proj, recompute_gate):
        ctx.set_materialize_grads(False)
        # Grad mode is off in here, so g, u and h are plain tensors that die with this frame.
        g = gate_proj(x16)
        u = up_proj(x16)
        h = _swiglu_fwd(g, u)
        del u
        if recompute_gate:
            ctx.save_for_backward(x16)
        else:
            ctx.save_for_backward(x16, g)
        del g
        out = down_proj(h)
        del h
        ctx.layers = (gate_proj, up_proj, down_proj)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        if grad_out is None or not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        saved = ctx.saved_tensors
        x16 = saved[0]
        g_saved = saved[1] if len(saved) > 1 else None
        del saved
        gate_proj, up_proj, down_proj = ctx.layers

        # 1. down_proj input gradient first, so its dense weight is freed before the recompute.
        # 2-4. Recompute g/u, SwiGLU backward (in place when g was recomputed), dx16.
        dx = _gate_up_input_grad([_down_input_grad(grad_out, down_proj)], x16, gate_proj, up_proj,
                                 g_saved)
        return dx, None, None, None, None


def _autocast_ok(device_type):
    """True when autocast is off or fp16 for device_type.

    The recompute runs with autocast off. Under fp16 autocast the forward's fp16 matmuls are not
    cast, so the bits agree; under bf16 autocast they would not. torch>=2.5 is pinned by
    transformers (`setup.py:150`), which has the device_type forms of both calls; the no-argument
    forms are the older CUDA-only API.
    """
    try:
        try:
            enabled = torch.is_autocast_enabled(device_type)
        except TypeError:
            enabled = torch.is_autocast_enabled()
        if not enabled:
            return True
        try:
            dtype = torch.get_autocast_dtype(device_type)
        except (AttributeError, TypeError):
            dtype = torch.get_autocast_gpu_dtype()
        return dtype == torch.float16
    except Exception:  # noqa: BLE001 - unknown torch API: stay on K4 (the tests would flag it)
        return False


def _global_hooks_present():
    module = sys.modules.get("torch.nn.modules.module")
    return any(getattr(module, name, None) for name in _GLOBAL_HOOKS)


def _reject_reason(module, x):
    """None when S4 may run, else a short reason (the K4 path then runs)."""
    if not isinstance(x, torch.Tensor) or not torch.is_grad_enabled() or not x.requires_grad:
        return _NOT_TRAINING
    if not _k4._fp16_path_ok(module, x):
        return "K4 guard (device, dtype, width, activation or layer type)"
    if not _autocast_ok(x.device.type):
        return "autocast with a dtype other than fp16"
    if _global_hooks_present():
        return "global module hooks (the recompute would fire them twice)"
    try:
        from bitsandbytes.nn import Params4bit
    except Exception:  # noqa: BLE001
        return "bitsandbytes.nn.Params4bit unavailable"
    for name in _PROJ_NAMES:
        layer = getattr(module, name)
        if getattr(layer, "bias", None) is not None:
            return f"{name} has a bias"
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, Params4bit) or weight.requires_grad:
            return f"{name}.weight is not a frozen Params4bit"
        if weight.device != x.device:
            return f"{name}.weight is on another device"
        state = getattr(weight, "quant_state", None)
        if state is None or getattr(state, "quant_type", None) != "nf4":
            return f"{name} is not nf4-quantized"
        # Forward hooks would fire twice; backward hooks would never fire for the recomputed calls.
        if any(getattr(layer, hooks, None) for hooks in _LAYER_HOOKS):
            return f"{name} has module hooks"
    return None


def _warn_once(reason, x):
    global _warned
    if _warned or reason is _NOT_TRAINING or not x.is_cuda:
        return
    _warned = True
    warnings.warn(f"S4 MLP recompute is on but skipped ({reason}); K4 path used",
                  RuntimeWarning, stacklevel=3)


def patch_qwen2_mlp_recompute():
    """Wrap the K4 forward with the S4 path. Idempotent; no CUDA work at call time.

    Installs K4 first if needed, and marks the S4 forward with K4's mark too, so calling
    patch_qwen2_mlp_fp16() after this is a no-op instead of wrapping S4 inside K4.
    """
    global _swiglu_forward, _swiglu_example_backward, _swiglu_example_kernel
    from transformers.models.qwen2 import modeling_qwen2

    from . import swiglu as example

    _k4.patch_qwen2_mlp_fp16()
    current = modeling_qwen2.Qwen2MLP.forward
    if getattr(current, _PATCH_MARK, False):
        return
    previous = current  # the K4 forward

    _swiglu_forward = example.swiglu
    backward = getattr(getattr(example, "SwiGLU", None), "backward", None)
    _swiglu_example_backward = backward if callable(backward) else None
    kernel = getattr(example, "example_swiglu_bwd_kernel", None)
    block = getattr(example, "BLOCK", None)
    _swiglu_example_kernel = (kernel, block) if kernel is not None and type(block) is int else None

    def forward(self, x):
        if USE_MLP_RECOMPUTE:
            reason = _reject_reason(self, x)
            if reason is None:
                # Same dtype contract as K4 (mlp.patch_qwen2_mlp_fp16's forward).
                x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
                out = SoupMLPRecompute.apply(
                    x16, self.gate_proj, self.up_proj, self.down_proj, bool(RECOMPUTE_GATE)
                )
                return out if out.dtype == x.dtype else out.to(x.dtype)
            if isinstance(x, torch.Tensor):
                _warn_once(reason, x)
        return previous(self, x)

    setattr(forward, _PATCH_MARK, True)
    setattr(forward, _k4._PATCH_MARK, True)
    forward.__wrapped__ = previous
    modeling_qwen2.Qwen2MLP.forward = forward
