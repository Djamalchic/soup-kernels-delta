"""v8 C2: SwiGLU, silu(gate) * up, forward and backward in one Triton kernel of ours.

With USE_SOUP_SWIGLU on (solution/__init__.py -> solution/mlp.py), K4 (solution/mlp.py) and S4
(solution/mlp_recompute.py) call this module instead of the starter kit's example
(solution/swiglu.py). The example stays in the repository as the reference and the flag-off path;
with the flag on it is never launched (solution/tests/test_soup_swiglu.py counts launches by name).

One @triton.jit, `soup_swiglu_kernel`, with a runtime `mode` in do_not_specialize: 0 is the
forward, 1 the backward. Training compiles one variant of one kernel where v7 compiled the
example's two kernels, and with C1 (solution/kernels/__init__.py) it shares the one gcc launcher
of all our kernels: slots 0-6 pointers, 7-10 int32, 11 fp32, no constexpr parameter.

Values. Per element, in fp32 from the loaded fp16 (or fp32) inputs, the same operations in the
same order as the example kernels (solution/swiglu.py:19-21 and :30-35):
    forward   s = sigmoid(g);  h = (g * s) * u
    backward  du = (dh * g) * s;  dg = ((dh * u) * s) * (1 + g * (1 - s))
then one cast to the output pointer's element type with Triton's default rounding (nearest even),
the cast the example writes as `.to(ptr.dtype.element_ty)`. tl.sigmoid is the same library
function and the compile options are the defaults, as for the example, so `1 + g * (1 - s)` gets
the same FMA treatment. The work is element-wise (no reduction), so the tile layout cannot change a
value. solution/tests/test_soup_swiglu.py checks torch.equal against the example kernels.

Layout. One program covers _TILE = 1024 consecutive elements (the example's BLOCK) at 4 warps, with
int64 element offsets. `n` stays specialised, as the example's n_elements: in training it is
tokens * 8960, a multiple of 16, so the bound mask is constant over each 16-byte fp16 vector and
the loads and stores stay vectorised; the value 1 (a one-element call) would be a separate variant,
as in the example.

In place (S4, swiglu_backward_into): mode 1 may write dg over dh and du over g. Each element is
loaded and stored by the same program, each store depends on the load of the buffer it overwrites
(dg needs dh, du needs g), and up is never written: the argument of
mlp_recompute._swiglu_backward_inplace for the example kernel holds here as well.

Idea credit: the operation and its derivative are those of the starter kit's example, which this
replaces in our hot path; fused SwiGLU kernels also exist in Liger-Kernel and Unsloth (idea only).
Code written by team Delta. Importing this module does no CUDA work.
"""

import torch
import triton
import triton.language as tl

# v10d: precompile.wait() joins a pending background compile (a module-global None check after it)
from .. import precompile as _precompile

_TILE = 1024  # elements per program, as the example's BLOCK
_NUM_WARPS = 4  # Triton's default, as the example's launch
_FORWARD, _BACKWARD = 0, 1
_INT32_END = 2**31  # n must stay an int32 argument (a larger value is another signature)
_CHUNK = 2**30  # elements per launch beyond that; a multiple of _TILE keeps chunk starts aligned

_TILE_C = tl.constexpr(_TILE)


@triton.jit(do_not_specialize=["spare_p5", "spare_p6", "mode", "spare_i2", "spare_i3", "spare_f"])
def soup_swiglu_kernel(
    Gate, Up, Hid, DGate, DUp, spare_p5, spare_p6,
    n, mode, spare_i2, spare_i3,
    spare_f,
):
    """mode 0: Hid = silu(Gate) * Up.  mode 1: Hid holds dL/dHid; DGate and DUp get the input
    gradients (DGate may be Hid and DUp may be Gate)."""
    first = tl.program_id(0).to(tl.int64) * _TILE_C
    at = tl.arange(0, _TILE_C) + first
    inside = at < n
    g16 = tl.load(Gate + at, mask=inside)
    u16 = tl.load(Up + at, mask=inside)
    # The incoming gradient (mode 1 only; no memory access in mode 0). All loads come before any
    # store, which the in-place backward relies on.
    dh16 = tl.load(Hid + at, mask=inside & (mode != 0))
    g = g16.to(tl.float32)
    u = u16.to(tl.float32)
    s = tl.sigmoid(g)
    if mode == 0:
        silu = g * s
        tl.store(Hid + at, (silu * u).to(Hid.dtype.element_ty), mask=inside)
    else:
        dh = dh16.to(tl.float32)
        rim = 1.0 - s
        lift = 1.0 + g * rim  # d silu / d g = s * (1 + g * (1 - s))
        dg = dh * u * s * lift
        du = dh * g * s
        tl.store(DGate + at, dg.to(DGate.dtype.element_ty), mask=inside)
        tl.store(DUp + at, du.to(DUp.dtype.element_ty), mask=inside)


def _launch(mode, gate, up, hid, dgate, dup):
    """One launch over gate.numel() elements. Every tensor is contiguous with gate's numel; the
    spare pointer slots get gate (never read, not specialised)."""
    _precompile.wait()  # v10d: a pending background compile finishes before our first launch
    n = gate.numel()
    if n == 0:
        return
    if n < _INT32_END:
        soup_swiglu_kernel[(triton.cdiv(n, _TILE),)](
            gate, up, hid, dgate, dup, gate, gate,
            n, mode, 0, 0,
            0.0,
            num_warps=_NUM_WARPS,
        )
        return
    flat = [t.view(-1) for t in (gate, up, hid, dgate, dup)]
    for start in range(0, n, _CHUNK):
        _launch(mode, *(t[start:start + _CHUNK] for t in flat))


class SoupSwiGLU(torch.autograd.Function):
    """silu(gate) * up. The same input handling as the example's SwiGLU (solution/swiglu.py:44-61):
    gate made contiguous, up cast to gate's dtype, the incoming gradient cast to gate's dtype."""

    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        hid = torch.empty_like(gate)
        # The two gradient slots get hid: aligned like the backward's fresh buffers, so the forward
        # and the backward use one compiled variant.
        _launch(_FORWARD, gate, up, hid, hid, hid)
        ctx.save_for_backward(gate, up)
        return hid

    @staticmethod
    def backward(ctx, dhid):
        gate, up = ctx.saved_tensors
        return swiglu_backward(dhid, gate, up)


def swiglu(gate, up):
    return SoupSwiGLU.apply(gate, up)


def swiglu_backward(dhid, gate, up):
    """(dgate, dup) in fresh buffers. gate and up as SoupSwiGLU.forward keeps them (contiguous,
    one dtype)."""
    dhid = dhid.to(gate.dtype).contiguous()
    dgate, dup = torch.empty_like(gate), torch.empty_like(up)
    _launch(_BACKWARD, gate, up, dhid, dgate, dup)
    return dgate, dup


def swiglu_backward_into(dhid, gate, up):
    """(dgate, dup) written over dhid and gate; returns (dhid, gate). The caller checks one dtype,
    one shape, one device, contiguity and three distinct buffers (mlp_recompute._inplace_ok)."""
    _launch(_BACKWARD, gate, up, dhid, dhid, gate)
    return dhid, gate
