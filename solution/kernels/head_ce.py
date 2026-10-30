"""K2: the LM head and the causal-LM cross-entropy without the [B, T, V] logits.

What base does (transformers 5.16.1, trl 0.29.1, accelerate 1.14.0 fp16 autocast):
- `Qwen2ForCausalLM.forward` runs `lm_head` on the fp32 hidden states under autocast
  (modeling_qwen2.py:452-465): logits = fp16(fp16(hidden) @ fp16(W).T), with a fresh fp16 copy of
  the frozen fp32 head that autograd keeps until the backward (445 MiB).
- `ForCausalLMLoss` takes `logits.float()`, shifts the labels inside each row with
  `pad(labels, (0, 1), -100)[..., 1:]` and returns `cross_entropy(sum) / num_items_in_batch`, or the
  mean when num_items_in_batch is None (loss/loss_utils.py:32-71).
- Backward: DivBackward gives a = grad_loss / num_items (grad_loss already carries the GradScaler
  factor); log_softmax backward gives fp32 a * softmax - a * onehot; that is cast to fp16 and goes
  through the head GEMM backward, and the hidden gradient goes back to fp32.

What this module does (one `soup_head_ce_kernel`, three runtime modes):
- Packing. Positions are ordered labelled first, then the other real positions (attention_mask != 0);
  padding that is neither is dropped. One host sync per step (`counts.tolist()`).
- The vocabulary is walked in tiles of VOCAB_TILE rows of the head. Per tile, mode 2 casts the fp32
  weight rows into a reused fp16 buffer (same round-to-nearest-even as autocast's cast), cuBLAS
  writes the tile's fp16 logits into a reused buffer (`torch.mm`, the GEMM `F.linear` runs), and
  mode 0 folds the tile into a per-row fp32 state: running max m, s = sum exp(z - m),
  u = sum exp(z - m) * (z - m) re-centred whenever m moves, the first argmax, the label's logit.
  The last tile leaves loss = (m - z_label) + log s, entropy = log s - u / s and the argmax hit.
  Nothing of size V x H in fp16 or T x V is ever kept: at most a 48 MiB weight tile and a 64 MiB
  logits tile at T = 512.
- When the head weight is already fp16 (solution/head_loss.py stores the frozen tied head in fp16
  after checking on the device that fp16 holds every value exactly), the tiles are plain row slices
  of it: no cast launches and no weight buffer. The fp16 operands are the same bits mode 2 makes.
- Each tile runs two GEMMs, one over the labelled rows and one over the others. The backward
  repeats the labelled one with the same operands and shapes, so cuBLAS returns the same logits the
  forward state was built from. Mode 1 overwrites them in place with
  fp16(exp((z - m) - log s) * a - a * onehot) with the real incoming gradient (the order of
  operations of torch's log_softmax forward epilogue and backward), and `torch.mm(g, W16_tile)`
  adds the tile's share of the hidden gradient into an fp32 accumulator.

Compile: one @triton.jit, one specialization in training. Pointers are fresh allocations or slices
at multiples of VOCAB_TILE rows (16-byte aligned); row0, col0 and mode vary and are in
do_not_specialize; n_cols is left specialized so Triton can prove 16-element alignment for vector
loads (16384, 4480 and both times 1536 are all multiples of 16, so training sees one variant). The
W32 argument of modes 0 and 1 is a placeholder (never read there): it is always an fp32 tensor, so
an fp16 head does not change the signature either.

v8 C1 (USE_UNIFIED_SIG, solution/kernels/__init__.py): `_launch` calls `soup_head_ce_usig`, the
same kernel inlined behind the launcher signature all our kernels share; n_cols stays specialised
there, row0 / col0 / mode stay in do_not_specialize, so the compiled body is the v7 one.

Ideas credited (no code taken): online softmax (Milakov & Gimelshein, 2018); chunked fused linear +
cross-entropy with logits recomputed in backward (Liger-Kernel `fused_linear_cross_entropy`; Apple,
"Cut Your Losses in Large-Vocabulary Language Models"); the logits-free step contract (Soup #1693,
soup_cli/utils/stream_loss_chunk.py:29-41).

Importing this module performs no CUDA work.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

IGNORE_INDEX = -100

# Head rows per tile. 151936 = 9 * 16384 + 4480: 10 tiles per pass. Peak extra memory at T = 512:
# fp16 weight tile 16384 x 1536 = 48 MiB (none with an fp16 head), logits tile <= 2048 x 16384
# fp16 = 64 MiB. v6 keeps v5's 16384 (the GEMM shapes validated on Kaggle): K2 is not the moment
# of the step's memory peak, so a smaller tile would not lower it.
VOCAB_TILE = 16384
# Columns per inner iteration (and elements per program in mode 2): 2048 fp32 values over 8 warps
# is 8 per thread, no spills expected on sm_75 (check `n_spills` of the compiled kernel on Kaggle).
BLOCK_V = 2048
NUM_WARPS = 8

# Per-row fp32 state: running max, sum exp(z - m), sum exp(z - m) * (z - m), first argmax (vocab id,
# exact in fp32 below 2**24), label logit, then the results of the last tile: entropy, loss, hit.
_NCOL = 8
_M, _S, _ENT, _LOSS, _HIT = 0, 1, 5, 6, 7
_STATS, _GRAD, _CAST = 0, 1, 2

# v8 C1: launch soup_head_ce_usig instead of soup_head_ce_kernel. solution.apply() sets it; the
# module default is the v7 launch.
USE_UNIFIED_SIG = False


@triton.jit(do_not_specialize=["row0", "col0", "mode"])
def soup_head_ce_kernel(
    Out16,  # *fp16 [rows, n_cols]: logits tile (modes 0, 1; overwritten in mode 1) | weight tile (2)
    Target,  # *int64 [n]: packed next-token labels, -100 = none
    State,  # *fp32 [n, NCOL]
    Scale,  # *fp32 [1]: grad_loss / num_items, read on the device (mode 1)
    W32,  # *fp32 [n_cols]: the head rows of this tile, flat (mode 2)
    row0,  # packed row of program 0 (modes 0, 1)
    n_cols,  # tile width = logits row stride (modes 0, 1); elements to cast (mode 2)
    col0,  # vocab id of the tile's first column; 0 starts a fresh state (mode 0)
    mode,  # 0: fold the tile into the statistics; 1: logits gradient in place; 2: weight cast
    BLOCK_V: tl.constexpr,
    NCOL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_V)
    if mode == 2:
        # fp32 -> fp16, round to nearest even: the cast autocast applies to the frozen head.
        widx = pid.to(tl.int64) * BLOCK_V + offs
        wmask = widx < n_cols
        w = tl.load(W32 + widx, mask=wmask, other=0.0)
        tl.store(Out16 + widx, w.to(tl.float16), mask=wmask)
    else:
        row = (row0 + pid).to(tl.int64)
        lrow = Out16 + pid.to(tl.int64) * n_cols
        st = State + row * NCOL
        label = tl.load(Target + row)
        local = label - col0  # the label's column inside this tile, when it is in this tile
        if mode == 0:
            fresh = col0 == 0
            # Finite sentinel instead of -inf: the first rescale is exp(-huge) * 0, never inf - inf.
            m = tl.where(fresh, -3.0e38, tl.load(st + 0))
            s = tl.where(fresh, 0.0, tl.load(st + 1))
            u = tl.where(fresh, 0.0, tl.load(st + 2))
            best = tl.where(fresh, 0.0, tl.load(st + 3))
            z_lab = tl.where(fresh, 0.0, tl.load(st + 4))
            for start in range(0, n_cols, BLOCK_V):
                cols = start + offs
                valid = cols < n_cols
                z = tl.load(lrow + cols, mask=valid, other=float("-inf")).to(tl.float32)
                block_max = tl.max(z, axis=0)
                # Smallest index of the block maximum, and a strict ">" across blocks and tiles
                # (visited in increasing vocab order): the first index of the row maximum, as
                # torch.argmax returns it.
                block_arg = tl.min(tl.where(z == block_max, cols, n_cols), axis=0)
                best = tl.where(block_max > m, (col0 + block_arg).to(tl.float32), best)
                m_new = tl.maximum(m, block_max)
                shift = m - m_new  # <= 0, finite
                alpha = tl.exp(shift)
                e = tl.exp(z - m_new)  # 0 on masked lanes (z = -inf)
                # e * (z - m_new) is 0 * -inf on masked lanes: select it away.
                t = tl.where(e > 0.0, e * (z - m_new), 0.0)
                # sum e^(z-m')(z-m') over the old terms = alpha * (u + (m - m') * s), OLD s.
                u = alpha * (u + shift * s) + tl.sum(t, axis=0)
                s = alpha * s + tl.sum(e, axis=0)
                m = m_new
            in_tile = (local >= 0) & (local < n_cols)
            z_new = tl.load(lrow + local, mask=in_tile, other=0.0).to(tl.float32)
            z_lab = tl.where(in_tile, z_new, z_lab)
            log_s = tl.log(s)
            labelled = label != -100
            tl.store(st + 0, m)
            tl.store(st + 1, s)
            tl.store(st + 2, u)
            tl.store(st + 3, best)
            tl.store(st + 4, z_lab)
            # Written after every tile; the last tile's values are the row's results. Stable forms:
            # entropy = log s - u / s (both terms >= 0), loss = (m - z_label) + log s.
            tl.store(st + 5, log_s - u / s)
            tl.store(st + 6, tl.where(labelled, (m - z_lab) + log_s, 0.0))
            tl.store(st + 7, tl.where(labelled & (best == label.to(tl.float32)), 1.0, 0.0))
        else:
            # torch's log_softmax: out = (z - max) - log(sum); its backward with nll's gO:
            # gO - exp(out) * sum(gO) = exp(out) * a - a * onehot, fp32, then the fp16 cast.
            g_m = tl.load(st + 0)
            g_log_s = tl.log(tl.load(st + 1))  # the same op on the same s as mode 0's log_s
            g_scale = tl.load(Scale)
            for g_start in range(0, n_cols, BLOCK_V):
                g_cols = g_start + offs
                g_valid = g_cols < n_cols
                g_z = tl.load(lrow + g_cols, mask=g_valid, other=0.0).to(tl.float32)
                prob = tl.exp((g_z - g_m) - g_log_s)
                grad = prob * g_scale - tl.where(g_cols == local, g_scale, 0.0)
                tl.store(lrow + g_cols, grad.to(tl.float16), mask=g_valid)


# v8 C1 entry: slots 0-6 pointers, 7-10 int32, 11 fp32, no constexpr parameter
# (solution/kernels/__init__.py). n_cols (slot 8) is specialised exactly as in the kernel above.
_BLOCK_V_C = tl.constexpr(BLOCK_V)
_NCOL_C = tl.constexpr(_NCOL)


@triton.jit(do_not_specialize=["spare_p5", "spare_p6", "row0", "col0", "mode", "spare_f"])
def soup_head_ce_usig(
    Out16, Target, State, Scale, W32, spare_p5, spare_p6,
    row0, n_cols, col0, mode,
    spare_f,
):
    soup_head_ce_kernel(Out16, Target, State, Scale, W32, row0, n_cols, col0, mode,
                        _BLOCK_V_C, _NCOL_C)


def _launch(grid, out16, target, state, scale, w32, row0, n_cols, col0, mode):
    if USE_UNIFIED_SIG:  # C1: `state` (fp32, always passed) fills the two spare pointer slots
        soup_head_ce_usig[grid](
            out16, target, state, scale, w32, state, state,
            row0, n_cols, col0, mode,
            0.0,
            num_warps=NUM_WARPS,
        )
        return
    soup_head_ce_kernel[grid](
        out16, target, state, scale, w32, row0, n_cols, col0, mode,
        BLOCK_V=BLOCK_V, NCOL=_NCOL, num_warps=NUM_WARPS,
    )


def _tiles(vocab):
    tile = min(VOCAB_TILE, vocab)
    return tile, [(v0, min(tile, vocab - v0)) for v0 in range(0, vocab, tile)]


def _cast_tile(w32, v0, nv, wbuf, target, state):
    """wbuf[:nv] = fp16(w32[v0:v0 + nv]) by mode 2. `target` and `state` only fill the signature."""
    wb = wbuf[:nv]
    numel = nv * w32.shape[1]
    _launch((triton.cdiv(numel, BLOCK_V),), wb, target, state, state, w32[v0:v0 + nv], 0, numel, 0,
            _CAST)
    return wb


def _weight_tiles(weight, H, device):
    """(fp16 tile getter, weight buffer or None). An fp16 head is sliced, an fp32 one is cast by
    mode 2 into one reused buffer. Any other dtype is refused (head_loss guards allow these two)."""
    w = weight.detach()
    if w.dtype == torch.float16:
        return (lambda v0, nv, target, state: w[v0:v0 + nv]), None
    if w.dtype != torch.float32:
        raise TypeError(f"SoupHeadCE: head weight must be fp32 or fp16, got {w.dtype}")
    tile = min(VOCAB_TILE, w.shape[0])
    wbuf = torch.empty((tile, H), dtype=torch.float16, device=device)
    return (lambda v0, nv, target, state: _cast_tile(w, v0, nv, wbuf, target, state)), wbuf


class SoupHeadCE(torch.autograd.Function):
    """loss, token_entropy, token_loss, correct, labelled = SoupHeadCE.apply(
    hidden [B,T,H] float, weight [V,H] frozen fp32 (or fp16 holding the same values),
    labels [B,T] int64 (unshifted, -100),
    attention_mask [B,T], num_items_in_batch (tensor, int or None))

    loss: fp32 0-dim, sum of labelled token losses / num_items_in_batch, or / the labelled count
    when None (fixed_cross_entropy's "mean", loss_utils.py:39-46). Only the loss is differentiable,
    and only with respect to hidden. The other four have the meaning of soup_cli's ChunkedStep
    fields (stream_loss_chunk.py:29-41): token_entropy [B,T] fp32 at every real or labelled
    position, token_loss [B,T] fp32 (position t holds the CE of labels[t + 1]), correct and
    labelled (int64 0-dim).
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, attention_mask, num_items_in_batch):
        ctx.set_materialize_grads(False)
        with torch.cuda.device(hidden.device), torch.autocast(device_type="cuda", enabled=False):
            return _forward(ctx, hidden, weight, labels, attention_mask, num_items_in_batch)

    @staticmethod
    def backward(ctx, grad_loss, *_unused):
        if grad_loss is None or not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        with torch.cuda.device(grad_loss.device), torch.autocast(device_type="cuda", enabled=False):
            return _backward(ctx, grad_loss), None, None, None, None


def _forward(ctx, hidden, weight, labels, attention_mask, num_items_in_batch):
    B, T, H = hidden.shape
    V = weight.shape[0]
    device = hidden.device

    target = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]  # loss_utils.py:63-64
    lab = target.ne(IGNORE_INDEX)
    real = attention_mask.ne(0)
    # Labelled positions (also outside the mask: base scores them), then the other real ones.
    group = torch.where(lab, 0, torch.where(real, 1, 2)).reshape(-1)
    order = torch.argsort(group, stable=True)
    labelled = lab.sum()
    counts = torch.stack((labelled, (real & ~lab).sum()))
    n_lab, n_other = (int(c) for c in counts.tolist())  # the one host sync of K2
    n = n_lab + n_other
    order = order[:n]

    h16 = hidden.reshape(B * T, H).index_select(0, order).to(torch.float16)  # = autocast's cast
    tgt = target.reshape(-1).index_select(0, order)
    state = torch.empty((n, _NCOL), dtype=torch.float32, device=device)
    if n > 0:
        tile, tiles = _tiles(V)
        weight_tile, wbuf = _weight_tiles(weight, H, device)
        lbuf = torch.empty(max(n_lab, n_other) * tile, dtype=torch.float16, device=device)
        wb = logits = None
        for v0, nv in tiles:
            wb = weight_tile(v0, nv, tgt, state)
            for row0, rows in ((0, n_lab), (n_lab, n_other)):
                if rows == 0:
                    continue
                logits = lbuf[:rows * nv].view(rows, nv)
                torch.mm(h16[row0:row0 + rows], wb.t(), out=logits)
                # W32 is not read in mode 0: `state` (fp32) keeps the signature of mode 2.
                _launch((rows,), logits, tgt, state, state, state, row0, nv, v0, _STATS)
        del weight_tile, wbuf, lbuf, wb, logits

    token_loss = torch.zeros(B * T, dtype=torch.float32, device=device)
    token_loss.index_copy_(0, order[:n_lab], state[:n_lab, _LOSS].contiguous())
    token_entropy = torch.zeros(B * T, dtype=torch.float32, device=device)
    token_entropy.index_copy_(0, order, state[:, _ENT].contiguous())
    correct = state[:n_lab, _HIT].sum().to(torch.int64)

    if num_items_in_batch is None:
        denominator = labelled  # reduction="mean": sum / labelled count
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(device)  # loss_utils.py:43-45
    else:
        denominator = num_items_in_batch
    # Sum, then divide, as fixed_cross_entropy: 0 / 0 = nan when nothing is labelled, as base.
    loss = state[:n_lab, _LOSS].sum() / denominator

    ctx.save_for_backward(h16[:n_lab], weight, tgt[:n_lab], state[:n_lab], order[:n_lab])
    ctx.denominator = denominator
    ctx.hidden_shape = hidden.shape
    ctx.hidden_dtype = hidden.dtype
    token_entropy = token_entropy.view(B, T)
    token_loss = token_loss.view(B, T)
    ctx.mark_non_differentiable(token_entropy, token_loss, correct, labelled)
    return loss, token_entropy, token_loss, correct, labelled


def _backward(ctx, grad_loss):
    h16, weight, tgt, state, order = ctx.saved_tensors
    B, T, H = ctx.hidden_shape
    n_lab = h16.shape[0]
    device = h16.device
    if n_lab == 0:
        # base: every target ignored, the logits gradient is 0
        return torch.zeros((B * T, H), dtype=ctx.hidden_dtype, device=device).view(B, T, H)
    # DivBackward of base's `sum / num_items_in_batch`, fp32.
    scale = (grad_loss.float() / ctx.denominator).to(torch.float32).reshape(1).contiguous()
    tile, tiles = _tiles(weight.shape[0])
    weight_tile, wbuf = _weight_tiles(weight, H, device)
    lbuf = torch.empty(n_lab * tile, dtype=torch.float16, device=device)
    part = torch.empty((n_lab, H), dtype=torch.float16, device=device)
    acc = torch.empty((n_lab, H), dtype=torch.float32, device=device)
    wb = logits = None
    for v0, nv in tiles:
        wb = weight_tile(v0, nv, tgt, state)
        logits = lbuf[:n_lab * nv].view(n_lab, nv)
        torch.mm(h16, wb.t(), out=logits)  # the forward's labelled GEMM, same operands and shape
        # W32 is not read in mode 1: `state` (fp32) keeps the signature of mode 2.
        _launch((n_lab,), logits, tgt, state, scale, state, 0, nv, v0, _GRAD)
        torch.mm(logits, wb, out=part)  # the head GEMM backward of base, restricted to the tile
        if v0 == 0:
            acc.copy_(part)
        else:
            acc.add_(part)
    # The tile buffers die before the [B*T, H] gradient is allocated (this runs at the start of
    # the backward, next to every saved activation; K2 is not the v6 peak moment, so ~0 gain).
    del weight_tile, wbuf, lbuf, part, wb, logits
    grad = torch.zeros((B * T, H), dtype=ctx.hidden_dtype, device=device)
    grad.index_copy_(0, order, acc.to(grad.dtype))
    return grad.view(B, T, H)
