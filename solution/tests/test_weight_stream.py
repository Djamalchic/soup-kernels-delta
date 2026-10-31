"""v12 W (solution/weight_stream.py, the resolve hooks of kernels/nf4.py, the pinned head of
head_loss.py / kernels/head_ce.py): decoder NF4 weights and the fp16 tied head in pinned host
memory, streamed through two GPU slots.

What must hold (asserted):
- install() moves every decoder Params4bit and both absmax levels to pinned host memory (code and
  offset stay on the GPU), frees the GPU originals, and every K1 dequant afterwards (through the
  Params4bit and its own quant_state, as every caller passes them) is a CUDA tensor equal bit for
  bit to the dequant before the move; the device guards (mlp_recompute, nf4.weight_device) still
  accept the weights;
- the two slots follow the layer order: forward 0..n-1 copies every layer once (prefetching the
  next), the reverse walk copies n - 2 (the two top layers are hits), a layer not in its slot is
  fetched on the spot and still reads the right bytes;
- a whole training step (forward + backward through K1's Function on every projection) gives the
  same loss, hidden states and gradients with streaming as without, with less GPU memory held;
- a model with a weight K1 refuses (fp32 quant state) is left alone: install() returns None,
  nothing moves, the registry does not change;
- K2 on a pinned fp16 head gives the device head's bits (forward, backward), the embedding patch
  returns F.embedding's rows, store_head_fp16 with STREAM_HEAD pins the tied weight and both the
  fused and the original forward still compute the same numbers.

CUDA only (Kaggle T4). Run from the repository root:
    python -m pytest solution/tests/test_weight_stream.py -q -rs
"""

import gc

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("triton")
pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.mlp_recompute as rc  # noqa: E402
from solution import head_loss, precompile, weight_stream as ws  # noqa: E402
from solution.kernels import head_ce, nf4  # noqa: E402
from solution.tests import test_head_fp16 as th  # noqa: E402
from solution.tests import test_peak_v6 as tp  # noqa: E402

DEV = "cuda"
HIDDEN, INTER, HEADS, KV_HEADS, LAYERS, VOCAB = 256, 512, 4, 2, 3, 512
NAMES = ws._NAMES


@pytest.fixture(scope="module", autouse=True)
def _setup():
    nf4.patch_bnb_matmul_4bit()  # K1 (idempotent; the hooks resolve streamed weights there)
    assert head_loss.patch_head_loss()


@pytest.fixture
def streamers(monkeypatch):
    """Every streamer a test installs is restored (weights back on the GPU, registry entries
    dropped); K1's I3 flags as apply() sets them; the counters are read relative to a snapshot."""
    for flag in ("USE_K1_FAST_HOST", "USE_DENSE_CACHE", "DENSE_CACHE_QO", "DENSE_CACHE_MIB"):
        monkeypatch.setattr(nf4, flag, getattr(nf4, flag))
    nf4.USE_K1_FAST_HOST, nf4.USE_DENSE_CACHE = True, True
    monkeypatch.setattr(ws, "USE_WEIGHT_STREAMING", True)
    # No install from a forward wrapper an earlier apply() left behind: these tests install.
    monkeypatch.setattr(precompile, "_STREAM_INSTALL", None)
    made = []
    yield made
    for s in made:
        if s.installed:
            s.restore()
    nf4.dense_cache_clear()


def _build(layers=LAYERS, fp16_states=True, seed=0):
    """A tiny Qwen2 whose seven projections per layer are bnb NF4 Linear4bit (fp16 states, as the
    real load leaves them, unless fp16_states=False: fp32 states, which K1 refuses). The RMSNorm
    weights train, so every dX path through every projection is exercised in backward."""
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
                      num_hidden_layers=layers, num_attention_heads=HEADS,
                      num_key_value_heads=KV_HEADS, max_position_embeddings=256,
                      tie_word_embeddings=True, use_cache=False, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(cfg)
    kv = KV_HEADS * (HIDDEN // HEADS)
    for layer in model.model.layers:
        att, mlp = layer.self_attn, layer.mlp
        for mod, name, n_in, n_out, bias in (
                (att, "q_proj", HIDDEN, HIDDEN, True), (att, "k_proj", HIDDEN, kv, True),
                (att, "v_proj", HIDDEN, kv, True), (att, "o_proj", HIDDEN, HIDDEN, False),
                (mlp, "gate_proj", HIDDEN, INTER, False), (mlp, "up_proj", HIDDEN, INTER, False),
                (mlp, "down_proj", INTER, HIDDEN, False)):
            if fp16_states:
                lin = tp.fp16_linear4bit(n_in, n_out, bias=bias)
            else:
                lin = bnb.nn.Linear4bit(n_in, n_out, bias=bias, compute_dtype=torch.float16,
                                        compress_statistics=True, quant_type="nf4")
            setattr(mod, name, lin)
    model = model.cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, mq.Qwen2RMSNorm):
            m.weight.requires_grad_(True)
    model.train()
    return model


def _weights(model):
    """[(layer, index, Params4bit)] in the streamer's order."""
    out = []
    for li, layer in enumerate(model.model.layers):
        for idx, name in enumerate(NAMES):
            out.append((li, idx, ws._base_module(layer, name).weight))
    return out


def _dequants(model):
    return {(li, idx): nf4.nf4_dequantize(w, w.quant_state).clone() for li, idx, w in _weights(model)}


def _batch(seed=1, B=2, T=40):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (B, T), generator=gen)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, 30:] = 0
    labels = ids.clone()
    labels[mask == 0] = -100
    return {"input_ids": ids.to(DEV), "attention_mask": mask.to(DEV), "labels": labels.to(DEV)}


def _step(model, batch, seed=1234):
    """One fp16-autocast forward + backward; loss, hidden states, grads, bytes held after forward."""
    model.zero_grad(set_to_none=True)
    rec = {}
    hook = model.model.register_forward_hook(
        lambda module, args, output: rec.__setitem__("hidden", output.last_hidden_state.detach().clone()))
    try:
        gc.collect()
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch, use_cache=False)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - start
        (out.loss * 1024.0).backward()
    finally:
        hook.remove()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    return {"loss": out.loss.detach().float(), "hidden": rec["hidden"], "grads": grads,
            "held": held}


# ---------------------------------------------------------------------------------------------
# t1: install moves the weights and K1 resolves them
# ---------------------------------------------------------------------------------------------

def test_install_moves_and_resolves(streamers):
    model = _build()
    before = _dequants(model)
    nf4.dense_cache_clear()
    weights = _weights(model)
    nbytes = sum(w.numel() + w.quant_state.absmax.numel() + w.quant_state.state2.absmax.numel() * 4
                 for _, _, w in weights)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    s = ws.install(model)
    assert s is not None and s.installed
    streamers.append(s)
    assert nf4.STREAMER is ws.REGISTRY
    assert s.n_layers == LAYERS and len(s.entries) == LAYERS * len(NAMES)
    # Everything streamed is on the host and pinned; code / offset stay on the GPU; the GPU gave
    # back the originals minus the two slots (allocator rounding: 512 B per tensor at most).
    for li, idx, w in weights:
        qs = w.quant_state
        assert type(w) is bnb.nn.Params4bit and not w.requires_grad
        assert not w.is_cuda and w.is_pinned() and w.dtype == torch.uint8
        assert not qs.absmax.is_cuda and qs.absmax.is_pinned()
        assert not qs.state2.absmax.is_cuda and qs.state2.absmax.is_pinned()
        assert qs.state2.code.is_cuda and qs.offset.is_cuda
        assert s.entries[(li, idx)]["weight"] is w and s.entries[(li, idx)]["real"] is qs
    torch.cuda.synchronize()
    freed = allocated - torch.cuda.memory_allocated()
    assert freed >= nbytes - 2 * s.layout_bytes - 512 * len(weights)
    # Every projection, asked as every caller asks (Params4bit + its own quant_state, or the S4
    # view form), dequantizes to the pre-install bits, on the GPU, through the slots.
    for li, idx, w in weights:
        dense = nf4.nf4_dequantize(w, w.quant_state)
        assert dense.is_cuda and torch.equal(dense, before[(li, idx)])
        dense_view = nf4.nf4_dequantize(w.view(-1, 1), w.quant_state)
        assert torch.equal(dense_view, before[(li, idx)])
        packed, shadow = nf4._streamed(w, w.quant_state)
        assert packed.is_cuda and shadow is s.entries[(li, idx)]["shadow"]
        assert nf4.weight_device(w, w.quant_state) == torch.device(DEV, torch.cuda.current_device())
        assert nf4.k1_dense_path(torch.empty(2048, w.quant_state.shape[1], device=DEV,
                                             dtype=torch.float16), w, w.quant_state)
    # The S4 device guard accepts the streamed MLP (K4's guard needs the exact Linear4bit type).
    mlp = model.model.layers[1].mlp
    x = torch.randn(1, 300, HIDDEN, device=DEV, dtype=torch.float16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        assert rc._reject_reason(mlp, x) is None
    # restore() puts the bits back on the GPU.
    s.restore()
    for li, idx, w in weights:
        assert w.is_cuda and w.quant_state.absmax.is_cuda
        assert torch.equal(nf4.nf4_dequantize(w, w.quant_state), before[(li, idx)])


# ---------------------------------------------------------------------------------------------
# t2: two slots, prefetch one layer ahead, misses fetched on the spot
# ---------------------------------------------------------------------------------------------

def test_two_slot_order_and_prefetch(streamers):
    model = _build()
    before = _dequants(model)
    s = ws.install(model)
    assert s is not None
    streamers.append(s)
    assert s.slot_layer == [None, None] and s.last_layer is None
    copies = ws.COUNTS["copies"]

    def read(layer):
        packed, shadow = s.get(layer, 0)
        assert torch.equal(nf4.nf4_dequantize(packed, shadow), before[(layer, 0)])

    read(0)  # miss: fetched on the spot; no direction yet
    assert s.slot_layer == [0, None] and ws.COUNTS["copies"] - copies == 1
    read(1)  # miss, then the direction says forward: 2 prefetched into slot 0
    assert s.slot_layer == [2, 1] and ws.COUNTS["copies"] - copies == 3
    read(2)  # hit; 3 is out of range
    assert s.slot_layer == [2, 1] and ws.COUNTS["copies"] - copies == 3
    read(2)  # the backward starts at the top layer: a hit, no direction change
    read(1)  # hit; backward direction: 0 prefetched into slot 0 (over 2, whose reads are issued)
    assert s.slot_layer == [0, 1] and ws.COUNTS["copies"] - copies == 4
    read(0)  # hit
    assert ws.COUNTS["copies"] - copies == 4 == LAYERS + (LAYERS - 2)
    # Every projection of every layer reads right in any order (misses included).
    misses = ws.COUNTS["misses"]
    for layer in (2, 0, 2, 1):
        for idx in range(len(NAMES)):
            packed, shadow = s.get(layer, idx)
            assert torch.equal(nf4.nf4_dequantize(packed, shadow), before[(layer, idx)])
    assert ws.COUNTS["misses"] > misses
    # A caller's transposed shape (Linear4bit passes weight.t()) gets one stable view per shape.
    w = _weights(model)[0][2]
    a, _ = nf4._streamed(w.t(), w.quant_state)
    b, _ = nf4._streamed(w.t(), w.quant_state)
    assert a is b and tuple(a.shape) == tuple(w.t().shape) and a.is_cuda and a.is_contiguous()


# ---------------------------------------------------------------------------------------------
# t3: a whole step, same numbers, less memory
# ---------------------------------------------------------------------------------------------

def test_full_step_same_values_and_less_memory(streamers):
    model = _build()
    batch = _batch()
    ref_a = _step(model, batch)
    ref_b = _step(model, batch)  # is the reference itself reproducible (SDPA backward)?
    s = ws.install(model)
    assert s is not None
    streamers.append(s)
    new = _step(model, batch)
    assert torch.equal(new["loss"], ref_a["loss"])
    assert torch.equal(new["hidden"], ref_a["hidden"])
    for name, g in new["grads"].items():
        if torch.equal(ref_a["grads"][name], ref_b["grads"][name]):
            assert torch.equal(g, ref_a["grads"][name]), name
        else:  # the reference's own spread (torch's SDPA backward): stay inside it
            spread = (ref_a["grads"][name] - ref_b["grads"][name]).abs().max().item()
            assert (g - ref_a["grads"][name]).abs().max().item() <= 4 * spread + 1e-6, name
    # A second step from the slots left by the first backward (layers 0 and 1 in place).
    again = _step(model, batch)
    assert torch.equal(again["loss"], ref_a["loss"])
    # The forward holds the same activations (the slots were allocated at install, before it).
    assert new["held"] <= ref_a["held"] + 2 * s.layout_bytes


def test_bnb_fallback_keeps_its_own_bytes(streamers):
    """m <= 4 sends soup_matmul_4bit to bnb's MatMul4Bit, which saves the weight it was given and
    dequantizes it in backward without asking the streamer: a streamed weight reaches it as a
    private copy, so a slot that changed layer in between does not change dX."""
    model = _build()
    before = _dequants(model)
    s = ws.install(model)
    assert s is not None
    streamers.append(s)
    w = ws._base_module(model.model.layers[0], NAMES[3]).weight  # o_proj: no bias
    dense = before[(0, 3)].float()  # [n, k]
    n, k = w.quant_state.shape
    a = torch.randn(2, k, device=DEV, dtype=torch.float16, requires_grad=True)
    assert not nf4.k1_dense_path(a, w, w.quant_state)  # m = 2: bnb's path
    out = nf4.soup_matmul_4bit(a, w.t(), w.quant_state)
    assert out.shape == (2, n) and out.is_cuda
    torch.testing.assert_close(out.float(), a.detach().float() @ dense.t(), rtol=1e-2, atol=1e-2)
    s.get(2, 3)  # layer 2 takes slot 0 over layer 0 (every read of layer 0 is issued)
    torch.cuda.synchronize()
    assert s.slot_layer[0] == 2
    out.sum().backward()
    torch.testing.assert_close(a.grad.float(), torch.ones(2, n, device=DEV) @ dense,
                               rtol=1e-2, atol=1e-2)
    assert not w.is_cuda  # the Params4bit itself stayed on the host


# ---------------------------------------------------------------------------------------------
# t4: refusal leaves the model alone
# ---------------------------------------------------------------------------------------------

def test_refuses_a_model_k1_refuses(streamers):
    model = _build(fp16_states=False)  # fp32 quant states: K1 refuses them, so does install
    registry_before = dict(ws.REGISTRY.by_qs)
    streamer_before = nf4.STREAMER
    assert ws.install(model) is None
    for _, _, w in _weights(model):
        assert w.is_cuda and w.quant_state.absmax.is_cuda
    assert ws.REGISTRY.by_qs == registry_before and nf4.STREAMER is streamer_before
    # maybe_install: one attempt per model, counted as a refusal, never raises.
    refused = ws.COUNTS["refused"]
    assert ws.maybe_install(model) is False and ws.COUNTS["refused"] == refused + 1
    assert ws.maybe_install(model) is False and ws.COUNTS["refused"] == refused + 1
    # One streamed-looking layer among fp16 ones is not enough: all or nothing.
    model = _build()
    bad = model.model.layers[1].mlp.up_proj
    bad.weight.quant_state.dtype = torch.float32
    assert ws.install(model) is None
    for _, _, w in _weights(model):
        assert w.is_cuda


def test_install_hook_runs_once_from_the_forward(streamers, monkeypatch):
    """precompile's forward wrapper calls maybe_install before the original forward; the second
    forward does not install again; a refused model is not retried."""
    monkeypatch.setattr(precompile, "_STREAM_INSTALL", ws.maybe_install)
    monkeypatch.setattr(mq.Qwen2ForCausalLM, "forward", mq.Qwen2ForCausalLM.forward)
    assert precompile.patch_forward_join()
    model = _build()
    batch = _batch()
    installed = ws.COUNTS["installed"]
    with torch.autocast("cuda", dtype=torch.float16):
        model(**batch, use_cache=False)
        assert ws.COUNTS["installed"] == installed + 1
        model(**batch, use_cache=False)
    assert ws.COUNTS["installed"] == installed + 1
    for _, _, w in _weights(model):
        assert not w.is_cuda
    hit = ws.REGISTRY.by_qs[id(_weights(model)[0][2].quant_state)]
    streamers.append(hit[0])


# ---------------------------------------------------------------------------------------------
# t5: the pinned head (K2 tiles from the host, the embedding lookup, store_head_fp16)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tile", [1024, 4096])
def test_k2_pinned_head_gives_the_device_head_bits(monkeypatch, tile):
    monkeypatch.setattr(head_ce, "VOCAB_TILE", tile)
    (_, w16), hidden, labels, mask = th._case(7)
    pinned = torch.empty(w16.shape, dtype=torch.float16, pin_memory=True)
    pinned.copy_(w16.detach())
    torch.cuda.synchronize()
    w_host = torch.nn.Parameter(pinned, requires_grad=False)
    num_items = labels[..., 1:].ne(-100).sum()
    copies = head_ce.HOST_TILE_COUNTS["copies"]
    outs_dev, grad_dev = th._run(w16, hidden, labels, mask, num_items)
    outs_host, grad_host = th._run(w_host, hidden, labels, mask, num_items)
    for a, b in zip(outs_host, outs_dev):
        assert a.dtype == b.dtype and a.is_cuda and torch.equal(a, b)
    assert torch.equal(grad_host, grad_dev)
    n_tiles = -(-w16.shape[0] // tile)
    assert head_ce.HOST_TILE_COUNTS["copies"] - copies == 2 * n_tiles  # forward + backward
    # A pageable host weight takes the synchronous path, same bits.
    w_page = torch.nn.Parameter(w16.detach().cpu(), requires_grad=False)
    sync = head_ce.HOST_TILE_COUNTS["sync"]
    outs_page, grad_page = th._run(w_page, hidden, labels, mask, num_items)
    assert head_ce.HOST_TILE_COUNTS["sync"] > sync
    assert torch.equal(outs_page[0], outs_dev[0]) and torch.equal(grad_page, grad_dev)


def test_pinned_embedding_lookup_matches_f_embedding():
    assert head_loss.patch_pinned_embedding() and head_loss.patch_pinned_embedding()
    torch.manual_seed(3)
    w16 = (torch.randn(300, 64) * 0.3).half()
    emb = torch.nn.Embedding(300, 64)
    pinned = torch.empty(w16.shape, dtype=torch.float16, pin_memory=True)
    pinned.copy_(w16)
    emb.weight.data = pinned
    emb.weight.requires_grad_(False)
    setattr(emb.weight, head_loss._PINNED_MARK, True)
    assert head_loss.is_pinned_head(emb.weight)
    for shape, dtype in (((2, 40), torch.int64), ((5,), torch.int32), ((3, 0), torch.int64)):
        ids = torch.randint(0, 300, shape, dtype=dtype, device=DEV)
        out = emb(ids)
        assert out.is_cuda and out.dtype == torch.float16 and out.shape == (*shape, 64)
        assert torch.equal(out, F.embedding(ids.long(), w16.to(DEV)))
    # Rows beyond the buffer's first allocation grow it; the lookup stays exact.
    ids = torch.randint(0, 300, (3, 1000), device=DEV)
    assert torch.equal(emb(ids), F.embedding(ids, w16.to(DEV)))
    # A device weight (unmarked, or marked but on the GPU) runs the original forward.
    plain = torch.nn.Embedding(300, 64).to(DEV)
    assert torch.equal(plain(ids), F.embedding(ids, plain.weight))


def test_store_head_fp16_pins_and_keeps_every_number(monkeypatch):
    monkeypatch.setattr(head_loss, "STREAM_HEAD", True)
    model = th._tiny_qwen(tied=True)
    batch = th._batch(model.config.vocab_size)
    before_orig = th._forward_all(model, batch, fused=False)
    before_fused = th._forward_all(model, batch, fused=True)
    pinned_count = head_loss.STREAM_HEAD_COUNTS["pinned"]
    assert head_loss.store_head_fp16(model)
    w = model.lm_head.weight
    assert w is model.model.embed_tokens.weight and head_loss.is_pinned_head(w) and w.is_pinned()
    assert head_loss._is_fp16_stored(w) and not w.requires_grad
    assert head_loss.STREAM_HEAD_COUNTS["pinned"] == pinned_count + 1
    lookups = head_loss.STREAM_HEAD_COUNTS["lookups"]
    after_fused = th._forward_all(model, batch, fused=True)  # K2 from the host, the embedding too
    assert head_loss.STREAM_HEAD_COUNTS["lookups"] == lookups + 2  # _forward_all embeds twice
    fallbacks = head_loss.STREAM_HEAD_COUNTS["gpu_fallback"]
    after_orig = th._forward_all(model, batch, fused=False)  # the copy-back safety net
    assert head_loss.STREAM_HEAD_COUNTS["gpu_fallback"] == fallbacks + 1
    assert head_loss.is_pinned_head(model.lm_head.weight)  # back on the host afterwards
    for before, after in ((before_orig, after_orig), (before_fused, after_fused)):
        loss_b, logits_b, emb_b, grad_b = before
        loss_a, logits_a, emb_a, grad_a = after
        assert emb_a.dtype == torch.float32 and torch.equal(emb_a, emb_b)
        assert torch.equal(loss_a, loss_b) and torch.equal(grad_a, grad_b)
        if logits_b is not None:
            assert torch.equal(logits_a, logits_b)


def test_store_head_fp16_without_the_flag_stays_on_the_gpu(monkeypatch):
    monkeypatch.setattr(head_loss, "STREAM_HEAD", False)
    model = th._tiny_qwen(tied=True)
    assert head_loss.store_head_fp16(model)
    w = model.lm_head.weight
    assert w.is_cuda and head_loss._is_fp16_stored(w) and not head_loss.is_pinned_head(w)
