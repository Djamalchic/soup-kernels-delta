"""A1 (solution/sdpa_recompute.py): the recompute SDPA gives base's output bit for bit, base's
gradients, keeps far less for backward, draws no random numbers, and leaves every call it does not
recognise on transformers' own sdpa_attention_forward.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from transformers.integrations import sdpa_attention as sa  # noqa: E402
from transformers.modeling_utils import AttentionInterface  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.sdpa_recompute as a1  # noqa: E402

DEV = "cuda"
HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128  # Qwen2.5-1.5B
LENGTHS = (96, 40, 1, 71)
SEQ = 96


@pytest.fixture
def patched(monkeypatch):
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward
    assert a1.patch_sdpa_recompute()
    assert a1.patch_sdpa_recompute()  # idempotent
    monkeypatch.setattr(a1, "USE_SDPA_RECOMPUTE", True)
    yield
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward


def module(n_rep=HEADS // KV_HEADS):
    return types.SimpleNamespace(num_key_value_groups=n_rep, is_causal=True, training=True)


def bool_mask(lengths=LENGTHS, seq=SEQ):
    real = torch.zeros(len(lengths), seq, dtype=torch.bool, device=DEV)
    for b, n in enumerate(lengths):
        real[b, :n] = True
    causal = torch.ones(seq, seq, dtype=torch.bool, device=DEV).tril()
    # The shape create_causal_mask hands to SDPA (masking_utils.py sdpa_mask: [B, 1, T, T] bool).
    return causal[None, None] & real[:, None, None, :]


def leaves(seed=0, dtype=torch.float32, batch=len(LENGTHS), seq=SEQ):
    """q, k, v as Qwen2Attention builds them: [B, T, H, D] buffers seen as [B, H, T, D] views."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(batch, seq, HEADS, HEAD_DIM, device=DEV, generator=gen, dtype=dtype)
    k = torch.randn(batch, seq, KV_HEADS, HEAD_DIM, device=DEV, generator=gen, dtype=dtype)
    v = torch.randn(batch, seq, KV_HEADS, HEAD_DIM, device=DEV, generator=gen, dtype=dtype)
    return [t.requires_grad_(True) for t in (q, k, v)]


def run(fn, qkv, mask, dout, autocast=True, **kw):
    q, k, v = (t.transpose(1, 2) for t in qkv)
    for t in qkv:
        t.grad = None
    saved = {}

    def pack(t):
        st = t.untyped_storage()
        saved[st.data_ptr()] = st.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            out, weights = fn(module(), q, k, v, mask, dropout=0.0, scaling=HEAD_DIM ** -0.5,
                              sliding_window=None, **kw)
    assert weights is None
    rng = torch.cuda.get_rng_state()
    out.backward(dout)
    assert torch.equal(torch.cuda.get_rng_state(), rng)  # backward draws nothing
    return out.detach(), [t.grad.clone() for t in qkv], sum(saved.values())


def test_matches_base_and_saves_less(patched):
    mask = bool_mask()
    qkv = leaves()
    dout = 0.1 * torch.randn(len(LENGTHS), SEQ, HEADS, HEAD_DIM, device=DEV, dtype=torch.float16)
    out0, g0, saved0 = run(sa.sdpa_attention_forward, qkv, mask, dout)
    before = dict(a1.COUNTS)
    out1, g1, saved1 = run(a1.sdpa_forward, qkv, mask, dout)
    assert a1.COUNTS["recompute"] == before["recompute"] + 1

    assert out1.dtype == out0.dtype == torch.float16 and out1.shape == out0.shape
    assert out1.is_contiguous()
    assert torch.equal(out1, out0)  # same inputs, same kernel
    for a, b in zip(g1, g0):
        assert a.dtype == b.dtype == torch.float32
        # Same kernel on the same inputs; only SDPA's own backward reduction order may differ.
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-4)
    # base keeps fp16 q + 12-head k, v + out + a float bias; we keep q + 2-head k, v (+ bool mask)
    per_token = HEADS * HEAD_DIM * 2
    rows = len(LENGTHS) * SEQ
    assert saved1 <= 0.45 * saved0, (saved1, saved0)
    assert saved0 - saved1 >= 2.0 * per_token * rows, (saved0 - saved1, per_token * rows)


def test_whole_qwen2_attention_matches_base(patched):
    """The real Qwen2Attention (looks the function up per call, modeling_qwen2.py:216-229)."""
    cfg = mq.Qwen2Config(hidden_size=HEADS * HEAD_DIM, num_attention_heads=HEADS,
                         num_key_value_heads=KV_HEADS, intermediate_size=256, num_hidden_layers=1,
                         attn_implementation="sdpa")
    torch.manual_seed(0)
    attn = mq.Qwen2Attention(cfg, layer_idx=0).cuda().train()
    rotary = mq.Qwen2RotaryEmbedding(config=cfg).cuda()
    x0 = torch.randn(len(LENGTHS), SEQ, cfg.hidden_size, device=DEV)
    pos = torch.arange(SEQ, device=DEV)[None]
    emb = rotary(x0, pos)
    mask = bool_mask()
    seed = 0.05 * torch.randn_like(x0)

    def go(enabled):
        a1.USE_SDPA_RECOMPUTE = enabled
        x = x0.clone().requires_grad_(True)
        attn.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            y, _ = attn(x, position_embeddings=emb, attention_mask=mask)
        y.float().backward(seed)
        return y.detach(), x.grad, attn.q_proj.weight.grad.clone()

    y0, dx0, dw0 = go(False)
    n = a1.COUNTS["recompute"]
    y1, dx1, dw1 = go(True)
    assert a1.COUNTS["recompute"] == n + 1
    assert torch.equal(y1, y0)
    torch.testing.assert_close(dx1, dx0, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(dw1, dw0, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("case", ["float_mask", "no_mask", "no_grad", "fp32_no_autocast",
                                  "dropout", "position_bias", "flag_off", "bf16_autocast"])
def test_fallbacks_call_the_original(patched, monkeypatch, case):
    calls = []
    genuine = a1._REFS["original"]

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return genuine(*args, **kwargs)

    monkeypatch.setitem(a1._REFS, "original", spy)
    mask = bool_mask()
    qkv = leaves()
    kw = dict(dropout=0.0, scaling=HEAD_DIM ** -0.5)
    autocast = dict(device_type="cuda", dtype=torch.float16, enabled=True)
    grad = True
    if case == "float_mask":
        mask = torch.zeros(mask.shape, device=DEV).masked_fill(~mask, float("-inf"))
    elif case == "no_mask":
        mask = None
    elif case == "no_grad":
        grad = False
    elif case == "fp32_no_autocast":
        autocast["enabled"] = False
    elif case == "dropout":
        kw["dropout"] = 0.1
    elif case == "position_bias":
        kw["position_bias"] = torch.zeros(1, HEADS, SEQ, SEQ, device=DEV)
    elif case == "flag_off":
        a1.USE_SDPA_RECOMPUTE = False
    elif case == "bf16_autocast":
        # On a T4 torch may switch a bf16 autocast off with a warning; either way not ours.
        autocast["dtype"] = torch.bfloat16
    q, k, v = (t.transpose(1, 2) for t in qkv)
    before = a1.COUNTS["recompute"]
    with torch.set_grad_enabled(grad), torch.autocast(**autocast):
        try:
            a1.sdpa_forward(module(), q, k, v, mask, **kw)
        except RuntimeError:
            pass  # bf16 on a T4 may refuse to run: only the routing is checked here
    assert len(calls) == 1 and a1.COUNTS["recompute"] == before
    for key, value in kw.items():
        assert calls[0][key] is value  # forwarded untouched


def test_someone_elses_sdpa_is_left_alone(monkeypatch):
    a1.unpatch_sdpa_recompute()

    def other(*args, **kwargs):
        return None

    monkeypatch.setitem(AttentionInterface._global_mapping, "sdpa", other)
    assert a1.patch_sdpa_recompute() is False
    assert AttentionInterface._global_mapping["sdpa"] is other


def test_full_model_step_matches_base(patched):
    """Tiny Qwen2 LM, padded batch, fp16 autocast: same loss bits, LoRA-free grads close."""
    cfg = mq.Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=512, use_cache=False, attn_implementation="sdpa")
    torch.manual_seed(0)
    model = mq.Qwen2ForCausalLM(cfg).cuda().train()
    ids = torch.randint(1, 256, (4, 64), device=DEV)
    am = torch.zeros(4, 64, dtype=torch.long, device=DEV)
    for b, n in enumerate((64, 30, 5, 47)):
        am[b, :n] = 1
    labels = ids.masked_fill(am == 0, -100)

    def go(enabled):
        a1.USE_SDPA_RECOMPUTE = enabled
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=am, labels=labels).loss
        (loss * 1024).backward()
        return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
                               if p.grad is not None}

    loss0, g0 = go(False)
    n = a1.COUNTS["recompute"]
    loss1, g1 = go(True)
    assert a1.COUNTS["recompute"] == n + cfg.num_hidden_layers
    assert torch.equal(loss1, loss0)
    assert g0.keys() == g1.keys()
    for name in g0:
        rel = (g1[name] - g0[name]).float().norm() / g0[name].float().norm().clamp_min(1e-30)
        assert rel < 5e-3, (name, rel.item())
