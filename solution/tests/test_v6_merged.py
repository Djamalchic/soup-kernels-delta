"""v6 merge, whole model: v5 (K1 + K3 + K4 + S4 + U) against v6 = v5 + B + A1 + A2 + D (A off),
as solution.apply() switches them, on a real-width Qwen2 stack whose q/k/v/o and MLP are
bitsandbytes NF4 and whose q/v carry peft's 4-bit LoRA with dropout 0.05 (the classes the real run
uses, so A2 is exercised, unlike in test_unpad's build where q/v are plain nn.Linear).

Checks: the same loss, hidden states and RNG stream (bit for bit; v5's forward is deterministic),
LoRA gradients bit for bit when v5 is itself reproducible and otherwise within v5's own spread,
every v6 piece fired the expected number of times, D packed only the two LoRA-A inputs per layer
(A1's 4-D saves and A2's absent masks are not touched), and autograd holds at least 80 % of the
expected saving after the forward.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import gc

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
pytest.importorskip("triton")

from peft.tuners.lora import bnb as lora_bnb  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from transformers.integrations import sdpa_attention as sa  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.fused_block as fb  # noqa: E402
import solution.lora_replay as a2  # noqa: E402
import solution.saved_compact as sc  # noqa: E402
import solution.sdpa_recompute as a1  # noqa: E402
import solution.unpad as unpad  # noqa: E402
from solution.tests import test_peak_v6 as tp  # noqa: E402
from solution.tests import test_unpad as tu  # noqa: E402  (also hides torchao from peft)

forwards = tu.forwards  # restores every patched class forward after each test

H = tp.H
HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128
INTER = 8960
LAYERS = 3


def build(layers=LAYERS, vocab=1024):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=H, intermediate_size=INTER,
                      num_hidden_layers=layers, num_attention_heads=HEADS,
                      num_key_value_heads=KV_HEADS, max_position_embeddings=1024,
                      tie_word_embeddings=False, use_cache=False, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(cfg)
    kv = KV_HEADS * HEAD_DIM
    for layer in model.model.layers:
        att, mlp = layer.self_attn, layer.mlp
        for mod, name, n_in, n_out, bias in (
                (att, "q_proj", H, H, True), (att, "k_proj", H, kv, True),
                (att, "v_proj", H, kv, True), (att, "o_proj", H, H, False),
                (mlp, "gate_proj", H, INTER, False), (mlp, "up_proj", H, INTER, False),
                (mlp, "down_proj", INTER, H, False)):
            setattr(mod, name, bnb.nn.Linear4bit(n_in, n_out, bias=bias, compute_dtype=torch.float16,
                                                 compress_statistics=True, quant_type="nf4"))
    model = model.cuda()
    # What from_pretrained(quantization_config=...) leaves on the model; peft's LoRA then wraps q/v
    # with its bnb Linear4bit (peft/tuners/lora/model.py:238, peft/tuners/lora/bnb.py:579-589) and
    # adds no input-require-grads hook (peft/peft_model.py:752-756), as in the real run.
    model.is_loaded_in_4bit = True
    lcfg = peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules=["q_proj", "v_proj"], init_lora_weights=False,
                           task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, lcfg)
    pm.train()
    for layer in tu.inner_model(pm).layers:
        assert type(layer.self_attn.q_proj) is lora_bnb.Linear4bit
        assert type(layer.self_attn.v_proj) is lora_bnb.Linear4bit
    return pm


@pytest.fixture
def v6_patches(forwards, monkeypatch):
    """K1 + K3 + K4 + S4 (as test_peak_v6) under U, plus A1 and A2 installed; flags set per run."""
    base, patched = forwards
    tp._install_k1_k3_s4(monkeypatch)
    # A1 in transformers' global registry, A nowhere (it would shadow A1, see solution/__init__.py).
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward
    assert "sdpa" not in ALL_ATTENTION_FUNCTIONS._local_mapping
    assert a1.patch_sdpa_recompute()
    cls = lora_bnb.Linear4bit
    monkeypatch.setattr(cls, "forward", getattr(cls.forward, "__wrapped__", cls.forward))
    assert a2.patch_lora_dropout_replay()
    monkeypatch.setattr(sc, "USE_SDPA_KV_COMPACT", False)
    for mod, flag in ((sc, "USE_LORA_ROW_COMPACT"), (fb, "USE_FUSED_BLOCK"),
                      (a1, "USE_SDPA_RECOMPUTE"), (a2, "USE_LORA_DROPOUT_REPLAY")):
        monkeypatch.setattr(mod, flag, getattr(mod, flag))
    yield patched
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward


def _set(on):
    sc.USE_LORA_ROW_COMPACT = on  # D
    fb.USE_FUSED_BLOCK = on  # B
    a1.USE_SDPA_RECOMPUTE = on  # A1
    a2.USE_LORA_DROPOUT_REPLAY = on  # A2
    sc.USE_SDPA_KV_COMPACT = False  # A: never together with A1


def _run(pm, batch, forward, extra, seed=1234):
    """One fp16-autocast training step without hooks on the LoRA dropout (they would send A2 to
    peft's forward). Returns loss, hidden, RNG after forward / backward, LoRA grads, bytes held."""
    mq.Qwen2Model.forward = forward
    pm.zero_grad(set_to_none=True)
    rec = {}

    def on_model(module, args, output):  # returns None: the output is not replaced
        rec["hidden"] = output.last_hidden_state.detach().clone()

    hook = tu.inner_model(pm).register_forward_hook(on_model)
    try:
        gc.collect()  # no earlier garbage freed inside the measured window
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16):
            out = pm(**batch, use_cache=False, **extra)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - start
        rng_fwd = torch.cuda.get_rng_state()
        (out.loss * tu.LOSS_SCALE).backward()
        rng_bwd = torch.cuda.get_rng_state()
    finally:
        hook.remove()
    grads = {n: p.grad.detach().clone() for n, p in pm.named_parameters() if p.requires_grad}
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads.values())
    grads = {n: g / tu.LOSS_SCALE for n, g in grads.items()}  # a power of two: exact
    return {"loss": out.loss.detach().float(), "hidden": rec["hidden"], "rng_fwd": rng_fwd,
            "rng_bwd": rng_bwd, "grads": grads, "held": held}


def _counts():
    return (dict(a1.COUNTS), dict(a2.COUNTS), dict(sc.COUNTS), dict(unpad.COUNTS))


def test_v6_merge_matches_v5_and_holds_less(v6_patches, monkeypatch):
    patched = v6_patches
    calls = []
    real_fused = fb.add_norm_mlp

    def counting(*a, **k):
        calls.append(1)
        return real_fused(*a, **k)

    monkeypatch.setattr(fb, "add_norm_mlp", counting)
    pm = build()
    n_layers = pm.config.num_hidden_layers
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}

    _set(False)  # v5
    ref_a = _run(pm, batch, patched, extra)
    ref_b = _run(pm, batch, patched, extra)
    assert not calls

    _set(True)  # v6
    c1, c2, c3, cu = _counts()
    new = _run(pm, batch, patched, extra)
    d1 = {k: a1.COUNTS[k] - c1[k] for k in c1}
    d2 = {k: a2.COUNTS[k] - c2[k] for k in c2}
    d3 = {k: sc.COUNTS[k] - c3[k] for k in c3}
    assert unpad.COUNTS["packed"] - cu["packed"] == 1
    assert len(calls) == n_layers  # B
    assert d1["recompute"] == n_layers and d1["original"] == 0  # A1
    # A2: layer 0's input has no grad (frozen embedding, no input-require-grads hook in 4-bit),
    # so peft's own dropout runs there and keeps no mask either; every later q/v call replays.
    assert d2["replay"] + d2["original"] == 2 * n_layers
    assert d2["replay"] >= 2 * (n_layers - 1)
    # D: exactly the two fp16 LoRA-A inputs per layer; A off.
    assert d3["rows_packed"] == 2 * n_layers, d3
    assert d3["sdpa_compact"] == 0 and d3["kv_packed"] == 0

    # Same RNG stream; A2's backward replay puts the generator back.
    assert torch.equal(new["rng_fwd"], ref_a["rng_fwd"])
    assert torch.equal(new["rng_bwd"], new["rng_fwd"]) and torch.equal(ref_a["rng_bwd"], ref_a["rng_fwd"])
    tp._assert_matches(ref_a["hidden"], ref_b["hidden"], new["hidden"], "hidden")
    tp._assert_matches(ref_a["loss"], ref_b["loss"], new["loss"], "loss")
    assert new["grads"].keys() == ref_a["grads"].keys()
    for name in ref_a["grads"]:
        tp._assert_matches(ref_a["grads"][name], ref_b["grads"][name], new["grads"][name], name)

    mask = batch["attention_mask"]
    rows, n = mask.numel(), int(mask.sum())
    # Layer 0 has no grad into the stack (frozen embeddings): v5 keeps no norm input there, so B's
    # (r, a) costs exactly v5's (r_mid, x16) on that layer, and its dropouts keep no mask (A2).
    b_saving = 4 * n * H * (n_layers - 1)           # B: r_mid fp32 + x16 fp16 - a fp16, real rows
    per_layer = (6 * rows * H - 4 * rows * KV_HEADS * HEAD_DIM  # A1: 12-head k, v, out -> 2-head
                 + 4 * (rows - n) * H)              # D: two fp16 LoRA-A inputs, padding rows
    a2_saving = 2 * rows * H * (n_layers - 1)       # A2: two bool masks where x needs grad
    expect = n_layers * per_layer + b_saving + a2_saving
    saved = ref_a["held"] - new["held"]
    # Before the leak fix the T4 measured 11 980 800 B: A1's throw-away SDPA graph (k, v, out, lse,
    # bias of every layer) stayed held through D's pass-through pack hook.
    assert saved >= 0.8 * expect, (ref_a["held"], new["held"], expect)


def test_a_and_a1_exclusion_in_the_registry():
    """Why apply() never installs A next to A1:
    - A installed first: its entry in the instance's local mapping is read before the class-wide
      one A1 registers later (transformers utils/generic.py:1110-1114), so A1 would be bypassed;
    - A1 installed first (apply()'s order): A sees a foreign "sdpa" and refuses by itself."""
    a1.unpatch_sdpa_recompute()
    assert "sdpa" not in ALL_ATTENTION_FUNCTIONS._local_mapping
    try:
        assert sc.patch_sdpa_kv_compact()  # A first, on the genuine function
        assert a1.patch_sdpa_recompute()  # A1 still registers class-wide...
        looked_up = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        assert getattr(looked_up, sc._MARK, False) and not getattr(looked_up, a1._MARK, False)
        ALL_ATTENTION_FUNCTIONS._local_mapping.pop("sdpa", None)
        looked_up = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        assert getattr(looked_up, a1._MARK, False)  # ...and is reached only without A's entry
        assert not sc.patch_sdpa_kv_compact()  # A1 first: A refuses
        assert "sdpa" not in ALL_ATTENTION_FUNCTIONS._local_mapping
    finally:
        ALL_ATTENTION_FUNCTIONS._local_mapping.pop("sdpa", None)
        a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward
