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
                      (a1, "USE_SDPA_RECOMPUTE"), (a2, "USE_LORA_DROPOUT_REPLAY"),
                      (a1, "USE_CAUSAL_PADDED")):
        monkeypatch.setattr(mod, flag, getattr(mod, flag))
    yield patched
    a1.unpatch_sdpa_recompute()
    assert AttentionInterface._global_mapping["sdpa"] is sa.sdpa_attention_forward


def _set(on, causal=False):
    # causal: v7 I1 (is_causal SDPA on the packed path). Off here keeps v6.1 vs v5 bit-exact;
    # solution/tests/test_sdpa_causal.py compares I1 against this v6.1 state.
    a1.USE_CAUSAL_PADDED = causal
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


# ---------------------------------------------------------------------------------------------
# v7 I4: S4 off on the first int(num_hidden_layers * unpad.S4_OFF_FRACTION) packed layers
# ---------------------------------------------------------------------------------------------

import math  # noqa: E402

import solution  # noqa: E402
from solution.tests.test_v6_apply import recorder  # noqa: E402,F401  (fixture reused)

GATE_UP_BYTES = 2 * 2 * INTER  # K4 keeps fp16 gate and up for the SwiGLU backward (swiglu.py:50)
# Extra bytes autograd holds per real token for an off layer, relative to B:
# layer 0: K3 keeps r_mid (4 H) + K4 gate/up, instead of B's packed embeddings (4 H) + a (2 H);
# later layers: r_mid (4 H) + gate/up instead of B's a (2 H) alone (r is input_layernorm's anyway).
OFF_LAYER0_BYTES = GATE_UP_BYTES - 2 * H          # 32 KiB at H = 1536, INTER = 8960
OFF_LATER_BYTES = 4 * H + GATE_UP_BYTES - 2 * H   # 38 KiB


def test_s4_off_count(monkeypatch):
    assert OFF_LAYER0_BYTES == 32 * 1024 and OFF_LATER_BYTES == 38 * 1024
    for frac, layers, want in ((0.25, 28, 7), (0.25, 3, 0), (0.25, 2, 0), (0.25, 1, 0),
                               (0.34, 3, 1), (0.67, 3, 2), (0.215, 28, 6), (0.29, 28, 8),
                               (0.0, 28, 0), (1.0, 28, 28)):
        monkeypatch.setattr(unpad, "S4_OFF_FRACTION", frac)
        assert unpad.s4_off_count(layers) == want, (frac, layers)
    for bad in (math.nan, -0.1, 1.5, True, "0.25", None):
        monkeypatch.setattr(unpad, "S4_OFF_FRACTION", bad)
        assert unpad.s4_off_count(28) == 0, bad
    monkeypatch.setattr(unpad, "S4_OFF_FRACTION", 0.25)
    for layers in (0, -3, 28.0, True, None):
        assert unpad.s4_off_count(layers) == 0, layers


def test_apply_forwards_s4_off_fraction(recorder, monkeypatch):  # noqa: F811
    monkeypatch.setattr(unpad, "S4_OFF_FRACTION", unpad.S4_OFF_FRACTION)
    monkeypatch.setattr(solution, "S4_OFF_FRACTION", solution.S4_OFF_FRACTION)
    assert solution.S4_OFF_FRACTION == 0.25
    solution.apply()
    assert unpad.S4_OFF_FRACTION == 0.25
    monkeypatch.setattr(solution, "USE_UNPAD", False)
    solution.apply()
    assert unpad.S4_OFF_FRACTION == 0.0


def _grad_scaler():
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        return amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def _run_step(pm, batch, forward, extra, seed=1234):
    """One training step as accelerate runs it in fp16: autocast forward, GradScaler-scaled
    backward, AdamW step through the scaler. Restores the LoRA weights afterwards."""
    mq.Qwen2Model.forward = forward
    named = [(n, p) for n, p in pm.named_parameters() if p.requires_grad]
    init = [p.detach().clone() for _, p in named]
    pm.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW([p for _, p in named], lr=1e-3)
    scaler = _grad_scaler()
    rec = {}

    def on_model(module, args, output):  # returns None: the output is not replaced
        rec["hidden"] = output.last_hidden_state.detach().clone()

    hook = tu.inner_model(pm).register_forward_hook(on_model)
    try:
        gc.collect()
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16):
            out = pm(**batch, use_cache=False, **extra)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - start
        rng_fwd = torch.cuda.get_rng_state()
        scaler.scale(out.loss).backward()
        rng_bwd = torch.cuda.get_rng_state()
    finally:
        hook.remove()
    grads = {n: p.grad.detach().clone() for n, p in named}
    assert all(torch.isfinite(g).all() for g in grads.values())
    scaler.step(opt)
    scaler.update()
    stepped = {n: p.detach().clone() for n, p in named}
    with torch.no_grad():
        for (_, p), p0 in zip(named, init):
            p.copy_(p0)
    pm.zero_grad(set_to_none=True)
    res = {"loss": out.loss.detach().float(), "logits": out.logits.detach().clone(),
           "hidden": rec["hidden"], "rng_fwd": rng_fwd, "rng_bwd": rng_bwd, "grads": grads,
           "stepped": stepped, "scale": float(scaler.get_scale()), "held": held}
    del out, opt
    return res


def test_s4_off_first_layers_bit_exact_and_holds_gate_up(v6_patches, monkeypatch):
    """S4_OFF_FRACTION 0.34 (layer 0 off) and 0.67 (layers 0-1 off) against 0.0 (v6.1) on the
    real-width 3-layer stack with every v6 piece on: the same loss, logits, hidden states, RNG,
    LoRA grads and GradScaler step bit for bit; B fires on the other layers only, the off ones are
    the first ones; autograd holds 32 KiB more per real token for layer 0 and 38 KiB for each later
    off layer (10 % tolerance)."""
    patched = v6_patches
    fused_calls, off_mlps = [], []
    real_fused, real_off = fb.add_norm_mlp, unpad._mlp_no_recompute

    def counting_fused(*a, **k):
        fused_calls.append(1)
        return real_fused(*a, **k)

    def recording_off(mlp, y):
        off_mlps.append(mlp)
        return real_off(mlp, y)

    monkeypatch.setattr(fb, "add_norm_mlp", counting_fused)
    monkeypatch.setattr(unpad, "_mlp_no_recompute", recording_off)
    monkeypatch.setattr(unpad, "S4_OFF_FRACTION", 0.0)
    pm = build()
    layers = tu.inner_model(pm).layers
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    n = int(batch["attention_mask"].sum())
    _set(True)  # v6.1: B + A1 + A2 + D on, A off

    ref_a = _run_step(pm, batch, patched, extra)
    ref_b = _run_step(pm, batch, patched, extra)
    assert len(fused_calls) == 2 * LAYERS and not off_mlps

    for frac, n_off, per_token in ((0.34, 1, OFF_LAYER0_BYTES),
                                   (0.67, 2, OFF_LAYER0_BYTES + OFF_LATER_BYTES)):
        monkeypatch.setattr(unpad, "S4_OFF_FRACTION", frac)
        del fused_calls[:], off_mlps[:]
        before = dict(unpad.COUNTS)
        new = _run_step(pm, batch, patched, extra)
        assert len(fused_calls) == LAYERS - n_off, frac
        assert [id(m) for m in off_mlps] == [id(layers[i].mlp) for i in range(n_off)]
        assert unpad.COUNTS["s4_off"] - before["s4_off"] == n_off
        assert unpad.COUNTS["s4_off_fallback"] == before["s4_off_fallback"]
        assert unpad.COUNTS["packed"] - before["packed"] == 1

        # Forward: deterministic, so bit for bit.
        assert torch.equal(new["rng_fwd"], ref_a["rng_fwd"])
        assert torch.equal(new["rng_bwd"], ref_a["rng_bwd"])
        for key in ("loss", "logits", "hidden"):
            assert tp._same(new[key], ref_a[key]), (frac, key)
        # Backward: bit for bit whenever v6.1 reproduces itself (A1's SDPA backward is only as
        # deterministic as torch's), else within v6.1's own spread.
        assert new["grads"].keys() == ref_a["grads"].keys()
        for name in ref_a["grads"]:
            tp._assert_matches(ref_a["grads"][name], ref_b["grads"][name], new["grads"][name], name)
            tp._assert_matches(ref_a["stepped"][name], ref_b["stepped"][name],
                               new["stepped"][name], name)
        assert new["scale"] == ref_a["scale"]

        grown = new["held"] - ref_a["held"]
        expect = per_token * n
        assert abs(grown - expect) <= 0.1 * expect, (frac, grown, expect)


def test_s4_off_guard_falls_back_to_the_module_call(v6_patches, monkeypatch):
    """A forward hook on the off layer's MLP: _mlp_no_recompute calls the module (v6.1's unfused
    S4 path, which fires the hook once); same values as v6.1 without the hook."""
    patched = v6_patches
    monkeypatch.setattr(unpad, "S4_OFF_FRACTION", 0.0)
    pm = build()
    batch = tu.make_batch(pm.config.vocab_size)
    extra = {"num_items_in_batch": tu.num_items(batch)}
    _set(True)
    ref = _run_step(pm, batch, patched, extra)

    fired = []
    mlp0 = tu.inner_model(pm).layers[0].mlp
    handle = mlp0.register_forward_hook(lambda mod, args, out: fired.append(1))
    try:
        assert unpad._k4_forward(mlp0) is None
        monkeypatch.setattr(unpad, "S4_OFF_FRACTION", 0.34)
        before = dict(unpad.COUNTS)
        new = _run_step(pm, batch, patched, extra)
    finally:
        handle.remove()
    assert fired == [1]
    assert unpad.COUNTS["s4_off_fallback"] - before["s4_off_fallback"] == 1
    assert unpad.COUNTS["s4_off"] == before["s4_off"]
    for key in ("loss", "logits", "hidden"):
        assert tp._same(new[key], ref[key]), key
    # Without the hook the guard passes and hands back S4's own fallback (K4's forward).
    assert unpad._k4_forward(mlp0) is mq.Qwen2MLP.forward.__wrapped__
