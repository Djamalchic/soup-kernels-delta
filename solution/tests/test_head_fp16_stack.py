"""K2 part 3 (STORE_HEAD_FP16) on the real-width stack, and K2's own cost with an fp16 head.

Why (v9 H16): the Kaggle bench of 31.10 (log12) ran v8b, then v8b with STORE_HEAD_FP16 = True
twice, in that order: median step 0.5985, 0.6285, 0.6509 s. The suspected cause was an fp16
residual stream: nn.Embedding on the fp16 weight returns fp16, K3's norm_ok wants fp32
(kernels/addnorm.py:242) and B's fused_ok calls it (fused_block.py:151), so every norm and every B
node would take the slow path. The code does not let that happen: store_head_fp16 registers
_embedding_output_to_fp32 on embed_tokens (head_loss.py:453-461, registered at :519 before the
storage swap) and U calls embed_tokens as a module (unpad.py:484; base too, modeling_qwen2.py:356),
so the hook widens the rows before pack_rows. Test 1 proves it on the device, through the patched
SFTTrainer.compute_loss as the run reaches it: with the flag on, every packed layer gets and returns
an fp32 residual, K3 launches exactly the kernels of the flag-off step, B fires on every layer, U
packs, K2 stays fused, and the forward is the flag-off one bit for bit (when that is reproducible).

Test 2 times K2 alone at Qwen2.5-1.5B's head with an fp32 and an fp16 head in alternating order,
which a sequential bench cannot do (the two identical h16 runs above differ by 3.6 %). The medians
are emitted as a UserWarning so they show in `pytest -q`'s warnings summary.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q
"""

import gc
import statistics
import warnings

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
trl = pytest.importorskip("trl")
pytest.importorskip("triton")
pytest.importorskip("soup_cli")  # K2's fused path needs soup_cli's ChunkedStep (head_loss._soup)

from peft.tuners.lora import bnb as lora_bnb  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.fused_block as fb  # noqa: E402
import solution.unpad as unpad  # noqa: E402
from solution import head_loss  # noqa: E402
from solution.kernels import addnorm, head_ce  # noqa: E402
from solution.tests import test_head_fp16 as th  # noqa: E402
from solution.tests import test_peak_v6 as tp  # noqa: E402
from solution.tests import test_unpad as tu  # noqa: E402  (also hides torchao from peft)
from solution.tests import test_v6_merged as tv  # noqa: E402

forwards = tu.forwards  # restores every patched class forward after each test
v6_patches = tv.v6_patches  # K1 + K3 + K4 + S4 + U, A1 and A2 installed; flags restored

DEV = "cuda"
F32, F16 = torch.float32, torch.float16
STEP_FIELDS = ("token_entropy", "token_loss", "correct", "labelled")  # ChunkedStep


def _build_tied(vocab=1024):
    """test_v6_merged.build with the tied embedding / lm_head of Qwen2.5-1.5B, holding the
    fp16-exact fp32 values Soup's fp16 load leaves after peft's fp32 widening."""
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=tv.H, intermediate_size=tv.INTER,
                      num_hidden_layers=tv.LAYERS, num_attention_heads=tv.HEADS,
                      num_key_value_heads=tv.KV_HEADS, max_position_embeddings=1024,
                      tie_word_embeddings=True, use_cache=False, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(cfg)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    with torch.no_grad():
        emb = model.model.embed_tokens.weight
        emb.copy_(emb.half().float())
    kv = tv.KV_HEADS * tv.HEAD_DIM
    H, INTER = tv.H, tv.INTER
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
    model.is_loaded_in_4bit = True
    lcfg = peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules=["q_proj", "v_proj"], init_lora_weights=False,
                           task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, lcfg)
    pm.train()
    for layer in tu.inner_model(pm).layers:
        assert type(layer.self_attn.q_proj) is lora_bnb.Linear4bit
    base = pm.get_base_model()
    head = base.lm_head.weight
    assert head is base.model.embed_tokens.weight  # the tie survives .cuda() and peft
    assert head.dtype == F32 and not head.requires_grad and not base.model.embed_tokens._forward_hooks
    return pm


class _Probe:
    """What the fast paths did, recorded through the module attributes their callers look up at
    every call (unpad._packed_forward -> _layer_packed, unpad._layer_packed -> fb.add_norm_mlp,
    K3's rmsnorm / add_rmsnorm / B -> addnorm._launch_fwd). The wrappers only record."""

    def __init__(self, monkeypatch):
        self.reset()
        real_layer, real_fused, real_k3 = unpad._layer_packed, fb.add_norm_mlp, addnorm._launch_fwd

        def layer(lay, r, *args, **kwargs):
            out = real_layer(lay, r, *args, **kwargs)
            self.layers.append((r.dtype, out.dtype))
            return out

        def fused(*args, **kwargs):
            self.fused += 1
            return real_fused(*args, **kwargs)

        def k3(x, delta, weight, eps, store_r=True):
            self.k3.append((x.dtype, None if delta is None else delta.dtype, bool(store_r)))
            return real_k3(x, delta, weight, eps, store_r=store_r)

        monkeypatch.setattr(unpad, "_layer_packed", layer)
        monkeypatch.setattr(fb, "add_norm_mlp", fused)
        monkeypatch.setattr(addnorm, "_launch_fwd", k3)

    def reset(self):
        self.layers, self.fused, self.k3 = [], 0, []


def _step(pm, batch, probe, seed=1234):
    """One training step through the patched SFTTrainer.compute_loss with an fp16-mixed-precision
    trainer (args.fp16 = True: part 3 may store), then the backward."""
    trainer = th._fake_trainer(trl.SFTTrainer, pm, True)
    pm.zero_grad(set_to_none=True)
    probe.reset()
    rec = {}

    def on_model(module, args, output):  # returns None: the output is not replaced
        rec["hidden"] = output.last_hidden_state.detach().clone()

    before_u, before_k2 = dict(unpad.COUNTS), dict(head_loss.COUNTS)
    hook = tu.inner_model(pm).register_forward_hook(on_model)
    try:
        gc.collect()
        torch.manual_seed(seed)  # the same LoRA dropout masks in every step
        loss = trl.SFTTrainer.compute_loss(trainer, pm, dict(batch),
                                           num_items_in_batch=tu.num_items(batch))
        fwd = {"layers": list(probe.layers), "fused": probe.fused, "k3": list(probe.k3)}
        (loss * tu.LOSS_SCALE).backward()
    finally:
        hook.remove()
    step = getattr(trainer, head_loss.STEP_ATTR)
    assert step is not None  # K2's logits-free path ran
    grads = {n: p.grad.detach().clone() for n, p in pm.named_parameters() if p.requires_grad}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    pm.zero_grad(set_to_none=True)
    return {
        "loss": loss.detach().float(),
        "hidden": rec["hidden"],
        "step": {f: getattr(step, f).detach().clone() for f in STEP_FIELDS},
        "metrics": {k: list(v) for k, v in trainer._metrics["train"].items()},
        "grads": {n: g / tu.LOSS_SCALE for n, g in grads.items()},  # a power of two: exact
        "fwd": fwd,
        "bwd_k3": probe.k3[len(fwd["k3"]):],
        "unpad": {k: unpad.COUNTS[k] - before_u[k] for k in before_u},
        "k2": {k: head_loss.COUNTS[k] - before_k2[k] for k in before_k2},
    }


def _assert_every_fast_path(run, n_layers):
    fwd = run["fwd"]
    # fp32 residual into and out of every packed layer; layer 0's r is the packed embedding output.
    assert fwd["layers"] == [(F32, F32)] * n_layers, fwd["layers"]
    assert fwd["fused"] == n_layers  # B on every layer
    # K3 forward launches, all on fp32 rows: input_layernorm then B per layer, model.norm last.
    # A fallback (K3's original Qwen2RMSNorm, or v5's add + norm instead of B) would drop entries.
    assert fwd["k3"] == [(F32, None, True), (F32, F16, True)] * n_layers + [(F32, None, True)], fwd["k3"]
    # B's backward rebuilt every layer's normed input with K3 on the fp32 residual.
    assert run["bwd_k3"] == [(F32, F16, False)] * n_layers, run["bwd_k3"]
    assert run["unpad"]["packed"] == 1 and run["unpad"]["original"] == 0, run["unpad"]
    assert run["k2"] == {"fused": 1, "logits": 0}, run["k2"]


def test_fp16_head_keeps_the_fp32_residual_and_every_fast_path(v6_patches, monkeypatch):
    patched = v6_patches
    # Qwen2ForCausalLM.forward is restored by the forwards fixture; the trainer method here.
    monkeypatch.setattr(trl.SFTTrainer, "compute_loss", trl.SFTTrainer.compute_loss)
    assert head_loss.patch_head_loss()
    probe = _Probe(monkeypatch)
    mq.Qwen2Model.forward = patched  # U (the forwards fixture restores the class afterwards)
    tv._set(True, causal=True)  # solution/__init__.py's defaults: B, A1, A2, D and I1 on, A off
    pm = th._accelerate_like(_build_tied())  # accelerate's fp16 autocast around the forward
    base = pm.get_base_model()
    head, emb = base.lm_head.weight, base.model.embed_tokens
    n_layers = base.config.num_hidden_layers
    batch = tu.make_batch(base.config.vocab_size)
    rows_ref = emb(batch["input_ids"]).detach()
    reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # as test_head_fp16
    try:
        monkeypatch.setattr(head_loss, "STORE_HEAD_FP16", False)
        ref_a = _step(pm, batch, probe)
        ref_b = _step(pm, batch, probe)
        assert head.dtype == F32
        stored = head_loss.HEAD_FP16_COUNTS["stored"]

        monkeypatch.setattr(head_loss, "STORE_HEAD_FP16", True)
        new = _step(pm, batch, probe)  # stores before its forward (head_loss.py:603)
        again = _step(pm, batch, probe)  # once per model
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced

    assert head_loss.HEAD_FP16_COUNTS["stored"] == stored + 1
    assert base.lm_head.weight is head and emb.weight is head
    assert head.dtype == F16 and head_loss._is_fp16_stored(head) and not head.requires_grad
    assert len(emb._forward_hooks) == 1
    rows = emb(batch["input_ids"]).detach()  # the hook: fp32 rows, the same values
    assert tp._same(rows, rows_ref)

    for run in (ref_a, ref_b, new, again):
        _assert_every_fast_path(run, n_layers)
    for run in (new, again):
        # Forward: the same operands everywhere (fp32 rows from the hook, the same fp16 head
        # tiles), so bit for bit whenever the flag-off forward is itself reproducible.
        tp._assert_matches(ref_a["loss"], ref_b["loss"], run["loss"], "loss")
        tp._assert_matches(ref_a["hidden"], ref_b["hidden"], run["hidden"], "hidden")
        for f in STEP_FIELDS:
            tp._assert_matches(ref_a["step"][f], ref_b["step"][f], run["step"][f], f)
        if ref_b["metrics"] == ref_a["metrics"]:
            assert run["metrics"] == ref_a["metrics"]
        # Backward: bit for bit or within tp.GRAD_RTOL (A1's SDPA backward, see test_peak_v6).
        assert run["grads"].keys() == ref_a["grads"].keys()
        for name in ref_a["grads"]:
            tp._assert_grad_close(ref_a["grads"][name], ref_b["grads"][name], run["grads"][name], name)


def test_k2_fp16_head_is_not_slower_at_qwen25_shapes():
    """SoupHeadCE forward + backward at V = 151936, H = 1536 and a ragged batch of 4 x 512 (1352
    real rows), fp32 head (a mode-2 cast per tile) against the fp16 head (row slices, no cast),
    alternating which goes first, 12 rounds after a warm-up. The fp16 head skips 2 x 10 cast
    launches that read 934 MB and write 467 MB per pass, ~2.8 GB per step (~10 ms on a T4 at
    ~280 GB/s [estimate]); nothing is added. So its median must not be above the fp32 one (2 %
    allowance for timer noise)."""
    V, H, B, T = 151936, 1536, 4, 512
    lengths = (512, 400, 260, 180)
    torch.manual_seed(7)
    w16 = (torch.randn(V, H, device=DEV) * 0.02).half()
    heads = {"fp32": torch.nn.Parameter(w16.float(), requires_grad=False),
             "fp16": torch.nn.Parameter(w16, requires_grad=False)}
    hidden = torch.randn(B, T, H, device=DEV)
    mask = torch.zeros(B, T, dtype=torch.long, device=DEV)
    labels = torch.full((B, T), -100, dtype=torch.long, device=DEV)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
        start = n * 3 // 10
        labels[b, start:n] = torch.randint(0, V, (n - start,), device=DEV)
    num_items = labels[..., 1:].ne(-100).sum()
    grad_out = torch.tensor(1024.0, device=DEV)

    def once(weight):
        h = hidden.detach().requires_grad_(True)
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0.record()
        with torch.autocast("cuda", dtype=torch.float16):
            out = head_ce.SoupHeadCE.apply(h, weight, labels, mask, num_items)
        out[0].backward(grad_out)
        t1.record()
        t1.synchronize()
        return t0.elapsed_time(t1)

    for name in ("fp32", "fp16", "fp32", "fp16"):  # compile, cuBLAS heuristics, allocator
        once(heads[name])
    times = {"fp32": [], "fp16": []}
    for i in range(12):
        for name in (("fp32", "fp16") if i % 2 == 0 else ("fp16", "fp32")):
            times[name].append(once(heads[name]))
    med = {k: statistics.median(v) for k, v in times.items()}
    msg = (f"K2 fwd+bwd at V=151936, 1352 rows: fp32 head {med['fp32']:.2f} ms, "
           f"fp16 head {med['fp16']:.2f} ms ({med['fp16'] / med['fp32'] - 1:+.1%})")
    warnings.warn(msg, UserWarning)
    assert med["fp16"] <= 1.02 * med["fp32"], msg
