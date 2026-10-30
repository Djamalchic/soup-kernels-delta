"""v6 leak regression: the Kaggle T4 run of 30.10 died at step 22/64 in both solution runs with
13.75 GiB allocated, ~0.5 GiB more after every step.

Cause: A1 (solution/sdpa_recompute.py) builds a throw-away SDPA graph inside its forward that is
never back-propagated, and D's pack hook (solution/saved_compact.py), active around every packed
self_attn, returned the saved tensors as they were. SDPA saves its own outputs, so each saved output
kept its own node alive through its grad_fn: a C++ reference cycle that only backward breaks and
Python's GC cannot see. Fix: A1 runs its inner calls under its own innermost hooks
(`_private_saves`), and D / A keep a detached alias, never the tensor itself.

The earlier v6 tests measured one forward or one step only, so they could not see a leak. These
fail on the old code:
- A1 under outer hooks (a foreign keep-everything hook, and D's real scope): outer hooks see only
  what A1 saves (q16, k16, v16, mask) and memory returns to its start after forward + backward;
- a real-width Qwen2 stack with every v6 patch on (as test_v6_merged builds it, but with the NF4
  weights quantized from fp16 as from_pretrained leaves them, so K1 runs too), trained for 8 steps
  with AdamW: memory after every step from step 2 on equals step 2's within 1 MiB, and the peak of
  a step does not grow. The old code leaks ~2 x 7.6 MB per step here (3 x 1.8 MB k/v/out + bias +
  lse per layer, rows = 600).

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
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.fused_block as fb  # noqa: E402
import solution.lora_replay as a2  # noqa: E402
import solution.saved_compact as sc  # noqa: E402
import solution.sdpa_recompute as a1  # noqa: E402
import solution.unpad as unpad  # noqa: E402
from solution.tests import test_peak_v6 as tp  # noqa: E402
from solution.tests import test_sdpa_recompute as ts  # noqa: E402
from solution.tests import test_unpad as tu  # noqa: E402  (also hides torchao from peft)
from solution.tests import test_v6_merged as tm  # noqa: E402

forwards = tu.forwards  # restores every patched class forward after each test
v6_patches = tm.v6_patches  # K1 + K3 + K4 + S4 under U, A1 and A2 installed, flags restored
patched = ts.patched  # A1 alone in transformers' registry

MiB = 1 << 20
STEPS = 8


def _allocated():
    gc.collect()  # only Python-collectable garbage; the leak this file guards against is not
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


# ---------------------------------------------------------------------------------------------
# A1 alone under outer saved-tensor hooks
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("outer", ["keep_everything", "d_scope"])
def test_a1_under_outer_hooks_frees_everything(patched, monkeypatch, outer):
    mask = ts.bool_mask()
    qkv = ts.leaves()
    batch, seq = len(ts.LENGTHS), ts.SEQ
    dout = 0.1 * torch.randn(batch, seq, ts.HEADS, ts.HEAD_DIM, device="cuda", dtype=torch.float16)
    real = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
    for b, n in enumerate(ts.LENGTHS):
        real[b, :n] = True
    idx = real.reshape(-1).nonzero().view(-1)
    monkeypatch.setattr(sc, "USE_LORA_ROW_COMPACT", True)
    monkeypatch.setattr(sc, "USE_SDPA_KV_COMPACT", False)
    seen = []

    def keep_everything(t):  # the worst outer hook there is: keeps the tensor itself
        seen.append(tuple(t.shape))
        return t

    def scope():
        if outer == "keep_everything":
            return torch.autograd.graph.saved_tensors_hooks(keep_everything, lambda t: t)
        return sc.attention_scope(idx, batch, seq, ts.HEADS * ts.HEAD_DIM)  # D, as unpad enters it

    def step():
        for t in qkv:
            t.grad = None
        q, k, v = (t.transpose(1, 2) for t in qkv)
        with scope(), torch.autocast("cuda", dtype=torch.float16):
            out, _ = a1.sdpa_forward(ts.module(), q, k, v, mask, dropout=0.0,
                                     scaling=ts.HEAD_DIM ** -0.5)
        out.backward(dout)

    n0 = a1.COUNTS["recompute"]
    step()  # warm-up (kernels, workspaces)
    for t in qkv:
        t.grad = None
    start = _allocated()
    for _ in range(3):
        step()
    for t in qkv:
        t.grad = None
    end = _allocated()
    assert a1.COUNTS["recompute"] == n0 + 4
    # Old code: every call left k/v (12 heads), out, lse and the bias behind (~3.6 MB here).
    assert abs(end - start) <= MiB, (start, end)
    if outer == "keep_everything":
        # The outer hook sees only what _SDPARecompute itself saves, nothing of its inner graph.
        q_shape = (batch, ts.HEADS, seq, ts.HEAD_DIM)
        kv_shape = (batch, ts.KV_HEADS, seq, ts.HEAD_DIM)
        expected = [q_shape, kv_shape, kv_shape, tuple(mask.shape)]
        assert sorted(seen) == sorted(expected * 4), seen


# ---------------------------------------------------------------------------------------------
# Real-width stack, every v6 patch, several optimizer steps
# ---------------------------------------------------------------------------------------------

def _build_fp16_nf4(layers=2, vocab=1024):
    """test_v6_merged.build with fp16-quantized NF4 weights (quant_state.dtype fp16, K1 runs)."""
    torch.manual_seed(0)
    h, kv = tm.H, tm.KV_HEADS * tm.HEAD_DIM
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=h, intermediate_size=tm.INTER,
                      num_hidden_layers=layers, num_attention_heads=tm.HEADS,
                      num_key_value_heads=tm.KV_HEADS, max_position_embeddings=1024,
                      tie_word_embeddings=False, use_cache=False, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(cfg)
    for layer in model.model.layers:
        att, mlp = layer.self_attn, layer.mlp
        for mod, name, n_in, n_out, bias in (
                (att, "q_proj", h, h, True), (att, "k_proj", h, kv, True),
                (att, "v_proj", h, kv, True), (att, "o_proj", h, h, False),
                (mlp, "gate_proj", h, tm.INTER, False), (mlp, "up_proj", h, tm.INTER, False),
                (mlp, "down_proj", tm.INTER, h, False)):
            setattr(mod, name, tp.fp16_linear4bit(n_in, n_out, bias=bias))
    model = model.cuda()
    model.is_loaded_in_4bit = True  # what from_pretrained(quantization_config=...) leaves
    lcfg = peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules=["q_proj", "v_proj"], init_lora_weights=False,
                           task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, lcfg)
    pm.train()
    for layer in tu.inner_model(pm).layers:
        assert type(layer.self_attn.q_proj) is lora_bnb.Linear4bit
        assert layer.mlp.gate_proj.weight.quant_state.dtype == torch.float16
    return pm


def test_training_steps_hold_flat_memory(v6_patches, monkeypatch):
    patched_forward = v6_patches
    calls = []
    real_fused = fb.add_norm_mlp

    def counting(*a, **k):
        calls.append(1)
        return real_fused(*a, **k)

    monkeypatch.setattr(fb, "add_norm_mlp", counting)
    tm._set(True)  # B + A1 + A2 + D on, A off: what solution.apply() switches on
    pm = _build_fp16_nf4()
    n_layers = pm.config.num_hidden_layers
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    vocab = pm.config.vocab_size
    batches = [tu.make_batch(vocab, seed=s) for s in (1, 2)]  # same lengths, other tokens
    mq.Qwen2Model.forward = patched_forward
    c1, c2, c3, cu = tm._counts()

    after, peaks = [], []
    for step in range(STEPS):
        batch = batches[step % 2]
        extra = {"num_items_in_batch": tu.num_items(batch)}
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            out = pm(**batch, use_cache=False, **extra)
        (out.loss * tu.LOSS_SCALE).backward()  # fp16 training's first GradScaler scale
        del out
        with torch.no_grad():
            for p in params:
                p.grad.div_(tu.LOSS_SCALE)  # a power of two: exact
        assert all(torch.isfinite(p.grad).all() for p in params)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 0:
            assert len(opt.state) == len(params)  # Adam's state exists from now on
        after.append(_allocated())
        peaks.append(torch.cuda.max_memory_allocated())

    # Every v6 piece really ran in every step.
    assert unpad.COUNTS["packed"] - cu["packed"] == STEPS
    assert len(calls) == n_layers * STEPS  # B
    assert a1.COUNTS["recompute"] - c1["recompute"] == n_layers * STEPS  # A1
    assert a1.COUNTS["original"] - c1["original"] == 0
    assert a2.COUNTS["replay"] - c2["replay"] >= 2 * (n_layers - 1) * STEPS  # A2
    assert sc.COUNTS["rows_packed"] - c3["rows_packed"] == 2 * n_layers * STEPS  # D

    mib = [round(a / MiB, 3) for a in after]
    peak_mib = [round(p / MiB, 3) for p in peaks]
    # Step 1 creates Adam's state (and first-use workspaces); from step 2 on nothing may stay.
    assert abs(after[5] - after[1]) <= MiB, mib  # step 6 vs step 2
    assert all(abs(a - after[1]) <= MiB for a in after[1:]), mib
    # The peak of a step does not grow (same shapes every step).
    assert all(p <= peaks[1] + MiB for p in peaks[2:]), peak_mib
