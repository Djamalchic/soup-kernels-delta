"""U (solution/unpad.py): the unpadded Qwen2Model.forward matches base on the real rows, keeps the
LoRA dropout masks and the CUDA RNG offsets identical, leaves anything unexpected on the original
forward, and composes with K1/K3/K4 (decoder side) and K2 (head).

Run from the repository root on a T4: python -m pytest solution -q
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

peft = pytest.importorskip("peft")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

import solution.unpad as unpad  # noqa: E402

DEV = "cuda"
SEQ = 150
LENGTHS = (7, 150, 1, 90)  # a length-1 row and a full row; N = 248 >= 129, B*T = 600


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def forwards(monkeypatch):
    """(base, patched) Qwen2Model.forward. Every class forward a test patches is restored after it."""
    for cls in (mq.Qwen2Model, mq.Qwen2DecoderLayer, mq.Qwen2RMSNorm, mq.Qwen2MLP,
                mq.Qwen2ForCausalLM):
        monkeypatch.setattr(cls, "forward", cls.forward)
    base = mq.Qwen2Model.forward
    while getattr(base, unpad._MARK, False):
        base = base.__wrapped__
    mq.Qwen2Model.forward = base
    assert unpad.patch_qwen2_model_unpad()
    patched = mq.Qwen2Model.forward
    assert patched is not base and getattr(patched, unpad._MARK, False)
    assert unpad.patch_qwen2_model_unpad() and mq.Qwen2Model.forward is patched  # idempotent
    return base, patched


def build(hidden=128, heads=4, kv_heads=2, inter=256, layers=2, vocab=512, four_bit_mlp=False):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
                      num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=kv_heads, max_position_embeddings=1024,
                      tie_word_embeddings=False, use_cache=False, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(cfg)
    if four_bit_mlp:
        import bitsandbytes as bnb

        for layer in model.model.layers:
            for name, (n_in, n_out) in {"gate_proj": (hidden, inter), "up_proj": (hidden, inter),
                                        "down_proj": (inter, hidden)}.items():
                setattr(layer.mlp, name, bnb.nn.Linear4bit(
                    n_in, n_out, bias=False, compute_dtype=torch.float16,
                    compress_statistics=True, quant_type="nf4"))
    model = model.cuda()
    assert model.config._attn_implementation == "sdpa"
    # init_lora_weights=False: lora_B is not zero, so lora_A gets a gradient (peft lora/layer.py:328-339).
    lcfg = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
                           init_lora_weights=False, task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, lcfg)
    pm.train()
    return pm


def inner_model(pm):
    model = pm.base_model.model.model
    assert type(model) is mq.Qwen2Model
    return model


def make_batch(vocab, lengths=LENGTHS, seq=SEQ, seed=1):
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, vocab, (len(lengths), seq), generator=gen)
    mask = torch.zeros(len(lengths), seq, dtype=torch.long)
    for row, n in enumerate(lengths):
        mask[row, :n] = 1
    labels = input_ids.clone()
    labels[:, :3] = -100  # a prompt
    labels[mask == 0] = -100
    input_ids[mask == 0] = 0  # pad id
    return {"input_ids": input_ids.to(DEV), "attention_mask": mask.to(DEV), "labels": labels.to(DEV)}


def run(pm, batch, forward, seed=1234, scope=None):
    """One training forward + backward with a given Qwen2Model.forward. Records the final hidden
    states, the RNG state before every LoRA dropout call and after the forward, the dropout masks
    and the LoRA gradients."""
    mq.Qwen2Model.forward = forward
    pm.zero_grad(set_to_none=True)
    rec = {"hidden": None, "rng_pre": [], "keep": [], "drop_in": []}

    # Every hook returns None, so no output is replaced.
    def on_model(module, args, output):
        rec["hidden"] = output.last_hidden_state.detach().clone()

    def before_dropout(module, args):
        rec["rng_pre"].append(torch.cuda.get_rng_state())

    def after_dropout(module, args, output):
        rec["keep"].append(output.detach() != 0)
        rec["drop_in"].append(args[0].detach().clone())

    hooks = [inner_model(pm).register_forward_hook(on_model)]
    for name, mod in pm.named_modules():
        if name.endswith("lora_dropout.default"):
            hooks.append(mod.register_forward_pre_hook(before_dropout))
            hooks.append(mod.register_forward_hook(after_dropout))
    try:
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.float16):
            out = pm(**batch, use_cache=False)
        rng_after = torch.cuda.get_rng_state()
        out.loss.backward()
    finally:
        for h in hooks:
            h.remove()
    grads = {n: p.grad.detach().clone() for n, p in pm.named_parameters() if p.requires_grad}
    assert grads and all(g is not None for g in grads.values())
    return out.loss.detach().float(), rec, rng_after, grads


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def counts():
    return dict(unpad.COUNTS)


# ---------------------------------------------------------------------------------------------
# Row gather / scatter
# ---------------------------------------------------------------------------------------------

def test_pack_unpack_roundtrip_and_gradients():
    batch, seq, width = 4, 33, 64
    mask = torch.zeros(batch, seq, dtype=torch.long, device=DEV)
    mask[0, :1] = 1  # length-1 row
    mask[1, :] = 1  # full row
    mask[2, :17] = 1
    idx = mask.reshape(-1).nonzero().view(-1)
    n = idx.numel()
    rng = torch.cuda.get_rng_state()

    x = torch.randn(n, width, device=DEV, requires_grad=True)
    y = unpad.unpack_rows(x, idx, batch, seq)
    assert y.shape == (batch, seq, width) and y.dtype == x.dtype
    flat = y.detach().reshape(-1, width)
    assert torch.equal(flat[idx], x.detach())
    assert (flat[mask.reshape(-1) == 0] == 0).all()
    g = torch.randn(batch, seq, width, device=DEV)
    y.backward(g)
    assert torch.equal(x.grad, g.reshape(-1, width)[idx])

    z = torch.randn(batch, seq, width, device=DEV, dtype=torch.float16, requires_grad=True)
    p = unpad.pack_rows(z, idx)
    assert p.dtype == torch.float16 and torch.equal(p.detach(), z.detach().reshape(-1, width)[idx])
    h = torch.randn(n, width, device=DEV, dtype=torch.float16)
    p.backward(h)
    zg = z.grad.reshape(-1, width)
    assert z.grad.dtype == torch.float16
    assert torch.equal(zg[idx], h) and (zg[mask.reshape(-1) == 0] == 0).all()

    assert torch.equal(torch.cuda.get_rng_state(), rng)  # no RNG use


# ---------------------------------------------------------------------------------------------
# Whole model against base
# ---------------------------------------------------------------------------------------------

def check_against_base(pm, batch, base_run, new_run, loss_rtol=1e-4, grad_rtol=2e-3):
    loss0, rec0, rng0, grads0 = base_run
    loss1, rec1, rng1, grads1 = new_run
    real = batch["attention_mask"].bool()

    # RNG: every LoRA dropout call starts from the same generator state, and the forward leaves
    # the generator where base leaves it.
    n_drop = 2 * pm.config.num_hidden_layers
    assert len(rec0["rng_pre"]) == len(rec1["rng_pre"]) == n_drop
    for s0, s1 in zip(rec0["rng_pre"], rec1["rng_pre"]):
        assert torch.equal(s0, s1)
    assert torch.equal(rng0, rng1)

    # Dropout inputs keep the base shape [B, T, C]; on the real rows the masks are identical.
    for k0, k1, i0, i1 in zip(rec0["keep"], rec1["keep"], rec0["drop_in"], rec1["drop_in"]):
        assert k0.shape == k1.shape == i0.shape == i1.shape
        nz = (i0 != 0) & (i1 != 0) & real[..., None]
        assert torch.equal(k0[nz], k1[nz])
        assert (i1[~real] == 0).all()  # our attention input is zero on padding
    # Layer 0 q_proj input: the same values on the real rows (row-wise norm on [N, C]).
    torch.testing.assert_close(rec1["drop_in"][0][real], rec0["drop_in"][0][real],
                               rtol=1e-6, atol=1e-6)

    # Final hidden states: close on the real rows, exact zeros on padding.
    h0, h1 = rec0["hidden"], rec1["hidden"]
    assert h0.shape == h1.shape and h0.dtype == h1.dtype
    torch.testing.assert_close(h1[real], h0[real], rtol=5e-3, atol=5e-3)
    assert (h1[~real] == 0).all()

    assert torch.isfinite(loss1)
    assert abs(loss1.item() - loss0.item()) <= loss_rtol * abs(loss0.item())
    assert grads0.keys() == grads1.keys()
    for name in grads0:
        assert rel(grads1[name], grads0[name]) <= grad_rtol, name


def test_matches_base_loss_grads_and_rng(forwards):
    base, patched = forwards
    pm = build()
    batch = make_batch(pm.config.vocab_size)
    before = counts()
    base_run = run(pm, batch, base)
    assert counts() == before  # the original forward does not count
    new_run = run(pm, batch, patched)
    assert counts()["packed"] == before["packed"] + 1
    check_against_base(pm, batch, base_run, new_run)


def test_inputs_are_not_mutated(forwards):
    _, patched = forwards
    pm = build()
    batch = make_batch(pm.config.vocab_size)
    copies = {k: v.clone() for k, v in batch.items()}
    run(pm, batch, patched)
    for k in batch:
        assert torch.equal(batch[k], copies[k]), k


@pytest.mark.parametrize("lengths,seq", [
    ((SEQ,) * 4, SEQ),  # no padding: N == B*T
    ((5, 3, 1, 2), 5),  # N = 11 < 129: bitsandbytes' small-M GEMM territory
    ((SEQ, SEQ, SEQ, SEQ - 10), SEQ),  # 10 of 600 rows padding (< 1/32)
])
def test_batches_without_enough_padding_use_the_original(forwards, lengths, seq):
    base, patched = forwards
    pm = build()
    batch = make_batch(pm.config.vocab_size, lengths=lengths, seq=seq)
    loss0, *_ = run(pm, batch, base)
    before = counts()
    loss1, *_ = run(pm, batch, patched)
    after = counts()
    assert after["packed"] == before["packed"] and after["original"] == before["original"] + 1
    assert torch.equal(loss0, loss1)  # the same code ran


# ---------------------------------------------------------------------------------------------
# Guards: everything unexpected goes to the original forward
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", [
    "eval", "no_grad", "output_hidden_states", "use_cache", "config_use_cache", "no_mask",
    "inputs_embeds", "position_ids", "tuple_output", "gradient_checkpointing", "is_causal",
    "hooked_layer", "unknown_layer_patch", "global_hook", "cpu"])
def test_guards_use_the_original(forwards, monkeypatch, case):
    _, patched = forwards
    mq.Qwen2Model.forward = patched
    pm = build()
    model = inner_model(pm)
    batch = make_batch(pm.config.vocab_size)
    kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
              "use_cache": False}
    grad_ctx = torch.enable_grad()
    handle = None
    if case == "eval":
        pm.eval()
    elif case == "no_grad":
        grad_ctx = torch.no_grad()
    elif case == "output_hidden_states":
        kwargs["output_hidden_states"] = True
    elif case == "use_cache":
        kwargs["use_cache"] = True
    elif case == "config_use_cache":
        monkeypatch.setattr(model.config, "use_cache", True)
        kwargs["use_cache"] = None
    elif case == "no_mask":
        kwargs["attention_mask"] = None
    elif case == "inputs_embeds":
        kwargs["inputs_embeds"] = model.embed_tokens(kwargs.pop("input_ids"))
    elif case == "position_ids":
        kwargs["position_ids"] = torch.arange(SEQ, device=DEV)[None]
    elif case == "tuple_output":
        kwargs["return_dict"] = False
    elif case == "gradient_checkpointing":
        monkeypatch.setattr(model, "gradient_checkpointing", True)
    elif case == "is_causal":
        kwargs["is_causal"] = True
    elif case == "hooked_layer":
        handle = model.layers[0].register_forward_hook(lambda *a: None)
    elif case == "unknown_layer_patch":
        orig = mq.Qwen2DecoderLayer.forward

        def other(self, *a, **k):
            return orig(self, *a, **k)

        monkeypatch.setattr(mq.Qwen2DecoderLayer, "forward", other)
    elif case == "global_hook":
        handle = torch.nn.modules.module.register_module_forward_hook(lambda *a: None)
    elif case == "cpu":
        pm.to("cpu")
        kwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}

    before = counts()
    try:
        with grad_ctx, torch.autocast("cuda", dtype=torch.float16, enabled=case != "cpu"):
            out = model(**kwargs)
    finally:
        if handle is not None:
            handle.remove()
    after = counts()
    assert after["packed"] == before["packed"]
    assert after["original"] == before["original"] + 1
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
    assert hidden.shape[:2] == batch["input_ids"].shape


def test_fake_objects_go_to_the_original():
    """Soup's tests pass fake modules: a guard that cannot be evaluated means the original."""

    class Fake:
        training = True
        config = None

    def original(model, **kwargs):
        return ("original", kwargs["attention_mask"])

    mask = torch.ones(2, 3, dtype=torch.long, device=DEV)
    out = unpad.forward_impl(original, Fake(), input_ids=mask, attention_mask=mask)
    assert out[0] == "original" and out[1] is mask


# ---------------------------------------------------------------------------------------------
# Composition with the other patches
# ---------------------------------------------------------------------------------------------

def test_composes_with_k1_k3_k4(forwards, monkeypatch):
    """Real widths (hidden 1536, 12 heads of 128, intermediate 8960), so K3's norm_ok and
    bitsandbytes' T4 routing (ops.py:703-708) are the production ones. Both runs use the same
    K1/K3/K4 patches; only Qwen2Model.forward differs."""
    bnb = pytest.importorskip("bitsandbytes")
    import solution.mlp as k4

    base, patched = forwards
    try:
        from solution.kernels import nf4 as k1
    except Exception:  # noqa: BLE001
        k1 = None
    try:
        import solution.kernels.addnorm as k3_kernels
        import solution.layers as k3
    except Exception:  # noqa: BLE001
        k3 = k3_kernels = None

    fused_calls = []
    if k1 is not None:
        monkeypatch.setattr(bnb, "matmul_4bit", bnb.matmul_4bit)  # restored after the test
        k1.patch_bnb_matmul_4bit()
    k4.patch_qwen2_mlp_fp16()
    if k3 is not None:
        real_fused = k3_kernels.add_rmsnorm

        def counting_fused(*a, **k):
            fused_calls.append(1)
            return real_fused(*a, **k)

        # Patched before K3 and U import it, so both layer paths call the counter. A K3 patch left
        # installed by another test module is unwrapped first (the fixture restores the classes).
        monkeypatch.setattr(k3_kernels, "add_rmsnorm", counting_fused)
        monkeypatch.delitem(unpad._REFS, "k3", raising=False)
        for cls in (mq.Qwen2DecoderLayer, mq.Qwen2RMSNorm):
            fn = cls.forward
            while getattr(fn, unpad._K3_MARK, False):
                fn = fn.__wrapped__
            cls.forward = fn
        k3.patch_qwen2_rmsnorm()
        k3.patch_qwen2_decoder_layer()

    pm = build(hidden=1536, heads=12, kv_heads=2, inter=8960, layers=2, vocab=1024,
               four_bit_mlp=True)
    batch = make_batch(pm.config.vocab_size)
    base_run = run(pm, batch, base)
    n_base = len(fused_calls)
    before = counts()
    new_run = run(pm, batch, patched)
    assert counts()["packed"] == before["packed"] + 1
    if k3 is not None:
        assert n_base == len(fused_calls) - n_base == pm.config.num_hidden_layers
    # Several fp16 GEMMs now see M = N instead of B*T: allow a little more than the plain test.
    check_against_base(pm, batch, base_run, new_run, loss_rtol=2e-4, grad_rtol=5e-3)


def test_composes_with_k2_head(forwards, monkeypatch):
    head_loss = pytest.importorskip("solution.head_loss")
    trl = pytest.importorskip("trl")

    base, patched = forwards
    monkeypatch.setattr(trl.SFTTrainer, "compute_loss", trl.SFTTrainer.compute_loss)
    if not head_loss.patch_head_loss():
        pytest.skip("K2 could not patch (transformers/trl missing)")

    pm = build()
    batch = make_batch(pm.config.vocab_size)
    results = []
    for forward in (base, patched):
        with head_loss.fast_path_scope(step_cls=dict) as holder:
            results.append(run(pm, batch, forward))
        assert holder.step is not None  # K2's logits-free path ran on top of both
    check_against_base(pm, batch, *results)
