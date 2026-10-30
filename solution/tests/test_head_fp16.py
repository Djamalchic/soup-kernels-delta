"""K2 part 3: lossless fp16 storage of the frozen tied head, and K2 on an fp16 head.

CUDA only (Kaggle T4). Run from the repository root: python -m pytest solution -q

What must hold:
- SoupHeadCE gives the same bits with an fp16 head as with the fp32 head holding the same values
  (the GEMMs see identical fp16 operands of identical shapes; only the cast launches are gone),
  and compiles no second variant of soup_head_ce_kernel.
- store_head_fp16 changes nothing the model computes (embedding rows, logits, loss, gradients),
  keeps the tie and the frozen Parameter, frees the fp32 storage, and refuses anything it cannot
  prove exact or does not recognise.
- The trainer hook does it once, only under fp16 mixed precision, before the first forward.
"""

from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from solution import head_loss  # noqa: E402
from solution.kernels import head_ce  # noqa: E402

DEV = "cuda"
GRAD_OUT = 1024.0
KERNEL = head_ce.soup_head_ce_kernel


@pytest.fixture(scope="module", autouse=True)
def _setup():
    assert head_loss.patch_head_loss()
    flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    yield
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag


def _compiled_count(kernel):
    caches = getattr(kernel, "device_caches", None)
    if isinstance(caches, dict):
        total = 0
        for entry in caches.values():
            entry = entry[0] if isinstance(entry, tuple) else entry
            if not isinstance(entry, dict):
                pytest.skip("unknown Triton cache layout")
            total += len(entry)
        return total
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):
        return sum(len(c) for c in cache.values())
    pytest.skip("unknown Triton cache layout")


def _case(seed, B=4, T=96, v=5000, H=256):
    """fp16-exact head (as Soup's fp16 load widened by peft), random hidden, ragged rows."""
    gen = torch.Generator().manual_seed(seed)
    w32 = (torch.randn(v, H, generator=gen) * 0.15).half().float()
    hidden = torch.randn(B, T, H, generator=gen).to(DEV)
    lengths = torch.randint(T // 3, T + 1, (B,), generator=gen)
    lengths[0] = T
    labels = torch.full((B, T), -100, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        n = int(lengths[b])
        mask[b, :n] = 1
        labels[b, int(n * 0.4):n] = torch.randint(0, v, (n - int(n * 0.4),), generator=gen)
    frozen = [torch.nn.Parameter(w32.to(DEV, dt).contiguous(), requires_grad=False)
              for dt in (torch.float32, torch.float16)]
    return frozen, hidden, labels.to(DEV), mask.to(DEV)


def _run(weight, hidden, labels, mask, num_items):
    h = hidden.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = head_ce.SoupHeadCE.apply(h, weight, labels, mask, num_items)
    out[0].backward(torch.tensor(GRAD_OUT, device=DEV))
    return [o.detach() for o in out], h.grad


# ---------------------------------------------------------------------------------------------
# SoupHeadCE on an fp16 head
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tile", [1024, 8192, 16384])
@pytest.mark.parametrize("with_num_items", [True, False])
def test_fp16_head_gives_the_fp32_head_bits(monkeypatch, tile, with_num_items):
    monkeypatch.setattr(head_ce, "VOCAB_TILE", tile)
    (w32, w16), hidden, labels, mask = _case(3)
    num_items = labels[..., 1:].ne(-100).sum() if with_num_items else None
    outs32, grad32 = _run(w32, hidden, labels, mask, num_items)
    outs16, grad16 = _run(w16, hidden, labels, mask, num_items)
    for a, b in zip(outs16, outs32):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert torch.equal(grad16, grad32)


def test_fp16_head_adds_no_compiled_variant(monkeypatch):
    """Modes 0 and 1 pass an fp32 placeholder as W32, so the head's dtype is not in the kernel
    signature; v = 4992 and tile 1024 keep every width a multiple of 16, as in training."""
    monkeypatch.setattr(head_ce, "VOCAB_TILE", 1024)
    (w32, w16), hidden, labels, mask = _case(4, B=3, T=77, v=4992)
    _run(w32, hidden, labels, mask, None)
    after_fp32 = _compiled_count(KERNEL)
    _run(w16, hidden, labels, mask, None)
    assert _compiled_count(KERNEL) == after_fp32


def test_other_head_dtypes_are_refused():
    (w32, _), hidden, labels, mask = _case(5, B=2, T=16, v=300)
    with pytest.raises(TypeError):
        _run(w32.detach().to(torch.bfloat16), hidden, labels, mask, None)


def test_fp16_head_allocates_no_weight_buffer(monkeypatch):
    """The fp16 head is sliced, not cast: no [tile, H] fp16 buffer and no cast launches."""
    monkeypatch.setattr(head_ce, "VOCAB_TILE", 1024)
    (_, w16), *_ = _case(6, v=4992)
    fetch, wbuf = head_ce._weight_tiles(w16, w16.shape[1], w16.device)
    assert wbuf is None
    tile = fetch(2048, 1024, None, None)
    assert tile.data_ptr() == w16[2048].data_ptr() and tile.shape == (1024, w16.shape[1])


# ---------------------------------------------------------------------------------------------
# store_head_fp16 on a real (tiny) Qwen2ForCausalLM
# ---------------------------------------------------------------------------------------------

def _tiny_qwen(exact=True, vocab=3000, hidden=64, tied=True):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=tied)
    model = Qwen2ForCausalLM(cfg).to(DEV, torch.float32)
    with torch.no_grad():
        for w in {id(p): p for p in (model.lm_head.weight, model.model.embed_tokens.weight)}.values():
            w.normal_(0.0, 0.5)
            if exact:  # what Soup's fp16 load + peft's fp32 widening leave in the weight
                w.copy_(w.half().float())
    for p in model.parameters():  # prepare_model_for_kbit_training (peft utils/other.py:185-202)
        p.requires_grad_(False)
    model.model.norm.weight.requires_grad_(True)  # something upstream that trains
    model.train()
    return model


def _batch(vocab, B=3, T=40, seed=1):
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab, (B, T), generator=gen)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, 25:] = 0
    mask[2, 10:] = 0
    labels = input_ids.clone()
    labels[:, :6] = -100
    labels[mask == 0] = -100
    input_ids[mask == 0] = 0
    return {"input_ids": input_ids.to(DEV), "attention_mask": mask.to(DEV),
            "labels": labels.to(DEV)}


def _forward_all(model, batch, fused):
    """(loss, logits or None, embedding rows, grad of the final norm weight)."""
    num_items = batch["labels"][..., 1:].ne(-100).sum()
    model.zero_grad(set_to_none=True)
    emb = model.model.embed_tokens(batch["input_ids"]).detach()
    scope = head_loss.fast_path_scope() if fused else _Null()
    with scope:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch, num_items_in_batch=num_items, use_cache=False)
    out.loss.backward(torch.tensor(GRAD_OUT, device=DEV))
    logits = None if out.logits is None else out.logits.detach()
    return out.loss.detach(), logits, emb, model.model.norm.weight.grad.clone()


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("tied", [True, False])
def test_store_is_lossless_for_every_path(tied):
    model = _tiny_qwen(tied=tied)
    batch = _batch(model.config.vocab_size)
    before_orig = _forward_all(model, batch, fused=False)
    before_fused = _forward_all(model, batch, fused=True)
    assert before_fused[1] is None and before_orig[1] is not None

    head_w, emb_w = model.lm_head.weight, model.model.embed_tokens.weight
    assert head_loss.store_head_fp16(model)
    # Same Parameter objects, tie kept, frozen, now fp16 and marked.
    assert model.lm_head.weight is head_w and model.model.embed_tokens.weight is emb_w
    assert (head_w is emb_w) == tied
    for w in (head_w, emb_w):
        assert w.dtype == torch.float16 and not w.requires_grad and head_loss._is_fp16_stored(w)
    assert len(model.model.embed_tokens._forward_hooks) == 1

    after_orig = _forward_all(model, batch, fused=False)
    after_fused = _forward_all(model, batch, fused=True)
    for before, after in ((before_orig, after_orig), (before_fused, after_fused)):
        loss_b, logits_b, emb_b, grad_b = before
        loss_a, logits_a, emb_a, grad_a = after
        assert emb_a.dtype == torch.float32 and torch.equal(emb_a, emb_b)
        assert torch.equal(loss_a, loss_b) and torch.equal(grad_a, grad_b)
        if logits_b is not None:
            assert logits_a.dtype == logits_b.dtype and torch.equal(logits_a, logits_b)


def test_store_frees_the_fp32_storage():
    """Exactly V x H x 2 bytes are released: nothing else held the fp32 weight."""
    model = _tiny_qwen(vocab=32768, hidden=512)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    assert head_loss.store_head_fp16(model)
    torch.cuda.synchronize()
    assert before - torch.cuda.memory_allocated() == 32768 * 512 * 2


def test_store_refuses_inexact_values():
    model = _tiny_qwen(exact=False)
    weight = model.lm_head.weight
    assert not head_loss.store_head_fp16(model)
    assert weight.dtype == torch.float32 and not head_loss._is_fp16_stored(weight)
    assert not model.model.embed_tokens._forward_hooks


@pytest.mark.parametrize("case", ["nan", "trainable", "hooked_embedding", "hooked_head", "cpu",
                                  "bias", "not_qwen2", "not_a_module"])
def test_store_guards_leave_the_model_alone(case):
    model = _tiny_qwen()
    target = model
    if case == "nan":
        with torch.no_grad():
            model.lm_head.weight[3, 5] = float("nan")
    elif case == "trainable":
        model.lm_head.weight.requires_grad_(True)
    elif case == "hooked_embedding":
        model.model.embed_tokens.register_forward_hook(lambda *a: None)
    elif case == "hooked_head":
        model.lm_head.register_forward_pre_hook(lambda *a: None)
    elif case == "cpu":
        model.to("cpu")
    elif case == "bias":
        model.lm_head.bias = torch.nn.Parameter(torch.zeros(model.config.vocab_size, device=DEV),
                                                requires_grad=False)
    elif case == "not_qwen2":
        target = model.model
    elif case == "not_a_module":
        target = SimpleNamespace(model=model.model, lm_head=model.lm_head, config=model.config)
    assert not head_loss.store_head_fp16(target)
    assert model.lm_head.weight.dtype == torch.float32
    assert model.model.embed_tokens.weight.dtype == torch.float32


def test_k2_guard_needs_the_mark():
    """An fp16 head that store_head_fp16 did not verify keeps the original forward."""
    model = _tiny_qwen()
    batch = _batch(model.config.vocab_size)
    with torch.autocast("cuda", dtype=torch.float16):
        assert head_loss._model_fast_path_ok(model, (), dict(batch))
    with torch.no_grad():
        model.lm_head.weight.data = model.lm_head.weight.data.half()  # unverified, unmarked
    with torch.autocast("cuda", dtype=torch.float16):
        assert not head_loss._model_fast_path_ok(model, (), dict(batch))
    setattr(model.lm_head.weight, head_loss._FP16_MARK, True)
    with torch.autocast("cuda", dtype=torch.float16):
        assert head_loss._model_fast_path_ok(model, (), dict(batch))


# ---------------------------------------------------------------------------------------------
# Trainer hook
# ---------------------------------------------------------------------------------------------

def _accelerate_like(model):
    from accelerate.utils import convert_outputs_to_fp32

    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    model.forward = MethodType(convert_outputs_to_fp32(autocast(type(model).forward)), model)
    return model


def _fake_trainer(cls, model, fp16):
    t = object.__new__(cls)
    t.args = SimpleNamespace(use_liger_kernel=False, average_tokens_across_devices=False, n_gpu=1,
                             fp16=fp16, bf16=False)
    t.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x, parallelism_config=None)
    t.model = model
    t.label_smoother = None
    t.compute_loss_func = None
    t.model_accepts_loss_kwargs = True
    t.num_virtual_tokens = 0
    t.aux_loss_enabled = False
    t._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    t._total_train_tokens = 0
    t.state = SimpleNamespace(global_step=0)
    return t


@pytest.mark.parametrize("fp16", [True, False])
def test_trainer_stores_once_and_only_under_fp16(monkeypatch, fp16):
    import trl

    monkeypatch.setattr(head_loss, "STORE_HEAD_FP16", True)
    tiny = _tiny_qwen()
    model = _accelerate_like(tiny)
    batch = _batch(tiny.config.vocab_size)
    num_items = batch["labels"][..., 1:].ne(-100).sum()

    def step():
        trainer = _fake_trainer(trl.SFTTrainer, model, fp16)
        loss = trl.SFTTrainer.compute_loss(trainer, model, dict(batch),
                                           num_items_in_batch=num_items)
        assert getattr(trainer, head_loss.STEP_ATTR) is not None  # the fused path ran
        loss.backward(torch.tensor(GRAD_OUT, device=DEV))
        grad = tiny.model.norm.weight.grad.clone()
        tiny.zero_grad(set_to_none=True)
        return loss.detach(), dict(trainer._metrics["train"]), grad

    counts = dict(head_loss.HEAD_FP16_COUNTS)
    first = step()
    second = step()
    stored = head_loss.HEAD_FP16_COUNTS["stored"] - counts["stored"]
    refused = head_loss.HEAD_FP16_COUNTS["refused"] - counts["refused"]
    assert (stored, refused) == ((1, 0) if fp16 else (0, 0))
    assert tiny.lm_head.weight.dtype == (torch.float16 if fp16 else torch.float32)
    # The first step already ran on the fp16 head (stored before its forward): same numbers.
    assert torch.equal(first[0], second[0]) and torch.equal(first[2], second[2])
    assert first[1] == second[1]


def test_trainer_flag_off_keeps_fp32(monkeypatch):
    import trl

    monkeypatch.setattr(head_loss, "STORE_HEAD_FP16", False)
    tiny = _tiny_qwen()
    model = _accelerate_like(tiny)
    batch = _batch(tiny.config.vocab_size)
    trainer = _fake_trainer(trl.SFTTrainer, model, True)
    trl.SFTTrainer.compute_loss(trainer, model, dict(batch))
    assert tiny.lm_head.weight.dtype == torch.float32
