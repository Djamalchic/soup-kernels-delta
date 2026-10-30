"""K2 (head + cross-entropy without logits): numerics against base, TRL metrics, Soup rewind, guards.

Run from the repository root on the T4: python -m pytest solution -q

Base reference = what transformers 5.16.1 does under accelerate's fp16 autocast: logits =
F.linear(hidden, W) under autocast (modeling_qwen2.py:465), then ForCausalLMLoss
(loss/loss_utils.py:49-71), backward with grad_output 1024 standing in for the GradScaler factor.

Two kinds of data:
- exact: hidden in multiples of 1/16, weights in multiples of 1/32, H = 256. Every fp16 product and
  every fp32 partial sum is exact, so with fp16 reduced-precision reductions off the fp16 logits are
  identical whatever GEMM shape or reduction order cuBLAS picks (vocab tiles, packed rows). Loss,
  per-token values, argmax ties and accuracy can then be compared to 1e-5 or exactly.
- random: Gaussian hidden and weight, for the loss and gradient tolerances on ordinary numbers.
"""

import inspect
from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402
from transformers.loss.loss_utils import ForCausalLMLoss  # noqa: E402
from trl.trainer.utils import entropy_from_logits  # noqa: E402

from solution import head_loss  # noqa: E402
from solution.kernels import head_ce  # noqa: E402

DEV = "cuda"
V, H = 5000, 256  # V is not a multiple of BLOCK_V or of the test tiles: ragged blocks and tiles
GRAD_OUT = 1024.0
KERNEL = head_ce.soup_head_ce_kernel


@pytest.fixture(scope="module", autouse=True)
def _setup():
    assert head_loss.patch_head_loss()
    flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    yield
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag


def rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(b), 1e-12)


def frozen(w):
    return torch.nn.Parameter(w.to(DEV, torch.float32).contiguous(), requires_grad=False)


def _labels_and_mask(gen, B, T, v, lengths=None, unlabelled_rows=()):
    if lengths is None:
        lengths = torch.randint(T // 3, T + 1, (B,), generator=gen)
        lengths[0] = T  # one row with no padding
    labels = torch.full((B, T), -100, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        length = int(lengths[b])
        prompt = int(length * 0.4)
        mask[b, :length] = 1
        if b not in unlabelled_rows:
            labels[b, prompt:length] = torch.randint(0, v, (length - prompt,), generator=gen)
    return labels.to(DEV), mask.to(DEV)


def random_case(seed, B=4, T=96, v=V):
    gen = torch.Generator().manual_seed(seed)
    weight = frozen(torch.randn(v, H, generator=gen) * 0.15)
    hidden = torch.randn(B, T, H, generator=gen).to(DEV)
    labels, mask = _labels_and_mask(gen, B, T, v)
    return weight, hidden, labels, mask


def exact_case(seed, B=4, T=64, lengths=None, unlabelled_rows=()):
    gen = torch.Generator().manual_seed(seed)
    weight = frozen(torch.randint(-8, 9, (V, H), generator=gen).float() / 32)
    hidden = (torch.randint(-16, 17, (B, T, H), generator=gen).float() / 16).to(DEV)
    labels, mask = _labels_and_mask(gen, B, T, V, lengths, unlabelled_rows)
    return weight, hidden, labels, mask


def base_reference(weight, hidden, labels, num_items):
    h = hidden.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = F.linear(h, weight)
        loss = ForCausalLMLoss(logits, labels, weight.shape[0], num_items_in_batch=num_items)
    assert logits.dtype == torch.float16
    loss.backward(torch.tensor(GRAD_OUT, device=DEV))
    return loss.detach(), logits.detach(), h.grad


def ours(weight, hidden, labels, mask, num_items):
    h = hidden.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):  # as inside the model forward
        out = head_ce.SoupHeadCE.apply(h, weight, labels, mask, num_items)
    out[0].backward(torch.tensor(GRAD_OUT, device=DEV))
    return out, h.grad


def token_reference(logits16, labels):
    logp = torch.log_softmax(logits16.float(), dim=-1)
    target = F.pad(labels, (0, 1), value=-100)[..., 1:]
    lab = target != -100
    tok = -logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    tok = torch.where(lab, tok, torch.zeros_like(tok))
    ent = -(logp.exp() * logp).sum(-1)
    return tok, ent, lab


def trl_accuracy(logits, labels):
    """sft_trainer.py:1292-1319."""
    predictions = logits[..., :-1, :].contiguous().argmax(dim=-1)
    shift = labels[..., 1:].contiguous()
    mask = shift != -100
    return int(((predictions == shift) & mask).sum()), int(mask.sum())


def trl_entropy(logits, mask):
    """sft_trainer.py:1241-1251."""
    per_token = entropy_from_logits(logits)
    return float(torch.sum(per_token * mask) / mask.sum())


def assert_grad_close(grad, ref_grad):
    scale = ref_grad.abs().max().clamp(min=1e-6).item()
    torch.testing.assert_close(grad.float(), ref_grad.float(), rtol=1e-2, atol=1e-2 * scale)


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


# ---------------------------------------------------------------------------------------------
# Kernel + autograd function
# ---------------------------------------------------------------------------------------------

def test_one_compiled_variant_for_all_modes_and_batches(monkeypatch):
    """The three modes, batch shapes, row offsets and tile offsets must not add compiles (compile
    time is inside the measured wall time). v = 4992 and tile 1024 keep every tile width and every
    tile size times H a multiple of 16, as 151936 with tile 16384 does in training."""
    monkeypatch.setattr(head_ce, "VOCAB_TILE", 1024)
    before = _compiled_count(KERNEL)
    counts = []
    for seed, (B, T) in enumerate([(2, 40), (3, 77), (4, 64)]):
        weight, hidden, labels, mask = random_case(10 + seed, B, T, v=4992)
        num_items = labels[..., 1:].ne(-100).sum() if seed else None
        ours(weight, hidden, labels, mask, num_items)
        counts.append(_compiled_count(KERNEL))
    assert counts[0] - before <= 1 and counts[0] == counts[1] == counts[2]


def test_kernel_name():
    name = KERNEL.fn.__name__
    assert name.startswith("soup_") and "example" not in name


def test_cast_mode_is_torch_cast(monkeypatch):
    """Mode 2 rounds fp32 -> fp16 exactly as `.to(torch.float16)` (autocast's cast), including
    values that need rounding, ties of the last bit and fp16 subnormals."""
    monkeypatch.setattr(head_ce, "VOCAB_TILE", 1024)
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(V, H, generator=gen) * 3
    w[0, :64] = torch.randn(64, generator=gen) * 1e-6  # fp16 subnormals
    w[1, :8] = torch.tensor([1 + 2**-11, 1 + 3 * 2**-11, 2**-24, 3 * 2**-25, 65504.0, -0.0,
                             1e-9, -1 - 2**-11])  # halfway cases round to even
    w = w.to(DEV)
    target = torch.zeros(1, dtype=torch.int64, device=DEV)
    state = torch.zeros((1, head_ce._NCOL), dtype=torch.float32, device=DEV)
    tile, tiles = head_ce._tiles(V)
    wbuf = torch.empty((tile, H), dtype=torch.float16, device=DEV)
    for v0, nv in tiles:
        wb = head_ce._cast_tile(w, v0, nv, wbuf, target, state)
        assert torch.equal(wb.view(torch.int16), w[v0:v0 + nv].half().view(torch.int16))


@pytest.mark.parametrize("tile", [1024, 16384])  # 5 tiles with a ragged last one; one tile
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("with_num_items", [True, False])
def test_matches_base_exact(monkeypatch, tile, dtype, with_num_items):
    monkeypatch.setattr(head_ce, "VOCAB_TILE", tile)
    lengths = torch.tensor([37, 20, 9, 30])
    weight, hidden, labels, mask = exact_case(1, B=4, T=37, lengths=lengths, unlabelled_rows=(2,))
    hidden[1, 20:] = 4.0  # garbage on padding must not matter (still exact: 4 is a 1/16 multiple)
    labels[3, 33] = 17  # a label on a padded position: base scores it (position 32 predicts it)
    hidden = hidden.to(dtype)
    num_items = labels[..., 1:].ne(-100).sum() if with_num_items else None

    ref_loss, logits16, ref_grad = base_reference(weight, hidden, labels, num_items)
    (loss, ent, tok, correct, labelled), grad = ours(weight, hidden, labels, mask, num_items)

    assert loss.dtype == torch.float32 and loss.dim() == 0
    assert rel(loss, ref_loss) <= 1e-5
    assert grad.dtype == dtype and grad.shape == hidden.shape
    assert_grad_close(grad, ref_grad)

    ref_tok, ref_ent, lab = token_reference(logits16, labels)
    computed = mask.bool() | lab
    assert bool(lab[3, 32]) and not bool(mask[3, 32])
    assert (grad[~lab] == 0).all() and (ref_grad[~lab] == 0).all()
    torch.testing.assert_close(tok, ref_tok, rtol=1e-5, atol=1e-5)  # position t = label t+1
    torch.testing.assert_close(ent[computed], ref_ent[computed], rtol=1e-4, atol=1e-4)
    assert (ent[~computed] == 0).all() and (tok[~lab] == 0).all()
    assert tok.dtype == torch.float32 and ent.shape == labels.shape

    # TRL accuracy, exactly (first-index argmax on identical logits); labelled = TRL's mask.sum().
    assert (int(correct), int(labelled)) == trl_accuracy(logits16.float(), labels)
    # TRL entropy: accelerate hands TRL fp32 logits (convert_outputs_to_fp32,
    # accelerate/utils/operations.py:889-939); if TRL ever saw fp16 logits the gap stays < 3e-3.
    ours_entropy = float((ent * mask).sum() / mask.sum())
    assert rel(ours_entropy, trl_entropy(logits16.float(), mask)) <= 1e-5
    assert rel(ours_entropy, trl_entropy(logits16, mask)) <= 3e-3


@pytest.mark.parametrize("tile", [1024, 16384])
@pytest.mark.parametrize("with_num_items", [True, False])
def test_random_data_matches_base(monkeypatch, tile, with_num_items):
    monkeypatch.setattr(head_ce, "VOCAB_TILE", tile)
    weight, hidden, labels, mask = random_case(0)
    num_items = labels[..., 1:].ne(-100).sum() if with_num_items else None
    ref_loss, _, ref_grad = base_reference(weight, hidden, labels, num_items)
    (loss, ent, tok, correct, labelled), grad = ours(weight, hidden, labels, mask, num_items)
    assert rel(loss, ref_loss) <= 1e-5
    assert_grad_close(grad, ref_grad)
    lab = F.pad(labels, (0, 1), value=-100)[..., 1:] != -100
    assert (grad[~lab] == 0).all() and int(labelled) == int(lab.sum())


def test_rewind_rows_match_soup_row_losses():
    rewind = pytest.importorskip("soup_cli.trainer.rewind_hf")
    # Row 2 is padded, row 3 has real tokens but no label at all.
    lengths = torch.tensor([64, 50, 20, 33])
    weight, hidden, labels, mask = exact_case(2, lengths=lengths, unlabelled_rows=(3,))
    num_items = labels[..., 1:].ne(-100).sum()
    _, logits16, ref_grad = base_reference(weight, hidden, labels, num_items)
    (loss, ent, tok, correct, labelled), grad = ours(weight, hidden, labels, mask, num_items)

    base_rows, base_tokens = rewind.row_losses(logits16.float(), labels)  # rewind_hf.py:183-223
    our_rows, our_tokens = rewind.row_losses_from_token_losses(tok, labels)  # :168-180
    assert our_tokens == base_tokens and base_tokens[3] == 0
    for a, b in zip(our_rows, base_rows):
        assert abs(a - b) <= 1e-5 * max(abs(b), 1.0)
    assert our_rows[3] == 0.0
    # The unlabelled row still has entropy at its real positions (TRL averages over the mask).
    assert (ent[3, :33] > 0).all() and (ent[3, 33:] == 0).all()
    assert (grad[3] == 0).all() and (ref_grad[3] == 0).all()
    assert (grad[2, 20:] == 0).all()


def test_argmax_ties_take_the_first_index(monkeypatch):
    monkeypatch.setattr(head_ce, "VOCAB_TILE", 1024)
    weight, hidden, labels, mask = exact_case(3, lengths=torch.tensor([64, 64, 64, 64]))
    hidden[..., -1] = 1.0
    # Four vocabulary rows share one vector with a large last weight, so they tie for the row
    # maximum at every position: 7 and 8 in the first block of the first tile, 2100 in the third
    # tile, 4999 in the ragged last tile. torch.argmax returns 7. Still exact data.
    ties = [7, 8, 2100, 4999]
    with torch.no_grad():
        shared = weight[7].clone()
        shared[-1] = 32.0
        weight[:, -1] = 0.0
        for i in ties:
            weight[i] = shared
    gen = torch.Generator().manual_seed(4)
    pick = torch.tensor(ties)[torch.randint(0, 4, labels.shape, generator=gen)].to(DEV)
    labels = torch.where(labels != -100, pick, labels)
    num_items = labels[..., 1:].ne(-100).sum()
    _, logits16, _ = base_reference(weight, hidden, labels, num_items)
    (loss, ent, tok, correct, labelled), _ = ours(weight, hidden, labels, mask, num_items)

    ref_correct, ref_total = trl_accuracy(logits16.float(), labels)
    assert ref_correct == int((labels[..., 1:] == 7).sum()) > 0
    assert (int(correct), int(labelled)) == (ref_correct, ref_total)


@pytest.mark.parametrize("with_num_items", [True, False])
def test_batch_without_labels_behaves_like_base(with_num_items):
    weight, hidden, labels, mask = exact_case(5)
    labels = torch.full_like(labels, -100)
    num_items = labels[..., 1:].ne(-100).sum() if with_num_items else None
    ref_loss, _, ref_grad = base_reference(weight, hidden, labels, num_items)
    (loss, ent, tok, correct, labelled), grad = ours(weight, hidden, labels, mask, num_items)
    # ForCausalLMLoss: sum 0 over 0 items (or the mean of nothing) is nan; the logits gradient is
    # exactly 0. If this fails, base differs here and GradScaler would treat the step differently.
    assert torch.isnan(ref_loss) and torch.isnan(loss)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0, equal_nan=True)
    assert int(correct) == 0 and int(labelled) == 0 and (tok == 0).all()


# ---------------------------------------------------------------------------------------------
# Model hook (Qwen2ForCausalLM.forward)
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=3000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg).to(DEV, torch.float32)
    with torch.no_grad():
        model.lm_head.weight.normal_(0.0, 0.5)  # peaked distributions, few near-ties
    for p in model.parameters():  # as prepare_model_for_kbit_training (peft utils/other.py:185-202)
        p.requires_grad_(False)
    model.model.norm.weight.requires_grad_(True)  # something upstream that trains
    assert model.lm_head.weight is model.model.embed_tokens.weight
    model.train()
    return model


def model_batch(vocab, B=3, T=40, seed=1):
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


def test_model_fast_path_matches_original(tiny_qwen):
    batch = model_batch(tiny_qwen.config.vocab_size)
    num_items = batch["labels"][..., 1:].ne(-100).sum()
    with torch.autocast("cuda", dtype=torch.float16):
        base = tiny_qwen(**batch, num_items_in_batch=num_items, use_cache=False)
    base.loss.backward(torch.tensor(GRAD_OUT, device=DEV))
    base_grad = tiny_qwen.model.norm.weight.grad.clone()
    tiny_qwen.zero_grad(set_to_none=True)

    with head_loss.fast_path_scope() as holder:
        with torch.autocast("cuda", dtype=torch.float16):
            out = tiny_qwen(**batch, num_items_in_batch=num_items, use_cache=False)
    assert out.logits is None and holder.step is not None
    assert rel(out.loss, base.loss) <= 1e-5
    out.loss.backward(torch.tensor(GRAD_OUT, device=DEV))
    grad = tiny_qwen.model.norm.weight.grad
    assert_grad_close(grad, base_grad)


class _nullscope:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("case", ["no_scope", "no_labels", "eval", "no_autocast", "cpu",
                                  "trainable_head", "tuple_output", "logits_to_keep",
                                  "hooked_head"])
def test_model_guards_use_the_original(tiny_qwen, case):
    batch = model_batch(tiny_qwen.config.vocab_size)
    kwargs = {"use_cache": False}
    use_autocast, scope = True, True
    if case == "no_scope":
        scope = False
    elif case == "no_labels":
        batch.pop("labels")
    elif case == "eval":
        tiny_qwen.eval()
    elif case == "no_autocast":
        use_autocast = False
    elif case == "cpu":
        tiny_qwen.to("cpu")
        batch = {k: v.cpu() for k, v in batch.items()}
        use_autocast = False
    elif case == "trainable_head":
        tiny_qwen.lm_head.weight.requires_grad_(True)
    elif case == "tuple_output":
        kwargs["return_dict"] = False
    elif case == "logits_to_keep":
        kwargs["logits_to_keep"] = 1
    elif case == "hooked_head":
        seen = []
        tiny_qwen.lm_head.register_forward_hook(lambda *a: seen.append(1))

    holder = None
    with head_loss.fast_path_scope() if scope else _nullscope() as holder:
        ctx = torch.autocast("cuda", dtype=torch.float16) if use_autocast else _nullscope()
        with ctx:
            out = tiny_qwen(**batch, **kwargs)
    logits = out[1] if isinstance(out, tuple) else out.logits
    assert logits is not None
    if holder is not None:
        assert holder.step is None
    if case == "hooked_head":
        assert seen


@pytest.mark.parametrize("extra", [{"ignore_index": -100}, {"shift_labels": None}])
def test_model_guard_refuses_loss_kwargs_the_kernel_ignores(tiny_qwen, extra):
    """ForCausalLMLoss honours ignore_index and shift_labels (loss_utils.py:49-64)."""
    batch = model_batch(tiny_qwen.config.vocab_size)
    with torch.autocast("cuda", dtype=torch.float16):
        assert head_loss._model_fast_path_ok(tiny_qwen, (), dict(batch))
        assert not head_loss._model_fast_path_ok(tiny_qwen, (), {**batch, **extra})


def test_signature_and_idempotence():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.utils.generic import find_labels

    import trl

    forward, compute_loss = Qwen2ForCausalLM.forward, trl.SFTTrainer.compute_loss
    assert getattr(forward, head_loss._PATCH_MARK) and getattr(compute_loss, head_loss._PATCH_MARK)
    params = inspect.signature(forward).parameters
    assert "labels" in params and "logits_to_keep" in params
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert find_labels(Qwen2ForCausalLM) == ["labels"]  # Trainer.label_names
    assert head_loss.patch_head_loss()
    assert Qwen2ForCausalLM.forward is forward and trl.SFTTrainer.compute_loss is compute_loss


# ---------------------------------------------------------------------------------------------
# Trainer hook (trl.SFTTrainer.compute_loss) and Soup's rewind recorder
# ---------------------------------------------------------------------------------------------

def accelerate_like(model):
    """What accelerate.prepare_model does for fp16 (accelerator.py:1818-1829): autocast the forward
    and convert its outputs to fp32."""
    from accelerate.utils import convert_outputs_to_fp32

    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    model.forward = MethodType(convert_outputs_to_fp32(autocast(type(model).forward)), model)
    return model


def fake_trainer(cls, model):
    """An instance of `cls` with only the attributes compute_loss reads (SFTTrainer.__init__ sets
    them at sft_trainer.py:788-792, 957-960; Trainer.compute_loss at trainer.py:1974-2061)."""
    t = object.__new__(cls)
    t.args = SimpleNamespace(use_liger_kernel=False, average_tokens_across_devices=False, n_gpu=1)
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


def test_trainer_fast_path_matches_trl(tiny_qwen):
    import trl

    model = accelerate_like(tiny_qwen)
    batch = model_batch(tiny_qwen.config.vocab_size)
    num_items = batch["labels"][..., 1:].ne(-100).sum()

    base_t = fake_trainer(trl.SFTTrainer, model)
    original = trl.SFTTrainer.compute_loss.__wrapped__
    loss_b, out_b = original(base_t, model, dict(batch), return_outputs=True,
                             num_items_in_batch=num_items)
    assert out_b.logits is not None and out_b.logits.dtype == torch.float32

    fused_before = head_loss.COUNTS["fused"]
    ours_t = fake_trainer(trl.SFTTrainer, model)
    loss_o, out_o = trl.SFTTrainer.compute_loss(ours_t, model, dict(batch), return_outputs=True,
                                                 num_items_in_batch=num_items)
    step = getattr(ours_t, head_loss.STEP_ATTR)
    assert out_o.logits is None and step is not None
    assert head_loss.COUNTS["fused"] == fused_before + 1
    stream = pytest.importorskip("soup_cli.utils.stream_loss_chunk")
    assert isinstance(step, stream.ChunkedStep)

    assert rel(loss_o, loss_b) <= 1e-5
    mb, mo = base_t._metrics["train"], ours_t._metrics["train"]
    assert list(mb) == list(mo) == ["entropy", "num_tokens", "mean_token_accuracy"]
    assert mo["num_tokens"] == mb["num_tokens"] == [int(batch["attention_mask"].sum())]
    assert rel(mo["entropy"][0], mb["entropy"][0]) <= 1e-4
    labelled = int(num_items)
    assert abs(mo["mean_token_accuracy"][0] - mb["mean_token_accuracy"][0]) <= 1.0 / labelled + 1e-7


def test_trainer_fallback_logs_trl_metrics(tiny_qwen):
    """Our compute_loss ran but the model declined the fused path (here: a hook on lm_head):
    TRL's own formulas run on the logits, the result equals TRL's, and no stale step is left."""
    import trl

    model = accelerate_like(tiny_qwen)
    tiny_qwen.lm_head.register_forward_hook(lambda *a: None)
    batch = model_batch(tiny_qwen.config.vocab_size)
    num_items = batch["labels"][..., 1:].ne(-100).sum()
    base_t = fake_trainer(trl.SFTTrainer, model)
    trl.SFTTrainer.compute_loss.__wrapped__(base_t, model, dict(batch), num_items_in_batch=num_items)
    ours_t = fake_trainer(trl.SFTTrainer, model)
    setattr(ours_t, head_loss.STEP_ATTR, "stale")
    logits_before = head_loss.COUNTS["logits"]
    trl.SFTTrainer.compute_loss(ours_t, model, dict(batch), num_items_in_batch=num_items)
    assert getattr(ours_t, head_loss.STEP_ATTR) is None
    assert head_loss.COUNTS["logits"] == logits_before + 1
    assert dict(ours_t._metrics["train"]) == dict(base_t._metrics["train"])


def test_rewind_recorder_records_the_same_rows(tiny_qwen, monkeypatch):
    """The real Soup RewindSFTTrainer.compute_loss -> super() -> our patch -> ChunkedStep ->
    row_losses_from_token_losses, against the same trainer on TRL's original path."""
    rewind = pytest.importorskip("soup_cli.trainer.rewind_hf")
    import trl

    model = accelerate_like(tiny_qwen)
    batch = model_batch(tiny_qwen.config.vocab_size)
    num_items = batch["labels"][..., 1:].ne(-100).sum()
    cls = rewind.make_rewind_trainer_class(trl.SFTTrainer)

    class Sink:
        def __init__(self):
            self.records = []

        def record_batch(self, **record):
            self.records.append(record)

    def run():
        sink = Sink()
        state = rewind.RewindState(sink)
        for i in range(batch["labels"].shape[0]):
            state.on_index(10 + i)
        trainer = fake_trainer(cls, model)
        setattr(trainer, rewind._STATE_ATTR, state)
        trainer.compute_loss(model, dict(batch), num_items_in_batch=num_items)
        assert not state.failed and state.dropped == 0
        return sink.records, trainer

    with monkeypatch.context() as m:
        m.setattr(trl.SFTTrainer, "compute_loss", trl.SFTTrainer.compute_loss.__wrapped__)
        base_records, _ = run()
    our_records, trainer = run()
    assert getattr(trainer, head_loss.STEP_ATTR) is not None  # the fused path really ran
    assert len(base_records) == len(our_records) == 1
    b, o = base_records[0], our_records[0]
    assert (o["step"], o["micro"], o["rows"], o["row_tokens"]) == \
        (b["step"], b["micro"], b["rows"], b["row_tokens"])
    for x, y in zip(o["row_loss"], b["row_loss"]):
        assert abs(x - y) <= 1e-4 * max(abs(y), 1.0)


def _llama_tiny():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2)
    return LlamaForCausalLM(cfg).to(DEV).train()


class _Exploding:
    """A trainer whose attributes raise something other than AttributeError."""

    def __getattr__(self, name):
        raise RuntimeError(name)


@pytest.mark.parametrize("case", ["mock", "exploding", "cpu", "no_labels", "eval", "liger",
                                  "shift_labels", "not_qwen2", "virtual_tokens"])
def test_trainer_guards_use_the_original(tiny_qwen, case):
    import trl

    model = tiny_qwen
    batch = model_batch(tiny_qwen.config.vocab_size)
    trainer = fake_trainer(trl.SFTTrainer, model)
    if case == "mock":
        trainer, model = MagicMock(), MagicMock()
    elif case == "exploding":
        trainer = _Exploding()
    elif case == "cpu":
        model.to("cpu")
        batch = {k: v.cpu() for k, v in batch.items()}
    elif case == "no_labels":
        batch.pop("labels")
    elif case == "eval":
        model.eval()
    elif case == "liger":
        trainer.args.use_liger_kernel = True
    elif case == "shift_labels":
        batch["shift_labels"] = batch["labels"]
    elif case == "not_qwen2":
        model = _llama_tiny()
        trainer.model = model
    elif case == "virtual_tokens":
        trainer.num_virtual_tokens = 4
    calls = []

    def original(*args, **kwargs):
        calls.append(1)
        return "original"

    assert head_loss.compute_loss_impl(original, trainer, model, batch) == "original"
    assert calls == [1]
