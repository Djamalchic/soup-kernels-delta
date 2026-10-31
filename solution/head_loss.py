"""K2 wiring: Qwen2ForCausalLM returns the loss without logits; TRL's metrics and Soup's rewind log
read the same numbers from a soup_cli ChunkedStep.

Two class-level patches, both guarded, both falling back to the untouched original:

1. `trl.SFTTrainer.compute_loss` (trl/trainer/sft_trainer.py:1200-1328). For a real Qwen2 causal LM
   on CUDA in training mode it does what Soup's own `ChunkedLossTrainer.compute_loss` does
   (soup_cli/utils/stream_loss_chunk.py:313-339): the two input edits TRL makes
   (`_prediction_loss_only` popped, `use_cache=False`, sft_trainer.py:1202, 1210), then
   `transformers.Trainer.compute_loss` with a context flag on, so the model may skip its logits.
   The step the model leaves goes to `trainer._soup_chunked_step` (STEP_ATTR,
   stream_loss_chunk.py:298), where Soup's rewind recorder reads `token_loss` when the outputs
   carry no logits (soup_cli/trainer/rewind_hf.py:345-355), and the three TRL metrics are logged by
   Soup's `_log_trl_metrics` (stream_loss_chunk.py:346-370), which has TRL's names, order and
   formulas (sft_trainer.py:1238-1320). `RewindSFTTrainer.compute_loss` reaches this through
   `super()` (rewind_hf.py:338).
2. `Qwen2ForCausalLM.forward` (modeling_qwen2.py:421-477). Only while that flag is on and every guard
   holds, the decoder runs exactly as in the original (lines 452-460) and the head plus loss run in
   `solution.kernels.head_ce` instead of `lm_head` + `ForCausalLMLoss` (lines 465-469); the output
   has `loss` and `logits=None`. Any other call is the original forward, unchanged.

Soup's config switch `training.stream_loss_chunk` is not touched, no trainer argument is changed and
the trainer class is not replaced. Importing this module and calling `patch_head_loss()` perform no
CUDA work. Any exception while a guard is evaluated (Soup's tests pass MagicMock trainers and fake
modules) sends the call to the original.

3. Lossless fp16 storage of the frozen tied embedding / lm_head (`store_head_fp16`, flag
   STORE_HEAD_FP16), done once, on the first training step that takes the fused path, before its
   forward. Why it is exact: on a T4 Soup loads a LoRA base in fp16 (`torch_dtype` at
   soup_cli/trainer/sft.py:1682 from `_resolve_load_dtype`, :1540-1581 -> utils/gpu.py:659-669 ->
   :634-656, `return torch.float16` at :655 on pre-Ampere cards), and
   `prepare_model_for_kbit_training` then widens every fp16 non-Params4bit parameter to fp32
   (peft/utils/other.py:197-202, called at sft.py:1747-1755), so the
   fp32 head holds fp16 values: 890 MiB storing 445 MiB of information. We re-check that on the
   device for every element (`fp16(w).float() == w`, NaN refuses) and only then set
   `param.data` to the fp16 copy (the same `param.data =` move peft makes, other.py:202), on the
   same Parameter object, so the tie, requires_grad=False and every reference stay. Every use
   then sees the same values as base:
   - the embedding lookup: a forward hook on embed_tokens returns `output.float()`, the exact fp32
     rows base's F.embedding on the fp32 weight returns (modeling_qwen2.py:356);
   - the head: base runs it under fp16 autocast (accelerate wraps the forward,
     accelerator.py:1818-1829), which casts the fp32 weight to fp16 anyway; K2 now slices the fp16
     weight instead of casting it tile by tile (`kernels/head_ce.py`), and the original lm_head
     under autocast gets the same fp16 operand.
   The weight is frozen and not in the optimizer; the adapter does not contain it (peft saves
   embeddings only when targeted or resized, peft/utils/save_and_load.py:421-457). Saves 445 MiB of
   static memory (151936 x 1536 x 2 bytes) and K2's per-step cast of the head (2 passes x ~1.3 GB
   of traffic). Any failed condition leaves the model untouched.

4. v12 W, STREAM_HEAD (needs part 3, K2 and solution/weight_stream.py's flag): the verified fp16
   copy goes to pinned host memory instead of the GPU (`p.data = host`, the same swap), so the
   445 MiB leave the GPU too. Its two readers:
   - the embedding lookup: a guarded class patch of torch.nn.Embedding.forward
     (patch_pinned_embedding) takes a weight marked _PINNED_MARK with CUDA int ids, gathers the
     needed rows on the host (ids to the CPU: one D2H sync of a few KiB; index_select into a
     run-long pinned fp16 buffer) and copies them to the ids' device in one non_blocking H2D on
     the current stream (~4 MiB at 2048 tokens), then the hook of part 3 widens them to fp32: the
     same rows base's F.embedding returns. The pinned row buffer is reused only after the event
     recorded behind the previous copy has completed (one event.synchronize per step, normally
     already done). Any other call runs the original forward.
   - K2's tiles: kernels/head_ce.py _weight_tiles feeds each vocabulary tile from the pinned weight
     through a double-buffered side-stream copy (its own stream / event synchronisation).
   Numerics: the same fp16 rows and the same tiles; nothing changes. Guards: the K2 fast path
   accepts a pinned head weight (it is what K2 reads); if the model ever declines the fast path,
   forward_impl runs the original forward with the head temporarily copied back to the GPU
   (_original_with_gpu_head: correct, slow, never in the measured run).
"""

import collections.abc
import contextlib
import contextvars
import functools
import logging
import weakref

import torch

from .kernels.head_ce import IGNORE_INDEX, SoupHeadCE

_PATCH_MARK = "_soup_delta_head_ce"
STEP_ATTR = "_soup_chunked_step"  # = soup_cli.utils.stream_loss_chunk.STEP_ATTR (line 298)

# Part 3 (see the module docstring). solution.apply() sets it from solution.STORE_HEAD_FP16.
STORE_HEAD_FP16 = False
# Set on a Parameter whose fp32 values we moved, verified, into fp16 storage.
_FP16_MARK = "_soup_delta_fp16_head"
# v12 W (part 4). solution.apply() sets it (only with STORE_HEAD_FP16, K2 and weight streaming).
STREAM_HEAD = False
# Set on a Parameter whose fp16 copy lives in pinned host memory (= solution/unpad.py _PINNED_MARK).
_PINNED_MARK = "_soup_delta_pinned_head"
_EMB_PATCH_MARK = "_soup_delta_pinned_embedding"
# The pinned fp16 row buffer of the embedding lookup and the event behind its last H2D copy.
_EMB_ROWS = {"buf": None, "event": None}
# "pinned": heads moved to the host; "lookups": pinned embedding lookups; "gpu_fallback": original
# forwards run with the head copied back (never expected in a run). Tests read them.
STREAM_HEAD_COUNTS = {"pinned": 0, "lookups": 0, "gpu_fallback": 0}
# Rows per verification chunk: an 8192 x 1536 fp32 temporary (48 MiB) at a time.
_STORE_CHUNK_ROWS = 8192
# Models store_head_fp16 already looked at (one attempt each; a refusal costs a sync and a copy).
_HEAD_TRIED = weakref.WeakSet()
# "stored" / "refused" attempts; tests read it.
HEAD_FP16_COUNTS = {"stored": 0, "refused": 0}

_log = logging.getLogger(__name__)
_warned = {"fallback": False}
# Steps that took our compute_loss: "fused" = logits-free head, "logits" = the model declined.
# Nothing prints it; tests read it to prove the fused path ran.
COUNTS = {"fused": 0, "logits": 0}

# Set (to a fresh holder) only inside our compute_loss, around Trainer.compute_loss.
_ACTIVE = contextvars.ContextVar("soup_delta_head_ce_holder", default=None)

# Real classes captured by patch_head_loss(); Soup's tests may replace `torch` in sys.modules
# (tests/test_callback.py:165-174), so nothing here re-imports transformers at call time.
_REFS = {}


class _StepHolder:
    __slots__ = ("step", "step_cls")

    def __init__(self, step_cls):
        self.step = None
        self.step_cls = step_cls


@contextlib.contextmanager
def fast_path_scope(step_cls=None):
    """Allow the model fast path for the duration; yields the holder the model fills."""
    holder = _StepHolder(step_cls or _soup()[0])
    token = _ACTIVE.set(holder)
    try:
        yield holder
    finally:
        _ACTIVE.reset(token)


def _soup():
    """(ChunkedStep, STEP_ATTR, _log_trl_metrics) of soup_cli 0.77.0 (stream_loss_chunk.py:29-41,
    298, 346-370), or None when soup_cli cannot be imported (then every call is the original)."""
    if "soup" not in _REFS:
        try:
            from soup_cli.utils import stream_loss_chunk as slc

            _REFS["soup"] = (slc.ChunkedStep, slc.STEP_ATTR, slc._log_trl_metrics)
        except Exception:  # noqa: BLE001
            _REFS["soup"] = None
    return _REFS["soup"]


def _log_metrics_from_logits(trainer, mode, inputs, labels, logits):
    """TRL's own metric code (sft_trainer.py:1238-1320) for a step where our compute_loss ran but
    the model declined the fast path and returned logits. Only the branches our trainer guard
    allows (attention_mask present, no shift_labels, no virtual tokens, no Liger, no aux loss)."""
    from trl.trainer.utils import entropy_from_logits  # utils.py:572, imported by sft_trainer.py:61

    if logits is None:
        raise RuntimeError("K2 head_loss: the model returned neither a ChunkedStep nor logits")
    gather = trainer.accelerator.gather_for_metrics
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        per_token_entropy = entropy_from_logits(logits)
        entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        entropy = gather(entropy).mean().item()
    trainer._metrics[mode]["entropy"].append(entropy)
    if mode == "train":
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    with torch.no_grad():
        predictions = logits[..., :-1, :].contiguous().argmax(dim=-1)
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != IGNORE_INDEX
        correct_tokens = gather(((predictions == shift_labels) & mask).sum())
        total_tokens = gather(mask.sum())
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


# ---------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------

def _is_int_scalar_tensor_or_none(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, torch.Tensor) and value.numel() == 1 and not value.is_floating_point()


def _no_module_hooks(module):
    for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
        if getattr(module, name, None):
            return False
    if "forward" in module.__dict__:  # an instance-level forward (accelerate hooks and the like)
        return False
    glob = torch.nn.modules.module
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                 "_global_backward_pre_hooks"):
        if getattr(glob, name, None):
            return False
    return True


def _model_fast_path_ok(model, args, kwargs):
    """Every condition under which head_ce reproduces lines 462-477 of modeling_qwen2.py."""
    refs = _REFS
    if args or type(model) is not refs["Qwen2ForCausalLM"]:
        return False
    if not model.training or not torch.is_grad_enabled():
        return False
    # Base's logits are fp16 only under fp16 autocast (accelerate wraps the model forward,
    # accelerate/accelerator.py:1818-1829).
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
        return False
    labels = kwargs.get("labels")
    input_ids = kwargs.get("input_ids")
    attention_mask = kwargs.get("attention_mask")
    if not (isinstance(labels, torch.Tensor) and isinstance(input_ids, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)):
        return False
    if labels.dim() != 2 or labels.dtype != torch.int64 or not labels.is_cuda:
        return False
    if input_ids.shape != labels.shape or attention_mask.shape != labels.shape:
        return False
    if not input_ids.is_cuda or not attention_mask.is_cuda:
        return False
    if kwargs.get("inputs_embeds") is not None or kwargs.get("past_key_values") is not None:
        return False
    # ForCausalLMLoss honours both (loss_utils.py:49-64); the kernel hard-codes -100 and the shift.
    if "shift_labels" in kwargs or "ignore_index" in kwargs:
        return False
    logits_to_keep = kwargs.get("logits_to_keep", 0)
    if type(logits_to_keep) is not int or logits_to_keep != 0:
        return False
    if not _is_int_scalar_tensor_or_none(kwargs.get("num_items_in_batch")):
        return False
    config = getattr(model, "config", None)
    # can_return_tuple (utils/generic.py:933-942): a tuple output is the original's business.
    return_dict = kwargs.get("return_dict")
    if return_dict is None:
        return_dict = getattr(config, "return_dict", True)
    if not return_dict:
        return False
    if getattr(config, "final_logit_softcapping", None) is not None:
        return False
    if model.loss_function is not refs["ForCausalLMLoss"]:  # modeling_utils.py:4700-4712
        return False
    head = getattr(model, "lm_head", None)
    if type(head) is not torch.nn.Linear or head.bias is not None or not _no_module_hooks(head):
        return False
    weight = head.weight
    # prepare_model_for_kbit_training: frozen (peft utils/other.py:185-187), fp32 (189-202); or
    # the verified fp16 copy store_head_fp16 put in its place.
    if not isinstance(weight, torch.nn.Parameter) or weight.requires_grad:
        return False
    if weight.dtype != torch.float32 and not _is_fp16_stored(weight):
        return False
    if weight.dim() != 2 or not weight.is_contiguous():
        return False
    # v12 W: a pinned host copy is what K2 streams its tiles from; else the labels' device.
    if not is_pinned_head(weight) and (not weight.is_cuda or weight.device != labels.device):
        return False
    return weight.shape[0] == getattr(config, "vocab_size", -1)


def _unwrapped_causal_lm(model):
    """The Qwen2ForCausalLM under a PEFT wrapper (peft_model.py:1099-1103), or the model itself."""
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _trainer_fast_path_ok(trainer, model, inputs):
    # TRL's collator returns a dict (torch_call, sft_trainer.py:164); any mutable mapping works.
    if not isinstance(model, torch.nn.Module):
        return False
    if not isinstance(inputs, collections.abc.MutableMapping):
        return False
    if "labels" not in inputs or "shift_labels" in inputs or "attention_mask" not in inputs:
        return False
    labels, attention_mask = inputs["labels"], inputs["attention_mask"]
    if not (isinstance(labels, torch.Tensor) and isinstance(attention_mask, torch.Tensor)):
        return False
    if not labels.is_cuda or labels.dim() != 2 or attention_mask.shape != labels.shape:
        return False
    if getattr(getattr(trainer, "args", None), "use_liger_kernel", True) is not False:
        return False
    # Set in SFTTrainer.__init__ (sft_trainer.py:788-792, 957): 0 and False for LoRA on Qwen2.
    # A missing attribute or a MagicMock is truthy and keeps the original.
    if getattr(trainer, "num_virtual_tokens", 1) or getattr(trainer, "aux_loss_enabled", True):
        return False
    if getattr(trainer, "compute_loss_func", None) is not None:
        return False
    if getattr(trainer, "label_smoother", None) is not None:
        return False
    if not isinstance(getattr(trainer, "_metrics", None), dict):
        return False
    if getattr(model, "training", False) is not True:
        return False
    if getattr(getattr(trainer, "model", None), "training", False) is not True:
        return False
    if not torch.is_grad_enabled():
        return False
    base = _unwrapped_causal_lm(model)
    if type(base) is not _REFS["Qwen2ForCausalLM"] or base.training is not True:
        return False
    weight = getattr(getattr(base, "lm_head", None), "weight", None)
    return isinstance(weight, torch.Tensor) and (weight.is_cuda or is_pinned_head(weight))


# ---------------------------------------------------------------------------------------------
# Part 3: lossless fp16 storage of the frozen tied head (see the module docstring)
# ---------------------------------------------------------------------------------------------

def _is_fp16_stored(weight):
    return weight.dtype == torch.float16 and getattr(weight, _FP16_MARK, False) is True


def _embedding_output_to_fp32(module, args, output):
    """Forward hook on an nn.Embedding whose weight store_head_fp16 moved to fp16: return the fp32
    rows base's F.embedding returns from the fp32 weight (the same values, verified). Anything
    else passes through (None keeps the output)."""
    weight = getattr(module, "weight", None)
    if (isinstance(output, torch.Tensor) and output.dtype == torch.float16
            and isinstance(weight, torch.Tensor) and _is_fp16_stored(weight)):
        return output.float()
    return None


def _fp16_exact_copy(w):
    """An fp16 tensor equal to fp32 `w` element for element, or None. One host sync."""
    w16 = torch.empty(w.shape, dtype=torch.float16, device=w.device)
    bad = torch.zeros((), dtype=torch.bool, device=w.device)
    for r0 in range(0, w.shape[0], _STORE_CHUNK_ROWS):
        src = w[r0:r0 + _STORE_CHUNK_ROWS]
        dst = w16[r0:r0 + _STORE_CHUNK_ROWS]
        dst.copy_(src)  # round to nearest even: the cast fp16 autocast applies to the head
        bad.logical_or_(dst.float().ne(src).any())  # NaN != NaN: refused as well
    if bool(bad):
        return None
    return w16


def _head_params_ok(params):
    for p in params:
        if not isinstance(p, torch.nn.Parameter) or p.requires_grad or p.grad is not None:
            return False
        if p.dtype != torch.float32 or p.dim() != 2 or not p.is_cuda or not p.is_contiguous():
            return False
    return True


def store_head_fp16(model):
    """Move the frozen embedding / lm_head weight of a Qwen2ForCausalLM from fp32 to fp16 storage
    when fp16 holds every one of its values exactly. True when it did. Never raises for an
    unexpected model; leaves it untouched unless every check passed."""
    if type(model) is not _REFS.get("Qwen2ForCausalLM"):
        return False
    config = getattr(model, "config", None)
    emb = getattr(getattr(model, "model", None), "embed_tokens", None)
    head = getattr(model, "lm_head", None)
    if type(emb) is not torch.nn.Embedding or type(head) is not torch.nn.Linear:
        return False
    # max_norm would make F.embedding renormalise the weight in place; hooks on either module
    # would see fp16 where they saw fp32.
    if head.bias is not None or emb.max_norm is not None or emb.sparse:
        return False
    if not _no_module_hooks(emb) or not _no_module_hooks(head):
        return False
    params = [emb.weight] if head.weight is emb.weight else [emb.weight, head.weight]
    if not _head_params_ok(params):
        return False
    vocab, hidden = getattr(config, "vocab_size", -1), getattr(config, "hidden_size", -1)
    if any(tuple(p.shape) != (vocab, hidden) for p in params):
        return False
    with torch.no_grad():
        copies = []
        for p in params:
            copy = _fp16_exact_copy(p)
            if copy is None:
                return False
            copies.append(copy)
    # v12 W (part 4): the fp16 copies go to pinned host memory when every step of that works
    # (pinned allocation, the embedding patch); otherwise the GPU copies, as in v10c.
    pinned = _pinned_copies(copies) if STREAM_HEAD else None
    if pinned is not None and not patch_pinned_embedding():
        pinned = None
    # Hook first: it passes outputs through until the weight is marked, so no order of failure
    # can leave an fp16 embedding output in the fp32 residual stream.
    emb.register_forward_hook(_embedding_output_to_fp32)
    for p, copy in zip(params, pinned if pinned is not None else copies):
        # The same in-place storage swap peft makes the other way (other.py:202): the Parameter
        # object, its tie and its requires_grad=False are kept; the fp32 storage is freed here.
        p.data = copy
        setattr(p, _FP16_MARK, True)
        if pinned is not None:
            setattr(p, _PINNED_MARK, True)
    if pinned is not None:
        STREAM_HEAD_COUNTS["pinned"] += 1
    return True


# ---------------------------------------------------------------------------------------------
# Part 4 (v12 W): the fp16 head on pinned host memory, the embedding looked up there
# ---------------------------------------------------------------------------------------------

def is_pinned_head(weight):
    """True for the fp16 head / embedding weight store_head_fp16 put in pinned host memory."""
    return (isinstance(weight, torch.Tensor) and getattr(weight, _PINNED_MARK, False) is True
            and not weight.is_cuda and weight.dtype == torch.float16)


def _pinned_copies(copies):
    """Pinned host fp16 tensors holding `copies` (GPU fp16), or None when pinning is not
    available. The D2H copies are complete when this returns (one synchronize)."""
    hosts = []
    try:
        for copy in copies:
            host = torch.empty(copy.shape, dtype=torch.float16, pin_memory=True)
            if not host.is_pinned():
                return None
            host.copy_(copy)
            hosts.append(host)
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - no pinned memory: keep the GPU copies
        return None
    return hosts


def _pinned_embedding_ok(module, ids):
    weight = getattr(module, "weight", None)
    if not is_pinned_head(weight) or weight.dim() != 2 or not weight.is_contiguous():
        return False
    if not isinstance(ids, torch.Tensor) or not ids.is_cuda:
        return False
    if ids.dtype not in (torch.int64, torch.int32):
        return False
    # max_norm renormalises the weight in place inside F.embedding (refused at store time too).
    return getattr(module, "max_norm", None) is None


def _pinned_rows(n, hidden):
    """The run-long pinned fp16 [>= n, hidden] row buffer, free for writing: the H2D copy of its
    previous contents has completed (event.synchronize; normally long done)."""
    buf, event = _EMB_ROWS["buf"], _EMB_ROWS["event"]
    if event is not None:
        event.synchronize()
    if buf is None or buf.shape[0] < n or buf.shape[1] != hidden:
        rows = max(n, 2048)  # the task's 2048 tokens per step [B x T] in one allocation
        buf = torch.empty((rows, hidden), dtype=torch.float16, pin_memory=True)
        _EMB_ROWS["buf"] = buf
    return buf


def pinned_embedding_lookup(weight, ids):
    """F.embedding(ids, weight) for a pinned fp16 weight and CUDA ids: the rows gathered on the
    host and copied to the ids' device in one non_blocking H2D on the current stream."""
    shape = tuple(ids.shape)
    hidden = weight.shape[1]
    flat = ids.reshape(-1)
    n = flat.numel()
    if n == 0:
        return torch.empty((*shape, hidden), dtype=weight.dtype, device=ids.device)
    host_ids = flat.to("cpu")  # one D2H sync of n x 8 bytes
    rows = _pinned_rows(n, hidden)[:n]
    torch.index_select(weight, 0, host_ids, out=rows)
    out = rows.to(ids.device, non_blocking=True)  # pinned -> device, ordered on the current stream
    event = _EMB_ROWS["event"]
    if event is None:
        event = _EMB_ROWS["event"] = torch.cuda.Event()
    event.record(torch.cuda.current_stream(ids.device))
    STREAM_HEAD_COUNTS["lookups"] += 1
    return out.view(*shape, hidden)


def patch_pinned_embedding():
    """torch.nn.Embedding.forward looks a pinned, marked weight up on the host; every other call
    is the original. Idempotent, no CUDA work; True when in place."""
    cls = torch.nn.Embedding
    current = cls.forward
    if getattr(current, _EMB_PATCH_MARK, False) is True:
        return True
    if not callable(current):
        return False
    original = current

    @functools.wraps(original)
    def forward(self, input):
        if _pinned_embedding_ok(self, input):
            try:
                return pinned_embedding_lookup(self.weight, input)
            except Exception:  # noqa: BLE001 - a full copy of the weight: correct, 445 MiB, slow
                return torch.nn.functional.embedding(input, self.weight.to(input.device))
        return original(self, input)

    setattr(forward, _EMB_PATCH_MARK, True)
    cls.forward = forward
    return True


def _original_with_gpu_head(original, model, args, kwargs):
    """The original forward with a pinned head temporarily copied back to the GPU (the lm_head
    and the embedding share the Parameter): correct and slow; a safety net for a declined fast
    path, never expected inside the measured run. The temporary GPU copy stays alive as long as
    autograd's graph of this forward references it."""
    param = model.lm_head.weight
    device = next((t.device for t in kwargs.values() if isinstance(t, torch.Tensor)),
                  torch.device("cuda"))
    host = param.data
    STREAM_HEAD_COUNTS["gpu_fallback"] += 1
    param.data = host.to(device)
    try:
        return original(model, *args, **kwargs)
    finally:
        param.data = host


def _maybe_store_head_fp16(trainer, model):
    """Once per model, on the fused training path, before the step's forward. Only under fp16
    mixed precision: then every forward runs in accelerate's fp16 autocast, where base itself
    feeds the head an fp16 copy of the weight."""
    if not STORE_HEAD_FP16:
        return
    args = getattr(trainer, "args", None)
    if getattr(args, "fp16", False) is not True or getattr(args, "bf16", False) is not False:
        return
    base = _unwrapped_causal_lm(model)
    if base in _HEAD_TRIED:
        return
    _HEAD_TRIED.add(base)
    try:
        stored = store_head_fp16(base)
    except Exception:  # noqa: BLE001 - e.g. out of memory for the copy: keep fp32, train on
        stored = False
    HEAD_FP16_COUNTS["stored" if stored else "refused"] += 1


# ---------------------------------------------------------------------------------------------
# The two patched functions (module level, so the tests can call them with any `original`)
# ---------------------------------------------------------------------------------------------

def _original_forward(original, model, args, kwargs):
    """The untouched forward; with a pinned head (v12 W) the lm_head is copied back first."""
    try:
        pinned = is_pinned_head(getattr(getattr(model, "lm_head", None), "weight", None))
    except Exception:  # noqa: BLE001
        pinned = False
    if pinned:
        return _original_with_gpu_head(original, model, args, kwargs)
    return original(model, *args, **kwargs)


def forward_impl(original, model, args, kwargs):
    holder = _ACTIVE.get()
    if holder is None:
        return _original_forward(original, model, args, kwargs)
    try:
        ok = _model_fast_path_ok(model, args, kwargs)
    except Exception:  # noqa: BLE001 - an unexpected object is never ours to handle
        ok = False
    if not ok:
        return _original_forward(original, model, args, kwargs)

    kw = dict(kwargs)
    kw.pop("return_dict", None)  # can_return_tuple pops it before the body runs (generic.py:936)
    labels = kw.pop("labels")
    input_ids = kw.pop("input_ids", None)
    attention_mask = kw.pop("attention_mask", None)
    position_ids = kw.pop("position_ids", None)
    past_key_values = kw.pop("past_key_values", None)
    inputs_embeds = kw.pop("inputs_embeds", None)
    use_cache = kw.pop("use_cache", None)
    kw.pop("logits_to_keep", None)
    # modeling_qwen2.py:452-460, the same call with the same **kwargs.
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kw,
    )
    loss, token_entropy, token_loss, correct, labelled = SoupHeadCE.apply(
        outputs.last_hidden_state, model.lm_head.weight, labels, attention_mask,
        kw.get("num_items_in_batch"))
    holder.step = holder.step_cls(
        token_entropy=token_entropy, token_loss=token_loss, correct=correct, labelled=labelled)
    return _REFS["CausalLMOutputWithPast"](
        loss=loss, logits=None, past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def compute_loss_impl(original, trainer, model, inputs, return_outputs=False,
                      num_items_in_batch=None):
    try:
        soup = _soup() if _trainer_fast_path_ok(trainer, model, inputs) else None
    except Exception:  # noqa: BLE001 - Soup's tests pass MagicMock trainers and models
        soup = None
    if soup is None:
        return original(trainer, model, inputs, return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch)
    step_cls, step_attr, log_trl_metrics = soup
    _maybe_store_head_fp16(trainer, model)  # part 3: once, before the first fused forward

    mode = "train" if trainer.model.training else "eval"  # sft_trainer.py:1201
    inputs.pop("_prediction_loss_only", None)  # sft_trainer.py:1202
    labels = inputs["labels"]  # sft_trainer.py:1207
    inputs["use_cache"] = False  # sft_trainer.py:1210
    with fast_path_scope(step_cls) as holder:
        loss, outputs = _REFS["Trainer"].compute_loss(
            trainer, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
    step = holder.step
    # Always written, so the rewind recorder never reads a step of an earlier batch.
    setattr(trainer, step_attr, step)
    if step is not None:
        COUNTS["fused"] += 1
        log_trl_metrics(trainer, mode, inputs, step)
    else:
        COUNTS["logits"] += 1
        if not _warned["fallback"]:
            _warned["fallback"] = True
            _log.warning("K2 head_ce: the model declined the fused path; using full logits.")
        _log_metrics_from_logits(trainer, mode, inputs, labels, getattr(outputs, "logits", None))
    return (loss, outputs) if return_outputs else loss


def patch_head_loss():
    """Install both patches. Idempotent; no CUDA work. Returns True when both are in place."""
    try:
        import trl
        from transformers import Trainer
        from transformers.loss.loss_utils import ForCausalLMLoss
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.qwen2 import modeling_qwen2

        sft_cls = trl.SFTTrainer
        qwen_cls = modeling_qwen2.Qwen2ForCausalLM
    except Exception:  # noqa: BLE001 - a missing library means nothing to patch
        return False

    _REFS.update(Qwen2ForCausalLM=qwen_cls, ForCausalLMLoss=ForCausalLMLoss,
                 CausalLMOutputWithPast=CausalLMOutputWithPast, Trainer=Trainer)

    current = qwen_cls.forward
    if not getattr(current, _PATCH_MARK, False):
        original_forward = current

        # functools.wraps keeps __wrapped__, so inspect.signature still shows `labels` etc.:
        # Trainer reads it for label_names, can_return_loss and model_accepts_loss_kwargs.
        @functools.wraps(original_forward)
        def forward(self, *args, **kwargs):
            return forward_impl(original_forward, self, args, kwargs)

        setattr(forward, _PATCH_MARK, True)
        qwen_cls.forward = forward

    current = sft_cls.compute_loss
    if not getattr(current, _PATCH_MARK, False):
        original_compute_loss = current

        @functools.wraps(original_compute_loss)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return compute_loss_impl(original_compute_loss, self, model, inputs,
                                     return_outputs=return_outputs,
                                     num_items_in_batch=num_items_in_batch)

        setattr(compute_loss, _PATCH_MARK, True)
        sft_cls.compute_loss = compute_loss
    return True
