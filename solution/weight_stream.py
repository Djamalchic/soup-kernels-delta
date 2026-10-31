"""v12 W (USE_WEIGHT_STREAMING): the decoder's NF4 weights live in pinned CPU memory and are
streamed through two GPU slots, one layer ahead of the compute stream.

What the GPU holds in v11 at the step's peak: ~680 MiB of frozen NF4 decoder weights (28 layers x
7 projections: packed nibbles + the two absmax levels of bitsandbytes' double quantization) that
each step reads exactly twice per layer (forward dequant, backward re-dequant for dX). Here, after
Soup has loaded and quantized the model and before the first forward, every one of those tensors
is copied into one pinned host buffer (`torch.empty(..., pin_memory=True)`) and the Params4bit /
QuantState objects are repointed at the host copy (`weight.data =`, `quant_state.absmax =`: the
same in-place swaps peft and head_loss.store_head_fp16 make). The GPU keeps two slot buffers of one
layer each (~25 MiB on Qwen2.5-1.5B); layer i lives in slot i % 2 for the whole run.

How reads find the GPU copy. Every base-weight read of ours, forward and backward, goes through
solution/kernels/nf4.py (nf4_dequantize, _checked_entry, k1_dense_path, soup_matmul_4bit): K1's
choke point. Those functions first ask `STREAMER.resolve(packed, quant_state)`, keyed by
id(quant_state) (an id that survives Params4bit.t() in Linear4bit.forward and `weight.view(-1, 1)`
in mlp_recompute), and get back a run-long GPU view of the slot plus a shadow QuantState whose
absmax tensors are views of the same slot. Everything else keeps calling the same functions with
the same arguments; nothing else in solution/ learns about the slots except three device guards
(mlp_recompute._reject_reason, merged_qkv._k1_dense / _build_weight, unpad's embedding check).

Prefetch and stream safety (get(), _prefetch() below). One side stream carries every H2D copy.
- Before the side stream writes slot s it waits on an event recorded on the compute stream at
  that moment, so every read of the slot's previous layer that has been issued completes first.
  In forward layer i's first read prefetches i + 1 into the slot of i - 1 (whose forward is
  issued); in backward layer i's first read prefetches i - 1 into the slot of i + 1 (all of whose
  weight-reading backward nodes are ancestors of layer i's, so they are issued by then).
- Before the compute stream reads slot s it waits on the slot's `ready` event, recorded on the
  side stream after the copy. A read of a layer that is not in its slot (first step, an eval
  forward after a forward, any order we did not predict) prefetches it right there and waits:
  always correct, only slower.
- Slot buffers, host buffer, views and shadow states live for the run; nothing is freed mid-step
  (so no record_stream is needed), and the pinned buffer is written once, at install.
- At the forward -> backward turn nothing is copied: slot 1 holds layer 27 and slot 0 still holds
  26 (prefetch(28) was out of range). Across steps: backward ends at 0 and 1 in the slots, which
  are the first two layers of the next forward.
Per step at 28 layers: 27 copies in forward (layer 0 is a hit), 26 in backward (27 and 26 are
hits), each ~25 MiB, ~2 ms on PCIe 3 x16 [estimate], hidden behind a layer's ~30 ms of compute.

Numerics: unchanged. The same bytes are dequantized by the same K1 kernel into the same GEMMs;
only tensor placement changes (solution/tests/test_weight_stream.py asserts torch.equal).

Install is all or nothing (install() returns None and touches nothing unless every projection of
every layer is an exact frozen Params4bit whose state passes K1's validation on the GPU, every
layer has the same layout, the host buffer is pinned and the shadow states validate on the
slots). It runs once per model, from the first Qwen2ForCausalLM.forward (precompile.
patch_forward_join calls maybe_install; solution.apply() wires it), never raises, and marks the
model so a refusal is not retried. Memory: pinned host 28 x ~25 MiB (~690 MiB, counted by
neither nvidia-smi nor torch), GPU 2 x ~25 MiB of slots instead of ~680 MiB of weights.

Expectation (set before the run): nvidia-smi samples the whole process, so the measured peak is
max(step peak, every transient before our hooks run). The step drops to ~1.4 GB [estimate]. Before
our install the GPU holds the NF4 weights (680 MiB) and, while peft's prepare_model_for_kbit_training
widens the fp16 tied embedding to fp32, both copies (445 + 890 MiB): ~2.1 GB with the context
[estimate]. head_loss._BEFORE_STORE installs the streamer before the fp16 head store, so that
store's own transient (fp32 + fp16 head side by side) no longer adds the 680 MiB; the widening
transient is Soup's / peft's and stays. So expect a peak near 2.1 GB, well above the step, until
the load path is hooked as well.

Importing this module performs no CUDA work.
"""

import copy
import weakref

import torch

from .kernels import nf4 as _nf4

# solution.apply() sets it; the module default keeps every model untouched.
USE_WEIGHT_STREAMING = False
# installed / refused: maybe_install outcomes; copies: H2D slot copies issued; misses: get() calls
# that found their layer not in its slot (the first step's layers 0 and 1, eval re-forwards).
COUNTS = {"installed": 0, "refused": 0, "copies": 0, "misses": 0}

_NAMES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_ALIGN = 256  # bytes; every region of a slot starts 256-byte aligned (fp32 views, 16-B loads)
_TRIED = weakref.WeakSet()  # models maybe_install looked at (one attempt each)


def _round_up(n, align=_ALIGN):
    return (n + align - 1) // align * align


def _base_module(layer, dotted):
    """The module `dotted` names under `layer`, unwrapped from a peft layer (its base_layer)."""
    mod = layer
    for part in dotted.split("."):
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    base = getattr(mod, "base_layer", None)
    return base if isinstance(base, torch.nn.Module) else mod


class _Registry:
    """What kernels/nf4.py holds as STREAMER: one dict lookup per resolve, several streamers (the
    tests install more than one tiny model; the run has one)."""

    def __init__(self):
        self.by_qs = {}  # id(real or shadow QuantState) -> (streamer, layer, index)

    def resolve(self, packed, quant_state):
        hit = self.by_qs.get(id(quant_state))
        if hit is None:
            return packed, quant_state
        streamer, layer, idx = hit
        if not streamer.owns(quant_state, layer, idx):
            return packed, quant_state  # an id reused by a foreign object: never ours
        return streamer.get(layer, idx, packed)

    def add(self, streamer):
        for key, value in streamer.keys():
            self.by_qs[key] = (streamer, *value)

    def remove(self, streamer):
        for key, _ in streamer.keys():
            if self.by_qs.get(key, (None,))[0] is streamer:
                del self.by_qs[key]


REGISTRY = _Registry()


class WeightStreamer:
    """One model's decoder weights on pinned host memory, two GPU slots, a side stream."""

    def __init__(self, device, n_layers, layout_bytes, host, entries):
        self.device = device
        self.n_layers = n_layers
        self.layout_bytes = layout_bytes
        self.host = host  # pinned uint8 [n_layers, layout_bytes]
        # entries[(layer, idx)] = dict(weight=Params4bit, real=QuantState, shadow=QuantState,
        #   packed=GPU uint8 view in the Params4bit's shape, shape=that shape, views={shape: view})
        self.entries = entries
        self.slots = [torch.empty(layout_bytes, dtype=torch.uint8, device=device) for _ in range(2)]
        self.slot_layer = [None, None]
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.stream = torch.cuda.Stream(device=device)
        self.last_layer = None
        self.installed = False

    # -- registry ------------------------------------------------------------------------------

    def keys(self):
        for (layer, idx), e in self.entries.items():
            yield id(e["real"]), (layer, idx)
            yield id(e["shadow"]), (layer, idx)

    def owns(self, quant_state, layer, idx):
        e = self.entries.get((layer, idx))
        return e is not None and (quant_state is e["real"] or quant_state is e["shadow"])

    # -- streaming -----------------------------------------------------------------------------

    def _prefetch(self, layer):
        """Issue the H2D copy of `layer` into its slot on the side stream (no-op when out of range
        or already there). The side stream first waits for everything the compute stream has
        issued so far, which includes every read of the slot's previous layer."""
        if layer < 0 or layer >= self.n_layers:
            return
        s = layer & 1
        if self.slot_layer[s] == layer:
            return
        compute = torch.cuda.current_stream(self.device)
        fence = torch.cuda.Event()
        fence.record(compute)
        self.stream.wait_event(fence)
        with torch.cuda.stream(self.stream):
            self.slots[s].copy_(self.host[layer], non_blocking=True)
        self.ready[s].record(self.stream)
        self.slot_layer[s] = layer
        COUNTS["copies"] += 1

    def get(self, layer, idx, packed=None):
        """(GPU packed view, shadow QuantState) of projection `idx` of `layer`, valid on the
        compute stream from now on; prefetches the next layer in the current direction."""
        s = layer & 1
        if self.slot_layer[s] != layer:
            COUNTS["misses"] += 1
            self._prefetch(layer)
        torch.cuda.current_stream(self.device).wait_event(self.ready[s])
        last = self.last_layer
        if last is not None and layer != last:
            self._prefetch(layer + (1 if layer > last else -1))
        self.last_layer = layer
        e = self.entries[(layer, idx)]
        view = e["packed"]
        if packed is not None:
            shape = tuple(packed.shape)
            if shape != e["shape"]:
                # Linear4bit.forward passes weight.t() ([1, n] of a [n, 1] weight): the same bytes
                # in the caller's shape, one view object per shape so K1's caches hit by identity.
                view = e["views"].get(shape)
                if view is None:
                    view = e["packed"].view(shape)
                    e["views"][shape] = view
        return view, e["shadow"]

    # -- tests ---------------------------------------------------------------------------------

    def restore(self):
        """Put every weight back on the GPU as it was (tests only): the Params4bit and QuantState
        objects get fresh GPU tensors with the host bytes; the registry forgets this streamer."""
        torch.cuda.synchronize()
        REGISTRY.remove(self)
        for e in self.entries.values():
            weight, real = e["weight"], e["real"]
            weight.data = e["host_packed"].to(self.device).view(e["shape"])
            real.absmax = e["host_absmax"].to(self.device)
            real.state2.absmax = e["host_abs2"].to(self.device)
        self.installed = False
        if _nf4.STREAMER is REGISTRY and not REGISTRY.by_qs:
            _nf4.STREAMER = None


def _layer_specs(model):
    """[(layer index, [(weight, quant_state), ...7])] or None when any projection is not an exact
    frozen Params4bit with an NF4 state K1 validates on the GPU."""
    try:
        from bitsandbytes.nn import Params4bit
    except Exception:  # noqa: BLE001
        return None
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    config = getattr(model, "config", None)
    n_layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or type(n_layers) is not int:
        return None
    if n_layers < 1 or len(layers) != n_layers:
        return None
    specs = []
    for li, layer in enumerate(layers):
        projections = []
        for name in _NAMES:
            base = _base_module(layer, name)
            weight = getattr(base, "weight", None)
            if type(weight) is not Params4bit or weight.requires_grad:
                return None
            if getattr(weight, "grad", None) is not None:
                return None
            qs = getattr(weight, "quant_state", None)
            if qs is None or getattr(qs, "state2", None) is None:
                return None
            if not weight.is_cuda or weight.dim() != 2:
                return None
            try:
                _nf4._validate_inputs(weight, qs)  # device, dtypes, sizes, nf4 / nested / fp16
            except (TypeError, ValueError):
                return None
            projections.append((weight, qs))
        specs.append((li, projections))
    return specs


def _layout(projections):
    """[(packed offset, packed bytes, absmax offset, absmax bytes, abs2 offset, abs2 bytes)] per
    projection and the slot size; every region 256-byte aligned."""
    regions = []
    offset = 0
    for weight, qs in projections:
        sizes = (weight.numel(), qs.absmax.numel(), qs.state2.absmax.numel() * 4)
        starts = []
        for size in sizes:
            starts.append(offset)
            offset = _round_up(offset + size)
        regions.append((starts[0], sizes[0], starts[1], sizes[1], starts[2], sizes[2]))
    return regions, offset


def _same_layout(specs, regions):
    first = specs[0][1]
    for _, projections in specs[1:]:
        for (w, qs), (w0, qs0) in zip(projections, first):
            if (tuple(w.shape) != tuple(w0.shape) or qs.absmax.numel() != qs0.absmax.numel()
                    or qs.state2.absmax.numel() != qs0.state2.absmax.numel()
                    or w.device != w0.device):
                return False
    return True


def install(model):
    """Move the decoder's NF4 weights of a Qwen2ForCausalLM to pinned host memory and return the
    WeightStreamer, or None with the model untouched. Raises only for a failure after the swap
    started (then it restores what it changed before re-raising); maybe_install catches that."""
    specs = _layer_specs(model)
    if specs is None:
        return None
    regions, layout_bytes = _layout(specs[0][1])
    if not _same_layout(specs, regions):
        return None
    device = specs[0][1][0][0].device
    n_layers = len(specs)
    try:
        host = torch.empty((n_layers, layout_bytes), dtype=torch.uint8, pin_memory=True)
    except Exception:  # noqa: BLE001 - no pinned memory: nothing to stream from
        return None
    if not host.is_pinned():
        return None
    streamer = WeightStreamer(device, n_layers, layout_bytes, host, {})
    entries = streamer.entries

    # 1. Host copies (D2H) and the GPU views; nothing on the model changes yet, so any failure
    # here (an unexpected QuantState, a view that K1 refuses) is a plain refusal.
    try:
        with torch.no_grad():
            _fill(streamer, specs, regions, host)
    except Exception:  # noqa: BLE001
        return None

    # 2. The swap: weight.data and the two absmax tensors now point at the host copies; the GPU
    # originals lose their last reference and go back to the allocator. K1's caches may hold
    # records of the old GPU tensors: drop them (perf only; they are re-made).
    done = []
    try:
        for e in entries.values():
            weight, real = e["weight"], e["real"]
            saved = (weight.data, real.absmax, real.state2.absmax)
            weight.data = e["host_packed"]
            real.absmax = e["host_absmax"]
            real.state2.absmax = e["host_abs2"]
            done.append((e, saved))
    except Exception:
        for e, (data, absmax, abs2) in done:
            e["weight"].data = data
            e["real"].absmax = absmax
            e["real"].state2.absmax = abs2
        raise
    _nf4._VALID.clear()
    _nf4.dense_cache_clear()
    streamer.installed = True
    REGISTRY.add(streamer)
    _nf4.STREAMER = REGISTRY
    return streamer


def _region(buf, start, nbytes, dtype):
    return buf[start:start + nbytes].view(dtype)


def _fill(streamer, specs, regions, host):
    """D2H copies of every tensor into its host region, the GPU slot views and shadow states."""
    entries = streamer.entries
    for li, projections in specs:
        slot = streamer.slots[li & 1]
        row = host[li]
        for idx, ((weight, qs), reg) in enumerate(zip(projections, regions)):
            p_off, p_n, a_off, a_n, b_off, b_n = reg
            host_packed = _region(row, p_off, p_n, torch.uint8)
            host_absmax = _region(row, a_off, a_n, torch.uint8)
            host_abs2 = _region(row, b_off, b_n, torch.float32)
            host_packed.copy_(weight.data.reshape(-1))
            host_absmax.copy_(qs.absmax.reshape(-1))
            host_abs2.copy_(qs.state2.absmax.reshape(-1))
            shape = tuple(weight.shape)
            packed_view = _region(slot, p_off, p_n, torch.uint8).view(shape)
            shadow = copy.copy(qs)
            shadow.state2 = copy.copy(qs.state2)
            shadow.absmax = _region(slot, a_off, a_n, torch.uint8)
            shadow.state2.absmax = _region(slot, b_off, b_n, torch.float32)
            _nf4._validate_inputs(packed_view, shadow)  # the pair K1 will see; raises if not
            entries[(li, idx)] = dict(
                weight=weight, real=qs, shadow=shadow, packed=packed_view, shape=shape,
                views={}, host_packed=host_packed.view(shape), host_absmax=host_absmax,
                host_abs2=host_abs2)
    torch.cuda.synchronize()


def maybe_install(model):
    """Once per model, never raises: install() when the flag is on; True when it installed."""
    if not USE_WEIGHT_STREAMING:
        return False
    try:
        if model in _TRIED:
            return False
        _TRIED.add(model)
    except TypeError:
        return False
    try:
        streamer = install(model)
    except Exception:  # noqa: BLE001 - the model is as it was (install restores on failure)
        streamer = None
    if streamer is None:
        COUNTS["refused"] += 1
        return False
    COUNTS["installed"] += 1
    return True
