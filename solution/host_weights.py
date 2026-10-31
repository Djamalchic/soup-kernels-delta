"""v12 H: the frozen base weights live in pinned host memory, not in GPU memory.

What moves: every NF4 weight (packed bytes and first-level absmax) and the tied embedding /
lm_head. They are frozen (requires_grad=False, not in the optimizer, not in the adapter), so the
GPU only ever reads them.

How the GPU reads them: CUDA's unified virtual addressing. Memory from cudaHostAlloc (torch's
`pin_memory=True`) is directly addressable from the device at the same pointer value (CUDA
Programming Guide, "Unified Virtual Address Space"). `alias()` wraps such a buffer as a CUDA tensor
through `__cuda_array_interface__`, so every existing consumer (K1's dequant kernel, bitsandbytes'
dequant, F.embedding) reads the same bytes over PCIe without a code change. The values are the
same bytes; nothing is recomputed or rounded differently.

Where they are put there, so they never sit in GPU memory as a whole:
- transformers' loader materializes each checkpoint tensor (core_model_loading._materialize_copy);
  a tensor of at least HOST_MIN_NUMEL elements (only the 151936 x 1536 embedding) is cast to the
  load dtype on the host instead of on the GPU (the same IEEE round-to-nearest-even cast);
- bitsandbytes quantizes each linear weight on the GPU as before (Bnb4bitQuantize.convert); right
  after, its packed bytes and absmax are copied to the host and the GPU copy is freed;
- peft's prepare_model_for_kbit_training widens fp16 parameters to fp32 (other.py:197-202); for
  a host parameter the same widening is done on the host before it runs, so it skips it.
K2 copies each vocab tile of the head into one GPU buffer before its GEMMs (kernels/head_ce.py);
head_loss's exact fp16 check of the head runs on the host copy.
"""

import torch

HOST_MIN_NUMEL = 64 * 2**20  # the embedding has 233 M elements, the largest linear 13.8 M
_TYPESTR = {torch.float16: "<f2", torch.float32: "<f4", torch.uint8: "|u1"}
_PINNED = {}  # alias data_ptr -> the pinned CPU tensor behind it
COUNTS = {"embedding": 0, "nf4": 0, "widened": 0}
_ORIG = {}


class _Interface:
    def __init__(self, cpu):
        self.__cuda_array_interface__ = {"shape": tuple(cpu.shape), "typestr": _TYPESTR[cpu.dtype],
                                         "data": (cpu.data_ptr(), False), "version": 2,
                                         "strides": None}


def alias(cpu):
    """A CUDA tensor over the memory of the pinned, contiguous CPU tensor `cpu`."""
    t = torch.as_tensor(_Interface(cpu), device="cuda")
    _PINNED[t.data_ptr()] = cpu
    return t


def to_host(t):
    """A pinned copy of `t` (any device), returned as its CUDA alias."""
    cpu = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    cpu.copy_(t)
    return alias(cpu)


def host_source(t):
    """The pinned CPU tensor `t` aliases, or None when `t` is not one of ours."""
    cpu = _PINNED.get(t.data_ptr()) if t.is_cuda else None
    if cpu is None or cpu.shape != t.shape or cpu.dtype != t.dtype or not t.is_contiguous():
        return None
    return cpu


def _is_cuda(device):
    try:
        return device is not None and torch.device(device).type == "cuda"
    except (TypeError, RuntimeError):
        return False


def _materialize_copy(tensor, device=None, dtype=None):
    tensor = tensor[...]  # reads the safetensors slice, as the original's first line
    if _is_cuda(device) and tensor.numel() >= HOST_MIN_NUMEL:
        out = tensor.to(dtype=dtype) if dtype is not None else tensor
        if out.dtype in _TYPESTR:
            COUNTS["embedding"] += 1
            return to_host(out.contiguous())
    return _ORIG["materialize"](tensor, device, dtype)


def _host_params4bit(p):
    if type(p).__name__ != "Params4bit" or not p.is_cuda or not getattr(p, "bnb_quantized", False):
        return
    if p.dtype not in _TYPESTR or host_source(p) is not None:
        return
    p.data = to_host(p.data)
    qs = getattr(p, "quant_state", None)
    absmax = getattr(qs, "absmax", None)
    if torch.is_tensor(absmax) and absmax.is_cuda and absmax.dtype in _TYPESTR:
        qs.absmax = to_host(absmax.contiguous())
    COUNTS["nf4"] += 1


def _convert(self, *args, **kwargs):
    result = _ORIG["convert"](self, *args, **kwargs)
    for value in result.values():
        _host_params4bit(value)
    return result


def _prepare(model, *args, **kwargs):
    for p in model.parameters():
        if p.dtype in (torch.float16, torch.bfloat16) and type(p).__name__ != "Params4bit":
            src = host_source(p)
            if src is not None:
                p.data = to_host(src.float())  # other.py:202's widening, in host memory
                COUNTS["widened"] += 1
    return _ORIG["prepare"](model, *args, **kwargs)


def install():
    """Patch the three loading points. False (nothing patched) when one is missing."""
    if _ORIG:
        return True
    try:
        import peft
        from transformers import core_model_loading
        from transformers.integrations import bitsandbytes as hf_bnb
        orig = {"materialize": core_model_loading._materialize_copy,
                "convert": hf_bnb.Bnb4bitQuantize.convert,
                "prepare": peft.prepare_model_for_kbit_training}
    except (ImportError, AttributeError):
        return False
    _ORIG.update(orig)
    core_model_loading._materialize_copy = _materialize_copy
    hf_bnb.Bnb4bitQuantize.convert = _convert
    peft.prepare_model_for_kbit_training = _prepare
    return True
