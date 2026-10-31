"""v10d: compile our five Triton kernels in one background thread while Soup loads the model.

Why. Every measured run starts with an empty Triton cache (harness/run_task.py:109 gives each run
its own TRITON_CACHE_DIR), so in v10c each kernel compiles at its first launch, one after the other,
inside training step 1: cuda_utils (gcc), the five kernels (Python frontend, MLIR, LLVM, ptxas) and
the one launcher they share (gcc, v8 C1). Kaggle T4, version 18: step_s[0] 3.71 s against 1.03 s
when the same compiles ran before step 1. v10 P1 ran them in a pool joined inside apply(): apply_s
went 7.79 -> 10.53 s and the wall did not move (the Triton frontend and apply()'s imports both hold
the GIL). Here they run after apply() has returned, overlapping what Soup does between apply() and
the first forward (CLI start, data, from_pretrained + 4-bit quantization, PEFT, trainer init), which
spends much of its time outside the GIL (file reads, CUDA copies, bitsandbytes' ctypes calls).

Rules. The organisers answered the team's question on 31.10.2026: a background thread
inside the training process that compiles our kernels with `kernel.warmup` and exact signatures
while Soup loads the model is allowed; everything stays inside the measured run with an empty
cache, nothing is carried between runs, no separate process. That is exactly this module: one
thread, the same five compiles step 1 would do, joined before the first forward.

What runs (solution.apply() -> start_overlap(), at the end of apply(), after every patch):
1. One worker thread (`soup_precompile`, daemon=False) calls, one kernel after the other,
   `JITFunction.warmup(*args, grid=(1,), **launch options)` with the argument kinds of the
   production launch (kernel_specs below): MockTensor(dtype) for pointers (data_ptr() 0, i.e.
   16-byte aligned, as every production pointer of these kernels is), a multiple of 16 other than 1
   for the two specialised int slots (head_ce n_cols, the SwiGLU count), int32-range values for the
   do_not_specialize ints, Python floats, the same num_warps and, for K1, enable_fp_fusion=False.
   Same specialisation and options -> same cache key: the compiled kernel lands in the
   JITFunction's in-memory cache (and the file cache), and the first `kernel[grid](...)` of step 1
   finds it there and only launches. Nothing is launched by warmup, nothing is allocated on the
   GPU, no stream, event or synchronisation.
2. After the first kernel, the worker also builds the gcc launcher all our kernels share (C1),
   with the call CompiledKernel._init_handles makes at the first launch,
   `driver.active.launcher_cls(src, metadata)` (a host-side .so; no CUDA call). Optional: any error
   leaves it to the first launch, as in v10c.
Only one thread, so two Triton compiles never run at the same time anywhere in the process.

Join. The worker is joined (wait()) before any of our kernels can launch, at the earliest of:
- the first `Qwen2ForCausalLM.forward` (patch_forward_join; transformers 5.16.1
  models/qwen2/modeling_qwen2.py:421-476; PEFT reaches it through peft_model.py:2086/2101 and
  tuners/tuners_utils.py:332-333), the entry of every training forward;
- every one of our launch helpers (kernels/nf4.py _launch_here, kernels/addnorm.py _launch_*,
  lora_epilogue_*, kernels/head_ce.py _launch, kernels/swiglu.py _launch): a safety net for a
  launch outside a model forward. When nothing is pending, wait() is one global read.
So the thread never outlives the first forward. A join waits at most JOIN_TIMEOUT_S; past that
(a hang, never seen) the worker is told to stop after the kernel in progress and the rest
compile at their first launch, as in v10c.

Harness counters. warmup goes through JITFunction.run, which harness/_train.py:60-74 wraps: each
warmup adds 1 to `triton_launches[name]` (:63) and is never timed (:66 skips calls with `warmup`;
timing is also off before step 3, :149). Check 3 only needs a count > 0 (check_rules.py:322-331)
and these five kernels launch on every step anyway; check 4 reads own_kernel_ms, which warmup never
enters.

Triton API. No Triton source on the team's machines; the calls below are those of v10 P1, which ran
on the Kaggle T4 with Triton 3.6.0 (version 18: solution/tests/test_precompile.py passed, step_s[0]
3.71 -> 1.03 s, i.e. no compile left at the first launch): `JITFunction.warmup(*args, grid,
**kwargs)` (= `run(..., warmup=True)`: compiles on a key miss, caches, returns the kernel without
launching), `triton.runtime.jit.MockTensor(dtype)`, `triton.runtime.driver.driver.active`
(`get_current_device()`, `launcher_cls`), `CompiledKernel.src` / `.metadata`.

Guards. start_overlap() does nothing unless the CUDA context already exists
(torch.cuda.is_initialized() is True: the harness creates it before apply(), _train.py:120 ->
:130; we never create one, and Soup's regression tests call apply() without it), Triton is real
(our kernels are JITFunctions with `warmup`) and C1 is on (the specs are the C1 signatures); once
per process. Any exception in the worker is caught: that kernel then compiles at its first launch.
Importing this module does nothing.
"""

import contextlib
import functools
import threading
import time

JOIN_TIMEOUT_S = 60.0  # a hang detector: the whole compile is ~3-4 s on the T4 without contention

# Diagnostics (read by tests; nothing prints them).
STATUS = {"started": False, "compiled": [], "failed": [], "errors": {}, "launcher": False,
          "busy_s": None, "joined_by": None, "wait_s": None, "timed_out": False}

_LOCK = threading.Lock()
_CANCEL = threading.Event()  # set by a timed-out join: the worker stops after its current kernel
_WORKER = None  # the worker thread until someone joined it; None otherwise (also: never started)
_STARTED = False  # one attempt per process: apply() runs once in training, many times in tests
_FORWARD_MARK = "_soup_delta_precompile_join"


def kernel_specs(addnorm=True, head_ce=True, swiglu=True, ew_k3=False):
    """[(jit function, warmup args, launch options)] for the C1 launches training makes.

    Each entry mirrors the production launch sites slot by slot (7 pointers, 4 ints, 1 float):
    - K1 kernels/nf4.py:482-494 (signature :188-194): packed u8, absmax u8, state2.absmax f32,
      state2.code f32, offset f32, out (int32 view), out again (spare); n_groups, 0, 0, 0 (all
      do_not_specialize); 0.0; num_warps=_NUM_WARPS, enable_fp_fusion=False.
    - with ew_k3 (v10 P2; v10c runs this way, EW_K3_FP16 on), the K3 pair the run launches:
      fwd v10 (signature kernels/addnorm.py:163-168, every int and the float in do_not_specialize),
      launched at :327-332 (norm, modes 0 / 1), :403-408 (LoRA epilogue, mode 2) and :426-431 (its
      backward, mode 3): X f32 (x / out / g), D fp16 (delta, dummy, base16, g16), W f32 (weight /
      out / g), Y f32 (y, rstd, out, g), R f32 (r, y or rstd, out, g), RSTD f32, Y16 fp16 (dummy,
      fp16 y, lora16, gl16); has_delta, store_r, mode, n16; eps or scale.
      bwd v10 (signature :210-215, all ints and the float in do_not_specialize), launched at
      :357-362: DY16 fp16 (dx16 or the dummy), DR f32 (d_next, dx or dy), X f32, D fp16 (delta or
      dummy), W f32, RSTD f32, DX f32; has_dr, has_delta, dy_mode, store_da; 0.0.
    - without ew_k3, the *_usig pair: fwd kernels/addnorm.py:264-269 (x f32, delta or the fp16
      dummy, weight f32, y f32, r or y f32, rstd f32, y; has_delta, store_r, 0, 0; eps), bwd
      :289-295 (dy f32, dr or dy f32, x f32, delta or dummy fp16, weight f32, rstd f32, dx f32;
      has_dr, has_delta, 0, 0; 0.0).
    - K2 kernels/head_ce.py:196-201 (signature :183-188): logits / weight tile fp16, targets int64,
      state f32, scale (or state) f32, w32 (or state) f32, state, state; row0, n_cols (SPECIALISED:
      16384, 4480 and their x1536 cast counts, all multiples of 16), col0, mode; 0.0; NUM_WARPS.
      With the fp16 head of v10c (STORE_HEAD_FP16) the cast mode 2 is not launched; modes 0 / 1
      keep these kinds.
    - K4 kernels/swiglu.py:95-100 (signature :55-60): gate, up, hid, dgate, dup, gate, gate, all
      fp16; n (SPECIALISED: tokens * 8960, a multiple of 16), mode, 0, 0; 0.0; _NUM_WARPS.
    Only the kinds matter for the key (dtype and 16-byte alignment of the specialised pointers,
    multiple-of-16 / == 1 of the specialised ints, int32 range of every int); the values are never
    read, nothing is launched.
    """
    import torch

    from .kernels import addnorm as k3, head_ce as k2, nf4 as k1, swiglu as k4

    f16, f32, u8, i32, i64 = torch.float16, torch.float32, torch.uint8, torch.int32, torch.int64
    p = _pointer
    specs = [(k1.soup_nf4_dequant_usig,
              (p(u8), p(u8), p(f32), p(f32), p(f32), p(i32), p(i32), 1536 * 1536 // 8, 0, 0, 0,
               0.0),
              {"num_warps": k1._NUM_WARPS, "enable_fp_fusion": False})]
    if addnorm and ew_k3:  # v10 P2: the only K3 entries the run launches
        specs.append((k3.soup_addnorm_fwd_v10,
                      (p(f32), p(f16), p(f32), p(f32), p(f32), p(f32), p(f16), 1, 1, 0, 0, 1e-6),
                      {"num_warps": k3.NUM_WARPS}))
        specs.append((k3.soup_addnorm_bwd_v10,
                      (p(f16), p(f32), p(f32), p(f16), p(f32), p(f32), p(f32), 1, 1, 0, 1, 0.0),
                      {"num_warps": k3.NUM_WARPS}))
    elif addnorm:
        specs.append((k3.soup_addnorm_fwd_usig,
                      (p(f32), p(f16), p(f32), p(f32), p(f32), p(f32), p(f32), 1, 1, 0, 0, 1e-6),
                      {"num_warps": k3.NUM_WARPS}))
        specs.append((k3.soup_addnorm_bwd_usig,
                      (p(f32), p(f32), p(f32), p(f16), p(f32), p(f32), p(f32), 1, 0, 0, 0, 0.0),
                      {"num_warps": k3.NUM_WARPS}))
    if head_ce:
        specs.append((k2.soup_head_ce_usig,
                      (p(f16), p(i64), p(f32), p(f32), p(f32), p(f32), p(f32),
                       0, k2.VOCAB_TILE_V8, 0, 2, 0.0),
                      {"num_warps": k2.NUM_WARPS}))
    if swiglu:
        specs.append((k4.soup_swiglu_kernel,
                      (p(f16), p(f16), p(f16), p(f16), p(f16), p(f16), p(f16),
                       16 * 8960, 0, 0, 0, 0.0),
                      {"num_warps": k4._NUM_WARPS}))
    return specs


def _pointer(dtype):
    """A pointer argument for warmup: Triton's MockTensor (dtype only, data_ptr() == 0)."""
    try:
        from triton.runtime.jit import MockTensor

        return MockTensor(dtype)
    except Exception:  # noqa: BLE001 - warmup wraps a bare dtype itself (MockTensor.wrap_dtype)
        return dtype


def _driver():
    from triton.runtime.driver import driver

    return driver.active


def _name(jit):
    return getattr(getattr(jit, "fn", None), "__name__", "?")


def _work(specs, device, began):
    """The worker: warm every spec in order; never raises."""
    try:
        import torch

        where = torch.cuda.device(device) if device is not None else contextlib.nullcontext()
        with where:  # the device the main thread launches on (the current device is per thread)
            _driver().get_current_device()  # first access builds / loads cuda_utils
            launcher_tried = False
            for jit, args, opts in specs:
                if _CANCEL.is_set():
                    break
                name = _name(jit)
                try:
                    kernel = jit.warmup(*args, grid=(1,), **opts)
                    if kernel is None:
                        raise RuntimeError("warmup returned no kernel")
                except Exception as exc:  # noqa: BLE001 - this kernel compiles at its first launch
                    STATUS["failed"].append(name)
                    STATUS["errors"][name] = repr(exc)[:300]
                    continue
                STATUS["compiled"].append(name)
                if not launcher_tried:  # the launcher source is the same for all five (C1)
                    launcher_tried = True
                    try:
                        cls = getattr(_driver(), "launcher_cls", None)
                        if cls is not None:
                            cls(kernel.src, kernel.metadata)  # gcc + dlopen of a host .so only
                            STATUS["launcher"] = True
                    except Exception as exc:  # noqa: BLE001 - the first launch builds it
                        STATUS["errors"]["launcher"] = repr(exc)[:300]
    except Exception as exc:  # noqa: BLE001 - e.g. no driver: every kernel compiles at its launch
        STATUS["errors"]["worker"] = repr(exc)[:300]
    finally:
        STATUS["busy_s"] = round(time.perf_counter() - began, 3)


def start_overlap(addnorm=True, head_ce=True, swiglu=True, unified=True, ew_k3=False):
    """Start the background compile; True when the worker thread runs. Never raises.

    The arguments name the kernels the run launches (solution.apply() passes its flags, and ew_k3 =
    solution._ew_k3(), the value it writes into kernels/addnorm.py USE_EW_K3)."""
    global _STARTED, _WORKER
    with _LOCK:
        if _STARTED or not unified:
            return False
        try:
            import torch

            # is_initialized() first: it never creates a context. `is True`: Soup's tests may
            # put a MagicMock torch in sys.modules.
            if torch.cuda.is_initialized() is not True:
                return False
            from triton.runtime.jit import JITFunction

            specs = kernel_specs(addnorm=addnorm, head_ce=head_ce, swiglu=swiglu,
                                 ew_k3=bool(ew_k3))
            if not all(isinstance(jit, JITFunction) and callable(getattr(jit, "warmup", None))
                       for jit, _, _ in specs):
                return False
            device = torch.cuda.current_device()
            if type(device) is not int:
                return False
        except Exception:  # noqa: BLE001 - fake modules in Soup's tests, no Triton, no CUDA
            return False
        _STARTED = True
        _CANCEL.clear()
        STATUS.update(started=True, compiled=[], failed=[], errors={}, launcher=False,
                      busy_s=None, joined_by=None, wait_s=None, timed_out=False)
        try:
            worker = threading.Thread(target=_work, args=(specs, device, time.perf_counter()),
                                      name="soup_precompile", daemon=False)
            worker.start()
        except Exception:  # noqa: BLE001 - no thread: every kernel compiles at its first launch
            STATUS["started"] = False
            return False
        _WORKER = worker
        return True


def wait(by="launch", timeout=None):
    """Join the worker if one is pending; True when nothing is left running. Never raises.

    Called by the first Qwen2ForCausalLM.forward (by="forward") and by every launch helper of ours
    (by="launch"). When nothing is pending this is one global read."""
    global _WORKER
    worker = _WORKER
    if worker is None:
        return True
    if worker is threading.current_thread():  # cannot happen (warmup never reaches our helpers)
        return False
    began = time.perf_counter()
    try:
        worker.join(JOIN_TIMEOUT_S if timeout is None else timeout)
    except Exception:  # noqa: BLE001
        pass
    alive = worker.is_alive()
    if alive:
        _CANCEL.set()  # stop after the kernel in progress; the others compile at first launch
    with _LOCK:
        if _WORKER is worker:  # the first joiner records; later callers see None
            _WORKER = None
            STATUS.update(joined_by=by, wait_s=round(time.perf_counter() - began, 3),
                          timed_out=alive)
    return not alive


def patch_forward_join():
    """Qwen2ForCausalLM.forward joins a pending compile first. Idempotent, no CUDA work; True when
    in place. functools.wraps keeps __wrapped__ (Trainer reads the forward signature through it,
    transformers trainer.py:502 and :1102) and the patch marks of the wrapped forward
    (head_loss)."""
    try:
        from transformers.models.qwen2 import modeling_qwen2

        cls = modeling_qwen2.Qwen2ForCausalLM
        current = cls.forward
        if getattr(current, _FORWARD_MARK, False) is True:
            return True
        if not callable(current):
            return False

        @functools.wraps(current)
        def forward(self, *args, **kwargs):
            if _WORKER is not None:
                wait(by="forward")
            return current(self, *args, **kwargs)

        setattr(forward, _FORWARD_MARK, True)
        cls.forward = forward
        return True
    except Exception:  # noqa: BLE001 - no transformers / no Qwen2: the launch helpers still join
        return False
