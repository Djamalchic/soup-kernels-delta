"""v10d: solution/precompile.py compiles in ONE background thread exactly what step 1 would compile,
and the first forward (or our first launch) joins it.

CPU part (torch and triton importable, no GPU needed):
- kernel_specs() covers the five launched C1 entries slot by slot (pointer slots get a dtype,
  specialised ints a multiple of 16 other than 1, every int in int32 range, the float slot a float),
  with the production launch options;
- apply() starts the thread after every patch with the run's flags, installs the forward-side join
  only when the thread runs, and USE_PRECOMPILE_OVERLAP = False is v10c's apply();
- start_overlap() does nothing without a CUDA context (fresh process);
- the worker compiles in order, survives a failing kernel, stops at a cancel and never raises;
  wait() joins, records who joined, times out without raising, and is free when idle;
- the Qwen2ForCausalLM.forward join runs before the original forward, keeps its signature and
  head_loss's mark, and is idempotent;
- every `soup_*[grid](...)` launch in solution/kernels/*.py is preceded by `_precompile.wait()` in
  its function, and the launch helpers call it before launching.

GPU part (Kaggle T4, Triton 3.6.0). Each case runs in a fresh Python process with its own empty
TRITON_CACHE_DIR and the harness's TritonWatch, as every measured run does:
- pre_ew: start_overlap(ew_k3=True) + wait(): one daemon=False thread compiled the five kernels and
  is gone; a production-like round of every launch adds no cubin and no launcher (no compile at the
  first launch); one cache entry per kernel; TritonWatch counts each warmup once and times none; no
  GPU memory and no CUDA RNG used by the compile;
- none_ew: the same round without the thread gives the same cache hashes and the same bits;
- apply: the real solution.apply() + wait(): the first real launch adds no cubin;
- apply_lazy: solution.apply() and no explicit join: the round's first launch helper joins;
- apply_forward: solution.apply(), then a (tiny) Qwen2ForCausalLM forward joins the thread.
Run from the repository root (-s prints the timings):
    python -m pytest solution/tests/test_precompile.py -q -s
"""

import ast
import inspect
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import solution  # noqa: E402
from solution import precompile  # noqa: E402

SOLUTION = Path(solution.__file__).resolve().parent
REPO = SOLUTION.parent
KERNELS = SOLUTION / "kernels"
LAUNCHED = {  # name -> file (solution/kernels/__init__.py), K3 pair without P2
    "soup_nf4_dequant_usig": "nf4.py",
    "soup_addnorm_fwd_usig": "addnorm.py",
    "soup_addnorm_bwd_usig": "addnorm.py",
    "soup_head_ce_usig": "head_ce.py",
    "soup_swiglu_kernel": "swiglu.py",
}
# v10 P2 (EW_K3_FP16, on in v10c): the K3 pair is the two *_v10 entries, still five kernels.
LAUNCHED_EW = {name: file for name, file in LAUNCHED.items() if not name.startswith("soup_addnorm")}
LAUNCHED_EW.update({"soup_addnorm_fwd_v10": "addnorm.py", "soup_addnorm_bwd_v10": "addnorm.py"})


def _launched_by_apply():
    """The five entries solution.apply() warms (and the run launches) with its defaults."""
    return LAUNCHED_EW if solution._ew_k3() else LAUNCHED


def _jit_node(file, name):
    tree = ast.parse((KERNELS / file).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            dec = node.decorator_list[0]
            dns = set()
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "do_not_specialize":
                        dns = {elt.value for elt in kw.value.elts}
            return [a.arg for a in node.args.args], dns
    raise AssertionError(f"{file}: {name} not found")


def _name(jit):
    return jit.fn.__name__


# ------------------------------------------------------------------------------------------------
# CPU: specs and wiring
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ew_k3", [False, True])
def test_specs_cover_the_launched_entries_slot_by_slot(ew_k3):
    from solution.kernels import addnorm, head_ce, nf4, swiglu

    launched = LAUNCHED_EW if ew_k3 else LAUNCHED
    specs = precompile.kernel_specs(ew_k3=ew_k3)
    assert sorted(_name(j) for j, _, _ in specs) == sorted(launched)
    for jit, args, opts in specs:
        params, dns = _jit_node(launched[_name(jit)], _name(jit))
        assert len(args) == len(params) == 12
        for slot, (param, value) in enumerate(zip(params, args)):
            if slot < 7:  # pointer: a MockTensor (or a bare dtype) of a torch dtype
                dtype = getattr(value, "dtype", value)
                assert isinstance(dtype, torch.dtype), (param, value)
            elif slot < 11:
                assert type(value) is int and -2**31 <= value < 2**31, param  # int32 slot
                if param not in dns:  # specialised: the production class (multiple of 16, not 1)
                    assert value % 16 == 0 and value != 1, param
            else:
                assert type(value) is float, param
    options = {_name(j): o for j, _, o in specs}
    assert options["soup_nf4_dequant_usig"] == {"num_warps": nf4._NUM_WARPS,
                                                "enable_fp_fusion": False}
    k3 = ("soup_addnorm_fwd_v10", "soup_addnorm_bwd_v10") if ew_k3 else (
        "soup_addnorm_fwd_usig", "soup_addnorm_bwd_usig")
    for name in k3:
        assert options[name] == {"num_warps": addnorm.NUM_WARPS}
    assert options["soup_head_ce_usig"] == {"num_warps": head_ce.NUM_WARPS}
    assert options["soup_swiglu_kernel"] == {"num_warps": swiglu._NUM_WARPS}
    # Optional kernels follow the flags; K1 is always patched.
    only_k1 = precompile.kernel_specs(addnorm=False, head_ce=False, swiglu=False, ew_k3=ew_k3)
    assert [_name(j) for j, _, _ in only_k1] == ["soup_nf4_dequant_usig"]
    f16, f32, u8, i32, i64 = torch.float16, torch.float32, torch.uint8, torch.int32, torch.int64
    dtypes = {_name(j): [getattr(a, "dtype", a) for a in args[:7]] for j, args, _ in specs}
    assert dtypes["soup_nf4_dequant_usig"] == [u8, u8, f32, f32, f32, i32, i32]  # nf4.py:482-494
    assert dtypes["soup_head_ce_usig"] == [f16, i64, f32, f32, f32, f32, f32]  # head_ce.py:196-201
    assert dtypes["soup_swiglu_kernel"] == [f16] * 7  # swiglu.py:95-100
    if ew_k3:  # v10 P2's slot dtypes, as kernels/addnorm.py launches them (see kernel_specs)
        assert dtypes["soup_addnorm_fwd_v10"] == [f32, f16, f32, f32, f32, f32, f16]
        assert dtypes["soup_addnorm_bwd_v10"] == [f16, f32, f32, f16, f32, f32, f32]
    else:
        assert dtypes["soup_addnorm_fwd_usig"] == [f32, f16, f32, f32, f32, f32, f32]
        assert dtypes["soup_addnorm_bwd_usig"] == [f32, f32, f32, f16, f32, f32, f32]


def test_ew_k3_is_the_one_switch(monkeypatch):
    """_ew_k3() is the one place both the warmed K3 pair and addnorm.USE_EW_K3 come from."""
    for master, k3 in ((True, True), (True, False), (False, True), (False, False)):
        monkeypatch.setattr(solution, "USE_ELEMENTWISE_FUSION", master)
        monkeypatch.setattr(solution, "EW_K3_FP16", k3)
        assert solution._ew_k3() is (master and k3)


@pytest.fixture
def wiring(monkeypatch):
    calls = []
    started = {"value": True}
    monkeypatch.setattr(solution, "_apply_patches", lambda: calls.append("patches"))
    monkeypatch.setattr(precompile, "start_overlap",
                        lambda **kw: calls.append(("start", kw)) or started["value"])
    monkeypatch.setattr(precompile, "patch_forward_join",
                        lambda: calls.append("forward_join") or True)
    monkeypatch.setattr(solution, "USE_PRECOMPILE_OVERLAP", solution.USE_PRECOMPILE_OVERLAP)
    return calls, started


def _steps(calls):
    return [c if isinstance(c, str) else c[0] for c in calls]


def test_apply_starts_the_thread_after_every_patch(wiring):
    calls, _ = wiring
    assert solution.USE_PRECOMPILE_OVERLAP is True  # v10d default
    solution.apply()
    assert _steps(calls) == ["patches", "start", "forward_join"]
    assert calls[1][1] == {"addnorm": True, "head_ce": True, "swiglu": True, "unified": True,
                           "ew_k3": solution._ew_k3()}


def test_no_forward_join_without_a_thread(wiring):
    calls, started = wiring
    started["value"] = False  # e.g. Soup's tests: no CUDA context
    solution.apply()
    assert _steps(calls) == ["patches", "start"]


def test_flag_off_is_v10c(wiring, monkeypatch):
    calls, _ = wiring
    monkeypatch.setattr(solution, "USE_PRECOMPILE_OVERLAP", False)
    solution.apply()
    assert calls == ["patches"]


def test_a_failing_patch_starts_nothing(wiring, monkeypatch):
    calls, _ = wiring

    def boom():
        raise RuntimeError("patch failed")

    monkeypatch.setattr(solution, "_apply_patches", boom)
    with pytest.raises(RuntimeError):
        solution.apply()
    assert calls == []


def test_a_failing_start_never_breaks_apply(monkeypatch):
    tried = []
    monkeypatch.setattr(solution, "_apply_patches", lambda: None)
    monkeypatch.setattr(solution, "USE_PRECOMPILE_OVERLAP", True)

    def boom(**kw):
        tried.append(kw)
        raise RuntimeError("no thread")

    monkeypatch.setattr(precompile, "start_overlap", boom)
    solution.apply()  # returns normally: every kernel then compiles at its first launch
    assert len(tried) == 1


def test_start_needs_an_existing_cuda_context():
    code = ("import threading, torch; import solution.precompile as p; "
            "assert p.start_overlap(unified=False) is False and p.start_overlap() is False; "
            "assert not torch.cuda.is_initialized(); "
            "assert p._WORKER is None and p.STATUS['started'] is False; "
            "assert not [t for t in threading.enumerate() if t.name == 'soup_precompile']")
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


# ------------------------------------------------------------------------------------------------
# CPU: the worker, wait() and the forward join
# ------------------------------------------------------------------------------------------------

class _FakeKernel:
    src, metadata = "launcher source", {"meta": 1}


class _FakeJit:
    def __init__(self, name, log, fail=False):
        self.fn = types.SimpleNamespace(__name__=name)
        self.log, self.fail = log, fail

    def warmup(self, *args, grid, **opts):
        self.log.append((self.fn.__name__, args, grid, opts))
        if self.fail:
            raise RuntimeError("no compiler")
        return _FakeKernel()


class _FakeDriver:
    def __init__(self):
        self.devices, self.launchers = 0, []

    def get_current_device(self):
        self.devices += 1
        return 0

    def launcher_cls(self, src, metadata):
        self.launchers.append((src, metadata))


@pytest.fixture
def fresh(monkeypatch):
    """Fresh module state for the worker / wait tests; restored afterwards."""
    monkeypatch.setattr(precompile, "STATUS", dict(precompile.STATUS, compiled=[], failed=[],
                                                   errors={}, launcher=False, joined_by=None,
                                                   wait_s=None, timed_out=False))
    monkeypatch.setattr(precompile, "_CANCEL", threading.Event())
    monkeypatch.setattr(precompile, "_WORKER", None)
    driver = _FakeDriver()
    monkeypatch.setattr(precompile, "_driver", lambda: driver)
    return driver


def test_worker_compiles_in_order_and_survives_a_failure(fresh):
    log = []
    specs = [(_FakeJit("a", log), (1, 2.0), {"num_warps": 4}),
             (_FakeJit("b", log, fail=True), (3,), {"num_warps": 8}),
             (_FakeJit("c", log), (4,), {"num_warps": 4, "enable_fp_fusion": False})]
    precompile._work(specs, None, time.perf_counter())  # never raises
    assert [entry[0] for entry in log] == ["a", "b", "c"]
    assert all(entry[2] == (1,) for entry in log)  # grid=(1,), as P1
    assert log[2][3] == {"num_warps": 4, "enable_fp_fusion": False}  # launch options passed on
    status = precompile.STATUS
    assert status["compiled"] == ["a", "c"] and status["failed"] == ["b"]
    assert "no compiler" in status["errors"]["b"]
    assert status["launcher"] is True and fresh.launchers == [("launcher source", {"meta": 1})]
    assert fresh.devices == 1 and status["busy_s"] is not None


def test_worker_stops_at_a_cancel(fresh):
    log = []
    precompile._CANCEL.set()
    precompile._work([(_FakeJit("a", log), (), {})], None, time.perf_counter())
    assert log == [] and precompile.STATUS["compiled"] == []


def test_worker_without_a_driver_never_raises(fresh, monkeypatch):
    def no_driver():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(precompile, "_driver", no_driver)
    precompile._work([(_FakeJit("a", []), (), {})], None, time.perf_counter())
    assert "no CUDA driver" in precompile.STATUS["errors"]["worker"]


def _held_thread(seconds=5.0):
    gate = threading.Event()
    thread = threading.Thread(target=gate.wait, args=(seconds,), name="soup_precompile_test")
    thread.start()
    return thread, gate


def test_wait_joins_once_and_records(fresh):
    assert precompile.wait() is True  # idle: nothing to join, nothing recorded
    assert precompile.STATUS["joined_by"] is None
    thread, gate = _held_thread()
    precompile._WORKER = thread
    threading.Timer(0.05, gate.set).start()
    assert precompile.wait(by="forward") is True
    assert not thread.is_alive() and precompile._WORKER is None
    assert precompile.STATUS["joined_by"] == "forward" and precompile.STATUS["timed_out"] is False
    assert precompile.wait(by="launch") is True and precompile.STATUS["joined_by"] == "forward"


def test_wait_times_out_without_raising(fresh):
    thread, gate = _held_thread()
    precompile._WORKER = thread
    try:
        assert precompile.wait(by="launch", timeout=0.05) is False
        assert precompile._CANCEL.is_set()  # the worker stops after its current kernel
        assert precompile._WORKER is None and precompile.STATUS["timed_out"] is True
        assert precompile.wait() is True  # never waits twice
    finally:
        gate.set()
        thread.join()


def test_forward_join_runs_first_and_keeps_the_signature(fresh, monkeypatch):
    mq = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    seen = []

    def original(self, input_ids=None, labels=None, **kwargs):
        seen.append(precompile._WORKER is None)
        return input_ids

    original._soup_delta_head_ce = True  # head_loss's mark must stay visible on the wrapper
    monkeypatch.setattr(mq.Qwen2ForCausalLM, "forward", original)  # restored after the test
    assert precompile.patch_forward_join() is True
    patched = mq.Qwen2ForCausalLM.forward
    assert patched is not original and patched.__wrapped__ is original
    assert precompile.patch_forward_join() is True and mq.Qwen2ForCausalLM.forward is patched
    assert getattr(patched, "_soup_delta_head_ce", False) is True
    assert inspect.signature(patched) == inspect.signature(original)  # Trainer reads it
    thread, gate = _held_thread()
    precompile._WORKER = thread
    threading.Timer(0.05, gate.set).start()
    assert patched(object(), 7) == 7
    assert seen == [True] and not thread.is_alive()  # joined before the original forward ran
    assert precompile.STATUS["joined_by"] == "forward"
    assert patched(object(), input_ids=3) == 3 and seen == [True, True]


def _launch_sites(tree):
    """(function, call) for every `soup_*[grid](...)` launch in a module."""
    sites = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id.startswith("soup_")):
                sites.append((fn, node))
    return sites


def _waits(fn):
    return [node.lineno for node in ast.walk(fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "wait" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_precompile"]


def test_every_launch_site_waits_for_the_background_compile():
    count = 0
    for path in sorted(KERNELS.glob("*.py")):
        for fn, call in _launch_sites(ast.parse(path.read_text(encoding="utf-8"))):
            count += 1
            waits = _waits(fn)
            assert waits and min(waits) < call.lineno, f"{path.name}:{call.lineno} ({fn.name})"
    assert count >= 13  # nf4 2, addnorm 8, head_ce 2, swiglu 1 (v10c)


def test_launch_helpers_join_before_launching(monkeypatch):
    from solution.kernels import head_ce, nf4, swiglu

    order = []
    monkeypatch.setattr(precompile, "wait", lambda *a, **k: order.append("wait") or True)

    class FakeLaunch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kwargs: order.append(self.name)

    t = torch.zeros(32, dtype=torch.float16)
    monkeypatch.setattr(swiglu, "soup_swiglu_kernel", FakeLaunch("swiglu"))
    swiglu._launch(0, t, t, t, t, t)
    monkeypatch.setattr(head_ce, "USE_UNIFIED_SIG", True)
    monkeypatch.setattr(head_ce, "soup_head_ce_usig", FakeLaunch("head_ce"))
    head_ce._launch((1,), t, t, t, t, t, 0, 16, 0, 0)
    monkeypatch.setattr(nf4, "USE_UNIFIED_SIG", True)
    monkeypatch.setattr(nf4, "soup_nf4_dequant_usig", FakeLaunch("nf4"))
    state2 = types.SimpleNamespace(absmax=t, code=t)
    state = types.SimpleNamespace(absmax=t, offset=t, state2=state2)
    nf4._launch_here(t, state, 64, t)
    assert order == ["wait", "swiglu", "wait", "head_ce", "wait", "nf4"]


# ------------------------------------------------------------------------------------------------
# GPU: one fresh process per case, empty Triton cache, the harness's TritonWatch
# ------------------------------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

PROBE = r'''
import glob, hashlib, json, os, sys, threading, time, types
import torch

repo, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
torch.empty(1, device="cuda")  # the harness creates the context before apply() (_train.py:120)
cache = os.environ["TRITON_CACHE_DIR"]

watch = None
harness = os.path.join(repo, "harness")
if not mode.startswith("none") and os.path.exists(os.path.join(harness, "_train.py")):
    sys.path.insert(0, harness)
    import _train  # the jury's TritonWatch, patching JITFunction.run as in the measured runs
    watch = _train.TritonWatch(torch)


def files():
    found = {}
    for pattern in ("*.cubin", "__triton_launcher*.so", "cuda_utils*.so"):
        paths = glob.glob(os.path.join(cache, "**", pattern), recursive=True)
        found[pattern] = sorted(os.path.basename(os.path.dirname(p)) for p in paths)
    return found


def ours_alive():
    return [t.name for t in threading.enumerate() if t.name == "soup_precompile" and t.is_alive()]


torch.cuda.synchronize()
mem0 = [torch.cuda.memory_allocated(), torch.cuda.memory_reserved()]
rng0 = torch.cuda.get_rng_state()
t0 = time.perf_counter()
if mode.startswith("apply"):
    import solution
    solution.apply()
elif mode == "pre_ew":
    from solution import precompile
    precompile.start_overlap(ew_k3=True)
start_s = round(time.perf_counter() - t0, 3)
from solution import precompile

worker = precompile._WORKER
alive_at_return = bool(worker is not None and worker.is_alive())
daemon = None if worker is None else worker.daemon
alive_after_forward = None
if mode == "apply_forward":
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                      use_cache=False, attn_implementation="sdpa")
    tiny = Qwen2ForCausalLM(cfg).to("cuda").eval()
    with torch.no_grad():  # not Qwen2.5 width: our patches fall back, only the join runs
        tiny(input_ids=torch.ones(1, 8, dtype=torch.long, device="cuda"))
    alive_after_forward = bool(ours_alive())
elif mode in ("apply", "pre_ew"):
    precompile.wait(by="explicit")
joined_s = round(time.perf_counter() - t0, 3)
lazy = mode == "apply_lazy"  # nothing joined yet: the round's first launch helper must do it
torch.cuda.synchronize()
mem1 = [torch.cuda.memory_allocated(), torch.cuda.memory_reserved()]
rng_same = bool(torch.equal(rng0, torch.cuda.get_rng_state()))
after_pre = None if lazy else files()

from solution.kernels import addnorm, head_ce, nf4
from solution.kernels import swiglu as ours

for mod in (addnorm, head_ce, nf4):
    mod.USE_UNIFIED_SIG = True  # C1, as apply() sets it
if mode.endswith("_ew"):
    addnorm.USE_EW_K3 = True  # v10 P2, as apply() sets it with EW_K3_FP16
ew = bool(getattr(addnorm, "USE_EW_K3", False))  # modes "apply*": what solution.apply() set
k3_pair = (("soup_addnorm_fwd_v10", "soup_addnorm_bwd_v10") if ew
           else ("soup_addnorm_fwd_usig", "soup_addnorm_bwd_usig"))
kernels = {"soup_nf4_dequant_usig": nf4.soup_nf4_dequant_usig,
           k3_pair[0]: getattr(addnorm, k3_pair[0]),
           k3_pair[1]: getattr(addnorm, k3_pair[1]),
           "soup_head_ce_usig": head_ce.soup_head_ce_usig,
           "soup_swiglu_kernel": ours.soup_swiglu_kernel}


def cache_sizes():
    """Compiled variants per kernel in the JITFunction's in-memory cache (device_caches: device ->
    tuple whose [0] is the kernel cache); None for an unknown layout."""
    out = {}
    for name, jit in kernels.items():
        caches = getattr(jit, "device_caches", None)
        total = 0 if isinstance(caches, dict) else None
        for entry in (list(caches.values()) if isinstance(caches, dict) else ()):
            entry = entry[0] if isinstance(entry, tuple) else entry
            total = total + len(entry) if isinstance(entry, dict) and total is not None else None
        out[name] = total
    return out


sizes_pre = None if lazy else cache_sizes()
launches_pre = dict(watch.launches) if (watch and not lazy) else None
events_pre = len(watch.events) if (watch and not lazy) else None

digest = hashlib.sha256()


def keep(*ts):
    for t in ts:
        digest.update(t.detach().float().cpu().numpy().tobytes())


took = {}


def timed(name, fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    took[name] = round(time.perf_counter() - start, 3)


gen = torch.Generator(device="cpu").manual_seed(0)
rows, width = 300, addnorm.HIDDEN_SIZE
x = torch.randn(rows, width, generator=gen).to("cuda")
d16 = torch.randn(rows, width, generator=gen).to("cuda", torch.float16)
w = (torch.rand(width, generator=gen) + 0.5).to("cuda")


def norms():
    for out in (addnorm._launch_fwd(x, d16, w, 1e-6), addnorm._launch_fwd(x, None, w, 1e-6),
                addnorm._launch_fwd(x, d16, w, 1e-6, store_r=False)):
        keep(*[t for t in out if t is not None])


timed("addnorm_fwd", norms)  # the first launch of the round (apply_lazy: joins the thread here)
_, _, rstd = addnorm._launch_fwd(x, None, w, 1e-6)
dy = torch.randn(rows, width, generator=gen).to("cuda")
timed("addnorm_bwd", lambda: keep(addnorm._launch_bwd(dy, dy, x, w, rstd),
                                  addnorm._launch_bwd(dy, None, x, w, rstd, delta=d16)))

if ew:  # v10 P2's other K3 call forms (B's fp16 forms, the LoRA epilogue): the same two entries
    v16 = d16[:, :256].contiguous()  # a v_proj-wide LoRA output

    def p2_forms():
        keep(*[t for t in addnorm.launch_fwd16(x, d16, w, 1e-6) if t is not None])
        keep(addnorm.launch_bwd16(d16.clone(), dy, x, w, rstd, delta=d16, store_da=True))
        for b16 in (d16, v16):
            out = addnorm.lora_epilogue_fwd(b16, b16.clone(), 2.0)
            keep(out, *addnorm.lora_epilogue_bwd(out, 2.0))

    timed("addnorm_v10", p2_forms)

tokens, hidden, vocab = 129, 128, 4096
hid = torch.randn(2, tokens, hidden, generator=gen).to("cuda").requires_grad_(True)
weight = (torch.randn(vocab, hidden, generator=gen) * 0.05).to("cuda")
labels = torch.randint(0, vocab, (2, tokens), generator=gen).to("cuda")
labels[:, :3] = -100
mask = torch.ones(2, tokens, device="cuda", dtype=torch.long)
mask[1, -5:] = 0


def head():
    loss = head_ce.SoupHeadCE.apply(hid, weight, labels, mask, None)[0]
    loss.backward()
    keep(loss, hid.grad)


timed("head_ce", head)

n_out, k_in = 1536, 1536
count = n_out * k_in
blocks = count // 64
packed = torch.randint(0, 256, (count // 2,), dtype=torch.uint8, generator=gen).to("cuda")
absmax = torch.randint(0, 256, (blocks,), dtype=torch.uint8, generator=gen).to("cuda")
state2 = types.SimpleNamespace(absmax=torch.rand((blocks + 255) // 256, generator=gen).to("cuda"),
                               code=torch.randn(256, generator=gen).to("cuda"))
state = types.SimpleNamespace(absmax=absmax, state2=state2,
                              offset=torch.rand(1, generator=gen).to("cuda"))
dense = torch.empty((n_out, k_in), device="cuda", dtype=torch.float16)
timed("nf4", lambda: (nf4._launch(packed, state, count, dense), keep(dense)))

g = torch.randn(tokens, 8960, generator=gen).to("cuda", torch.float16).requires_grad_(True)
u = torch.randn(tokens, 8960, generator=gen).to("cuda", torch.float16).requires_grad_(True)
dh = torch.randn(tokens, 8960, generator=gen).to("cuda", torch.float16)


def act():
    h = ours.swiglu(g, u)
    h.backward(dh)
    keep(h, g.grad, u.grad)


timed("swiglu", act)
g2, u2, dh2 = g.detach().clone(), u.detach().clone(), dh.clone()
timed("swiglu_inplace", lambda: keep(*ours.swiglu_backward_into(dh2, g2, u2)))

print("PROBE " + json.dumps({
    "start_s": start_s, "joined_s": joined_s, "status": precompile.STATUS, "ew": ew,
    "alive_at_return": alive_at_return, "daemon": daemon,
    "alive_after_forward": alive_after_forward, "alive_after": ours_alive(),
    "worker_left": precompile._WORKER is not None, "mem0": mem0, "mem1": mem1,
    "rng_same": rng_same, "after_pre": after_pre, "after_round": files(), "sizes_pre": sizes_pre,
    "sizes_round": cache_sizes(), "launches_pre": launches_pre, "events_pre": events_pre,
    "round": took, "digest": digest.hexdigest()}))
'''


def _probe(tmp_path, mode):
    cache = tmp_path / f"triton_cache_{mode}"
    cache.mkdir()
    env = dict(os.environ, TRITON_CACHE_DIR=str(cache))
    proc = subprocess.run([sys.executable, "-c", PROBE, str(REPO), mode], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [s for s in proc.stdout.splitlines() if s.startswith("PROBE ")][-1]
    result = json.loads(line[len("PROBE "):])
    status = result["status"]
    print(f"\n{mode}: start {result['start_s']} s (thread alive at return: "
          f"{result['alive_at_return']}), joined after {result['joined_s']} s, worker busy "
          f"{status['busy_s']} s, join waited {status['wait_s']} s by {status['joined_by']}, "
          f"launcher prebuilt {status['launcher']}, errors {status['errors']}; first round "
          f"{sum(result['round'].values()):.3f} s {result['round']}")
    return result


def _gone(r):
    """The thread is joined and finished; nothing of it is left."""
    assert not r["alive_after"] and not r["worker_left"], r
    assert r["status"]["timed_out"] is False


@cuda
def test_first_launch_after_the_thread_compiles_nothing(tmp_path):
    r = _probe(tmp_path, "pre_ew")
    status = r["status"]
    assert sorted(status["compiled"]) == sorted(LAUNCHED_EW) and not status["failed"], status
    assert r["daemon"] is False and status["joined_by"] == "explicit"
    _gone(r)
    assert len(r["after_pre"]["*.cubin"]) == len(LAUNCHED_EW)
    assert len(r["after_pre"]["__triton_launcher*.so"]) == (1 if status["launcher"] else 0)
    assert len(r["after_pre"]["cuda_utils*.so"]) <= 1
    # The production round compiles nothing: no new cubin; at most the launcher if the prebuild
    # was skipped. Exactly one launcher in all (C1).
    assert r["after_round"]["*.cubin"] == r["after_pre"]["*.cubin"]
    assert len(r["after_round"]["__triton_launcher*.so"]) == 1
    if all(v is not None for v in r["sizes_pre"].values()):
        assert r["sizes_pre"] == r["sizes_round"] == {name: 1 for name in LAUNCHED_EW}
    if r["launches_pre"] is not None:  # the harness's TritonWatch: once each, never timed
        print(f"TritonWatch after the thread: {r['launches_pre']} (warmup goes through run: "
              f"expected 1 per kernel, harness/_train.py:63)")
        assert set(r["launches_pre"]) <= set(LAUNCHED_EW)
        assert all(count == 1 for count in r["launches_pre"].values())
        assert r["events_pre"] == 0
    assert r["mem1"] == r["mem0"] and r["rng_same"]  # no GPU memory, no CUDA RNG draw


@cuda
def test_same_binaries_and_bits_as_without_the_thread(tmp_path):
    pre = _probe(tmp_path, "pre_ew")
    none = _probe(tmp_path, "none_ew")
    assert pre["ew"] is True and none["ew"] is True
    assert none["status"]["started"] is False
    # Same cache keys (directory = hash of source, specialisation, options) -> same binaries.
    assert pre["after_round"]["*.cubin"] == none["after_round"]["*.cubin"]
    launcher = "__triton_launcher*.so"
    assert pre["after_round"][launcher] == none["after_round"][launcher]
    assert pre["digest"] == none["digest"]


def _apply_case(tmp_path, mode, joined_by):
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    r = _probe(tmp_path, mode)
    launched = _launched_by_apply()  # the v10 K3 pair while P2's EW_K3_FP16 is on (v10c)
    assert r["ew"] is solution._ew_k3()  # the round launches what apply() switched on
    status = r["status"]
    assert status["started"] is True and r["daemon"] is False
    assert sorted(status["compiled"]) == sorted(launched) and not status["failed"], status
    assert status["joined_by"] == joined_by
    _gone(r)
    if r["after_pre"] is not None:  # joined before the round: everything already on disk
        assert len(r["after_pre"]["*.cubin"]) == len(launched)
        assert r["after_round"]["*.cubin"] == r["after_pre"]["*.cubin"]
    assert len(r["after_round"]["*.cubin"]) == len(launched)  # nothing compiled at a launch
    assert len(r["after_round"]["__triton_launcher*.so"]) == 1
    if all(v is not None for v in r["sizes_round"].values()):
        assert r["sizes_round"] == {name: 1 for name in launched}
    return r


@cuda
def test_apply_then_join_first_launch_adds_no_cubin(tmp_path):
    _apply_case(tmp_path, "apply", "explicit")


@cuda
def test_apply_without_join_first_launch_joins(tmp_path):
    _apply_case(tmp_path, "apply_lazy", "launch")


@cuda
def test_apply_first_forward_joins(tmp_path):
    pytest.importorskip("transformers")
    r = _apply_case(tmp_path, "apply_forward", "forward")
    assert r["alive_after_forward"] is False  # the thread never outlives the first forward
