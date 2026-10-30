"""v6 merge: what solution.apply() switches on, with every patch function replaced by a recorder.

No GPU and no real patching: the patch functions are monkeypatched, so this only checks the wiring
in solution/__init__.py (A1 and A are never both installed, A2 / B / D / head flags as documented).
Run from the repository root: python -m pytest solution -q
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import solution  # noqa: E402
from solution import (  # noqa: E402
    fused_block,
    head_loss,
    lora_recompute,
    lora_replay,
    mlp_recompute,
    saved_compact,
    sdpa_recompute,
    unpad,
)


@pytest.fixture
def recorder(monkeypatch):
    """Replace every patch function apply() calls; restore every flag it writes afterwards."""
    calls = []
    result = {"a1": True, "a": True, "m2": True}

    def rec(name, value=True):
        def fn(*a, **k):
            calls.append(name)
            return value() if callable(value) else value
        return fn

    for name in ("patch_bnb_matmul_4bit", "patch_qwen2_mlp_fp16", "patch_qwen2_rmsnorm",
                 "patch_qwen2_decoder_layer", "patch_head_loss"):
        monkeypatch.setattr(solution, name, rec(name))
    monkeypatch.setattr(mlp_recompute, "patch_qwen2_mlp_recompute", rec("s4"))
    monkeypatch.setattr(unpad, "patch_qwen2_model_unpad", rec("u"))
    monkeypatch.setattr(sdpa_recompute, "patch_sdpa_recompute", rec("a1", lambda: result["a1"]))
    monkeypatch.setattr(saved_compact, "patch_sdpa_kv_compact", rec("a", lambda: result["a"]))
    monkeypatch.setattr(lora_replay, "patch_lora_dropout_replay", rec("a2"))
    monkeypatch.setattr(lora_recompute, "patch_lora_input_recompute",  # v9 M2
                        rec("m2", lambda: result["m2"]))
    for mod, flag in ((mlp_recompute, "USE_MLP_RECOMPUTE"), (mlp_recompute, "S4_DEQUANT_REUSE"),
                      (unpad, "USE_UNPAD"), (unpad, "S4_OFF_FRACTION"),
                      (fused_block, "USE_FUSED_BLOCK"), (saved_compact, "USE_LORA_ROW_COMPACT"),
                      (saved_compact, "USE_SDPA_KV_COMPACT"), (sdpa_recompute, "USE_SDPA_RECOMPUTE"),
                      (sdpa_recompute, "USE_CAUSAL_PADDED"), (sdpa_recompute, "USE_ROW_COMPACT"),
                      (lora_replay, "USE_LORA_DROPOUT_REPLAY"), (head_loss, "STORE_HEAD_FP16"),
                      (lora_recompute, "USE_LORA_INPUT_RECOMPUTE")):
        monkeypatch.setattr(mod, flag, getattr(mod, flag))
    from solution import mlp
    from solution.kernels import addnorm, head_ce, nf4  # v7 I3 / v8 C1: apply() writes these too

    for flag in ("USE_K1_FAST_HOST", "USE_DENSE_CACHE", "DENSE_CACHE_QO", "DENSE_CACHE_MIB"):
        monkeypatch.setattr(nf4, flag, getattr(nf4, flag))
    for mod in (nf4, addnorm, head_ce):  # v8 C1
        monkeypatch.setattr(mod, "USE_UNIFIED_SIG", getattr(mod, "USE_UNIFIED_SIG"))
    monkeypatch.setattr(mlp, "USE_SOUP_SWIGLU", mlp.USE_SOUP_SWIGLU)  # v8 C2
    for flag in ("BWD_ROW_CHUNKS", "BWD_CHUNK_MIN_ROWS"):  # v9 M5
        monkeypatch.setattr(mlp_recompute, flag, getattr(mlp_recompute, flag))
    monkeypatch.setattr(head_ce, "VOCAB_TILE", head_ce.VOCAB_TILE)
    for flag in ("USE_ADDNORM", "USE_UNPAD", "USE_MLP_RECOMPUTE", "USE_HEAD_CE", "USE_FUSED_BLOCK",
                 "USE_SDPA_RECOMPUTE", "USE_SDPA_CAUSAL_PADDED", "USE_LORA_DROPOUT_REPLAY",
                 "USE_LORA_ROW_COMPACT", "USE_SDPA_KV_COMPACT", "STORE_HEAD_FP16", "S4_DEQUANT_REUSE",
                 "S4_OFF_FRACTION", "USE_K1_FAST_HOST", "USE_DENSE_CACHE", "DENSE_CACHE_QO",
                 "DENSE_CACHE_MIB", "USE_UNIFIED_SIG", "USE_SOUP_SWIGLU",
                 "USE_LORA_INPUT_RECOMPUTE",  # v9 M2
                 "USE_SDPA_ROW_COMPACT",  # v9 M3
                 "USE_BWD_CHUNKING", "BWD_ROW_CHUNKS", "BWD_CHUNK_MIN_ROWS", "K2_VOCAB_TILE"):  # M5
        monkeypatch.setattr(solution, flag, getattr(solution, flag))
    return calls, result


@pytest.mark.parametrize("unified, soup", [(True, True), (False, False), (True, False)])
def test_v8_flags_reach_the_modules(recorder, monkeypatch, unified, soup):
    """C1 / C2: apply() copies USE_UNIFIED_SIG into K1 / K2 / K3 and USE_SOUP_SWIGLU into K4
    (which S4 reads); the module defaults are the v7 launches."""
    from solution import mlp
    from solution.kernels import addnorm, head_ce, nf4

    assert solution.USE_UNIFIED_SIG is True and solution.USE_SOUP_SWIGLU is True
    monkeypatch.setattr(solution, "USE_UNIFIED_SIG", unified)
    monkeypatch.setattr(solution, "USE_SOUP_SWIGLU", soup)
    solution.apply()
    assert nf4.USE_UNIFIED_SIG is addnorm.USE_UNIFIED_SIG is head_ce.USE_UNIFIED_SIG is unified
    assert mlp.USE_SOUP_SWIGLU is soup


@pytest.mark.parametrize("on", [True, False])
def test_m5_flags_reach_the_modules(recorder, monkeypatch, on):
    """v9 M5: apply() writes S4's row chunks and K2's vocabulary tile; off = v8b's values. The
    module defaults (before apply()) are v8b's."""
    from solution.kernels import head_ce

    assert solution.USE_BWD_CHUNKING is False and solution.BWD_ROW_CHUNKS == 4  # M5 off in v9b
    assert solution.BWD_CHUNK_MIN_ROWS == 512 and solution.K2_VOCAB_TILE == 8192
    assert head_ce.VOCAB_TILE_V8 == 16384
    assert mlp_recompute.BWD_ROW_CHUNKS == 1 and head_ce.VOCAB_TILE == 16384
    monkeypatch.setattr(solution, "USE_BWD_CHUNKING", on)
    solution.apply()
    assert mlp_recompute.BWD_ROW_CHUNKS == (4 if on else 1)
    assert mlp_recompute.BWD_CHUNK_MIN_ROWS == 512
    assert head_ce.VOCAB_TILE == (8192 if on else 16384)


def test_m5_row_bounds(monkeypatch):
    """v9 M5 chunk boundaries (CPU tensors: _rows_bounds only reads metadata)."""
    import torch

    def bounds(n, inter=8960, width=1536, dtype=torch.float16, lead=None):
        lead = lead or (n,)
        return mlp_recompute._rows_bounds(torch.empty(*lead, inter, dtype=dtype),
                                          torch.empty(*lead, width, dtype=dtype))

    monkeypatch.setattr(mlp_recompute, "BWD_CHUNK_MIN_ROWS", 512)
    monkeypatch.setattr(mlp_recompute, "BWD_ROW_ALIGN", 128)
    monkeypatch.setattr(mlp_recompute, "BWD_ROW_CHUNKS", 1)
    assert bounds(1350) is None  # v8b
    monkeypatch.setattr(mlp_recompute, "BWD_ROW_CHUNKS", True)
    assert bounds(1350) is None
    monkeypatch.setattr(mlp_recompute, "BWD_ROW_CHUNKS", 4)
    assert bounds(511) is None  # under BWD_CHUNK_MIN_ROWS
    assert bounds(512) == [0, 128, 256, 384, 512]
    assert bounds(513) == [0, 256, 513]  # a 1-row tail is merged, never its own GEMM
    assert bounds(1350) == [0, 384, 768, 1152, 1350]
    assert bounds(2048) == [0, 512, 1024, 1536, 2048]
    assert bounds(700, lead=(2, 350)) == [0, 256, 512, 700]  # [B, T, C] flattens to rows
    assert bounds(1350, dtype=torch.float32) is None
    x = torch.empty(1350, 1536, dtype=torch.float16)
    dh = torch.empty(1350, 8960, dtype=torch.float16)
    assert mlp_recompute._rows_bounds(dh, x.t().contiguous().t()) is None  # non-contiguous x16
    assert mlp_recompute._rows_bounds(dh[:1349], x) is None  # row counts differ
    monkeypatch.setattr(mlp_recompute, "BWD_ROW_CHUNKS", 2)
    assert bounds(600) == [0, 384, 600]
    monkeypatch.setattr(mlp_recompute, "BWD_CHUNK_MIN_ROWS", 0)
    assert bounds(129) is None  # one 128-row chunk + a 1-row tail -> merged -> nothing to chunk


def test_defaults_are_the_v6_merge():
    assert solution.USE_FUSED_BLOCK and solution.USE_SDPA_RECOMPUTE
    assert solution.USE_LORA_DROPOUT_REPLAY and solution.USE_LORA_ROW_COMPACT
    assert solution.USE_SDPA_KV_COMPACT is False  # A: fallback for A1 only
    assert solution.STORE_HEAD_FP16 is False  # open rule question (PLAN Q2)
    # Module defaults of the head store and A stay off until apply() runs.
    assert head_loss.STORE_HEAD_FP16 is False


def test_default_apply_installs_a1_and_never_a(recorder):
    calls, _ = recorder
    solution.apply()
    assert "a1" in calls and "a" not in calls
    assert sdpa_recompute.USE_SDPA_RECOMPUTE is True
    assert saved_compact.USE_SDPA_KV_COMPACT is False
    assert saved_compact.USE_LORA_ROW_COMPACT is True
    assert fused_block.USE_FUSED_BLOCK is True
    assert lora_replay.USE_LORA_DROPOUT_REPLAY is True and "a2" in calls
    assert head_loss.STORE_HEAD_FP16 is False
    assert mlp_recompute.S4_DEQUANT_REUSE == "both"  # v7 I2
    assert calls.index("u") < calls.index("a1")  # U (and with it A / D's scope) before attention patches


def test_a1_refused_and_a_off_installs_neither(recorder):
    calls, result = recorder
    result["a1"] = False
    solution.apply()
    assert "a1" in calls and "a" not in calls
    assert sdpa_recompute.USE_SDPA_RECOMPUTE is False
    assert saved_compact.USE_SDPA_KV_COMPACT is False


def test_a_is_the_fallback_when_a1_is_refused(recorder, monkeypatch):
    calls, result = recorder
    result["a1"] = False
    monkeypatch.setattr(solution, "USE_SDPA_KV_COMPACT", True)
    solution.apply()
    assert calls.count("a") == 1
    assert sdpa_recompute.USE_SDPA_RECOMPUTE is False
    assert saved_compact.USE_SDPA_KV_COMPACT is True


def test_a_never_with_a1_even_if_both_flags_are_on(recorder, monkeypatch):
    calls, _ = recorder
    monkeypatch.setattr(solution, "USE_SDPA_KV_COMPACT", True)
    solution.apply()
    assert "a" not in calls
    assert saved_compact.USE_SDPA_KV_COMPACT is False and sdpa_recompute.USE_SDPA_RECOMPUTE is True


def test_a1_off_is_not_even_patched(recorder, monkeypatch):
    calls, _ = recorder
    monkeypatch.setattr(solution, "USE_SDPA_RECOMPUTE", False)
    solution.apply()
    assert "a1" not in calls and "a" not in calls
    assert sdpa_recompute.USE_SDPA_RECOMPUTE is False


def test_without_u_b_d_and_a_stay_off(recorder, monkeypatch):
    calls, result = recorder
    result["a1"] = False
    monkeypatch.setattr(solution, "USE_UNPAD", False)
    monkeypatch.setattr(solution, "USE_SDPA_KV_COMPACT", True)
    solution.apply()
    assert "u" not in calls and "a" not in calls
    assert fused_block.USE_FUSED_BLOCK is False
    assert saved_compact.USE_LORA_ROW_COMPACT is False
    assert saved_compact.USE_SDPA_KV_COMPACT is False


def test_b_needs_k3_and_s4(recorder, monkeypatch):
    monkeypatch.setattr(solution, "USE_MLP_RECOMPUTE", False)
    solution.apply()
    assert fused_block.USE_FUSED_BLOCK is False


def test_head_flag_is_forwarded(recorder, monkeypatch):
    monkeypatch.setattr(solution, "STORE_HEAD_FP16", True)
    solution.apply()
    assert head_loss.STORE_HEAD_FP16 is True


# v7 I1: causal-only SDPA on U's packed path, wired as A1 + U + its own flag.

def test_i1_default_is_on_with_a1_and_u(recorder):
    assert solution.USE_SDPA_CAUSAL_PADDED is True
    solution.apply()
    assert sdpa_recompute.USE_CAUSAL_PADDED is True


@pytest.mark.parametrize("case", ["a1_refused", "a1_off", "u_off", "flag_off"])
def test_i1_off_without_its_prerequisites(recorder, monkeypatch, case):
    _, result = recorder
    if case == "a1_refused":
        result["a1"] = False
    elif case == "a1_off":
        monkeypatch.setattr(solution, "USE_SDPA_RECOMPUTE", False)
    elif case == "u_off":
        monkeypatch.setattr(solution, "USE_UNPAD", False)
    else:
        monkeypatch.setattr(solution, "USE_SDPA_CAUSAL_PADDED", False)
    solution.apply()
    assert sdpa_recompute.USE_CAUSAL_PADDED is False


# v9 M2: LoRA-A input recompute, wired after A2 (its class patch must be the outer one), needs U
# and K3 (the source scope lives in U's packed path and replays K3's norm).

def test_m2_default_is_on_and_installed_after_a2(recorder):
    calls, _ = recorder
    assert solution.USE_LORA_INPUT_RECOMPUTE is True
    solution.apply()
    assert lora_recompute.USE_LORA_INPUT_RECOMPUTE is True
    assert calls.index("a2") < calls.index("m2")


@pytest.mark.parametrize("case", ["refused", "u_off", "k3_off", "flag_off"])
def test_m2_off_without_its_prerequisites(recorder, monkeypatch, case):
    calls, result = recorder
    if case == "refused":
        result["m2"] = False
    elif case == "u_off":
        monkeypatch.setattr(solution, "USE_UNPAD", False)
    elif case == "k3_off":
        monkeypatch.setattr(solution, "USE_ADDNORM", False)
    else:
        monkeypatch.setattr(solution, "USE_LORA_INPUT_RECOMPUTE", False)
    solution.apply()
    assert lora_recompute.USE_LORA_INPUT_RECOMPUTE is False
    assert ("m2" in calls) == (case == "refused")


# v9 M3: A1 saves real rows only on U's packed path, wired as A1 + U + its own flag; I1 independent.

def test_m3_default_is_on_with_a1_and_u(recorder):
    assert solution.USE_SDPA_ROW_COMPACT is True
    solution.apply()
    assert sdpa_recompute.USE_ROW_COMPACT is True


def test_m3_works_without_i1(recorder, monkeypatch):
    monkeypatch.setattr(solution, "USE_SDPA_CAUSAL_PADDED", False)
    solution.apply()
    assert sdpa_recompute.USE_CAUSAL_PADDED is False
    assert sdpa_recompute.USE_ROW_COMPACT is True


@pytest.mark.parametrize("case", ["a1_refused", "a1_off", "u_off", "flag_off"])
def test_m3_off_without_its_prerequisites(recorder, monkeypatch, case):
    _, result = recorder
    if case == "a1_refused":
        result["a1"] = False
    elif case == "a1_off":
        monkeypatch.setattr(solution, "USE_SDPA_RECOMPUTE", False)
    elif case == "u_off":
        monkeypatch.setattr(solution, "USE_UNPAD", False)
    else:
        monkeypatch.setattr(solution, "USE_SDPA_ROW_COMPACT", False)
    solution.apply()
    assert sdpa_recompute.USE_ROW_COMPACT is False
