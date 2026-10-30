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
    result = {"a1": True, "a": True}

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
    for mod, flag in ((mlp_recompute, "USE_MLP_RECOMPUTE"), (unpad, "USE_UNPAD"),
                      (fused_block, "USE_FUSED_BLOCK"), (saved_compact, "USE_LORA_ROW_COMPACT"),
                      (saved_compact, "USE_SDPA_KV_COMPACT"), (sdpa_recompute, "USE_SDPA_RECOMPUTE"),
                      (lora_replay, "USE_LORA_DROPOUT_REPLAY"), (head_loss, "STORE_HEAD_FP16")):
        monkeypatch.setattr(mod, flag, getattr(mod, flag))
    for flag in ("USE_ADDNORM", "USE_UNPAD", "USE_MLP_RECOMPUTE", "USE_HEAD_CE", "USE_FUSED_BLOCK",
                 "USE_SDPA_RECOMPUTE", "USE_LORA_DROPOUT_REPLAY", "USE_LORA_ROW_COMPACT",
                 "USE_SDPA_KV_COMPACT", "STORE_HEAD_FP16"):
        monkeypatch.setattr(solution, flag, getattr(solution, flag))
    return calls, result


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
