"""K4: the fp16 MLP path gives the same outputs and input gradients as base Qwen2MLP on real
bitsandbytes 4-bit layers, and leaves anything unexpected on the original path.

Run from the repository root: python -m pytest solution -q
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

bnb = pytest.importorskip("bitsandbytes")

from transformers.models.qwen2 import modeling_qwen2  # noqa: E402
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # noqa: E402

import solution.mlp as mlp_patch  # noqa: E402

HIDDEN, INTER = 256, 512


def make_mlp(act="silu", four_bit=True):
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=HIDDEN, intermediate_size=INTER, hidden_act=act)
    mlp = modeling_qwen2.Qwen2MLP(cfg)
    if four_bit:
        for name, (n_in, n_out) in {"gate_proj": (HIDDEN, INTER), "up_proj": (HIDDEN, INTER),
                                    "down_proj": (INTER, HIDDEN)}.items():
            layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                                      compress_statistics=True, quant_type="nf4")
            setattr(mlp, name, layer)
    return mlp.cuda()


@pytest.fixture
def original_forward(monkeypatch):
    current = modeling_qwen2.Qwen2MLP.forward
    original = getattr(current, "__wrapped__", current)
    # Registered with monkeypatch, so the class is restored after each test.
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", original)
    return original


@pytest.mark.parametrize("shape", [(2, 37, HIDDEN), (4, 512, HIDDEN)])
def test_fp16_path_matches_base(original_forward, shape):
    mlp = make_mlp()
    x_ref = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)

    ref = original_forward(mlp, x_ref)
    ref.backward(grad)

    mlp_patch.patch_qwen2_mlp_fp16()
    assert modeling_qwen2.Qwen2MLP.forward is not original_forward
    out = mlp(x_new)
    out.backward(grad)

    assert out.dtype == ref.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x_new.grad, x_ref.grad, rtol=2e-3, atol=2e-3)


def test_patch_is_idempotent(original_forward):
    mlp_patch.patch_qwen2_mlp_fp16()
    first = modeling_qwen2.Qwen2MLP.forward
    mlp_patch.patch_qwen2_mlp_fp16()
    assert modeling_qwen2.Qwen2MLP.forward is first


@pytest.mark.parametrize("case", ["gelu", "plain_linear", "cpu"])
def test_unexpected_inputs_take_the_original_path(original_forward, monkeypatch, case):
    calls = []

    def spy(self, x):
        calls.append(case)
        return original_forward(self, x)

    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", spy)
    mlp_patch.patch_qwen2_mlp_fp16()

    if case == "cpu":
        mlp = make_mlp(four_bit=False)  # plain nn.Linear on CPU
        mlp = mlp.cpu()
        x = torch.randn(1, 3, HIDDEN)
    else:
        mlp = make_mlp(act="gelu" if case == "gelu" else "silu", four_bit=case != "plain_linear")
        x = torch.randn(1, 3, HIDDEN, device="cuda")
    mlp(x)
    assert calls == [case]
