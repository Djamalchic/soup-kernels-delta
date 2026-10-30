"""Run from the repository root with: python -m pytest solution/tests/test_nf4.py"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

pytest.importorskip("triton")
bnb = pytest.importorskip("bitsandbytes")

import solution.kernels.nf4 as nf4


ORIGINAL = nf4._get_original_matmul_4bit()

SHAPES = (
    (1536, 1536),
    (256, 1536),
    (8960, 1536),
    (1536, 8960),
)


def quantize(weight, *, blocksize=64, nested=True):
    return bnb.functional.quantize_4bit(
        weight,
        blocksize=blocksize,
        compress_statistics=nested,
        quant_type="nf4",
    )


def assert_bit_exact(actual, expected):
    assert actual.dtype == expected.dtype == torch.float16
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    # torch.equal alone does not distinguish positive and negative zero.
    assert torch.equal(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )


@pytest.fixture(scope="module")
def quantized_weights():
    generator = torch.Generator(device="cuda").manual_seed(20261030)
    result = {}
    for shape in SHAPES:
        weight = torch.randn(
            shape,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        result[shape] = quantize(weight)
        del weight
    return result


@pytest.mark.parametrize("shape", SHAPES)
def test_dequant_bit_exact(quantized_weights, shape):
    packed, state = quantized_weights[shape]
    actual = nf4.nf4_dequantize(packed, state)
    expected = bnb.functional.dequantize_4bit(packed, quant_state=state)
    assert_bit_exact(actual, expected)


def test_tail_and_odd_element_count():
    # Quantize an even-sized input, then interpret its first 349 elements as
    # a logical weight. The last byte's low nibble is deliberately unused.
    # This also avoids relying on the quantizer's odd-length input support.
    import copy

    generator = torch.Generator(device="cuda").manual_seed(41)
    weight = torch.randn(
        (1, 350),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    packed, state = quantize(weight)
    expected = bnb.functional.dequantize_4bit(
        packed, quant_state=state
    )[:, :349].contiguous()

    odd_state = copy.copy(state)
    odd_state.shape = torch.Size((1, 349))

    # Sentinels check that the final partially filled program stays in bounds.
    backing = torch.full(
        (351,), 123.0, device="cuda", dtype=torch.float16
    )
    output = backing[1:-1].view(1, 349)
    result = nf4.nf4_dequantize(packed, odd_state, out=output)

    assert result is output
    assert_bit_exact(result, expected)
    assert backing[0].item() == 123.0
    assert backing[-1].item() == 123.0


def test_forward_backward_matches_original(quantized_weights, monkeypatch):
    packed, state = quantized_weights[(256, 1536)]
    generator = torch.Generator(device="cuda").manual_seed(73)

    A_ref = torch.randn(
        (4, 512, 1536),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
        requires_grad=True,
    )
    A_soup = A_ref.detach().clone().requires_grad_(True)

    bias_ref = torch.randn(
        (256,),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
        requires_grad=True,
    )
    bias_soup = bias_ref.detach().clone().requires_grad_(True)

    reference = ORIGINAL(A_ref, packed, quant_state=state, bias=bias_ref)

    # M=2048 must use our path without consulting the private heuristic.
    import bitsandbytes.backends.cuda.ops as cuda_ops

    def forbidden(*args, **kwargs):
        raise AssertionError("M > 1536 must take the Soup path")

    monkeypatch.setattr(
        cuda_ops, "_gemm_4bit_use_custom_fn", forbidden, raising=False
    )
    monkeypatch.setattr(nf4, "_ORIGINAL_BNB_MATMUL_4BIT", forbidden)

    actual = nf4.soup_matmul_4bit(
        A_soup, packed, quant_state=state, bias=bias_soup
    )

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)

    grad = torch.randn(
        reference.shape,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    reference.backward(grad)
    actual.backward(grad)

    torch.testing.assert_close(
        A_soup.grad, A_ref.grad, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        bias_soup.grad, bias_ref.grad, rtol=1e-3, atol=1e-3
    )
    assert packed.grad is None


@pytest.mark.parametrize(
    "case", ("non_nested", "blocksize_128", "fp32", "m4", "out")
)
def test_guards_delegate_unchanged(case, monkeypatch):
    weight = torch.randn((8, 128), device="cuda", dtype=torch.float16)
    packed, state = quantize(
        weight,
        blocksize=128 if case == "blocksize_128" else 64,
        nested=case != "non_nested",
    )

    m = 4 if case == "m4" else 2048
    dtype = torch.float32 if case == "fp32" else torch.float16
    A = torch.empty((m, 128), device="cuda", dtype=dtype)
    bias = torch.empty((8,), device="cuda", dtype=dtype)
    out = (
        torch.empty((m, 8), device="cuda", dtype=dtype)
        if case == "out"
        else None
    )

    sentinel = object()
    calls = []

    def original_spy(a, b, quant_state, out=None, bias=None):
        calls.append((a, b, quant_state, out, bias))
        return sentinel

    monkeypatch.setattr(
        nf4, "_ORIGINAL_BNB_MATMUL_4BIT", original_spy
    )

    result = nf4.soup_matmul_4bit(
        A, packed, quant_state=state, out=out, bias=bias
    )

    assert result is sentinel
    assert len(calls) == 1
    a, b, qs, received_out, received_bias = calls[0]
    assert a is A
    assert b is packed
    assert qs is state
    assert received_out is out
    assert received_bias is bias


@pytest.mark.parametrize("decision", (True, False, "raises"))
def test_private_heuristic_routing(
    quantized_weights, monkeypatch, decision
):
    import bitsandbytes.backends.cuda.ops as cuda_ops

    packed, state = quantized_weights[(256, 1536)]
    A = torch.empty((8, 1536), device="cuda", dtype=torch.float16)

    original_result = object()
    soup_result = object()
    heuristic_calls = []
    route_calls = []

    def heuristic(device_index, dtype, m, n, k):
        heuristic_calls.append((device_index, dtype, m, n, k))
        if decision == "raises":
            raise RuntimeError("simulated private API failure")
        return decision

    def original_spy(a, b, quant_state, out=None, bias=None):
        assert a is A and b is packed and quant_state is state
        assert out is None and bias is None
        route_calls.append("original")
        return original_result

    def soup_spy(a, b, quant_state, bias):
        assert a is A and b is packed and quant_state is state
        assert bias is None
        route_calls.append("soup")
        return soup_result

    monkeypatch.setattr(
        cuda_ops, "_gemm_4bit_use_custom_fn", heuristic, raising=False
    )
    monkeypatch.setattr(
        nf4, "_ORIGINAL_BNB_MATMUL_4BIT", original_spy
    )
    monkeypatch.setattr(
        nf4.SoupNF4Matmul, "apply", staticmethod(soup_spy)
    )

    result = nf4.soup_matmul_4bit(A, packed, state)

    assert heuristic_calls == [
        (A.device.index, torch.float16, 8, 256, 1536)
    ]
    if decision is False:
        assert result is soup_result
        assert route_calls == ["soup"]
    else:
        assert result is original_result
        assert route_calls == ["original"]


def test_gate_up_shared_buffer(quantized_weights):
    gate, gate_state = quantized_weights[(8960, 1536)]

    generator = torch.Generator(device="cuda").manual_seed(97)
    up_weight = torch.randn(
        (8960, 1536),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    up, up_state = quantize(up_weight)
    del up_weight

    shared = torch.empty(
        (17920, 1536), device="cuda", dtype=torch.float16
    )
    gate_out = shared[:8960]
    up_out = shared[8960:]

    assert nf4.nf4_dequantize(gate, gate_state, out=gate_out) is gate_out
    assert nf4.nf4_dequantize(up, up_state, out=up_out) is up_out

    assert_bit_exact(gate_out, nf4.nf4_dequantize(gate, gate_state))
    assert_bit_exact(up_out, nf4.nf4_dequantize(up, up_state))


def test_validation_errors(quantized_weights):
    import copy

    packed, state = quantized_weights[(256, 1536)]

    invalid = copy.copy(state)
    invalid.dtype = torch.float32
    with pytest.raises(TypeError, match="quant_state.dtype"):
        nf4.nf4_dequantize(packed, invalid)

    invalid = copy.copy(state)
    invalid.blocksize = 128
    with pytest.raises(ValueError, match="blocksize"):
        nf4.nf4_dequantize(packed, invalid)

    wrong_dtype = torch.empty(
        state.shape, device="cuda", dtype=torch.float32
    )
    with pytest.raises(TypeError, match="out"):
        nf4.nf4_dequantize(packed, state, out=wrong_dtype)

    noncontiguous = torch.empty(
        (1536, 256), device="cuda", dtype=torch.float16
    ).t()
    with pytest.raises(ValueError, match="contiguous"):
        nf4.nf4_dequantize(packed, state, out=noncontiguous)


def test_patch_is_idempotent(monkeypatch):
    # Register the attribute with monkeypatch so teardown restores it.
    monkeypatch.setattr(bnb, "matmul_4bit", ORIGINAL)
    saved_original = nf4._get_original_matmul_4bit()

    nf4.patch_bnb_matmul_4bit()
    nf4.patch_bnb_matmul_4bit()

    assert bnb.matmul_4bit is nf4.soup_matmul_4bit
    assert nf4._get_original_matmul_4bit() is saved_original


# --- K1 v3s: kernel layout limits, fallback, validation cache, rules -------

# (3, 40): 120 elements, one program with 15 of its 256 rows live and a
# partial 64-block. (1, 8): a single 4-byte group. (5, 1536): partial program.
# (7, 9): 63 elements, numel % 8 != 0, takes the torch fallback.
RAGGED_SHAPES = ((3, 40), (1, 8), (5, 1536), (7, 9))


@pytest.mark.parametrize("shape", RAGGED_SHAPES)
def test_ragged_shapes_bit_exact(shape):
    generator = torch.Generator(device="cuda").manual_seed(1000 + shape[1])
    weight = torch.randn(
        shape, device="cuda", dtype=torch.float16, generator=generator
    )
    packed, state = quantize(weight)
    expected = bnb.functional.dequantize_4bit(packed, quant_state=state)
    assert_bit_exact(nf4.nf4_dequantize(packed, state), expected)


@pytest.mark.parametrize("offset", (0, 1, 2, 8))
def test_out_slice_offsets_stay_in_bounds(offset):
    # offset 1: odd start, goes through a temporary; offset 2: 4-byte aligned
    # int32 view (extra Triton specialization, tests only); 0 and 8: aligned.
    generator = torch.Generator(device="cuda").manual_seed(5 + offset)
    weight = torch.randn(
        (3, 40), device="cuda", dtype=torch.float16, generator=generator
    )
    packed, state = quantize(weight)
    expected = bnb.functional.dequantize_4bit(packed, quant_state=state)

    backing = torch.full(
        (offset + 120 + 8,), 123.0, device="cuda", dtype=torch.float16
    )
    output = backing[offset:offset + 120].view(3, 40)
    result = nf4.nf4_dequantize(packed, state, out=output)

    assert result is output
    assert_bit_exact(result, expected)
    assert bool((backing[:offset] == 123.0).all())
    assert bool((backing[offset + 120:] == 123.0).all())


def test_model_shapes_take_the_kernel(quantized_weights, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Qwen2.5 shapes must not take the torch fallback")

    monkeypatch.setattr(nf4, "_torch_dequant", forbidden)
    for shape in SHAPES:
        packed, state = quantized_weights[shape]
        expected = bnb.functional.dequantize_4bit(packed, quant_state=state)
        assert_bit_exact(nf4.nf4_dequantize(packed, state), expected)


def test_validation_cache_revalidates_changes(quantized_weights):
    import copy

    packed, state = quantized_weights[(256, 1536)]
    expected = bnb.functional.dequantize_4bit(packed, quant_state=state)

    cached = copy.copy(state)
    assert_bit_exact(nf4.nf4_dequantize(packed, cached), expected)
    assert_bit_exact(nf4.nf4_dequantize(packed, cached), expected)  # hit

    cached.dtype = torch.float32
    with pytest.raises(TypeError, match="quant_state.dtype"):
        nf4.nf4_dequantize(packed, cached)
    cached.dtype = torch.float16
    assert_bit_exact(nf4.nf4_dequantize(packed, cached), expected)

    original_absmax = cached.absmax
    cached.absmax = original_absmax[:-1]
    with pytest.raises(ValueError, match="quant_state.absmax"):
        nf4.nf4_dequantize(packed, cached)
    cached.absmax = original_absmax.clone()
    assert_bit_exact(nf4.nf4_dequantize(packed, cached), expected)

    cached.shape = torch.Size((128, 1536))
    with pytest.raises(ValueError, match="packed size"):
        nf4.nf4_dequantize(packed, cached)
    cached.shape = state.shape

    with pytest.raises(ValueError, match="packed size"):
        nf4.nf4_dequantize(packed[:-1], cached)
    assert_bit_exact(nf4.nf4_dequantize(packed, cached), expected)


def test_single_soup_kernel_far_from_example(monkeypatch):
    # Uses the harness's own rule code (read-only import).
    import importlib.util
    import pathlib
    import sys

    repo = pathlib.Path(nf4.__file__).resolve().parents[2]
    rules_path = repo / "harness" / "check_rules.py"
    if not rules_path.exists():
        pytest.skip("harness/check_rules.py not found")

    monkeypatch.setattr(sys, "path", list(sys.path))
    had_common = "common" in sys.modules
    spec = importlib.util.spec_from_file_location("_soup_check_rules", rules_path)
    rules = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(rules)
        _, kernels = rules.triton_kernels(nf4.__file__)
        assert [node.name for node in kernels] == ["soup_nf4_dequant_kernel"]
        assert rules.example_likeness(kernels[0]) < rules.EXAMPLE_REVIEW
    finally:
        if not had_common:
            sys.modules.pop("common", None)
