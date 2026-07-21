import os
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile


DEVICE = torch.device(os.getenv("MUSA_TEST_DEVICE", "musa:0"))


def _sequence_lengths(name: str, default: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in os.getenv(name, default).split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must be a comma-separated list of positive integers")

    return values


TEXT_SEQUENCE_LENGTHS = _sequence_lengths("MUSA_SDPA_TEXT_SEQUENCE_LENGTHS", "1024,2048")
VISION_SEQUENCE_LENGTHS = _sequence_lengths("MUSA_SDPA_VISION_SEQUENCE_LENGTHS", "1024")
REAL_TEXT_SEQUENCE_LENGTH = int(os.getenv("MUSA_SDPA_REAL_TEXT_SEQUENCE_LENGTH", "32000"))
RUN_REAL_SHAPE_MATH_PROBE = os.getenv("MUSA_SDPA_RUN_REAL_SHAPE_MATH_PROBE", "0").lower() in {
    "1",
    "true",
    "y",
}
RUN_REAL_SHAPE_DEFAULT_PROBE = os.getenv("MUSA_SDPA_RUN_REAL_SHAPE_DEFAULT_PROBE", "0").lower() in {
    "1",
    "true",
    "y",
}


def _require_idle_musa(min_free_gib: float = 16.0) -> None:
    if os.getenv("MUSA_UNIT_TEST_ALLOW_BUSY_DEVICE", "0").lower() in {"1", "true", "y"}:
        return

    free_bytes, total_bytes = torch.musa.mem_get_info(DEVICE)
    required_bytes = max(int(min_free_gib * 1024**3), int(total_bytes * 0.8))
    if free_bytes < required_bytes:
        pytest.skip(
            f"MUSA SDPA precision test requires an idle device: free={free_bytes / 1024**3:.2f} GiB, "
            f"total={total_bytes / 1024**3:.2f} GiB. Set MUSA_UNIT_TEST_ALLOW_BUSY_DEVICE=1 to override."
        )


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().float().cpu().flatten()
    reference = reference.detach().float().cpu().flatten()
    error = actual - reference
    return {
        "max_abs": error.abs().max().item(),
        "mean_abs": error.abs().mean().item(),
        "nrmse": (error.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)).item(),
        "cosine": F.cosine_similarity(actual, reference, dim=0).item(),
    }


def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    backend: SDPBackend | None,
    is_causal: bool,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_run, k_run, v_run = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)]
    context = sdpa_kernel(backend) if backend is not None else nullcontext()
    with context:
        output = F.scaled_dot_product_attention(
            q_run,
            k_run,
            v_run,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
    output.backward(grad_output)
    torch.musa.synchronize(DEVICE)
    result = tuple(tensor.detach().float().cpu() for tensor in (output, q_run.grad, k_run.grad, v_run.grad))
    del output, q_run, k_run, v_run
    torch.musa.empty_cache()
    return result


def _assert_agreement(
    case: str,
    actual: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
) -> None:
    thresholds = {
        "output": (0.02, 0.999),
        "q_grad": (0.03, 0.998),
        "k_grad": (0.03, 0.998),
        "v_grad": (0.03, 0.998),
    }
    failures = []
    for name, actual_tensor, reference_tensor in zip(("output", "q_grad", "k_grad", "v_grad"), actual, reference):
        assert torch.isfinite(actual_tensor).all().item(), f"{case}.{name} contains NaN or Inf"
        assert torch.isfinite(reference_tensor).all().item(), f"{case}.math.{name} contains NaN or Inf"
        metric = _metrics(actual_tensor, reference_tensor)
        max_nrmse, min_cosine = thresholds[name]
        print(
            f"case={case} tensor={name} max_abs={metric['max_abs']:.8g} "
            f"mean_abs={metric['mean_abs']:.8g} nrmse={metric['nrmse']:.8g} "
            f"cosine={metric['cosine']:.10f}"
        )
        if metric["nrmse"] > max_nrmse or metric["cosine"] < min_cosine:
            failures.append((name, metric))

    assert not failures, f"MUSA default SDPA precision regression in {case}: {failures}"


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
def test_qwen3_vl_sdpa_selectors_use_musa_kernels() -> None:
    """Guard against accidentally toggling CUDA flags without selecting the intended MUSA kernels."""
    _require_idle_musa()
    torch.manual_seed(20260716)
    q = torch.randn((1, 32, 64, 128), device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn((1, 4, 64, 128), device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn((1, 4, 64, 128), device=DEVICE, dtype=torch.bfloat16)
    with profile(activities=[ProfilerActivity.CPU]) as default_profiler:
        F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        torch.musa.synchronize(DEVICE)
    with profile(activities=[ProfilerActivity.CPU]) as math_profiler:
        with sdpa_kernel(SDPBackend.MATH):
            F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        torch.musa.synchronize(DEVICE)

    default_operator_names = {event.key for event in default_profiler.key_averages()}
    math_operator_names = {event.key for event in math_profiler.key_averages()}
    assert "aten::_scaled_dot_product_attention_flash_musa" in default_operator_names
    assert "aten::_scaled_dot_product_attention_math_musa" not in default_operator_names
    assert "aten::_scaled_dot_product_attention_math_musa" in math_operator_names
    assert "aten::_scaled_dot_product_attention_flash_musa" not in math_operator_names


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
def test_qwen3_vl_real_shape_math_attention_matrix_consumes_most_device_memory() -> None:
    """The configured 32k GQA shape leaves little memory for model state and backward."""
    free_bytes, total_bytes = torch.musa.mem_get_info(DEVICE)
    score_elements = 32 * REAL_TEXT_SEQUENCE_LENGTH**2
    attention_weight_bytes = score_elements * torch.tensor([], dtype=torch.bfloat16).element_size()
    print(
        f"sequence_length={REAL_TEXT_SEQUENCE_LENGTH} score_elements={score_elements} "
        f"musa_math_attention_weight_gib={attention_weight_bytes / 1024**3:.2f} "
        f"device_free_gib={free_bytes / 1024**3:.2f} device_total_gib={total_bytes / 1024**3:.2f}"
    )
    assert attention_weight_bytes > total_bytes * 0.7


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.skipif(
    not RUN_REAL_SHAPE_DEFAULT_PROBE,
    reason="set MUSA_SDPA_RUN_REAL_SHAPE_DEFAULT_PROBE=1 to run the real-shape default-backend probe",
)
def test_qwen3_vl_real_shape_default_forward_backward_is_finite() -> None:
    """Run Qwen3-VL's training shape through the default MUSA flash SDPA backend."""
    _require_idle_musa()
    torch.manual_seed(20260716)
    q = torch.randn(
        (1, 32, REAL_TEXT_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        (1, 4, REAL_TEXT_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        (1, 4, REAL_TEXT_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, enable_gqa=True)
    output.backward(torch.ones_like(output) / 128)
    torch.musa.synchronize(DEVICE)
    for name, tensor in (("output", output), ("q_grad", q.grad), ("k_grad", k.grad), ("v_grad", v.grad)):
        assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf"


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.skipif(
    not RUN_REAL_SHAPE_MATH_PROBE,
    reason="set MUSA_SDPA_RUN_REAL_SHAPE_MATH_PROBE=1 to run the isolated real-shape OOM probe",
)
def test_qwen3_vl_real_shape_math_forward_backward_probe() -> None:
    """Opt-in training-like forward/backward probe; an expected OOM is reported as xfail."""
    _require_idle_musa(min_free_gib=64.0)
    torch.manual_seed(20260716)
    q = torch.randn(
        (1, 32, REAL_TEXT_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        (1, 4, REAL_TEXT_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        (1, 4, REAL_TEXT_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    try:
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=True,
            )
        output.backward(torch.ones_like(output) / 128)
        torch.musa.synchronize(DEVICE)
    except RuntimeError as error:
        if "memory" in str(error).lower():
            torch.musa.empty_cache()
            pytest.xfail(f"real-shape MUSA math SDPA confirmed OOM: {error}")
        raise

    for name, tensor in (("output", output), ("q_grad", q.grad), ("k_grad", k.grad), ("v_grad", v.grad)):
        assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf"


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.parametrize("sequence_length", TEXT_SEQUENCE_LENGTHS)
def test_qwen3_vl_text_gqa_default_sdpa_against_math(sequence_length: int) -> None:
    """Qwen3-VL text attention: 32 query heads, 4 KV heads, head_dim=128, causal GQA."""
    _require_idle_musa()
    torch.manual_seed(20260716 + sequence_length)
    q = torch.randn((1, 32, sequence_length, 128), device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn((1, 4, sequence_length, 128), device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn((1, 4, sequence_length, 128), device=DEVICE, dtype=torch.bfloat16)
    grad_output = torch.randn(q.shape, device=DEVICE, dtype=torch.bfloat16) / 128

    actual = _run_sdpa(q, k, v, grad_output, backend=None, is_causal=True, enable_gqa=True)
    reference = _run_sdpa(q, k, v, grad_output, backend=SDPBackend.MATH, is_causal=True, enable_gqa=True)
    _assert_agreement(f"text_gqa_seq_{sequence_length}", actual, reference)


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.parametrize("sequence_length", VISION_SEQUENCE_LENGTHS)
def test_qwen3_vl_vision_default_sdpa_against_math(sequence_length: int) -> None:
    """Qwen3-VL vision attention: 16 heads, head_dim=72, non-causal per-image attention."""
    _require_idle_musa()
    torch.manual_seed(20260716 + sequence_length)
    shape = (1, 16, sequence_length, 72)
    q = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    grad_output = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) / 72

    actual = _run_sdpa(q, k, v, grad_output, backend=None, is_causal=False, enable_gqa=False)
    reference = _run_sdpa(q, k, v, grad_output, backend=SDPBackend.MATH, is_causal=False, enable_gqa=False)
    _assert_agreement(f"vision_seq_{sequence_length}", actual, reference)
