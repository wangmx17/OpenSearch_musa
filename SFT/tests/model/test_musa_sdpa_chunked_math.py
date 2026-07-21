import os

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


DEVICE = torch.device(os.getenv("MUSA_TEST_DEVICE", "musa:0"))
REAL_SEQUENCE_LENGTH = int(os.getenv("MUSA_SDPA_REAL_TEXT_SEQUENCE_LENGTH", "32000"))
REAL_CHUNK_SIZE = int(os.getenv("MUSA_SDPA_MATH_QUERY_CHUNK_SIZE", "512"))
RUN_REAL_SHAPE_PROBE = os.getenv("MUSA_SDPA_RUN_CHUNKED_MATH_PROBE", "0").lower() in {
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
            f"MUSA precision test requires an idle device: free={free_bytes / 1024**3:.2f} GiB, "
            f"total={total_bytes / 1024**3:.2f} GiB. Set MUSA_UNIT_TEST_ALLOW_BUSY_DEVICE=1 to override."
        )


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().flatten()
    reference = reference.detach().flatten()
    max_abs = 0.0
    sum_abs = 0.0
    sum_error_sq = 0.0
    sum_actual_sq = 0.0
    sum_reference_sq = 0.0
    dot = 0.0
    metric_chunk_size = 1_000_000
    for start in range(0, actual.numel(), metric_chunk_size):
        actual_chunk = actual[start : start + metric_chunk_size].double()
        reference_chunk = reference[start : start + metric_chunk_size].double()
        error = actual_chunk - reference_chunk
        max_abs = max(max_abs, error.abs().max().item())
        sum_abs += error.abs().sum().item()
        sum_error_sq += error.square().sum().item()
        sum_actual_sq += actual_chunk.square().sum().item()
        sum_reference_sq += reference_chunk.square().sum().item()
        dot += (actual_chunk * reference_chunk).sum().item()

    return {
        "max_abs": max_abs,
        "mean_abs": sum_abs / actual.numel(),
        "nrmse": (sum_error_sq / max(sum_reference_sq, 1e-24)) ** 0.5,
        "cosine": dot / max((sum_actual_sq * sum_reference_sq) ** 0.5, 1e-24),
    }


def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    math_backend: bool,
) -> tuple[tuple[torch.Tensor, ...], float]:
    q_run, k_run, v_run = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)]
    torch.musa.reset_peak_memory_stats(DEVICE)
    if math_backend:
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                q_run,
                k_run,
                v_run,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=True,
            )
    else:
        output = F.scaled_dot_product_attention(
            q_run,
            k_run,
            v_run,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )
    output.backward(grad_output)
    torch.musa.synchronize(DEVICE)
    peak_gib = torch.musa.max_memory_allocated(DEVICE) / 1024**3
    result = tuple(tensor.detach().cpu() for tensor in (output, q_run.grad, k_run.grad, v_run.grad))
    del output, q_run, k_run, v_run
    torch.musa.empty_cache()
    return result, peak_gib


def _run_chunked_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    chunk_size: int,
    reverse: bool,
) -> tuple[tuple[torch.Tensor, ...], float]:
    """Evaluate causal math SDPA one query block at a time and immediately backpropagate it."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    sequence_length = q.shape[-2]
    q_run, k_run, v_run = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)]
    output_cpu = torch.empty(q.shape, dtype=q.dtype, device="cpu")
    blocks = [(start, min(start + chunk_size, sequence_length)) for start in range(0, sequence_length, chunk_size)]
    if reverse:
        blocks.reverse()

    torch.musa.reset_peak_memory_stats(DEVICE)
    for start, end in blocks:
        query_positions = torch.arange(start, end, device=DEVICE).unsqueeze(1)
        key_positions = torch.arange(end, device=DEVICE).unsqueeze(0)
        causal_mask = key_positions <= query_positions
        with sdpa_kernel(SDPBackend.MATH):
            output_chunk = F.scaled_dot_product_attention(
                q_run[:, :, start:end],
                k_run[:, :, :end],
                v_run[:, :, :end],
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        output_chunk.backward(grad_output[:, :, start:end])
        output_cpu[:, :, start:end].copy_(output_chunk.detach().cpu())
        del causal_mask, key_positions, output_chunk, query_positions

    torch.musa.synchronize(DEVICE)
    peak_gib = torch.musa.max_memory_allocated(DEVICE) / 1024**3
    result = output_cpu, q_run.grad.detach().cpu(), k_run.grad.detach().cpu(), v_run.grad.detach().cpu()
    del q_run, k_run, v_run
    torch.musa.empty_cache()
    return result, peak_gib


def _assert_agreement(
    case: str,
    actual: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
    *,
    max_nrmse: float,
    min_cosine: float,
) -> None:
    failures = []
    for name, actual_tensor, reference_tensor in zip(
        ("output", "q_grad", "k_grad", "v_grad"), actual, reference
    ):
        assert torch.isfinite(actual_tensor).all().item(), f"{case}.{name} contains NaN or Inf"
        metric = _metrics(actual_tensor, reference_tensor)
        print(
            f"case={case} tensor={name} max_abs={metric['max_abs']:.8g} "
            f"mean_abs={metric['mean_abs']:.8g} nrmse={metric['nrmse']:.8g} "
            f"cosine={metric['cosine']:.10f}"
        )
        if metric["nrmse"] > max_nrmse or metric["cosine"] < min_cosine:
            failures.append((name, metric))

    assert not failures, f"chunked MUSA math SDPA precision regression in {case}: {failures}"


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.parametrize("reverse", [False, True], ids=["forward_blocks", "reverse_blocks"])
def test_chunked_math_matches_monolithic_math(reverse: bool) -> None:
    """Validate the causal mask, immediate backward, and both K/V accumulation orders."""
    _require_idle_musa()
    torch.manual_seed(20260716)
    sequence_length = 256
    q = torch.randn((1, 32, sequence_length, 128), device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn((1, 4, sequence_length, 128), device=DEVICE, dtype=torch.bfloat16) * 0.1
    v = torch.randn((1, 4, sequence_length, 128), device=DEVICE, dtype=torch.bfloat16) * 0.1
    grad_output = torch.randn(q.shape, device=DEVICE, dtype=torch.bfloat16) / 128

    reference, reference_peak_gib = _run_sdpa(q, k, v, grad_output, math_backend=True)
    actual, chunked_peak_gib = _run_chunked_math(
        q,
        k,
        v,
        grad_output,
        chunk_size=64,
        reverse=reverse,
    )
    print(
        f"reverse={reverse} monolithic_peak_gib={reference_peak_gib:.3f} "
        f"chunked_peak_gib={chunked_peak_gib:.3f}"
    )
    _assert_agreement(
        f"seq_{sequence_length}_chunk_64_reverse_{reverse}",
        actual,
        reference,
        max_nrmse=0.02,
        min_cosine=0.999,
    )


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.skipif(
    not RUN_REAL_SHAPE_PROBE,
    reason="set MUSA_SDPA_RUN_CHUNKED_MATH_PROBE=1 to compare real-shape chunked math with default SDPA",
)
def test_real_shape_chunked_math_against_default_sdpa() -> None:
    """Use bounded-memory math SDPA as a 32k forward/backward precision oracle."""
    _require_idle_musa(min_free_gib=64.0)
    torch.manual_seed(20260716)
    q = torch.randn((1, 32, REAL_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn((1, 4, REAL_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16) * 0.1
    v = torch.randn((1, 4, REAL_SEQUENCE_LENGTH, 128), device=DEVICE, dtype=torch.bfloat16) * 0.1
    grad_output = torch.randn(q.shape, device=DEVICE, dtype=torch.bfloat16) / 128

    actual, default_peak_gib = _run_sdpa(q, k, v, grad_output, math_backend=False)
    reference, chunked_peak_gib = _run_chunked_math(
        q,
        k,
        v,
        grad_output,
        chunk_size=REAL_CHUNK_SIZE,
        reverse=True,
    )
    print(
        f"sequence_length={REAL_SEQUENCE_LENGTH} chunk_size={REAL_CHUNK_SIZE} "
        f"default_peak_gib={default_peak_gib:.3f} chunked_math_peak_gib={chunked_peak_gib:.3f}"
    )
    _assert_agreement(
        f"real_seq_{REAL_SEQUENCE_LENGTH}_chunk_{REAL_CHUNK_SIZE}",
        actual,
        reference,
        max_nrmse=0.02,
        min_cosine=0.998,
    )
