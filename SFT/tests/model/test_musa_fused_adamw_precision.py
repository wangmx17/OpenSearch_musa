import os

import pytest
import torch
import torch.nn.functional as F
from torch_musa.optim import FusedAdamW


DEVICE = torch.device(os.getenv("MUSA_TEST_DEVICE", "musa:0"))


def _require_idle_musa(min_free_gib: float = 8.0) -> None:
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
    actual = actual.detach().float().cpu().flatten()
    reference = reference.detach().float().cpu().flatten()
    error = actual - reference
    return {
        "max_abs": error.abs().max().item(),
        "mean_abs": error.abs().mean().item(),
        "nrmse": (error.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)).item(),
        "cosine": F.cosine_similarity(actual, reference, dim=0).item(),
    }


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf"


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("numel", [1, 4096, 262147])
def test_fused_adamw_zero_lr_first_step_is_exact_noop(numel: int, dtype: torch.dtype) -> None:
    """The first warmup step uses lr=0 and must not corrupt model parameters."""
    _require_idle_musa()
    generator = torch.Generator().manual_seed(20260716 + numel)
    initial = (torch.randn(numel, generator=generator, dtype=torch.float32) * 0.02).to(dtype)
    grad = (torch.randn(numel, generator=generator, dtype=torch.float32) * 0.01).to(dtype)
    parameter = torch.nn.Parameter(initial.to(DEVICE))
    parameter.grad = grad.to(DEVICE)
    optimizer = FusedAdamW([parameter], lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    optimizer.step()
    torch.musa.synchronize(DEVICE)

    state = optimizer.state[parameter]
    actual = parameter.detach().cpu()
    max_abs = (actual - initial).abs().max().item()
    print(
        f"dtype={dtype} numel={numel} zero_lr_param_max_abs={max_abs:.8g} "
        f"state_step={state['step'].item():.8g}"
    )
    _assert_finite("parameter", parameter)
    _assert_finite("exp_avg", state["exp_avg"])
    _assert_finite("exp_avg_sq", state["exp_avg_sq"])
    _assert_finite("state_step", state["step"])
    torch.testing.assert_close(actual, initial, rtol=0.0, atol=0.0)
    assert state["step"].item() == 1.0


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_adamw_warmup_schedule_against_cpu_adamw(dtype: torch.dtype) -> None:
    """Compare the MUSA multi-tensor kernel with a CPU AdamW oracle across the real warmup boundary."""
    _require_idle_musa()
    generator = torch.Generator().manual_seed(20260716)
    sizes = (17, 4096, 262147)
    initial = [(torch.randn(size, generator=generator, dtype=torch.float32) * 0.02).to(dtype) for size in sizes]
    musa_parameters = [torch.nn.Parameter(value.to(DEVICE)) for value in initial]
    cpu_parameters = [torch.nn.Parameter(value.clone()) for value in initial]
    fused = FusedAdamW(musa_parameters, lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    reference = torch.optim.AdamW(
        cpu_parameters,
        lr=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    learning_rates = (0.0, 8.734e-8, 1.747e-7, 2.0e-5)

    for step, learning_rate in enumerate(learning_rates, 1):
        fused.param_groups[0]["lr"] = learning_rate
        reference.param_groups[0]["lr"] = learning_rate
        for musa_parameter, cpu_parameter in zip(musa_parameters, cpu_parameters):
            grad = (torch.randn(cpu_parameter.shape, generator=generator, dtype=torch.float32) * 0.01).to(dtype)
            musa_parameter.grad = grad.to(DEVICE)
            cpu_parameter.grad = grad.clone()

        fused.step()
        reference.step()
        torch.musa.synchronize(DEVICE)

        for index, (musa_parameter, cpu_parameter) in enumerate(zip(musa_parameters, cpu_parameters)):
            fused_state = fused.state[musa_parameter]
            reference_state = reference.state[cpu_parameter]
            _assert_finite(f"step_{step}.parameter_{index}", musa_parameter)
            _assert_finite(f"step_{step}.exp_avg_{index}", fused_state["exp_avg"])
            _assert_finite(f"step_{step}.exp_avg_sq_{index}", fused_state["exp_avg_sq"])

            parameter_metrics = _metrics(musa_parameter, cpu_parameter)
            exp_avg_metrics = _metrics(fused_state["exp_avg"], reference_state["exp_avg"])
            exp_avg_sq_metrics = _metrics(fused_state["exp_avg_sq"], reference_state["exp_avg_sq"])
            print(
                f"dtype={dtype} step={step} lr={learning_rate:.8g} tensor={index} "
                f"param_max_abs={parameter_metrics['max_abs']:.8g} param_nrmse={parameter_metrics['nrmse']:.8g} "
                f"exp_avg_max_abs={exp_avg_metrics['max_abs']:.8g} "
                f"exp_avg_nrmse={exp_avg_metrics['nrmse']:.8g} "
                f"exp_avg_sq_max_abs={exp_avg_sq_metrics['max_abs']:.8g} "
                f"exp_avg_sq_nrmse={exp_avg_sq_metrics['nrmse']:.8g}"
            )
            if dtype == torch.float32:
                torch.testing.assert_close(
                    musa_parameter.detach().cpu(), cpu_parameter.detach(), rtol=3e-5, atol=2e-7
                )
                torch.testing.assert_close(
                    fused_state["exp_avg"].detach().cpu(), reference_state["exp_avg"], rtol=3e-5, atol=2e-7
                )
                torch.testing.assert_close(
                    fused_state["exp_avg_sq"].detach().cpu(),
                    reference_state["exp_avg_sq"],
                    rtol=5e-5,
                    atol=2e-9,
                )
            else:
                assert parameter_metrics["nrmse"] <= 1e-3
                assert parameter_metrics["cosine"] >= 0.9999
                assert exp_avg_metrics["nrmse"] <= 2e-2
                assert exp_avg_metrics["cosine"] >= 0.999
                assert exp_avg_sq_metrics["nrmse"] <= 3e-2
                assert exp_avg_sq_metrics["cosine"] >= 0.999
            assert fused_state["step"].item() == float(step)
