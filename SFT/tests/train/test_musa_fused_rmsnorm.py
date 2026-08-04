from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llamafactory.model.model_utils import musa_fused_rmsnorm as rmsnorm_impl
from llamafactory.model.model_utils.musa_fused_rmsnorm import (
    apply_rms_norm_eager,
    apply_rms_norm_musa,
    patch_qwen3_vl_moe_fused_rmsnorm,
)


try:
    import torch_musa  # noqa: F401

    _MUSA_AVAILABLE = torch.musa.is_available()
except ImportError:
    _MUSA_AVAILABLE = False


class FakeRMSNorm:
    def __init__(self, hidden_size: int, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.weight = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
        self.variance_epsilon = 1e-6


def test_musa_rmsnorm_falls_back_for_cpu() -> None:
    norm = FakeRMSNorm(32)
    hidden_states = torch.randn(2, 17, 32)

    actual = apply_rms_norm_musa(norm, hidden_states)
    expected = apply_rms_norm_eager(norm, hidden_states)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_musa_rmsnorm_disables_fused_path_after_kernel_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rmsnorm_impl, "_FUSED_RMSNORM_DISABLED", False)
    norm = FakeRMSNorm(32, device="musa", dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 17, 32, device="musa", dtype=torch.bfloat16)
    fused_calls = 0

    def failing_rms_norm(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        raise RuntimeError("injected fused RMSNorm failure")

    monkeypatch.setattr(F, "rms_norm", failing_rms_norm)
    expected = apply_rms_norm_eager(norm, hidden_states)
    first = apply_rms_norm_musa(norm, hidden_states)
    second = apply_rms_norm_musa(norm, hidden_states)

    assert fused_calls == 1
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen3_vl_moe_fused_rmsnorm_patch_and_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    monkeypatch.setenv("OPENSEARCH_MUSA_FUSED_RMSNORM", "1")
    hidden_size = 2048
    norm = modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm(hidden_size).to(device="musa", dtype=torch.bfloat16)
    expected_norm = modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm(hidden_size).to(
        device="musa", dtype=torch.bfloat16
    )
    expected_norm.load_state_dict(norm.state_dict())

    class FakeModel:
        config = SimpleNamespace(model_type="qwen3_vl_moe")

        @staticmethod
        def modules():
            return [norm]

    assert patch_qwen3_vl_moe_fused_rmsnorm(FakeModel()) == 1

    hidden_states = torch.randn(1, 1025, hidden_size, device="musa", dtype=torch.bfloat16)
    actual_input = hidden_states.clone().requires_grad_(True)
    expected_input = hidden_states.clone().requires_grad_(True)
    grad_output = torch.randn_like(hidden_states)

    actual = norm(actual_input)
    expected = apply_rms_norm_eager(expected_norm, expected_input)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.0625)

    actual.backward(grad_output)
    expected.backward(grad_output)
    assert actual_input.grad is not None and torch.isfinite(actual_input.grad).all()
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0.01, atol=0.0625)
    torch.testing.assert_close(norm.weight.grad, expected_norm.weight.grad, rtol=0.01, atol=0.5)
