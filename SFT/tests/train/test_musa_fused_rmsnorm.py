from types import SimpleNamespace

import pytest
import torch

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
def test_qwen3_vl_moe_fused_rmsnorm_patch_and_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    monkeypatch.setenv("OPENSEARCH_MUSA_FUSED_RMSNORM", "1")
    hidden_size = 2048
    norm = modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm(hidden_size).to(device="musa", dtype=torch.bfloat16)

    class FakeModel:
        config = SimpleNamespace(model_type="qwen3_vl_moe")

        @staticmethod
        def modules():
            return [norm]

    assert patch_qwen3_vl_moe_fused_rmsnorm(FakeModel()) == 1

    hidden_states = torch.randn(1, 4097, hidden_size, device="musa", dtype=torch.bfloat16, requires_grad=True)
    actual = norm(hidden_states)
    expected = apply_rms_norm_eager(norm, hidden_states)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.0625)

    actual.float().square().mean().backward()
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
