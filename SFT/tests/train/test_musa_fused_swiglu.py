from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llamafactory.model.model_utils.musa_fused_swiglu import (
    apply_swiglu_eager,
    apply_swiglu_musa,
    patch_qwen3_vl_moe_fused_swiglu,
)


try:
    import torch_musa  # noqa: F401

    _MUSA_AVAILABLE = torch.musa.is_available()
except ImportError:
    _MUSA_AVAILABLE = False


def test_musa_swiglu_falls_back_for_cpu() -> None:
    gate_up = torch.randn(3, 17, 64)

    actual = apply_swiglu_musa(gate_up, F.silu)
    expected = apply_swiglu_eager(gate_up, F.silu)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
@pytest.mark.parametrize(("enabled", "hidden_act"), [("0", "silu"), ("1", "gelu")])
def test_musa_swiglu_respects_dispatch_guards(
    monkeypatch: pytest.MonkeyPatch,
    enabled: str,
    hidden_act: str,
) -> None:
    monkeypatch.setenv("OPENSEARCH_MUSA_FUSED_SWIGLU", enabled)
    gate_up = torch.randn(8, 64, device="musa", dtype=torch.bfloat16)

    def unexpected_fused_call(*args, **kwargs):
        raise AssertionError("F.swish_glu should not be called")

    monkeypatch.setattr(F, "swish_glu", unexpected_fused_call)
    actual = apply_swiglu_musa(gate_up, F.silu, hidden_act=hidden_act)
    expected = apply_swiglu_eager(gate_up, F.silu)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen3_vl_moe_fused_swiglu_patch_and_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    monkeypatch.setenv("OPENSEARCH_MUSA_FUSED_SWIGLU", "1")
    config = modeling_qwen3_vl_moe.Qwen3VLMoeTextConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    experts = modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts(config).to(device="musa", dtype=torch.bfloat16)
    expected_experts = modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts(config).to(
        device="musa", dtype=torch.bfloat16
    )
    with torch.no_grad():
        experts.gate_up_proj.normal_(mean=0.0, std=0.02)
        experts.down_proj.normal_(mean=0.0, std=0.02)
        expected_experts.load_state_dict(experts.state_dict())

    hidden_states = torch.randn(19, 32, device="musa", dtype=torch.bfloat16)
    actual_input = hidden_states.clone().requires_grad_(True)
    expected_input = hidden_states.clone().requires_grad_(True)
    top_k_index = torch.randint(0, config.num_experts, (19, config.num_experts_per_tok), device="musa")
    top_k_weights = torch.rand(19, config.num_experts_per_tok, device="musa", dtype=torch.float32)
    top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
    grad_output = torch.randn_like(hidden_states)
    expected = expected_experts(expected_input, top_k_index, top_k_weights)

    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl_moe", text_config=config),
        modules=lambda: [experts],
    )
    assert patch_qwen3_vl_moe_fused_swiglu(fake_model) == 1

    actual = experts(actual_input, top_k_index, top_k_weights)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.015625)

    actual.backward(grad_output)
    expected.backward(grad_output)
    assert actual_input.grad is not None and torch.isfinite(actual_input.grad).all()
    assert experts.gate_up_proj.grad is not None and torch.isfinite(experts.gate_up_proj.grad).all()
    assert experts.down_proj.grad is not None and torch.isfinite(experts.down_proj.grad).all()
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0.05, atol=0.01)
    torch.testing.assert_close(
        experts.gate_up_proj.grad, expected_experts.gate_up_proj.grad, rtol=0.05, atol=0.01
    )
    torch.testing.assert_close(experts.down_proj.grad, expected_experts.down_proj.grad, rtol=0.05, atol=0.01)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen3_vl_moe_fused_swiglu_does_not_override_transformers_grouped_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    monkeypatch.setenv("OPENSEARCH_MUSA_FUSED_SWIGLU", "1")
    config = modeling_qwen3_vl_moe.Qwen3VLMoeTextConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    config._experts_implementation = "grouped_mm"
    experts = modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts(config).to(device="musa", dtype=torch.bfloat16)
    original_forward = experts.forward
    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl_moe", text_config=config),
        modules=lambda: [experts],
    )

    assert patch_qwen3_vl_moe_fused_swiglu(fake_model) == 0
    assert experts.forward == original_forward
