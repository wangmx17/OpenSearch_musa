import pytest
import torch

from llamafactory.model.model_utils import musa_fused_rope as rope_impl
from llamafactory.model.model_utils.musa_fused_rope import (
    MUSA_ROPE_FREQ_CIS_ATTR,
    apply_rotary_pos_emb_eager,
    apply_rotary_pos_emb_musa,
)
from llamafactory.model.model_utils.rope import _qwen3_vl_moe_rope_forward


try:
    import torch_musa  # noqa: F401

    _MUSA_AVAILABLE = torch.musa.is_available()
except ImportError:
    _MUSA_AVAILABLE = False


def test_musa_rope_falls_back_for_cpu_gqa() -> None:
    batch_size, sequence_length, head_dim = 2, 17, 16
    q = torch.randn(batch_size, 8, sequence_length, head_dim)
    k = torch.randn(batch_size, 2, sequence_length, head_dim)
    cos = torch.randn(batch_size, sequence_length, head_dim)
    sin = torch.randn(batch_size, sequence_length, head_dim)

    actual = apply_rotary_pos_emb_musa(q, k, cos, sin)
    expected = apply_rotary_pos_emb_eager(q, k, cos, sin)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_musa_rope_disables_fused_path_after_kernel_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rope_impl, "_FUSED_ROPE_DISABLED", False)
    batch_size, sequence_length, head_dim = 1, 17, 16
    q = torch.randn(batch_size, 8, sequence_length, head_dim, device="musa", dtype=torch.bfloat16)
    k = torch.randn(batch_size, 2, sequence_length, head_dim, device="musa", dtype=torch.bfloat16)
    freq_cis = torch.randn(sequence_length, head_dim, device="musa", dtype=torch.float32)
    cos = freq_cis.cos().unsqueeze(0).to(torch.bfloat16)
    sin = freq_cis.sin().unsqueeze(0).to(torch.bfloat16)
    setattr(cos, MUSA_ROPE_FREQ_CIS_ATTR, freq_cis)
    fused_calls = 0

    def failing_rope(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        raise RuntimeError("injected fused RoPE failure")

    monkeypatch.setattr(torch, "rope", failing_rope)
    expected = apply_rotary_pos_emb_eager(q, k, cos, sin)
    first = apply_rotary_pos_emb_musa(q, k, cos, sin)
    second = apply_rotary_pos_emb_musa(q, k, cos, sin)

    assert fused_calls == 1
    torch.testing.assert_close(first[0], expected[0])
    torch.testing.assert_close(first[1], expected[1])
    torch.testing.assert_close(second[0], expected[0])
    torch.testing.assert_close(second[1], expected[1])


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen3_vl_moe_fused_rope_preserves_broadcast_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_MUSA_FUSED_ROPE", "1")
    monkeypatch.setenv("OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND", "1")
    sequence_length, head_dim = 2051, 128

    class FakeRotaryEmbedding:
        inv_freq = 1.0 / (5_000_000.0 ** (torch.arange(0, head_dim, 2, device="musa", dtype=torch.float32) / head_dim))
        mrope_section = [24, 20, 20]
        attention_scaling = 1.0

        @staticmethod
        def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
            return freqs[0]

    x = torch.empty(1, sequence_length, 2048, device="musa", dtype=torch.bfloat16)
    position_ids = torch.arange(sequence_length, device="musa").view(1, 1, -1).expand(3, 1, -1)

    cos, _ = _qwen3_vl_moe_rope_forward(FakeRotaryEmbedding(), x, position_ids)
    freq_cis = getattr(cos, MUSA_ROPE_FREQ_CIS_ATTR)
    expected_half = torch.outer(position_ids[0, 0].float(), FakeRotaryEmbedding.inv_freq)
    expected = torch.cat((expected_half, expected_half), dim=-1)

    assert freq_cis.dtype == torch.float32
    torch.testing.assert_close(freq_cis, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_qwen3_vl_moe_fused_rope_preserves_bmm_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_MUSA_FUSED_ROPE", "1")
    monkeypatch.setenv("OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND", "0")
    sequence_length, head_dim = 2051, 128

    class FakeRotaryEmbedding:
        inv_freq = 1.0 / (5_000_000.0 ** (torch.arange(0, head_dim, 2, device="musa", dtype=torch.float32) / head_dim))
        mrope_section = [24, 20, 20]
        attention_scaling = 1.0

        @staticmethod
        def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
            return freqs[0]

    x = torch.empty(1, sequence_length, 2048, device="musa", dtype=torch.bfloat16)
    position_ids = torch.arange(sequence_length, device="musa").view(1, 1, -1).expand(3, 1, -1)

    cos, _ = _qwen3_vl_moe_rope_forward(FakeRotaryEmbedding(), x, position_ids)
    freq_cis = getattr(cos, MUSA_ROPE_FREQ_CIS_ATTR)
    inv_freq_expanded = FakeRotaryEmbedding.inv_freq[None, None, :, None].expand(3, 1, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    expected_half = (inv_freq_expanded.float() @ position_ids_expanded).transpose(2, 3)[0, 0]
    expected = torch.cat((expected_half, expected_half), dim=-1)

    torch.testing.assert_close(freq_cis, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _MUSA_AVAILABLE, reason="MUSA device is required")
def test_musa_fused_rope_supports_gqa_and_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(7)
    batch_size, sequence_length, head_dim = 1, 513, 128
    q_base = torch.randn(batch_size, 32, sequence_length, head_dim, device="musa", dtype=torch.bfloat16)
    k_base = torch.randn(batch_size, 4, sequence_length, head_dim, device="musa", dtype=torch.bfloat16)
    q = q_base.clone().requires_grad_(True)
    k = k_base.clone().requires_grad_(True)
    q_reference = q_base.clone().requires_grad_(True)
    k_reference = k_base.clone().requires_grad_(True)
    position = torch.arange(sequence_length, device="musa", dtype=torch.float32)
    inv_freq = 1.0 / (5_000_000.0 ** (torch.arange(0, head_dim, 2, device="musa", dtype=torch.float32) / head_dim))
    freq_half = torch.outer(position, inv_freq)
    freq_cis = torch.cat((freq_half, freq_half), dim=-1).contiguous()
    cos_fp32 = freq_cis.cos()
    sin_fp32 = freq_cis.sin()
    cos = cos_fp32.unsqueeze(0).to(torch.bfloat16)
    sin = sin_fp32.unsqueeze(0).to(torch.bfloat16)
    setattr(cos, MUSA_ROPE_FREQ_CIS_ATTR, freq_cis)

    original_rope = torch.rope
    rope_calls = 0

    def counted_rope(*args, **kwargs):
        nonlocal rope_calls
        rope_calls += 1
        return original_rope(*args, **kwargs)

    monkeypatch.setattr(torch, "rope", counted_rope)
    q_actual, k_actual = apply_rotary_pos_emb_musa(q, k, cos, sin)
    q_expected, k_expected = apply_rotary_pos_emb_eager(
        q_reference.float(), k_reference.float(), cos_fp32.unsqueeze(0), sin_fp32.unsqueeze(0)
    )

    assert rope_calls == 2
    torch.testing.assert_close(q_actual, q_expected.to(torch.bfloat16), rtol=0, atol=0.02)
    torch.testing.assert_close(k_actual, k_expected.to(torch.bfloat16), rtol=0, atol=0.02)

    q_grad_output = torch.randn_like(q_actual)
    k_grad_output = torch.randn_like(k_actual)
    actual_grads = torch.autograd.grad((q_actual, k_actual), (q, k), (q_grad_output, k_grad_output))
    expected_grads = torch.autograd.grad(
        (q_expected, k_expected),
        (q_reference, k_reference),
        (q_grad_output.float(), k_grad_output.float()),
    )
    assert torch.isfinite(actual_grads[0]).all()
    assert torch.isfinite(actual_grads[1]).all()
    torch.testing.assert_close(actual_grads[0], expected_grads[0], rtol=0, atol=0.02)
    torch.testing.assert_close(actual_grads[1], expected_grads[1], rtol=0, atol=0.02)
