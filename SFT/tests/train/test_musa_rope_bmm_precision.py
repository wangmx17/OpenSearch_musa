import pytest
import torch


torch_musa = pytest.importorskip("torch_musa")
compare_tool = pytest.importorskip("torch_musa.utils.compare_tool")


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is required")
def test_float32_bmm_preserves_long_rope_positions() -> None:
    """Regression test for torch_musa 2.7.1 RoPE frequency corruption after position 2048."""
    sequence_length = 8048
    head_dim = 128
    inv_freq = 1.0 / (
        1_000_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    position_ids = torch.arange(sequence_length, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)

    inv_freq_expanded = inv_freq[None, None, :, None].expand(3, 1, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    expected = inv_freq_expanded @ position_ids_expanded
    actual = (
        inv_freq_expanded.to("musa").float() @ position_ids_expanded.to("musa").float()
    ).cpu()

    mismatch = compare_tool.compare_tensors(actual, expected, atol=1e-6, rtol=1e-6)
    assert mismatch.sum().item() == 0


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is required")
def test_broadcast_mul_preserves_long_rope_positions() -> None:
    """The model-side workaround must match the CPU RoPE reference on affected releases."""
    sequence_length = 8048
    head_dim = 128
    inv_freq = 1.0 / (
        1_000_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    position_ids = torch.arange(sequence_length, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)

    inv_freq_expanded = inv_freq[None, None, :, None].expand(3, 1, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    expected = inv_freq_expanded * position_ids_expanded
    actual = (
        inv_freq_expanded.to("musa").float() * position_ids_expanded.to("musa").float()
    ).cpu()

    mismatch = compare_tool.compare_tensors(actual, expected, atol=1e-6, rtol=1e-6)
    assert mismatch.sum().item() == 0
