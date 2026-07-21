import gc

import pytest
import torch
import torch_musa  # noqa: F401


pytestmark = pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA is required")


@pytest.mark.xfail(
    reason="torch_musa _grouped_mm leaves zero-sized expert outputs uninitialized",
    strict=True,
)
def test_grouped_mm_zero_sized_expert_output_is_initialized() -> None:
    r"""An empty expert must produce a zero weight gradient, not allocator contents."""
    device = torch.device("musa:0")
    dtype = torch.bfloat16
    counts = torch.tensor([4, 0, 4, 4], dtype=torch.int32)
    offsets = counts.cumsum(0, dtype=torch.int32).to(device)
    output_shape = (counts.numel(), 512, 512)

    torch.manual_seed(7)
    lhs_base = torch.randn(
        (int(offsets[-1].item()), output_shape[1]), device=device, dtype=dtype
    )
    lhs = lhs_base.transpose(0, 1)
    rhs = torch.randn((lhs.shape[1], output_shape[2]), device=device, dtype=dtype)

    poisons = [
        torch.full(output_shape, float("nan"), device=device, dtype=dtype)
        for _ in range(16)
    ]
    torch.musa.synchronize()
    poison_ptrs = {poison.data_ptr() for poison in poisons}
    del poisons
    gc.collect()

    output = torch.ops.aten._grouped_mm(lhs, rhs, offsets)
    torch.musa.synchronize()

    assert output.data_ptr() in poison_ptrs, "The test must reuse a poisoned allocator block."
    assert torch.count_nonzero(output[1]).item() == 0
    assert torch.isfinite(torch.cat((output[0].flatten(), output[2:].flatten()))).all().item()
