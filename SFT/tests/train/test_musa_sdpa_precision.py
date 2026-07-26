import math
import os

import pytest
import torch


torch_musa = pytest.importorskip("torch_musa")
compare_tool = pytest.importorskip("torch_musa.utils.compare_tool")


def _chunked_causal_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    query = query.float()
    key = key.float()
    value = value.float()
    sequence_length = query.shape[-2]
    key_positions = torch.arange(sequence_length)
    outputs = []
    for start in range(0, sequence_length, chunk_size):
        stop = min(start + chunk_size, sequence_length)
        scores = torch.matmul(query[..., start:stop, :], key.transpose(-2, -1)) * scale
        query_positions = torch.arange(start, stop).unsqueeze(-1)
        scores.masked_fill_(key_positions.unsqueeze(0) > query_positions, float("-inf"))
        outputs.append(torch.matmul(torch.softmax(scores, dim=-1), value))

    return torch.cat(outputs, dim=-2).to(torch.bfloat16)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is required")
def test_flash_sdpa_matches_chunked_cpu_reference() -> None:
    # Use OPENSEARCH_SDPA_UT_SEQ_LEN=8048 and OPENSEARCH_SDPA_UT_HEADS=32
    # for the exact Qwen3-VL long-context regression shape. Defaults keep the
    # regular test inexpensive while exercising the same MUSA flash operator.
    sequence_length = int(os.getenv("OPENSEARCH_SDPA_UT_SEQ_LEN", "128"))
    heads = int(os.getenv("OPENSEARCH_SDPA_UT_HEADS", "4"))
    head_dim = 128
    generator = torch.Generator().manual_seed(20260723)
    shape = (1, heads, sequence_length, head_dim)
    query = torch.randn(shape, generator=generator).to(torch.bfloat16)
    key = torch.randn(shape, generator=generator).to(torch.bfloat16)
    value = torch.randn(shape, generator=generator).to(torch.bfloat16)
    scale = 1 / math.sqrt(head_dim)

    reference = _chunked_causal_reference(query, key, value, scale)
    causal_mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=torch.float32)
    causal_mask.masked_fill_(
        torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    actual = torch.nn.functional.scaled_dot_product_attention(
        query.to("musa"),
        key.to("musa"),
        value.to("musa"),
        attn_mask=causal_mask.to(torch.bfloat16).to("musa"),
        dropout_p=0.0,
        scale=scale,
        is_causal=False,
    ).cpu()

    mismatch = compare_tool.compare_tensors(actual, reference, atol=1e-2, rtol=1e-2)
    assert mismatch.sum().item() == 0
