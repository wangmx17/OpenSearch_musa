import pytest
import torch

from llamafactory.model.model_utils.moe import stable_topk


def _tied_router_scores(device: torch.device) -> torch.Tensor:
    scores = torch.zeros((2, 128), dtype=torch.float32, device=device)
    scores[:, 81] = 1.0
    scores[:, 111] = 1.0
    return scores


def test_stable_topk_breaks_ties_by_original_index() -> None:
    values, indices = stable_topk(_tied_router_scores(torch.device("cpu")), k=1)

    torch.testing.assert_close(values, torch.ones_like(values))
    torch.testing.assert_close(indices, torch.full_like(indices, 81))


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is unavailable")
def test_stable_topk_matches_cpu_on_musa() -> None:
    cpu_values, cpu_indices = stable_topk(_tied_router_scores(torch.device("cpu")), k=2)
    musa_values, musa_indices = stable_topk(_tied_router_scores(torch.device("musa")), k=2)

    torch.testing.assert_close(musa_values.cpu(), cpu_values, rtol=0, atol=0)
    torch.testing.assert_close(musa_indices.cpu(), cpu_indices, rtol=0, atol=0)
