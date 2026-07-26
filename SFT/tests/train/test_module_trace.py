import json

import torch

from llamafactory.train.module_trace import install_module_precision_trace, tensor_summary


class _ToyAttention(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _ToyAttention()

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.attention(input_ids.float())}


def test_tensor_summary_has_exact_hash_for_small_tensor() -> None:
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    summary = tensor_summary(tensor)

    assert summary["shape"] == [2, 2]
    assert summary["sample_sha256"] == summary["full_sha256"]
    assert summary["sample_values"] == [1, 2, 3, 4]


def test_tensor_summary_hashes_scalar_tensor() -> None:
    summary = tensor_summary(torch.tensor(1.25))

    assert summary["shape"] == []
    assert summary["sample_sha256"] == summary["full_sha256"]
    assert summary["sample_values"] == [1.25]


def test_tensor_summary_serializes_nonfinite_values() -> None:
    tensor = torch.tensor([float("nan"), float("inf"), float("-inf"), 2.0])
    summary = tensor_summary(tensor)

    assert summary["sample_values"] == ["nan", "+inf", "-inf", 2.0]
    assert summary["sample_nan_count"] == 1
    assert summary["sample_posinf_count"] == 1
    assert summary["sample_neginf_count"] == 1
    json.dumps(summary, allow_nan=False)


def test_tensor_summary_serializes_overflowed_statistic() -> None:
    maximum = torch.finfo(torch.float32).max
    summary = tensor_summary(torch.tensor([maximum, maximum]))

    assert summary["sample_l2"] == "+inf"
    json.dumps(summary, allow_nan=False)


def test_module_trace_records_first_root_call(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENSEARCH_MODULE_TRACE", "1")
    monkeypatch.setenv("OPENSEARCH_MODULE_TRACE_RANKS", "0")
    monkeypatch.setenv("OPENSEARCH_MODULE_TRACE_DIR", str(tmp_path))
    model = _ToyModel()

    assert install_module_precision_trace(model) == 1
    model(torch.tensor([[1, 2]], dtype=torch.int64))
    model(torch.tensor([[3, 4]], dtype=torch.int64))

    records = [json.loads(line) for line in (tmp_path / "module_rank0.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "metadata",
        "root_input",
        "module_forward",
        "root_output",
    ]
    module_record = records[2]
    assert module_record["module"] == "attention"
    assert module_record["output"]["sample_values"] == [2.0, 3.0]
