import json
from types import SimpleNamespace

import pytest

from llamafactory.train.zero_init_probe import (
    ZeroInitProbeCallback,
    record_zero_init_event,
    skip_diagnostic_final_save,
)


def _enable_probe(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENSEARCH_ZERO_INIT_PROBE", "1")
    monkeypatch.setenv("OPENSEARCH_ZERO_INIT_PROBE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENSEARCH_ZERO_INIT_ATTEMPT_ID", "attempt_007")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "32")


def test_record_zero_init_event_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("OPENSEARCH_ZERO_INIT_PROBE", raising=False)
    monkeypatch.setenv("OPENSEARCH_ZERO_INIT_PROBE_DIR", str(tmp_path))

    assert record_zero_init_event("model_load_begin") is None
    assert list(tmp_path.iterdir()) == []


def test_record_zero_init_done_writes_rank_timeline_and_marker(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _enable_probe(monkeypatch, tmp_path)

    timeline = record_zero_init_event("zero_init_done", global_step=0)

    assert timeline == tmp_path / "rank_00003.jsonl"
    payload = json.loads(timeline.read_text(encoding="utf-8"))
    assert payload["attempt_id"] == "attempt_007"
    assert payload["event"] == "zero_init_done"
    assert payload["local_rank"] == 1
    assert payload["rank"] == 3
    assert payload["world_size"] == 32
    assert json.loads((tmp_path / "done" / "rank_00003.json").read_text())["event"] == "zero_init_done"


def test_callback_releases_rank_without_entering_training(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _enable_probe(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENSEARCH_ZERO_INIT_PROBE_HOLD", "1")
    (tmp_path / "RELEASE").touch()

    with pytest.raises(SystemExit) as exc_info:
        ZeroInitProbeCallback().on_train_begin(
            SimpleNamespace(output_dir=str(tmp_path / "output")),
            SimpleNamespace(global_step=0),
            SimpleNamespace(),
        )

    assert exc_info.value.code == 0
    events = [json.loads(line)["event"] for line in (tmp_path / "rank_00003.jsonl").read_text().splitlines()]
    assert events == ["zero_init_done", "release_seen"]


def test_skip_final_save_is_explicitly_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENSEARCH_DIAGNOSTIC_SKIP_FINAL_SAVE", raising=False)
    assert not skip_diagnostic_final_save()

    monkeypatch.setenv("OPENSEARCH_DIAGNOSTIC_SKIP_FINAL_SAVE", "1")
    assert skip_diagnostic_final_save()
