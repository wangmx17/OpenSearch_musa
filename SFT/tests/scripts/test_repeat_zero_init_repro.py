import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "repeat_zero_init_repro.py"
SPEC = importlib.util.spec_from_file_location("repeat_zero_init_repro", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
repro = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repro
SPEC.loader.exec_module(repro)


def test_parse_hostfile_supports_slots_and_comments(tmp_path: Path) -> None:
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.1 slots=8\n# spare\n10.0.0.2\n", encoding="utf-8")

    assert repro.parse_hostfile(hostfile) == ["10.0.0.1", "10.0.0.2"]


def test_parse_hostfile_rejects_duplicates(tmp_path: Path) -> None:
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.1\n10.0.0.1 slots=8\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        repro.parse_hostfile(hostfile)


def test_parser_accepts_a_pool_specific_lock_file() -> None:
    args = repro._build_parser().parse_args(["hostfile.runtime", "--lock-file", ".zero_init_repro.pool_a.lock"])

    assert args.lock_file == Path(".zero_init_repro.pool_a.lock")


def test_summarize_latest_events_counts_done_ranks(tmp_path: Path) -> None:
    latest = tmp_path / "latest"
    done = tmp_path / "done"
    latest.mkdir()
    done.mkdir()
    (latest / "rank_00000.json").write_text(json.dumps({"rank": 0, "event": "zero_init_done"}))
    (latest / "rank_00001.json").write_text(json.dumps({"rank": 1, "event": "model_load_begin"}))
    (done / "rank_00000.json").write_text("{}")

    assert repro.summarize_latest_events(tmp_path) == {
        "done_count": 1,
        "event_counts": {"model_load_begin": 1, "zero_init_done": 1},
        "latest_count": 2,
        "ranks": [0, 1],
    }


def test_gpu_busy_count_checks_utilization_and_memory() -> None:
    output = """
0 X10000 | 0%    0MiB(81920MiB)
1 X10000 | 0%    12MiB(81920MiB)
2 X10000 | 80%    100MiB(81920MiB)
"""

    assert repro._gpu_busy_count(output) == 2


def test_start_launcher_is_non_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(repro.subprocess, "Popen", fake_popen)
    stream = io.StringIO()

    result = repro._start_launcher(["bash", "launch.sh", "hostfile"], env={"TEST": "1"}, log_stream=stream)

    assert result is sentinel
    assert captured["command"] == ["bash", "launch.sh", "hostfile"]
    assert captured["env"] == {"TEST": "1"}
    assert captured["stdout"] is stream
    assert captured["stderr"] is repro.subprocess.STDOUT


@pytest.mark.parametrize(("returncode", "failed"), [(None, False), (0, False), (1, True)])
def test_launcher_failure_only_rejects_nonzero_exit(returncode: int | None, failed: bool) -> None:
    assert repro._launcher_failed(returncode) is failed
