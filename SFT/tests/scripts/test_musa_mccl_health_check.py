# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "musa_mccl_health_check.py"
SPEC = importlib.util.spec_from_file_location("musa_mccl_health_check", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
health_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = health_check
SPEC.loader.exec_module(health_check)


def test_parse_hostfile_uses_one_entry_per_node(tmp_path: Path) -> None:
    hostfile = tmp_path / "hostfile"
    hostfile.write_text(
        "10.121.33.83 slots=8\n"
        "10.121.32.15 slots=8 # current fourth worker\n"
        "\n"
        "worker33032\n",
        encoding="utf-8",
    )

    entries = health_check.parse_hostfile(hostfile)

    assert [entry.host for entry in entries] == ["10.121.33.83", "10.121.32.15", "worker33032"]
    assert [entry.slots for entry in entries] == [8, 8, None]


def test_parse_hostfile_rejects_duplicate_nodes(tmp_path: Path) -> None:
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("worker-a slots=8\nworker-a slots=8\n", encoding="utf-8")

    with pytest.raises(health_check.HealthCheckError, match="duplicate host"):
        health_check.parse_hostfile(hostfile)


def test_parse_and_validate_successful_mccl_output() -> None:
    output = """
       1048576       262144     float     sum      -1    120.0    8.73    16.90      0    119.0    8.81    17.06      0
    # Out of bounds values : 0 OK
    """

    parsed = health_check.parse_mccl_test_output(output)
    failures = health_check.validate_mccl_result(parsed, returncode=0, timed_out=False, min_busbw_gbps=0)

    assert len(parsed.rows) == 1
    assert parsed.out_of_bounds == 0
    assert failures == []


@pytest.mark.parametrize(
    ("output", "returncode", "timed_out", "expected"),
    [
        (
            "1048576 262144 float sum -1 120 8.7 16.9 2 119 8.8 17.0 0\n# Out of bounds values : 2 FAIL",
            0,
            False,
            "wrong values",
        ),
        ("", 124, True, "external timeout"),
        ("# Out of bounds values : 0 OK", 0, False, "no MCCL performance rows"),
        (
            "1048576 262144 float sum -1 120 8.7 nan 0 119 8.8 inf 0\n# Out of bounds values : 0 OK",
            0,
            False,
            "non-finite or non-positive",
        ),
    ],
)
def test_validate_mccl_output_reports_hard_failures(
    output: str, returncode: int, timed_out: bool, expected: str
) -> None:
    parsed = health_check.parse_mccl_test_output(output)
    failures = health_check.validate_mccl_result(
        parsed,
        returncode=returncode,
        timed_out=timed_out,
        min_busbw_gbps=0,
    )

    assert expected in "\n".join(failures)


def test_mpirun_command_covers_all_local_gpus_with_one_process_per_node() -> None:
    command = health_check.build_mpirun_command(
        mpirun="/usr/local/openmpi/bin/mpirun",
        test_binary="/shared/mccl-test/build/all_reduce_perf",
        smoke_hostfile="/tmp/hostfile.smoke",
        node_count=4,
        gpus_per_node=8,
        message_bytes="1M",
        warmup_iterations=1,
        iterations=2,
        stream_timeout_seconds=60,
        mpi_timeout_seconds=90,
        env={"PATH": "/usr/bin", "MCCL_PROTOS": "2"},
    )

    assert command[command.index("-np") + 1] == "4"
    assert command[command.index("--map-by") + 1] == "ppr:1:node"
    assert command[command.index("-g") + 1] == "8"
    assert command[command.index("-b") + 1] == "1M"
    assert command[command.index("-e") + 1] == "1M"
    assert command[command.index("-c") + 1] == "1"


def test_health_environment_matches_training_channel_and_traffic_class_defaults() -> None:
    env = health_check.build_mccl_environment({"PATH": "/usr/bin"})

    assert env["MCCL_MAX_NCHANNELS"] == "14"
    assert env["MCCL_IB_TC"] == "41"
