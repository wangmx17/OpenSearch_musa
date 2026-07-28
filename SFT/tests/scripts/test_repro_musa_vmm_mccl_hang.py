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


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "repro_musa_vmm_mccl_hang.py"
SPEC = importlib.util.spec_from_file_location("repro_musa_vmm_mccl_hang", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
repro = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repro
SPEC.loader.exec_module(repro)


def test_parse_sizes_mb() -> None:
    assert repro.parse_sizes_mb("1, 3,7,15") == [1, 3, 7, 15]
    with pytest.raises(Exception, match="positive"):
        repro.parse_sizes_mb("1,0,3")


def test_shards_cover_tensor_without_overlap() -> None:
    numel = 103
    bounds = [repro.shard_bounds(numel, rank, 8) for rank in range(8)]
    covered = [index for start, length in bounds for index in range(start, start + length)]

    assert covered == list(range(numel))


def test_rank_info_accepts_openmpi_environment() -> None:
    info = repro.rank_info_from_env(
        {
            "OMPI_COMM_WORLD_RANK": "17",
            "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            "OMPI_COMM_WORLD_SIZE": "32",
        }
    )

    assert info == repro.RankInfo(rank=17, local_rank=1, world_size=32)


def test_openmpi_rank_overrides_stale_generic_rank_environment() -> None:
    info = repro.rank_info_from_env(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "OMPI_COMM_WORLD_RANK": "17",
            "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            "OMPI_COMM_WORLD_SIZE": "32",
        }
    )

    assert info == repro.RankInfo(rank=17, local_rank=1, world_size=32)
