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

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch


pytest.importorskip("torch_musa")


_CHILD_PROGRAM = textwrap.dedent(
    r"""
    import gc
    import json

    import torch
    import torch_musa  # noqa: F401


    assert "expandable_segments:True" in __import__("os").environ["PYTORCH_MUSA_ALLOC_CONF"]
    assert torch.musa.is_available()
    torch.musa.set_device(0)
    baseline_memory_allocated = int(torch.musa.memory_allocated())

    sizes_mb = (1, 3, 7, 15, 31, 63)
    iterations = 4
    for cycle in range(iterations):
        live = []
        for index, size_mb in enumerate(sizes_mb):
            expected = (cycle + index) % 251
            tensor = torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8, device="musa")
            tensor.fill_(expected)
            torch.musa.synchronize()
            assert int(tensor[0].item()) == expected
            assert int(tensor[-1].item()) == expected
            live.append(tensor)
            if len(live) > 2:
                del live[0]

        live.clear()
        del tensor
        gc.collect()
        torch.musa.empty_cache()
        torch.musa.synchronize()

    result = {
        "cycles": iterations,
        "baseline_memory_allocated": baseline_memory_allocated,
        "max_memory_allocated": int(torch.musa.max_memory_allocated()),
        "memory_allocated": int(torch.musa.memory_allocated()),
    }
    print("MUSA_VMM_RESULT=" + json.dumps(result, sort_keys=True))
    """
)


@pytest.mark.slow
@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is required")
def test_expandable_segments_allocation_churn_completes_in_fresh_process() -> None:
    """Exercise VMM map/unmap churn without sharing allocator state with pytest."""
    env = os.environ.copy()
    env["PYTORCH_MUSA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )

    prefix = "MUSA_VMM_RESULT="
    result_line = next((line for line in completed.stdout.splitlines() if line.startswith(prefix)), None)
    assert result_line is not None, completed.stdout + completed.stderr
    result = json.loads(result_line.removeprefix(prefix))
    assert result["cycles"] == 4
    assert result["max_memory_allocated"] > result["baseline_memory_allocated"]
    assert result["memory_allocated"] <= result["baseline_memory_allocated"]
