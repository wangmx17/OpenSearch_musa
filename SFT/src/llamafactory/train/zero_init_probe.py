# Copyright 2026 the LlamaFactory team.
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

import faulthandler
import json
import os
import signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Optional

from transformers import TrainerCallback
from typing_extensions import override

from ..extras.misc import is_env_enabled


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


_STARTED_AT = time.monotonic()
_FAULT_HANDLER_READY = False


def zero_init_probe_enabled() -> bool:
    return is_env_enabled("OPENSEARCH_ZERO_INIT_PROBE")


def skip_diagnostic_final_save() -> bool:
    return is_env_enabled("OPENSEARCH_DIAGNOSTIC_SKIP_FINAL_SAVE")


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as err:
        raise ValueError(f"{name} must be an integer, got: {raw_value!r}.") from err

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got: {value}.")

    return value


def _rank() -> int:
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))


def _probe_dir(output_dir: Optional[str] = None) -> Path:
    configured = os.getenv("OPENSEARCH_ZERO_INIT_PROBE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if output_dir:
        return Path(output_dir).expanduser().resolve() / "zero_init_probe"
    return (Path.cwd() / "logs" / "zero_init_probe").resolve()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _register_fault_handler() -> None:
    global _FAULT_HANDLER_READY
    if _FAULT_HANDLER_READY or not hasattr(signal, "SIGUSR1"):
        return

    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (OSError, RuntimeError, ValueError):
        return

    _FAULT_HANDLER_READY = True


def record_zero_init_event(event: str, *, output_dir: Optional[str] = None, **fields: Any) -> Optional[Path]:
    if not zero_init_probe_enabled():
        return None

    _register_fault_handler()
    rank = _rank()
    payload = {
        "attempt_id": os.getenv("OPENSEARCH_ZERO_INIT_ATTEMPT_ID", "unknown"),
        "elapsed_seconds": round(time.monotonic() - _STARTED_AT, 6),
        "event": event,
        "hostname": socket.gethostname(),
        "local_rank": int(os.getenv("LOCAL_RANK", "0")),
        "node_rank": int(os.getenv("NODE_RANK", "0")),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "rank": rank,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 -- MUSA image uses Python 3.10
        "world_size": int(os.getenv("WORLD_SIZE", "1")),
        **fields,
    }
    probe_dir = _probe_dir(output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    timeline = probe_dir / f"rank_{rank:05d}.jsonl"
    with timeline.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()

    _atomic_write_json(probe_dir / "latest" / f"rank_{rank:05d}.json", payload)
    if event == "zero_init_done":
        _atomic_write_json(probe_dir / "done" / f"rank_{rank:05d}.json", payload)
    return timeline


class ZeroInitProbeCallback(TrainerCallback):
    r"""Mark ZeRO initialization completion and optionally hold each rank for an external supervisor."""

    def _hold_until_release(self, output_dir: str) -> NoReturn:
        probe_dir = _probe_dir(output_dir)
        release_file = probe_dir / "RELEASE"
        timeout_seconds = _int_env("OPENSEARCH_ZERO_INIT_PROBE_HOLD_TIMEOUT_SECONDS", 1800)
        heartbeat_seconds = _int_env("OPENSEARCH_ZERO_INIT_PROBE_HEARTBEAT_SECONDS", 10)
        deadline = time.monotonic() + timeout_seconds
        rank = _rank()

        while not release_file.exists():
            now = time.monotonic()
            if now >= deadline:
                record_zero_init_event("hold_timeout", output_dir=output_dir, timeout_seconds=timeout_seconds)
                raise TimeoutError(f"ZeRO-init probe rank {rank} timed out waiting for {release_file}.")

            _atomic_write_json(
                probe_dir / "heartbeat" / f"rank_{rank:05d}.json",
                {
                    "elapsed_seconds": round(now - _STARTED_AT, 6),
                    "event": "hold_heartbeat",
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "rank": rank,
                    "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 -- Python 3.10
                },
            )
            time.sleep(min(heartbeat_seconds, max(0.1, deadline - now)))

        record_zero_init_event("release_seen", output_dir=output_dir)
        raise SystemExit(0)

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        record_zero_init_event("zero_init_done", output_dir=args.output_dir, global_step=int(state.global_step))
        if is_env_enabled("OPENSEARCH_ZERO_INIT_PROBE_HOLD"):
            self._hold_until_release(args.output_dir)
