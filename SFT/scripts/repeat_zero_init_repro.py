#!/usr/bin/env python3
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

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAIN_PROCESS_MARKERS = ("train_30b_trace.sh", "llamafactory.cli", "torchrun")
GPU_STATUS_RE = re.compile(r"(?P<util>\d+)%\s+(?P<memory>\d+)MiB")


def parse_hostfile(path: Path) -> list[str]:
    hosts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line:
            hosts.append(line.split()[0])
    if not hosts:
        raise ValueError(f"no valid hosts found in: {path}")
    if len(set(hosts)) != len(hosts):
        raise ValueError(f"duplicate hosts found in: {path}")
    return hosts


def summarize_latest_events(probe_dir: Path) -> dict[str, Any]:
    latest: list[dict[str, Any]] = []
    for path in sorted((probe_dir / "latest").glob("rank_*.json")):
        try:
            latest.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue

    event_counts = Counter(str(record.get("event", "unknown")) for record in latest)
    done_count = len(list((probe_dir / "done").glob("rank_*.json")))
    return {
        "done_count": done_count,
        "event_counts": dict(sorted(event_counts.items())),
        "latest_count": len(latest),
        "ranks": sorted(int(record["rank"]) for record in latest if "rank" in record),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- the target MUSA image uses Python 3.10


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run(
    command: Sequence[str], *, timeout: int, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _start_launcher(
    command: Sequence[str], *, env: dict[str, str], log_stream: Any
) -> subprocess.Popen[str]:
    """Start the launcher without blocking the probe supervisor."""
    return subprocess.Popen(
        command,
        env=env,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _launcher_failed(returncode: int | None) -> bool:
    """A zero exit means the detached node jobs were dispatched successfully."""
    return returncode is not None and returncode != 0


def _ssh(host: str, command: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            host,
            command,
        ],
        timeout=timeout,
    )


def _gpu_busy_count(output: str) -> int:
    busy = 0
    for match in GPU_STATUS_RE.finditer(output):
        if int(match.group("util")) > 0 or int(match.group("memory")) > 0:
            busy += 1
    return busy


def _relevant_process_lines(output: str, workdir: Path) -> list[str]:
    result = []
    root = str(workdir)
    for line in output.splitlines():
        if root in line and any(marker in line for marker in TRAIN_PROCESS_MARKERS):
            result.append(line)
    return result


def inspect_host(host: str, workdir: Path) -> dict[str, Any]:
    command = "ps -eo pid=,pgid=,comm=,stat=,etime=,args=; mthreads-gmi 2>&1"
    completed = _ssh(host, command)
    if completed.returncode != 0:
        raise RuntimeError(f"host inspection failed for {host}:\n{completed.stdout}")
    return {
        "busy_gpus": _gpu_busy_count(completed.stdout),
        "host": host,
        "processes": _relevant_process_lines(completed.stdout, workdir),
        "raw": completed.stdout,
    }


def require_idle_hosts(hosts: Sequence[str], workdir: Path) -> None:
    problems = []
    for host in hosts:
        state = inspect_host(host, workdir)
        print(
            f"preflight host={host} busy_gpus={state['busy_gpus']} training_processes={len(state['processes'])}",
            flush=True,
        )
        if state["busy_gpus"] or state["processes"]:
            problems.append(state)
    if problems:
        raise RuntimeError("refusing to start: at least one target host is not idle")


def all_training_processes_exited(hosts: Sequence[str], workdir: Path) -> bool:
    return all(not inspect_host(host, workdir)["processes"] for host in hosts)


def collect_evidence(hosts: Sequence[str], workdir: Path, attempt_dir: Path, probe_dir: Path) -> None:
    evidence_dir = attempt_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(evidence_dir / "rank_summary.json", summarize_latest_events(probe_dir))

    snapshot = """
date -Ins
hostname
echo '=== processes ==='
ps -eo pid,ppid,pgid,stat,etime,wchan:32,args \
  | grep -E 'train_30b_trace|llamafactory.cli|torchrun|deepspeed' \
  | grep -v grep || true
echo '=== gpu ==='
mthreads-gmi 2>&1 || true
echo '=== network ==='
ip -br addr 2>&1 || true
echo '=== infiniband ==='
for p in /sys/class/infiniband/*/ports/*; do
  [ -d "$p" ] || continue
  printf '%s state=' "$p"
  cat "$p/state" 2>/dev/null || true
done
echo '=== kernel tail ==='
dmesg --ctime 2>&1 | tail -200 || true
""".strip()
    for host in hosts:
        completed = _ssh(host, snapshot, timeout=60)
        (evidence_dir / f"host_{host}.log").write_text(completed.stdout, encoding="utf-8")


def _write_attempt_status(
    attempt_dir: Path,
    *,
    attempt: int,
    status: str,
    started_at: str,
    probe_dir: Path,
    reason: str = "",
) -> None:
    _atomic_json(
        attempt_dir / "status.json",
        {
            "attempt": attempt,
            "probe": summarize_latest_events(probe_dir),
            "reason": reason,
            "started_at": started_at,
            "status": status,
            "updated_at": _utc_now(),
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeatedly launch 4-node training and detect ZeRO-init hangs.")
    parser.add_argument("hostfile", type=Path)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--base-port", type=int, default=35200)
    parser.add_argument("--cooldown-seconds", type=int, default=15)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--hold-timeout-seconds", type=int, default=1800)
    parser.add_argument("--launch-timeout-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--release-timeout-seconds", type=int, default=180)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--launcher", type=Path, default=Path("launch_train_30b_trace.sh"))
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path(".zero_init_repro.lock"),
        help="Lock path, relative to --workdir unless absolute. Use a distinct lock for disjoint host pools.",
    )
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--disable-mccl-preflight", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "attempts",
        "cooldown_seconds",
        "gpus_per_node",
        "hold_timeout_seconds",
        "launch_timeout_seconds",
        "poll_seconds",
        "release_timeout_seconds",
        "timeout_seconds",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.base_port < 1024 or args.base_port + args.attempts > 65535:
        raise ValueError("the requested master-port range is outside 1024..65535")


def run(args: argparse.Namespace) -> int:
    if os.getenv("OPENSEARCH_ZERO_INIT_REPRO_ALLOW") != "1":
        raise RuntimeError("refusing to start repeated four-node training without OPENSEARCH_ZERO_INIT_REPRO_ALLOW=1")

    _validate_args(args)
    workdir = args.workdir.expanduser().resolve()
    hostfile = args.hostfile.expanduser().resolve()
    launcher = args.launcher
    if not launcher.is_absolute():
        launcher = workdir / launcher
    launcher = launcher.resolve()
    if not hostfile.is_file():
        raise FileNotFoundError(hostfile)
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    if not (workdir / "src" / "llamafactory").is_dir():
        raise RuntimeError(f"not an SFT project directory: {workdir}")

    hosts = parse_hostfile(hostfile)
    expected_ranks = len(hosts) * args.gpus_per_node
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = (args.log_root or (workdir / "logs" / f"zero_init_repro_{run_id}")).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=False)
    lock_path = args.lock_file.expanduser()
    if not lock_path.is_absolute():
        lock_path = workdir / lock_path
    lock_path = lock_path.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, f"pid={os.getpid()} log_root={log_root}\n".encode())
    os.close(lock_fd)

    summary_path = log_root / "summary.jsonl"
    print(f"run_id={run_id} hosts={len(hosts)} expected_ranks={expected_ranks} log_root={log_root}")
    try:
        require_idle_hosts(hosts, workdir)
        for attempt in range(1, args.attempts + 1):
            attempt_id = f"{run_id}_{attempt:03d}"
            attempt_dir = log_root / f"attempt_{attempt:03d}"
            probe_dir = attempt_dir / "probe"
            node_log_dir = attempt_dir / "node_logs"
            attempt_dir.mkdir(parents=True)
            started_at = _utc_now()
            env = os.environ.copy()
            env.update(
                {
                    "LOG_DIR": str(node_log_dir),
                    "MASTER_PORT": str(args.base_port + attempt - 1),
                    "MCCL_DEBUG": env.get("MCCL_DEBUG", "INFO"),
                    "MCCL_DEBUG_SUBSYS": env.get("MCCL_DEBUG_SUBSYS", "INIT,COLL,NET,ENV"),
                    "NPROC_PER_NODE": str(args.gpus_per_node),
                    "OPENSEARCH_DIAGNOSTIC_SKIP_FINAL_SAVE": "1",
                    "OPENSEARCH_MCCL_PREFLIGHT": "0" if args.disable_mccl_preflight else "1",
                    "OPENSEARCH_ZERO_INIT_ATTEMPT_ID": attempt_id,
                    "OPENSEARCH_ZERO_INIT_PROBE": "1",
                    "OPENSEARCH_ZERO_INIT_PROBE_DIR": str(probe_dir),
                    "OPENSEARCH_ZERO_INIT_PROBE_HEARTBEAT_SECONDS": str(args.poll_seconds),
                    "OPENSEARCH_ZERO_INIT_PROBE_HOLD": "1",
                    "OPENSEARCH_ZERO_INIT_PROBE_HOLD_TIMEOUT_SECONDS": str(args.hold_timeout_seconds),
                    "WORKDIR": str(workdir),
                }
            )
            print(f"attempt={attempt} state=launching port={env['MASTER_PORT']} probe_dir={probe_dir}", flush=True)
            launcher_log_path = attempt_dir / "launcher.log"
            with launcher_log_path.open("w", encoding="utf-8") as launcher_log:
                launched = _start_launcher(["bash", str(launcher), str(hostfile)], env=env, log_stream=launcher_log)
                deadline = time.monotonic() + args.timeout_seconds
                launch_deadline = time.monotonic() + args.launch_timeout_seconds
                previous_summary: dict[str, Any] | None = None
                outcome = ""
                reason = ""
                while time.monotonic() < deadline:
                    summary = summarize_latest_events(probe_dir)
                    if summary != previous_summary:
                        print(
                            f"attempt={attempt} done={summary['done_count']}/{expected_ranks} "
                            f"latest={summary['event_counts']}",
                            flush=True,
                        )
                        previous_summary = summary
                    if summary["done_count"] == expected_ranks:
                        outcome = "zero_init_pass"
                        break

                    launcher_returncode = launched.poll()
                    if _launcher_failed(launcher_returncode):
                        outcome = "launch_failed"
                        reason = f"launcher exited with status {launcher_returncode} before ZeRO init completed"
                        break
                    if summary["latest_count"] == 0 and time.monotonic() >= launch_deadline:
                        outcome = "launch_failed"
                        reason = f"no rank probe appeared within {args.launch_timeout_seconds}s"
                        break
                    if len(list((probe_dir / "nodes").glob("node_*.exit_code"))) == len(hosts):
                        outcome = "early_exit"
                        reason = "all node launchers exited before every rank completed ZeRO init"
                        break
                    time.sleep(args.poll_seconds)

                if not outcome:
                    outcome = "hang"
                    reason = f"fewer than {expected_ranks} ranks completed ZeRO init within {args.timeout_seconds}s"

                if outcome != "zero_init_pass":
                    collect_evidence(hosts, workdir, attempt_dir, probe_dir)
                    _write_attempt_status(
                        attempt_dir,
                        attempt=attempt,
                        status=outcome,
                        started_at=started_at,
                        probe_dir=probe_dir,
                        reason=reason,
                    )
                    with summary_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"attempt": attempt, "reason": reason, "status": outcome}) + "\n")
                    print(f"attempt={attempt} state={outcome}; preserving live state and stopping", flush=True)
                    return 2

                (probe_dir / "RELEASE").touch()
                release_deadline = time.monotonic() + args.release_timeout_seconds
                while time.monotonic() < release_deadline:
                    if launched.poll() is not None and all_training_processes_exited(hosts, workdir):
                        break
                    time.sleep(min(args.poll_seconds, 5))
                else:
                    reason = "launcher or training processes did not exit after the probe release"
                    collect_evidence(hosts, workdir, attempt_dir, probe_dir)
                    _write_attempt_status(
                        attempt_dir,
                        attempt=attempt,
                        status="release_failed",
                        started_at=started_at,
                        probe_dir=probe_dir,
                        reason=reason,
                    )
                    print(f"attempt={attempt} state=release_failed; preserving live state and stopping", flush=True)
                    return 3

            require_idle_hosts(hosts, workdir)
            _write_attempt_status(
                attempt_dir,
                attempt=attempt,
                status="passed",
                started_at=started_at,
                probe_dir=probe_dir,
            )
            with summary_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"attempt": attempt, "status": "passed"}) + "\n")
            print(f"attempt={attempt} state=passed", flush=True)
            if attempt < args.attempts:
                time.sleep(args.cooldown_seconds)
    finally:
        lock_path.unlink(missing_ok=True)

    print(f"all {args.attempts} attempts passed ZeRO initialization")
    return 0


def main() -> int:
    try:
        return run(_build_parser().parse_args())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, subprocess.TimeoutExpired) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
