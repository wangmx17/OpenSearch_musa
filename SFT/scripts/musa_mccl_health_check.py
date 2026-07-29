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

"""Run a small, correctness-first MCCL preflight before multi-node training."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_MCCL_TEST_BINARIES = (
    "/home/jd/liang.geng/mccl-test-master/build/all_reduce_perf",
    "/usr/local/musa/mccl_test/all_reduce_perf",
)
DEFAULT_MPIRUN_BINARIES = ("/usr/local/openmpi/bin/mpirun", "mpirun")
MCCL_ENV_DEFAULTS = {
    "MCCL_PROTOS": "2",
    "MCCL_ALGOS": "1",
    "MCCL_BUFFSIZE": "20971520",
    "MCCL_MAX_NCHANNELS": "14",
    "MCCL_CHECK_POINTERS": "0",
    "MCCL_IB_GID_INDEX": "3",
    "MCCL_IB_TC": "41",
    "MCCL_IB_TIMEOUT": "22",
    "MCCL_DEBUG": "WARN",
}
PROPAGATED_ENV_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "MCCL_PROTOS",
    "MCCL_ALGOS",
    "MCCL_BUFFSIZE",
    "MCCL_MAX_NCHANNELS",
    "MCCL_CHECK_POINTERS",
    "MCCL_IB_GID_INDEX",
    "MCCL_IB_TC",
    "MCCL_IB_TIMEOUT",
    "MCCL_IB_RETRY_CNT",
    "MCCL_CROSS_NIC",
    "MCCL_SOCKET_IFNAME",
    "MCCL_IB_HCA",
    "MCCL_DEBUG",
    "MCCL_DEBUG_SUBSYS",
)


class HealthCheckError(RuntimeError):
    """Raised when the preflight configuration or result is invalid."""


@dataclass(frozen=True)
class HostEntry:
    host: str
    slots: int | None


@dataclass(frozen=True)
class PerfRow:
    size_bytes: int
    out_of_place_busbw_gbps: float
    out_of_place_wrong: int
    in_place_busbw_gbps: float
    in_place_wrong: int


@dataclass(frozen=True)
class ParsedOutput:
    rows: list[PerfRow]
    out_of_bounds: int | None
    avg_busbw_gbps: float | None


@dataclass(frozen=True)
class RunResult:
    returncode: int
    output: str
    elapsed_seconds: float
    timed_out: bool


def parse_hostfile(path: str | Path) -> list[HostEntry]:
    """Parse OpenMPI-style hosts while rejecting an empty or ambiguous topology."""
    hostfile = Path(path)
    if not hostfile.is_file():
        raise HealthCheckError(f"hostfile not found: {hostfile}")

    entries: list[HostEntry] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(hostfile.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue

        fields = line.split()
        host = fields[0]
        slots: int | None = None
        for field in fields[1:]:
            if field.startswith("slots="):
                try:
                    slots = int(field.removeprefix("slots="))
                except ValueError as exc:
                    raise HealthCheckError(f"invalid slots value on hostfile line {line_number}: {field}") from exc
                if slots < 1:
                    raise HealthCheckError(f"slots must be positive on hostfile line {line_number}: {field}")

        if host in seen:
            raise HealthCheckError(f"duplicate host in hostfile: {host}")
        seen.add(host)
        entries.append(HostEntry(host=host, slots=slots))

    if not entries:
        raise HealthCheckError(f"hostfile has no usable hosts: {hostfile}")
    return entries


def parse_mccl_test_output(output: str) -> ParsedOutput:
    """Extract correctness and bus-bandwidth fields from MCCL tests output."""
    rows: list[PerfRow] = []
    out_of_bounds: int | None = None
    avg_busbw_gbps: float | None = None

    for line in output.splitlines():
        match = re.search(r"Out of bounds values\s*:\s*(\d+)", line, flags=re.IGNORECASE)
        if match:
            out_of_bounds = int(match.group(1))

        match = re.search(
            r"Avg bus bandwidth\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            avg_busbw_gbps = float(match.group(1))

        fields = line.split()
        if len(fields) < 13 or not fields[0].isdigit():
            continue

        try:
            row = PerfRow(
                size_bytes=int(fields[0]),
                out_of_place_busbw_gbps=float(fields[7]),
                out_of_place_wrong=int(fields[8]),
                in_place_busbw_gbps=float(fields[11]),
                in_place_wrong=int(fields[12]),
            )
        except ValueError:
            continue
        rows.append(row)

    return ParsedOutput(rows=rows, out_of_bounds=out_of_bounds, avg_busbw_gbps=avg_busbw_gbps)


def validate_mccl_result(
    parsed: ParsedOutput,
    returncode: int,
    timed_out: bool,
    min_busbw_gbps: float,
) -> list[str]:
    """Return all hard-failure reasons for a completed health check."""
    failures: list[str] = []
    if timed_out:
        failures.append("mpirun exceeded the external timeout")
    if returncode != 0:
        failures.append(f"all_reduce_perf exited with code {returncode}")
    if not parsed.rows and parsed.avg_busbw_gbps is None:
        failures.append("no MCCL performance rows or average bandwidth summary were found")
    if parsed.out_of_bounds is None:
        failures.append("the correctness summary '# Out of bounds values' is missing")
    elif parsed.out_of_bounds != 0:
        failures.append(f"out-of-bounds/corrupt values: {parsed.out_of_bounds}")

    wrong_values = sum(row.out_of_place_wrong + row.in_place_wrong for row in parsed.rows)
    if wrong_values:
        failures.append(f"MCCL reported {wrong_values} wrong values")

    observed_bandwidths = [
        bandwidth
        for row in parsed.rows
        for bandwidth in (row.out_of_place_busbw_gbps, row.in_place_busbw_gbps)
    ]
    if not observed_bandwidths and parsed.avg_busbw_gbps is not None:
        # MCCL INFO output can be interleaved into a nccl-tests row when many
        # local devices share stdout. The final average is emitted atomically
        # and remains a valid liveness/correctness and bandwidth observation.
        observed_bandwidths.append(parsed.avg_busbw_gbps)
    bandwidths = [bandwidth for bandwidth in observed_bandwidths if math.isfinite(bandwidth)]
    invalid_bandwidths = [
        bandwidth
        for bandwidth in observed_bandwidths
        if not math.isfinite(bandwidth) or bandwidth <= 0
    ]
    if invalid_bandwidths:
        failures.append(f"MCCL reported {len(invalid_bandwidths)} non-finite or non-positive bus bandwidth values")
    if min_busbw_gbps > 0:
        if not bandwidths:
            failures.append("no finite bus bandwidth value was found")
        elif min(bandwidths) < min_busbw_gbps:
            failures.append(
                f"minimum bus bandwidth {min(bandwidths):.3f} GB/s is below {min_busbw_gbps:.3f} GB/s"
            )
    return failures


def _resolve_executable(explicit: str | None, candidates: tuple[str, ...], name: str) -> str:
    search = (explicit,) if explicit else candidates
    for candidate in search:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return str(candidate_path.resolve())
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    checked = ", ".join(str(item) for item in search)
    raise HealthCheckError(f"{name} was not found; checked: {checked}")


def _pick_existing_path(root: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        if (Path(root) / name).exists():
            return name
    return None


def build_mccl_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Build the same baseline MCCL environment used by the 30B training script."""
    env = dict(os.environ if source is None else source)
    for key, value in MCCL_ENV_DEFAULTS.items():
        env.setdefault(key, value)

    if not env.get("MCCL_SOCKET_IFNAME"):
        socket_ifname = _pick_existing_path(
            "/sys/class/net", ("bond1", "bond0", "ib0", "ib1", "eth0", "eno1", "ens1", "enp1s0")
        )
        if socket_ifname:
            env["MCCL_SOCKET_IFNAME"] = socket_ifname

    if not env.get("MCCL_IB_HCA"):
        ib_hca = _pick_existing_path(
            "/sys/class/infiniband", ("mubd0", "mubd1", "mlx5_bond_0", "mlx5_bond_1")
        )
        if ib_hca:
            env["MCCL_IB_HCA"] = ib_hca
    return env


def build_mpirun_command(
    *,
    mpirun: str,
    test_binary: str,
    smoke_hostfile: str,
    node_count: int,
    gpus_per_node: int,
    message_bytes: str,
    warmup_iterations: int,
    iterations: int,
    stream_timeout_seconds: int,
    mpi_timeout_seconds: int,
    env: dict[str, str],
) -> list[str]:
    """Build a four-node/32-rank command with one MPI process per node."""
    command = [
        mpirun,
        "--allow-run-as-root",
        "--hostfile",
        smoke_hostfile,
        "--map-by",
        "ppr:1:node",
        "-np",
        str(node_count),
        "--bind-to",
        "none",
        "--timeout",
        str(mpi_timeout_seconds),
        "--prtemca",
        "prte_keep_fqdn_hostnames",
        "1",
    ]
    if env.get("MCCL_SOCKET_IFNAME"):
        command.extend(("--mca", "btl_tcp_if_include", env["MCCL_SOCKET_IFNAME"]))

    for key in PROPAGATED_ENV_KEYS:
        value = env.get(key)
        if value:
            command.extend(("-x", f"{key}={value}"))

    command.extend(
        (
            test_binary,
            "-b",
            message_bytes,
            "-e",
            message_bytes,
            "-f",
            "2",
            "-g",
            str(gpus_per_node),
            "-w",
            str(warmup_iterations),
            "-n",
            str(iterations),
            "-c",
            "1",
            "-z",
            "1",
            "-T",
            str(stream_timeout_seconds),
        )
    )
    return command


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: int = 10) -> str:
    if process.poll() is not None:
        return ""

    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        output, _ = process.communicate(timeout=grace_seconds)
        return output
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        output, _ = process.communicate()
        return output


def run_command(command: list[str], cwd: str, env: dict[str, str], timeout_seconds: int) -> RunResult:
    """Run mpirun under an outer process-group timeout."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        return RunResult(
            returncode=process.returncode,
            output=output,
            elapsed_seconds=time.monotonic() - started,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        remaining = _terminate_process_group(process)
        return RunResult(
            returncode=124,
            output=remaining or partial,
            elapsed_seconds=time.monotonic() - started,
            timed_out=True,
        )


def _validate_message_size(value: str) -> str:
    if not re.fullmatch(r"[1-9]\d*(?:[KMG])?", value, flags=re.IGNORECASE):
        raise argparse.ArgumentTypeError(f"invalid byte size: {value}")
    return value.upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hostfile", required=True, help="Current runtime hostfile; comments and slots=N are accepted.")
    parser.add_argument("--workdir", default=os.getcwd(), help="Shared working directory used by remote MPI ranks.")
    parser.add_argument("--log-dir", default=None, help="Output directory (default: logs/mccl_health_<timestamp>).")
    parser.add_argument("--test-bin", default=None, help="Explicit all_reduce_perf binary path.")
    parser.add_argument("--mpirun", default=None, help="Explicit OpenMPI mpirun path.")
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--message-bytes", type=_validate_message_size, default="1M")
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--stream-timeout-seconds", type=int, default=60)
    parser.add_argument("--mpi-timeout-seconds", type=int, default=90)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--min-busbw-gbps",
        type=float,
        default=0.0,
        help="Optional hard bandwidth threshold. Zero checks liveness and correctness only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without using a GPU.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        hosts = parse_hostfile(args.hostfile)
        if args.gpus_per_node < 1:
            raise HealthCheckError("--gpus-per-node must be positive")
        if min(
            args.warmup_iterations,
            args.iterations,
            args.stream_timeout_seconds,
            args.mpi_timeout_seconds,
            args.timeout_seconds,
        ) < 1:
            raise HealthCheckError("iteration counts and timeouts must be positive")
        if args.min_busbw_gbps < 0:
            raise HealthCheckError("--min-busbw-gbps cannot be negative")

        workdir = str(Path(args.workdir).resolve())
        if not Path(workdir).is_dir():
            raise HealthCheckError(f"workdir not found: {workdir}")
        mpirun = _resolve_executable(args.mpirun, DEFAULT_MPIRUN_BINARIES, "mpirun")
        test_binary = _resolve_executable(args.test_bin, DEFAULT_MCCL_TEST_BINARIES, "all_reduce_perf")
        env = build_mccl_environment()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(args.log_dir or f"logs/mccl_health_{timestamp}").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="musa_mccl_health_") as temporary_dir:
            smoke_hostfile = Path(temporary_dir) / "hostfile.smoke"
            smoke_hostfile.write_text(
                "".join(f"{entry.host} slots=1\n" for entry in hosts),
                encoding="utf-8",
            )
            command = build_mpirun_command(
                mpirun=mpirun,
                test_binary=test_binary,
                smoke_hostfile=str(smoke_hostfile),
                node_count=len(hosts),
                gpus_per_node=args.gpus_per_node,
                message_bytes=args.message_bytes,
                warmup_iterations=args.warmup_iterations,
                iterations=args.iterations,
                stream_timeout_seconds=args.stream_timeout_seconds,
                mpi_timeout_seconds=args.mpi_timeout_seconds,
                env=env,
            )

            print(
                f"[MCCL preflight] nodes={len(hosts)} ranks={len(hosts) * args.gpus_per_node} "
                f"message={args.message_bytes}",
                flush=True,
            )
            print(f"[MCCL preflight] log_dir={log_dir}", flush=True)
            print(f"[MCCL preflight] command={shlex.join(command)}", flush=True)
            if args.dry_run:
                return 0

            result = run_command(command, cwd=workdir, env=env, timeout_seconds=args.timeout_seconds)

        raw_log = log_dir / "all_reduce_perf.log"
        raw_log.write_text(result.output, encoding="utf-8")
        parsed = parse_mccl_test_output(result.output)
        failures = validate_mccl_result(
            parsed,
            returncode=result.returncode,
            timed_out=result.timed_out,
            min_busbw_gbps=args.min_busbw_gbps,
        )
        bandwidths = [
            bandwidth
            for row in parsed.rows
            for bandwidth in (row.out_of_place_busbw_gbps, row.in_place_busbw_gbps)
            if math.isfinite(bandwidth)
        ]
        if not bandwidths and parsed.avg_busbw_gbps is not None and math.isfinite(parsed.avg_busbw_gbps):
            bandwidths.append(parsed.avg_busbw_gbps)
        summary = {
            "status": "fail" if failures else "pass",
            "hosts": [asdict(entry) for entry in hosts],
            "world_size": len(hosts) * args.gpus_per_node,
            "elapsed_seconds": result.elapsed_seconds,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "out_of_bounds": parsed.out_of_bounds,
            "avg_busbw_gbps": parsed.avg_busbw_gbps,
            "rows": [asdict(row) for row in parsed.rows],
            "min_busbw_gbps": min(bandwidths) if bandwidths else None,
            "max_busbw_gbps": max(bandwidths) if bandwidths else None,
            "failures": failures,
            "raw_log": str(raw_log),
        }
        (log_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if failures:
            print(f"[MCCL preflight] FAIL ({result.elapsed_seconds:.2f}s)", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            print(f"  - raw log: {raw_log}", file=sys.stderr)
            return 1

        minimum_bandwidth = summary["min_busbw_gbps"]
        bandwidth_text = f"{minimum_bandwidth:.3f} GB/s" if minimum_bandwidth is not None else "n/a"
        print(f"[MCCL preflight] PASS ({result.elapsed_seconds:.2f}s, min busbw={bandwidth_text})", flush=True)
        return 0
    except HealthCheckError as exc:
        print(f"[MCCL preflight] configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
