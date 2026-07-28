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

"""Stress the MUSA expandable allocator, MCCL, or their ZeRO-like interaction."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
DEFAULT_ALLOCATOR_CONF = "expandable_segments:True,garbage_collection_threshold:0.8"


class ProbeError(RuntimeError):
    """Raised when the probe cannot safely run or detects corruption."""


@dataclass(frozen=True)
class RankInfo:
    rank: int
    local_rank: int
    world_size: int


def parse_sizes_mb(value: str) -> list[int]:
    """Parse a comma-separated positive MiB allocation pattern."""
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid MiB list: {value}") from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("allocation sizes must be positive MiB values")
    return sizes


def shard_bounds(numel: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return the ZeRO-like contiguous shard start and length for one rank."""
    if numel < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid shard geometry")
    chunk = (numel + world_size - 1) // world_size
    start = min(rank * chunk, numel)
    length = min(chunk, numel - start)
    return start, length


def rank_info_from_env(env: dict[str, str] | None = None) -> RankInfo:
    """Accept either torchrun or OpenMPI rank variables."""
    source = os.environ if env is None else env

    def read(key: str, default: int) -> int:
        raw = source.get(key, str(default))
        try:
            return int(raw)
        except ValueError as exc:
            raise ProbeError(f"{key} must be an integer, got {raw!r}") from exc

    # mpirun can inherit stale generic RANK variables from its parent shell.
    # OMPI_* uniquely identifies the process spawned by this reproduction, so
    # it must win whenever OpenMPI provided a complete rank tuple.
    ompi_keys = ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_SIZE")
    if all(key in source for key in ompi_keys):
        rank = read("OMPI_COMM_WORLD_RANK", 0)
        local_rank = read("OMPI_COMM_WORLD_LOCAL_RANK", 0)
        world_size = read("OMPI_COMM_WORLD_SIZE", 1)
    else:
        rank = read("RANK", 0)
        local_rank = read("LOCAL_RANK", 0)
        world_size = read("WORLD_SIZE", 1)
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ProbeError(f"invalid rank tuple: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    return RankInfo(rank=rank, local_rank=local_rank, world_size=world_size)


def normalize_distributed_env(info: RankInfo) -> None:
    os.environ["RANK"] = str(info.rank)
    os.environ["LOCAL_RANK"] = str(info.local_rank)
    os.environ["WORLD_SIZE"] = str(info.world_size)


class EventLogger:
    """Write one flushed JSONL timeline per rank so the last completed phase survives a hang."""

    def __init__(self, log_dir: Path, info: RankInfo):
        log_dir.mkdir(parents=True, exist_ok=True)
        hostname = socket.gethostname().split(".", maxsplit=1)[0]
        self.path = log_dir / f"rank_{info.rank:04d}_{hostname}_{os.getpid()}.jsonl"
        self.info = info
        self.started = time.monotonic()

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "elapsed_seconds": round(time.monotonic() - self.started, 6),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "rank": self.info.rank,
            "local_rank": self.info.local_rank,
            "world_size": self.info.world_size,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
            output.flush()
        print(f"MUSA_VMM_MCCL_EVENT={line}", flush=True)


def _memory_fields(torch_module: Any, device: int) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        function = getattr(torch_module.musa, name, None)
        try:
            result[f"{name}_mib"] = round(float(function(device)) / MIB, 3) if function else None
        except Exception:
            result[f"{name}_mib"] = None
    mem_get_info = getattr(torch_module.musa, "mem_get_info", None)
    try:
        free_bytes, total_bytes = mem_get_info(device) if mem_get_info else (None, None)
        result["device_free_mib"] = round(float(free_bytes) / MIB, 3) if free_bytes is not None else None
        result["device_total_mib"] = round(float(total_bytes) / MIB, 3) if total_bytes is not None else None
    except Exception:
        result["device_free_mib"] = None
        result["device_total_mib"] = None
    return result


def _synchronize(torch_module: Any) -> None:
    torch_module.musa.synchronize()


def _validate_edges(tensor: Any, expected: int | float, label: str) -> None:
    if tensor.numel() == 0:
        return
    first = float(tensor[0].float().item())
    last = float(tensor[-1].float().item())
    expected_float = float(expected)
    if first != expected_float or last != expected_float:
        raise ProbeError(f"{label} corruption: expected {expected_float}, got first={first}, last={last}")


def run_allocator_probe(torch_module: Any, args: argparse.Namespace, info: RankInfo, logger: EventLogger) -> None:
    live: list[Any] = []
    for iteration in range(args.iterations):
        size_mb = args.sizes_mb[iteration % len(args.sizes_mb)]
        logger.write("alloc_begin", iteration=iteration, size_mb=size_mb)
        tensor = torch_module.empty(size_mb * MIB, dtype=torch_module.uint8, device="musa")
        expected = (info.rank + iteration) % 251
        tensor.fill_(expected)
        _synchronize(torch_module)
        _validate_edges(tensor, expected, "allocator")
        live.append(tensor)
        while len(live) > args.retained_allocations:
            del live[0]
        logger.write("alloc_done", iteration=iteration, size_mb=size_mb, **_memory_fields(torch_module, info.local_rank))

        if args.empty_cache_every and (iteration + 1) % args.empty_cache_every == 0:
            live.clear()
            gc.collect()
            logger.write("empty_cache_begin", iteration=iteration)
            torch_module.musa.empty_cache()
            _synchronize(torch_module)
            logger.write("empty_cache_done", iteration=iteration, **_memory_fields(torch_module, info.local_rank))

    live.clear()
    gc.collect()
    torch_module.musa.empty_cache()
    _synchronize(torch_module)


def run_mccl_probe(
    torch_module: Any,
    dist: Any,
    args: argparse.Namespace,
    info: RankInfo,
    logger: EventLogger,
) -> None:
    numel = max(1, args.collective_mb * MIB // 4)
    tensor = torch_module.empty(numel, dtype=torch_module.float32, device="musa")
    expected_sum = info.world_size * (info.world_size + 1) / 2

    for iteration in range(args.iterations):
        tensor.fill_(info.rank + 1)
        logger.write("all_reduce_begin", iteration=iteration, size_mb=args.collective_mb)
        dist.all_reduce(tensor)
        _synchronize(torch_module)
        _validate_edges(tensor, expected_sum, "all_reduce")
        logger.write("all_reduce_done", iteration=iteration, **_memory_fields(torch_module, info.local_rank))

        source = iteration % info.world_size
        expected_broadcast = source + iteration + 1
        tensor.fill_(expected_broadcast if info.rank == source else -1)
        logger.write("broadcast_begin", iteration=iteration, source=source)
        dist.broadcast(tensor, src=source)
        _synchronize(torch_module)
        _validate_edges(tensor, expected_broadcast, "broadcast")
        logger.write("broadcast_done", iteration=iteration, source=source)


def run_combined_probe(
    torch_module: Any,
    dist: Any,
    args: argparse.Namespace,
    info: RankInfo,
    logger: EventLogger,
    overlap: bool,
) -> None:
    retained_shards: list[Any] = []
    for iteration in range(args.iterations):
        size_mb = args.sizes_mb[iteration % len(args.sizes_mb)]
        numel = max(1, size_mb * MIB // 2)
        expected = iteration % 127 + 1
        source = iteration % info.world_size

        logger.write("alloc_begin", iteration=iteration, size_mb=size_mb)
        full_parameter = torch_module.empty(numel, dtype=torch_module.bfloat16, device="musa")
        if info.rank == source:
            full_parameter.fill_(expected)
        logger.write("alloc_done", iteration=iteration, size_mb=size_mb, **_memory_fields(torch_module, info.local_rank))

        logger.write("broadcast_begin", iteration=iteration, source=source, async_op=overlap)
        work = dist.broadcast(full_parameter, src=source, async_op=overlap)
        scratch = None
        if overlap:
            scratch_numel = max(1, numel // 4)
            logger.write("overlap_alloc_begin", iteration=iteration)
            scratch = torch_module.empty(scratch_numel, dtype=torch_module.bfloat16, device="musa")
            scratch.fill_(info.rank + 1)
            logger.write("overlap_alloc_done", iteration=iteration)
            work.wait()
        _synchronize(torch_module)
        _validate_edges(full_parameter, expected, "broadcasted parameter")
        logger.write("broadcast_done", iteration=iteration, source=source)

        start, length = shard_bounds(numel, info.rank, info.world_size)
        logger.write("partition_begin", iteration=iteration, shard_numel=length)
        shard = full_parameter.narrow(0, start, length).clone()
        _synchronize(torch_module)
        _validate_edges(shard, expected, "parameter shard")
        retained_shards.append(shard)
        while len(retained_shards) > args.retained_shards:
            del retained_shards[0]
        logger.write("partition_done", iteration=iteration, shard_numel=length)

        del full_parameter
        if scratch is not None:
            del scratch
        if args.empty_cache_every and (iteration + 1) % args.empty_cache_every == 0:
            gc.collect()
            logger.write("empty_cache_begin", iteration=iteration)
            torch_module.musa.empty_cache()
            _synchronize(torch_module)
            logger.write("empty_cache_done", iteration=iteration)
        logger.write("iteration_done", iteration=iteration, **_memory_fields(torch_module, info.local_rank))

    retained_shards.clear()
    gc.collect()
    torch_module.musa.empty_cache()
    _synchronize(torch_module)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("allocator", "mccl", "combined", "overlap"), default="combined")
    parser.add_argument("--sizes-mb", type=parse_sizes_mb, default=parse_sizes_mb("1,3,7,15,31,63"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--collective-mb", type=int, default=1)
    parser.add_argument("--retained-allocations", type=int, default=2)
    parser.add_argument("--retained-shards", type=int, default=64)
    parser.add_argument("--empty-cache-every", type=int, default=2)
    parser.add_argument("--dist-timeout-seconds", type=int, default=60)
    parser.add_argument("--stall-dump-seconds", type=int, default=30)
    parser.add_argument("--log-dir", default="logs/musa_vmm_mccl_repro")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.iterations,
        args.collective_mb,
        args.retained_allocations,
        args.retained_shards,
        args.dist_timeout_seconds,
        args.stall_dump_seconds,
    )
    if min(positive) < 1:
        raise ProbeError("iteration, size, retention, and timeout values must be positive")
    if args.empty_cache_every < 0:
        raise ProbeError("--empty-cache-every cannot be negative")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    # This must be set before torch/torch_musa initialize the caching allocator.
    os.environ.setdefault("PYTORCH_MUSA_ALLOC_CONF", DEFAULT_ALLOCATOR_CONF)
    info = rank_info_from_env()
    normalize_distributed_env(info)
    logger = EventLogger(Path(args.log_dir).resolve(), info)
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(args.stall_dump_seconds, repeat=True)

    dist = None
    dist_initialized = False
    try:
        import torch
        import torch_musa  # noqa: F401

        if not torch.musa.is_available():
            raise ProbeError("torch.musa is not available")
        device_count = torch.musa.device_count()
        if info.local_rank >= device_count:
            raise ProbeError(f"local rank {info.local_rank} exceeds visible MUSA device count {device_count}")
        torch.musa.set_device(info.local_rank)
        logger.write(
            "probe_start",
            mode=args.mode,
            allocator_conf=os.environ.get("PYTORCH_MUSA_ALLOC_CONF"),
            torch_version=torch.__version__,
            torch_musa_version=getattr(torch_musa, "__version__", "unknown"),
            sizes_mb=args.sizes_mb,
        )

        if args.mode != "allocator":
            if info.world_size < 2:
                raise ProbeError(f"mode {args.mode!r} needs at least two ranks")
            import torch.distributed as torch_dist

            dist = torch_dist
            logger.write("process_group_init_begin", backend="mccl")
            dist.init_process_group(
                backend="mccl",
                init_method="env://",
                timeout=timedelta(seconds=args.dist_timeout_seconds),
            )
            dist_initialized = True
            logger.write("process_group_init_done", backend="mccl")

        if args.mode == "allocator":
            run_allocator_probe(torch, args, info, logger)
        elif args.mode == "mccl":
            run_mccl_probe(torch, dist, args, info, logger)
        else:
            run_combined_probe(torch, dist, args, info, logger, overlap=args.mode == "overlap")

        if dist_initialized:
            logger.write("final_barrier_begin")
            dist.barrier()
            _synchronize(torch)
            logger.write("final_barrier_done")
        logger.write("probe_pass", **_memory_fields(torch, info.local_rank))
        return 0
    except Exception as exc:
        logger.write("probe_fail", error=repr(exc), traceback=traceback.format_exc())
        print(f"rank {info.rank} failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        if dist_initialized and dist is not None:
            try:
                dist.destroy_process_group()
            except Exception as exc:
                logger.write("process_group_destroy_fail", error=repr(exc))


if __name__ == "__main__":
    raise SystemExit(main())
