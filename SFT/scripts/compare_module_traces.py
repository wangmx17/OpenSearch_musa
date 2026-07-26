#!/usr/bin/env python3
"""Locate the first tensor divergence between two module trace JSONL files."""

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _is_tensor_summary(value: Any) -> bool:
    return isinstance(value, dict) and {"shape", "dtype", "sample_sha256"}.issubset(value)


def _tensor_summaries(value: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    if _is_tensor_summary(value):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _tensor_summaries(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _tensor_summaries(child, f"{path}[{index}]")


def _finite_numbers(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def _sample_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float] | None:
    reference_values = reference.get("sample_values")
    candidate_values = candidate.get("sample_values")
    if not isinstance(reference_values, list) or not isinstance(candidate_values, list):
        return None
    if len(reference_values) != len(candidate_values):
        return None

    pairs = [
        (float(left), float(right))
        for left, right in zip(reference_values, candidate_values, strict=True)
        if isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isfinite(left)
        and math.isfinite(right)
    ]
    if not pairs:
        return None

    absolute = [abs(left - right) for left, right in pairs]
    relative = [difference / max(abs(left), 1e-12) for difference, (left, _) in zip(absolute, pairs, strict=True)]
    return {"max_abs": max(absolute), "max_rel": max(relative), "compared_values": len(pairs)}


def _compare_payload(reference: Any, candidate: Any) -> list[dict[str, Any]]:
    reference_tensors = dict(_tensor_summaries(reference))
    candidate_tensors = dict(_tensor_summaries(candidate))
    mismatches: list[dict[str, Any]] = []
    for path in sorted(reference_tensors.keys() | candidate_tensors.keys()):
        left = reference_tensors.get(path)
        right = candidate_tensors.get(path)
        if left is None or right is None:
            mismatches.append({"tensor_path": path, "reason": "missing_tensor"})
            continue
        if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
            mismatches.append(
                {
                    "tensor_path": path,
                    "reason": "metadata",
                    "reference_shape": left["shape"],
                    "candidate_shape": right["shape"],
                    "reference_dtype": left["dtype"],
                    "candidate_dtype": right["dtype"],
                }
            )
            continue

        hash_key = "full_sha256" if "full_sha256" in left and "full_sha256" in right else "sample_sha256"
        if left[hash_key] != right[hash_key]:
            mismatch = {
                "tensor_path": path,
                "reason": "value",
                "hash_kind": hash_key,
                "reference_hash": left[hash_key],
                "candidate_hash": right[hash_key],
                "reference_nonfinite": {
                    "nan": left.get("sample_nan_count"),
                    "posinf": left.get("sample_posinf_count"),
                    "neginf": left.get("sample_neginf_count"),
                },
                "candidate_nonfinite": {
                    "nan": right.get("sample_nan_count"),
                    "posinf": right.get("sample_posinf_count"),
                    "neginf": right.get("sample_neginf_count"),
                },
            }
            delta = _sample_delta(left, right)
            if delta is not None:
                mismatch["sample_delta"] = delta
            mismatches.append(mismatch)
    return mismatches


def _module_records(records: list[dict[str, Any]]) -> dict[tuple[int, str, int], dict[str, Any]]:
    return {
        (record["root_call"], record["module"], record["call_index"]): record
        for record in records
        if record.get("event") == "module_forward"
    }


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = _read_records(reference_path)
    candidate = _read_records(candidate_path)
    reference_metadata = next(record for record in reference if record.get("event") == "metadata")
    candidate_metadata = next(record for record in candidate if record.get("event") == "metadata")

    config_keys = ("torch_version", "model_class", "attention_implementation", "experts_implementation")
    config_differences = {
        key: {"reference": reference_metadata.get(key), "candidate": candidate_metadata.get(key)}
        for key in config_keys
        if reference_metadata.get(key) != candidate_metadata.get(key)
    }

    reference_root = next(record for record in reference if record.get("event") == "root_input")
    candidate_root = next(record for record in candidate if record.get("event") == "root_input")
    root_mismatches = _compare_payload(reference_root, candidate_root)

    reference_modules = _module_records(reference)
    candidate_modules = _module_records(candidate)
    ordered_keys = [
        (record["root_call"], record["module"], record["call_index"])
        for record in reference
        if record.get("event") == "module_forward"
    ]
    first_divergence = None
    for key in ordered_keys:
        if key not in candidate_modules:
            first_divergence = {"module_key": key, "reason": "missing_candidate_module"}
            break
        mismatches = _compare_payload(reference_modules[key], candidate_modules[key])
        if mismatches:
            first_divergence = {
                "module_key": key,
                "module_class": reference_modules[key].get("module_class"),
                "mismatches": mismatches,
            }
            break

    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "config_differences": config_differences,
        "root_input_mismatches": root_mismatches,
        "reference_module_records": len(reference_modules),
        "candidate_module_records": len(candidate_modules),
        "first_module_divergence": first_divergence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference module_rank*.jsonl")
    parser.add_argument("candidate", type=Path, help="Candidate module_rank*.jsonl")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = compare(args.reference, args.candidate)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(encoded)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
