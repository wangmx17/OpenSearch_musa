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
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import TrainerCallback
from typing_extensions import override

from ..extras.misc import is_env_enabled


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


def _rank() -> int:
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from error

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")

    return value


def _string_list_env(name: str) -> Optional[list[str]]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None

    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _rank_list_env(name: str) -> Optional[list[int]]:
    raw_value = os.getenv(name, "all").strip().lower()
    if raw_value in {"all", "*"}:
        return None

    if not raw_value:
        raise ValueError(f"{name} must be 'all', '*', or a comma-separated rank list.")

    try:
        return [int(value.strip()) for value in raw_value.split(",")]
    except ValueError as error:
        raise ValueError(f"{name} must be 'all', '*', or a comma-separated rank list, got {raw_value!r}.") from error


def _int_list_env(name: str, default: str) -> list[int]:
    raw_value = os.getenv(name, default).strip()
    try:
        return [int(value.strip()) for value in raw_value.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list, got {raw_value!r}.") from error


def _output_dir(args: "TrainingArguments", child: str = "") -> str:
    root = os.path.abspath(
        os.getenv("OPENSEARCH_PRECISION_DIR", os.path.join(args.output_dir, "precision_debug"))
    )
    output_dir = os.path.join(root, child) if child else root
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def append_precision_record(args: "TrainingArguments", event: str, **payload: Any) -> None:
    output_path = os.path.join(_output_dir(args), f"boundary_rank{_rank()}.jsonl")
    record = {"event": event, "rank": _rank(), **payload}
    with open(output_path, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")


def record_precision_loss(trainer: Any, inputs: dict[str, Any], result: Any) -> None:
    if not is_env_enabled("OPENSEARCH_PRECISION_DEBUG"):
        return

    loss = result[0] if isinstance(result, tuple) else result
    if not isinstance(loss, torch.Tensor):
        return

    input_summary: dict[str, Any] = {}
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue

        summary: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        if name == "labels":
            summary["valid_count"] = int(value.ne(-100).sum().item())
        elif name in {"image_grid_thw", "video_grid_thw"} and value.numel() <= 128:
            summary["values"] = value.detach().cpu().tolist()
        elif value.is_floating_point():
            summary["finite"] = bool(torch.isfinite(value).all().item())

        input_summary[name] = summary

    loss_finite = bool(torch.isfinite(loss).all().item())
    append_precision_record(
        trainer.args,
        "loss",
        optimizer_step=int(trainer.state.global_step),
        micro_step=int(getattr(trainer, "_opensearch_precision_micro_step", 0)),
        loss=float(loss.detach().float().item()),
        loss_finite=loss_finite,
        inputs=input_summary,
    )
    trainer._opensearch_precision_micro_step = int(
        getattr(trainer, "_opensearch_precision_micro_step", 0)
    ) + 1

    if not loss_finite and is_env_enabled("OPENSEARCH_PRECISION_ABORT_ON_NONFINITE", "1"):
        raise FloatingPointError(
            f"Non-finite loss detected on rank {_rank()} at optimizer step {trainer.state.global_step}."
        )


def _optimizer_chain(optimizer: Any) -> list[Any]:
    chain = []
    seen = set()
    current = optimizer
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = getattr(current, "optimizer", None)

    return chain


def _append_tensor(
    groups: dict[str, list[torch.Tensor]], name: str, value: Any, seen: set[int]
) -> None:
    if isinstance(value, torch.Tensor):
        if id(value) not in seen:
            seen.add(id(value))
            groups.setdefault(name, []).append(value)
        return

    if isinstance(value, dict):
        for child_name, child_value in value.items():
            _append_tensor(groups, f"{name}.{child_name}", child_value, seen)
    elif isinstance(value, (list, tuple)):
        for child_value in value:
            _append_tensor(groups, name, child_value, seen)


def _optimizer_tensors(
    optimizer: Any,
) -> tuple[dict[str, list[torch.Tensor]], list[str], list[float]]:
    tensor_groups: dict[str, list[torch.Tensor]] = {}
    types = []
    learning_rates = []
    seen_tensors: set[int] = set()
    for item in _optimizer_chain(optimizer):
        types.append(f"{type(item).__module__}.{type(item).__name__}")
        for group in getattr(item, "param_groups", []):
            if "lr" in group:
                learning_rates.append(float(group["lr"]))
            for parameter in group.get("params", []):
                _append_tensor(tensor_groups, "param_group", parameter, seen_tensors)
                if isinstance(parameter, torch.Tensor) and parameter.grad is not None:
                    _append_tensor(tensor_groups, "param_group_grad", parameter.grad, seen_tensors)

        for attribute in (
            "fp32_partitioned_groups_flat",
            "fp16_partitioned_groups_flat",
            "grad_partitions_flat_buffer",
            "averaged_gradients",
        ):
            if hasattr(item, attribute):
                _append_tensor(tensor_groups, attribute, getattr(item, attribute), seen_tensors)

        for state in getattr(item, "state", {}).values():
            if isinstance(state, dict):
                for state_name, value in state.items():
                    _append_tensor(tensor_groups, f"optimizer_state.{state_name}", value, seen_tensors)

    return tensor_groups, list(dict.fromkeys(types)), list(dict.fromkeys(learning_rates))


def _finite_summary(tensors: list[torch.Tensor]) -> dict[str, Any]:
    checked = 0
    total_numel = 0
    first_bad = None
    chunk_numel = _int_env(
        "OPENSEARCH_PRECISION_SCAN_CHUNK_NUMEL", 8 * 1024 * 1024, minimum=1
    )
    for index, tensor in enumerate(tensors):
        if tensor is None or tensor.numel() == 0:
            continue

        checked += 1
        total_numel += tensor.numel()
        flat = tensor.detach().reshape(-1)
        for offset in range(0, flat.numel(), chunk_numel):
            chunk = flat[offset : offset + chunk_numel]
            if not bool(torch.isfinite(chunk).all().item()):
                first_bad = {
                    "index": index,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "first_bad_chunk_offset": offset,
                    "nan_count_in_chunk": int(torch.isnan(chunk).sum().item()),
                    "inf_count_in_chunk": int(torch.isinf(chunk).sum().item()),
                }
                break

        if first_bad is not None:
            break

    return {
        "checked_tensors": checked,
        "checked_numel": total_numel,
        "finite": first_bad is None,
        "first_bad": first_bad,
    }


def _group_finite_summaries(
    tensor_groups: dict[str, list[torch.Tensor]],
) -> dict[str, dict[str, Any]]:
    return {name: _finite_summary(tensors) for name, tensors in tensor_groups.items()}


def _take_parameter_samples(
    tensor_groups: dict[str, list[torch.Tensor]],
) -> list[tuple[str, int, torch.Tensor]]:
    max_tensors = _int_env("OPENSEARCH_PRECISION_SAMPLE_TENSORS", 16, minimum=1)
    samples = []
    for group_name in (
        "fp32_partitioned_groups_flat",
        "fp16_partitioned_groups_flat",
        "param_group",
    ):
        for index, tensor in enumerate(tensor_groups.get(group_name, [])):
            if len(samples) >= max_tensors:
                return samples
            if tensor.numel() == 0:
                continue

            flat = tensor.detach().reshape(-1)
            positions = torch.linspace(
                0, flat.numel() - 1, steps=min(64, flat.numel()), device=flat.device
            ).long()
            samples.append((group_name, index, flat[positions].clone()))

    return samples


def _any_nonfinite(summaries: dict[str, dict[str, Any]], names: tuple[str, ...]) -> bool:
    return any(not summary["finite"] for name, summary in summaries.items() if name in names)


def _zero3_subgroup_summaries(
    zero3_optimizer: Any, sub_group_id: int
) -> dict[str, dict[str, Any]]:
    tensor_groups: dict[str, list[torch.Tensor]] = {
        "fp32_parameter": [zero3_optimizer.fp32_partitioned_groups_flat[sub_group_id]],
        "bf16_parameter": [zero3_optimizer.fp16_partitioned_groups_flat[sub_group_id]],
    }
    fp32_parameter = tensor_groups["fp32_parameter"][0]
    if fp32_parameter.grad is not None:
        tensor_groups["fp32_gradient"] = [fp32_parameter.grad]

    optimizer_state = zero3_optimizer.optimizer.state.get(fp32_parameter, {})
    for state_name, value in optimizer_state.items():
        if isinstance(value, torch.Tensor):
            tensor_groups[f"optimizer_state.{state_name}"] = [value]

    return _group_finite_summaries(tensor_groups)


class PrecisionDebugCallback(TrainerCallback):
    r"""Log finite-state boundaries around the real DeepSpeed optimizer step."""

    def __init__(self) -> None:
        self.parameter_samples: list[tuple[str, int, torch.Tensor]] = []

    def _set_experts_implementation(self, args: "TrainingArguments", model: Any) -> None:
        requested = os.getenv("OPENSEARCH_PRECISION_EXPERTS_IMPLEMENTATION", "").strip()
        if not requested:
            return

        target = model
        seen = set()
        while hasattr(target, "module") and id(target) not in seen:
            seen.add(id(target))
            target = target.module

        if not hasattr(target, "set_experts_implementation"):
            raise RuntimeError(
                f"Model {type(target)} does not support changing experts_implementation."
            )

        target.set_experts_implementation(requested)
        append_precision_record(
            args,
            "experts_implementation_set",
            requested=requested,
            effective=getattr(target.config, "_experts_implementation", None),
        )

    def _install_zero3_probe(self, args: "TrainingArguments", optimizer: Any) -> None:
        selected_subgroups = set(_int_list_env("OPENSEARCH_PRECISION_ZERO3_SUBGROUPS", "0"))
        for item in _optimizer_chain(optimizer):
            if not (
                hasattr(item, "fp32_partitioned_groups_flat")
                and hasattr(item, "fp16_partitioned_groups_flat")
                and callable(getattr(item, "_optimizer_step", None))
            ):
                continue

            original_optimizer_step = item._optimizer_step

            def traced_optimizer_step(
                sub_group_id: int, *, _zero3=item, _original=original_optimizer_step
            ):
                if sub_group_id not in selected_subgroups:
                    return _original(sub_group_id)

                param_group_id = _zero3.sub_group_to_group_id[sub_group_id]
                learning_rate = float(_zero3.optimizer.param_groups[param_group_id]["lr"])
                append_precision_record(
                    args,
                    "zero3_subgroup_pre_fused_adamw",
                    sub_group_id=sub_group_id,
                    learning_rate=learning_rate,
                    global_grad_norm=float(_zero3._global_grad_norm),
                    tensor_summaries=_zero3_subgroup_summaries(_zero3, sub_group_id),
                )
                result = _original(sub_group_id)
                append_precision_record(
                    args,
                    "zero3_subgroup_post_fused_adamw",
                    sub_group_id=sub_group_id,
                    learning_rate=learning_rate,
                    global_grad_norm=float(_zero3._global_grad_norm),
                    tensor_summaries=_zero3_subgroup_summaries(_zero3, sub_group_id),
                )
                return result

            item._optimizer_step = traced_optimizer_step
            append_precision_record(
                args,
                "zero3_probe_installed",
                optimizer_type=f"{type(item).__module__}.{type(item).__name__}",
                selected_subgroups=sorted(selected_subgroups),
                subgroup_count=len(item.fp32_partitioned_groups_flat),
            )
            return

        append_precision_record(args, "zero3_probe_not_found")

    @override
    def on_train_begin(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        output_path = os.path.join(_output_dir(args), f"boundary_rank{_rank()}.jsonl")
        if os.path.exists(output_path):
            os.remove(output_path)
        append_precision_record(args, "train_begin", model_type=str(type(kwargs.get("model"))))
        self._set_experts_implementation(args, kwargs.get("model"))
        self._install_zero3_probe(args, kwargs.get("optimizer"))

    @override
    def on_pre_optimizer_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        tensor_groups, optimizer_types, learning_rates = _optimizer_tensors(kwargs.get("optimizer"))
        if is_env_enabled("OPENSEARCH_PRECISION_ZERO3_INNER_ONLY"):
            tensor_groups = {}
        tensor_summaries = _group_finite_summaries(tensor_groups)
        if state.global_step == 0:
            self.parameter_samples = _take_parameter_samples(tensor_groups)

        append_precision_record(
            args,
            "pre_optimizer",
            optimizer_step=int(state.global_step),
            optimizer_types=optimizer_types,
            learning_rates=learning_rates,
            tensor_summaries=tensor_summaries,
            zero_lr_sample_count=len(self.parameter_samples) if state.global_step == 0 else None,
        )
        if _any_nonfinite(
            tensor_summaries,
            ("param_group_grad", "grad_partitions_flat_buffer", "averaged_gradients"),
        ) and is_env_enabled("OPENSEARCH_PRECISION_ABORT_ON_NONFINITE", "1"):
            raise FloatingPointError(
                f"Non-finite optimizer gradient detected on rank {_rank()} at step {state.global_step}."
            )

    @override
    def on_optimizer_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        tensor_groups, optimizer_types, learning_rates = _optimizer_tensors(kwargs.get("optimizer"))
        if is_env_enabled("OPENSEARCH_PRECISION_ZERO3_INNER_ONLY"):
            tensor_groups = {}
        tensor_summaries = _group_finite_summaries(tensor_groups)
        sample_equal = True
        sample_max_abs = 0.0
        if state.global_step == 0 and self.parameter_samples:
            for group_name, index, before in self.parameter_samples:
                tensors = tensor_groups.get(group_name, [])
                if index >= len(tensors) or tensors[index].numel() == 0:
                    sample_equal = False
                    continue

                flat = tensors[index].detach().reshape(-1)
                positions = torch.linspace(
                    0, flat.numel() - 1, steps=min(64, flat.numel()), device=flat.device
                ).long()
                after = flat[positions]
                sample_equal = sample_equal and bool(torch.equal(before, after))
                sample_max_abs = max(sample_max_abs, float((before.float() - after.float()).abs().max().item()))

        append_precision_record(
            args,
            "post_optimizer",
            optimizer_step=int(state.global_step),
            optimizer_types=optimizer_types,
            learning_rates=learning_rates,
            tensor_summaries=tensor_summaries,
            zero_lr_sample_count=len(self.parameter_samples) if state.global_step == 0 else None,
            zero_lr_sample_equal=sample_equal if state.global_step == 0 else None,
            zero_lr_sample_max_abs=sample_max_abs if state.global_step == 0 else None,
        )
        self.parameter_samples = []
        if _any_nonfinite(
            tensor_summaries,
            ("param_group", "fp32_partitioned_groups_flat", "fp16_partitioned_groups_flat"),
        ) and is_env_enabled("OPENSEARCH_PRECISION_ABORT_ON_NONFINITE", "1"):
            raise FloatingPointError(
                f"Non-finite optimizer parameter detected on rank {_rank()} at step {state.global_step}."
            )

    @override
    def on_step_end(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        stop_after_steps = _int_env("OPENSEARCH_PRECISION_STOP_AFTER_STEPS", 0)
        if stop_after_steps and state.global_step >= stop_after_steps:
            append_precision_record(args, "intentional_stop", optimizer_step=int(state.global_step))
            raise RuntimeError(f"Precision diagnostic intentionally stopped after {state.global_step} steps.")


class MusaNanInfTrackerCallback(TrainerCallback):
    r"""Enable torch_musa's operator-level NaN/Inf tracker for selected optimizer steps and ranks."""

    def __init__(self) -> None:
        self.tracker: Optional[Any] = None

    @override
    def on_train_begin(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        from torch_musa.utils.compare_tool import NanInfTracker, open_module_tracker

        enable_ranks = _rank_list_env("OPENSEARCH_NAN_INF_RANKS")
        rank_selected = enable_ranks is None or _rank() in enable_ranks
        if rank_selected and is_env_enabled("OPENSEARCH_NAN_INF_MODULE_TRACKER", "1"):
            open_module_tracker(kwargs["model"])

        tracker_dir = _output_dir(args, "nan_inf")
        self.tracker = NanInfTracker(
            target_list=_string_list_env("OPENSEARCH_NAN_INF_TARGET_LIST"),
            white_list=_string_list_env("OPENSEARCH_NAN_INF_WHITE_LIST"),
            enable_ranks=enable_ranks,
            should_log_to_file=True,
            output_dir=tracker_dir,
            dump_error_data=is_env_enabled("OPENSEARCH_NAN_INF_DUMP"),
            start_step=_int_env("OPENSEARCH_NAN_INF_START_STEP", 1),
            end_step=_int_env("OPENSEARCH_NAN_INF_END_STEP", 2),
        )
        self.tracker.__enter__()
        append_precision_record(
            args,
            "nan_inf_tracker_begin",
            enable_ranks=enable_ranks,
            target_list=_string_list_env("OPENSEARCH_NAN_INF_TARGET_LIST"),
            tracker_dir=tracker_dir,
        )

    @override
    def on_step_end(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.tracker is not None:
            self.tracker.step()

    @override
    def on_train_end(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.tracker is not None:
            try:
                self.tracker.__exit__(None, None, None)
            finally:
                self.tracker = None
