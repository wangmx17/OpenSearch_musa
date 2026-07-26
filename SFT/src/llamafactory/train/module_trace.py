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

import hashlib
import json
import math
import os
from collections.abc import Mapping
from typing import Any

import torch

from ..extras.misc import is_env_enabled


_DEFAULT_CLASS_TARGETS = (
    "Attention",
    "DecoderLayer",
    "RMSNorm",
    "SparseMoeBlock",
    "TopKRouter",
    "VisionBlock",
    "VisionPatchEmbed",
    "VisionTransformer",
)
_DEFAULT_NAME_TARGETS = ("embed_tokens", "lm_head", "model.norm", "visual")


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


def _selected_rank() -> bool:
    raw_value = os.getenv("OPENSEARCH_MODULE_TRACE_RANKS", "0").strip().lower()
    if raw_value in {"all", "*"}:
        return True

    try:
        return _rank() in {int(value.strip()) for value in raw_value.split(",") if value.strip()}
    except ValueError as error:
        raise ValueError(
            "OPENSEARCH_MODULE_TRACE_RANKS must be 'all', '*', or a comma-separated rank list."
        ) from error


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous()
    if value.device.type != "cpu":
        value = value.cpu()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _sample_tensor(tensor: torch.Tensor, sample_numel: int) -> torch.Tensor:
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= sample_numel:
        return flat

    positions = torch.linspace(0, flat.numel() - 1, steps=sample_numel, device=flat.device).long()
    return flat[positions]


def _json_safe_float(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, float):
        return _json_safe_float(value)
    if isinstance(value, dict):
        return {key: _json_safe_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(child) for child in value]
    return value


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    sample_numel = _int_env("OPENSEARCH_MODULE_TRACE_SAMPLE_NUMEL", 64, minimum=1)
    full_hash_numel = _int_env("OPENSEARCH_MODULE_TRACE_FULL_HASH_NUMEL", 131072, minimum=0)
    detached = tensor.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": detached.numel(),
        "requires_grad": bool(tensor.requires_grad),
    }
    if detached.numel() == 0:
        summary["sample_sha256"] = hashlib.sha256(b"").hexdigest()
        return summary

    sample = _sample_tensor(detached, sample_numel)
    summary["sample_sha256"] = hashlib.sha256(_raw_bytes(sample)).hexdigest()
    if full_hash_numel and detached.numel() <= full_hash_numel:
        summary["full_sha256"] = hashlib.sha256(_raw_bytes(detached)).hexdigest()

    if sample.is_floating_point():
        sample_float = sample.float()
        finite = torch.isfinite(sample_float)
        summary.update(
            {
                "sample_finite": bool(finite.all().item()),
                "sample_nan_count": int(torch.isnan(sample_float).sum().item()),
                "sample_inf_count": int(torch.isinf(sample_float).sum().item()),
                "sample_posinf_count": int(torch.isposinf(sample_float).sum().item()),
                "sample_neginf_count": int(torch.isneginf(sample_float).sum().item()),
            }
        )
        if bool(finite.any().item()):
            finite_values = sample_float[finite]
            summary.update(
                {
                    "sample_min": _json_safe_float(float(finite_values.min().item())),
                    "sample_max": _json_safe_float(float(finite_values.max().item())),
                    "sample_mean": _json_safe_float(float(finite_values.mean().item())),
                    "sample_l2": _json_safe_float(float(torch.linalg.vector_norm(finite_values).item())),
                }
            )

        summary["sample_values"] = [_json_safe_float(value) for value in sample_float.cpu().tolist()]
    else:
        summary["sample_values"] = sample.cpu().tolist()

    return summary


def _value_summary(value: Any, depth: int = 0) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, float):
        return _json_safe_float(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if depth >= 3:
        return {"type": f"{type(value).__module__}.{type(value).__name__}"}
    if isinstance(value, Mapping):
        return {
            str(key): _value_summary(child, depth + 1)
            for key, child in list(value.items())[:32]
        }
    if isinstance(value, (list, tuple)):
        return [_value_summary(child, depth + 1) for child in value[:16]]
    if hasattr(value, "items") and callable(value.items):
        try:
            return {
                str(key): _value_summary(child, depth + 1)
                for key, child in list(value.items())[:32]
            }
        except Exception:
            pass

    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def _module_selected(name: str, module: torch.nn.Module) -> bool:
    configured_targets = [
        value.strip()
        for value in os.getenv("OPENSEARCH_MODULE_TRACE_TARGETS", "").split(",")
        if value.strip()
    ]
    class_name = type(module).__name__
    if configured_targets:
        return any(target in name or target in class_name for target in configured_targets)

    return any(target in class_name for target in _DEFAULT_CLASS_TARGETS) or any(
        name == target or name.endswith(f".{target}") for target in _DEFAULT_NAME_TARGETS
    )


class _ModulePrecisionTrace:
    def __init__(self, model: torch.nn.Module, output_dir: str) -> None:
        self.model = model
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, f"module_rank{_rank()}.jsonl")
        self.max_root_calls = _int_env("OPENSEARCH_MODULE_TRACE_MAX_ROOT_CALLS", 1, minimum=1)
        self.root_calls = 0
        self.active = False
        self.sequence = 0
        self.call_counts: dict[str, int] = {}
        self.handles: list[Any] = []
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def _write(self, event: str, **payload: Any) -> None:
        record = {"event": event, "rank": _rank(), "sequence": self.sequence, **payload}
        self.sequence += 1
        with open(self.output_path, "a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(_json_safe_value(record), ensure_ascii=False, allow_nan=False) + "\n")

    def _root_pre_hook(self, _module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.active = self.root_calls < self.max_root_calls
        if not self.active:
            return

        self._write(
            "root_input",
            root_call=self.root_calls,
            args=_value_summary(args),
            kwargs=_value_summary(kwargs),
        )

    def _root_post_hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if not self.active:
            return

        self._write("root_output", root_call=self.root_calls, output=_value_summary(output))
        self.root_calls += 1
        self.active = False

    def _module_hook(
        self,
        name: str,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if not self.active:
            return

        call_index = self.call_counts.get(name, 0)
        self.call_counts[name] = call_index + 1
        self._write(
            "module_forward",
            root_call=self.root_calls,
            module=name,
            module_class=f"{type(module).__module__}.{type(module).__name__}",
            call_index=call_index,
            args=_value_summary(args),
            kwargs=_value_summary(kwargs),
            output=_value_summary(output),
        )

    def install(self) -> int:
        selected_modules = []
        for name, module in self.model.named_modules():
            if module is self.model or not _module_selected(name, module):
                continue

            selected_modules.append((name, module))
            self.handles.append(
                module.register_forward_hook(
                    lambda current_module, args, kwargs, output, module_name=name: self._module_hook(
                        module_name, current_module, args, kwargs, output
                    ),
                    with_kwargs=True,
                )
            )

        self.handles.append(self.model.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self._root_post_hook, with_kwargs=True))
        config = getattr(self.model, "config", None)
        self._write(
            "metadata",
            torch_version=torch.__version__,
            model_class=f"{type(self.model).__module__}.{type(self.model).__name__}",
            model_type=getattr(config, "model_type", None),
            attention_implementation=getattr(config, "_attn_implementation", None),
            experts_implementation=getattr(config, "_experts_implementation", None),
            selected_module_count=len(selected_modules),
            selected_modules=[
                {"name": name, "class": f"{type(module).__module__}.{type(module).__name__}"}
                for name, module in selected_modules
            ],
        )
        return len(selected_modules)


def install_module_precision_trace(model: torch.nn.Module) -> int:
    if not is_env_enabled("OPENSEARCH_MODULE_TRACE") or not _selected_rank():
        return 0
    if getattr(model, "_opensearch_module_precision_trace", None) is not None:
        return 0

    output_dir = os.path.abspath(
        os.getenv("OPENSEARCH_MODULE_TRACE_DIR", os.path.join(os.getcwd(), "module_trace"))
    )
    trace = _ModulePrecisionTrace(model, output_dir)
    selected_module_count = trace.install()
    model._opensearch_module_precision_trace = trace
    return selected_module_count
