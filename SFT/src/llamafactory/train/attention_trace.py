import functools
import importlib
import json
import os
from typing import Any

import torch

from ..extras.misc import is_env_enabled
from .module_trace import _json_safe_value, _rank, _selected_rank, _value_summary


class _AttentionBoundaryTrace:
    def __init__(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, f"attention_rank{_rank()}.jsonl")
        self.rope_recorded = False
        self.sdpa_recorded = False
        self.sequence = 0
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def _write(self, event: str, **payload: Any) -> None:
        record = _json_safe_value({"event": event, "rank": _rank(), "sequence": self.sequence, **payload})
        self.sequence += 1
        with open(self.output_path, "a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _target(query: Any) -> bool:
        if not isinstance(query, torch.Tensor) or query.ndim != 4:
            return False
        minimum_length = int(os.getenv("OPENSEARCH_ATTENTION_TRACE_MIN_SEQ_LEN", "1024"))
        return query.shape[-2] >= minimum_length and query.shape[-1] == 128

    def install(self) -> None:
        modeling = importlib.import_module("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
        original_rope = modeling.apply_rotary_pos_emb
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        @functools.wraps(original_rope)
        def traced_rope(query, key, cos, sin, *args, **kwargs):
            output = original_rope(query, key, cos, sin, *args, **kwargs)
            if not self.rope_recorded and self._target(query):
                self._write(
                    "rope",
                    query_input=_value_summary(query),
                    key_input=_value_summary(key),
                    cos=_value_summary(cos),
                    sin=_value_summary(sin),
                    query_output=_value_summary(output[0]),
                    key_output=_value_summary(output[1]),
                )
                self.rope_recorded = True
            return output

        @functools.wraps(original_sdpa)
        def traced_sdpa(query, key, value, *args, **kwargs):
            output = original_sdpa(query, key, value, *args, **kwargs)
            if not self.sdpa_recorded and self._target(query):
                self._write(
                    "sdpa",
                    query=_value_summary(query),
                    key=_value_summary(key),
                    value=_value_summary(value),
                    args=_value_summary(args),
                    kwargs=_value_summary(kwargs),
                    output=_value_summary(output),
                )
                self.sdpa_recorded = True
            return output

        modeling.apply_rotary_pos_emb = traced_rope
        torch.nn.functional.scaled_dot_product_attention = traced_sdpa
        self._write("metadata", torch_version=torch.__version__)


def install_attention_precision_trace() -> bool:
    if not is_env_enabled("OPENSEARCH_ATTENTION_TRACE") or not _selected_rank():
        return False

    output_dir = os.path.abspath(
        os.getenv("OPENSEARCH_ATTENTION_TRACE_DIR", os.path.join(os.getcwd(), "attention_trace"))
    )
    trace = _AttentionBoundaryTrace(output_dir)
    trace.install()
    return True
