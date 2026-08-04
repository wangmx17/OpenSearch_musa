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

"""OpenSearch-owned MUSA fused SwiGLU runtime patch."""

import os
from collections.abc import Callable
from types import MethodType
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from packaging.version import Version

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)
_FUSED_SWIGLU_DISABLED = False
_FUSED_SWIGLU_LOGGED = False


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


def _is_supported_transformers_version() -> bool:
    import transformers

    return Version(transformers.__version__).release[:2] == (5, 2)


def apply_swiglu_eager(gate_up: torch.Tensor, act_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return act_fn(gate) * up


def apply_swiglu_musa(
    gate_up: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    hidden_act: str | None = "silu",
) -> torch.Tensor:
    """Consume the contiguous ``[gate, up]`` projection with torch-musa's fused kernel."""
    global _FUSED_SWIGLU_DISABLED, _FUSED_SWIGLU_LOGGED

    if (
        _FUSED_SWIGLU_DISABLED
        or not _env_flag("OPENSEARCH_MUSA_FUSED_SWIGLU", "0")
        or hidden_act not in {"silu", "swish"}
        or gate_up.device.type != "musa"
        or not hasattr(F, "swish_glu")
    ):
        return apply_swiglu_eager(gate_up, act_fn)

    try:
        output = F.swish_glu(gate_up)
        if not _FUSED_SWIGLU_LOGGED:
            logger.info_rank0("MUSA fused SwiGLU fast path is active.")
            _FUSED_SWIGLU_LOGGED = True
        return output
    except Exception as err:
        _FUSED_SWIGLU_DISABLED = True
        logger.warning_rank0_once(f"MUSA fused SwiGLU disabled after kernel failure; using eager fallback: {err}")
        return apply_swiglu_eager(gate_up, act_fn)


def qwen3_vl_moe_text_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Qwen3-VL-MoE eager experts forward with fused SwiGLU activation."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
        current_hidden_states = apply_swiglu_musa(
            gate_up,
            self.act_fn,
            hidden_act=getattr(self.config, "hidden_act", None),
        )
        current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def patch_qwen3_vl_moe_fused_swiglu(model: "PreTrainedModel") -> int:
    if not _env_flag("OPENSEARCH_MUSA_FUSED_SWIGLU", "0"):
        return 0
    if getattr(model.config, "model_type", None) != "qwen3_vl_moe":
        return 0

    text_config = getattr(model.config, "text_config", None)
    if getattr(text_config, "hidden_act", None) not in {"silu", "swish"}:
        return 0
    if not hasattr(torch, "musa") or not torch.musa.is_available() or not hasattr(F, "swish_glu"):
        return 0
    if not _is_supported_transformers_version():
        logger.warning_rank0_once("MUSA fused SwiGLU eager patch supports Transformers 5.2.x only; skipped.")
        return 0

    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    experts_cls = modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts
    patched_modules = 0
    for module in model.modules():
        if not isinstance(module, experts_cls):
            continue
        if getattr(module.config, "_experts_implementation", "eager") not in {None, "eager"}:
            continue
        if not all(hasattr(module, name) for name in ("gate_up_proj", "down_proj", "act_fn", "num_experts")):
            continue
        module.forward = MethodType(qwen3_vl_moe_text_experts_forward, module)
        patched_modules += 1

    return patched_modules


__all__ = [
    "apply_swiglu_eager",
    "apply_swiglu_musa",
    "patch_qwen3_vl_moe_fused_swiglu",
    "qwen3_vl_moe_text_experts_forward",
]
