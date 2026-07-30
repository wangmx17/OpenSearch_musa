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

"""OpenSearch-owned MUSA fused RMSNorm runtime patch."""

import os
from types import MethodType
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


def apply_rms_norm_eager(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


def apply_rms_norm_musa(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Route MUSA tensors to torch_musa's fused muDNN RMSNorm kernel."""
    if hidden_states.device.type != "musa" or not hasattr(F, "rms_norm"):
        return apply_rms_norm_eager(self, hidden_states)

    try:
        return F.rms_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            self.weight,
            self.variance_epsilon,
        )
    except Exception as err:
        logger.warning_rank0_once(f"MUSA fused RMSNorm fell back to eager: {err}")
        return apply_rms_norm_eager(self, hidden_states)


def patch_qwen3_vl_moe_fused_rmsnorm(model: "PreTrainedModel") -> int:
    if os.getenv("OPENSEARCH_MUSA_FUSED_RMSNORM", "0").lower() in {"0", "false", "no", "off"}:
        return 0
    if getattr(model.config, "model_type", None) != "qwen3_vl_moe":
        return 0
    if not hasattr(torch, "musa") or not torch.musa.is_available() or not hasattr(F, "rms_norm"):
        return 0

    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    rmsnorm_cls = modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm
    patched_modules = 0
    for module in model.modules():
        if isinstance(module, rmsnorm_cls):
            module.forward = MethodType(apply_rms_norm_musa, module)
            patched_modules += 1

    return patched_modules


__all__ = ["apply_rms_norm_eager", "apply_rms_norm_musa", "patch_qwen3_vl_moe_fused_rmsnorm"]
