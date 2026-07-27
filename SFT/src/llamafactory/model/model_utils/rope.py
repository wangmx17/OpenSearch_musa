# Copyright 2025 LMSYS and the LlamaFactory team.
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
# This code is inspired by the LMSYS's FastChat library.
# https://github.com/lm-sys/FastChat/blob/v0.2.30/fastchat/train/train.py
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

import math
import os
from types import MethodType
from typing import TYPE_CHECKING

import torch

from ...extras import logging
from ...extras.constants import RopeScaling


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _qwen3_vl_moe_rope_forward_without_bmm(self, x: torch.Tensor, position_ids: torch.Tensor):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    with modeling_qwen3_vl_moe.maybe_autocast(device_type=device_type, enabled=False):
        # The contraction dimension is one, so broadcast multiplication is
        # mathematically identical to matmul and avoids the torch-musa 2.7.1
        # float32 bmm kernel that loses position precision after index 2048.
        freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def patch_qwen3_vl_moe_rope_bmm(model: "PreTrainedModel") -> int:
    if os.getenv("OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND", "1").lower() in {"0", "false", "no", "off"}:
        return 0
    if getattr(model.config, "model_type", None) != "qwen3_vl_moe":
        return 0

    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    if not torch_version.startswith("2.7.") or not hasattr(torch, "musa") or not torch.musa.is_available():
        return 0

    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    Qwen3VLMoeTextRotaryEmbedding = modeling_qwen3_vl_moe.Qwen3VLMoeTextRotaryEmbedding

    patched_modules = 0
    for module in model.modules():
        if isinstance(module, Qwen3VLMoeTextRotaryEmbedding):
            forward = torch.no_grad()(
                modeling_qwen3_vl_moe.dynamic_rope_update(_qwen3_vl_moe_rope_forward_without_bmm)
            )
            module.forward = MethodType(forward, module)
            patched_modules += 1

    return patched_modules


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    if model_args.rope_scaling is None:
        return

    if not hasattr(config, "rope_scaling"):
        logger.warning_rank0("Current model does not support RoPE scaling.")
        return

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "original_max_position_embeddings" in rope_scaling:
        old_max_length = rope_scaling["original_max_position_embeddings"]
    elif hasattr(config, "max_position_embeddings"):
        old_max_length = getattr(config, "max_position_embeddings", None)
    else:
        logger.warning_rank0("Cannot find the max position embeddings in the config.")
        return

    if model_args.model_max_length is not None:  # training
        if model_args.model_max_length <= old_max_length:
            logger.warning_rank0("Input length is smaller than max length. Disabling rope scaling.")
            return

        if model_args.rope_scaling == RopeScaling.DYNAMIC:
            logger.warning_rank0(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        rope_factor = float(math.ceil(model_args.model_max_length / old_max_length))
    else:  # inference
        rope_factor = 2.0

    rope_kwargs = {
        "rope_type": getattr(model_args.rope_scaling, "value", model_args.rope_scaling),  # handle enum
        "factor": rope_factor,
    }
    setattr(config, "max_position_embeddings", old_max_length * rope_factor)
    logger.info_rank0(f"Enlarge max model length from {old_max_length} to {old_max_length * rope_factor}.")

    if model_args.rope_scaling in [RopeScaling.DYNAMIC, RopeScaling.YARN]:
        rope_kwargs["original_max_position_embeddings"] = old_max_length
    elif model_args.rope_scaling == RopeScaling.LLAMA3:
        rope_kwargs["original_max_position_embeddings"] = old_max_length
        rope_kwargs["low_freq_factor"] = 1.0
        rope_kwargs["high_freq_factor"] = 4.0

    setattr(config, "rope_scaling", rope_kwargs)
    logger.info_rank0(
        f"Using {rope_kwargs['rope_type']} scaling strategy and setting scaling factor to {rope_kwargs['factor']}."
    )
