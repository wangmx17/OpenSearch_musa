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
from packaging.version import Version

from ...extras import logging
from ...extras.constants import RopeScaling
from .musa_fused_rope import MUSA_ROPE_FREQ_CIS_ATTR, apply_rotary_pos_emb_musa


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


def _is_supported_qwen3_vl_moe_rope_version() -> bool:
    import transformers

    return Version(transformers.__version__).release[:2] == (5, 2)


def _qwen3_vl_moe_rope_forward(self, x: torch.Tensor, position_ids: torch.Tensor):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    with modeling_qwen3_vl_moe.maybe_autocast(device_type=device_type, enabled=False):
        if _env_flag("OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND", "1"):
            # The contraction dimension is one, so broadcast multiplication is
            # mathematically identical to matmul and avoids the torch-musa 2.7.1
            # float32 bmm kernel that loses position precision after index 2048.
            freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(2, 3)
        else:
            # Keep the original Transformers BMM frequency-generation path. This
            # allows BMM + fused RoPE A/B runs while preserving its exact output
            # phase for torch.rope instead of reconstructing it from BF16 cos/sin.
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    if (
        _env_flag("OPENSEARCH_MUSA_FUSED_ROPE", "0")
        and x.device.type == "musa"
        and emb.shape[0] == 1
        and isinstance(self.attention_scaling, (int, float))
        and float(self.attention_scaling) == 1.0
    ):
        # Keep the exact FP32 phase produced before cos/sin quantization. Reconstructing
        # it later with atan2(cos_bf16, sin_bf16) is slower and measurably less accurate.
        setattr(cos, MUSA_ROPE_FREQ_CIS_ATTR, emb.squeeze(0).contiguous())

    return cos, sin


def patch_qwen3_vl_moe_rope_bmm(model: "PreTrainedModel") -> tuple[int, bool, bool]:
    use_bmm_workaround = _env_flag("OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND", "1")
    use_fused_rope = _env_flag("OPENSEARCH_MUSA_FUSED_ROPE", "0")
    if not use_bmm_workaround and not use_fused_rope:
        return 0, False, False
    if getattr(model.config, "model_type", None) != "qwen3_vl_moe":
        return 0, False, False

    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    if not torch_version.startswith("2.7.") or not hasattr(torch, "musa") or not torch.musa.is_available():
        return 0, False, False
    if not _is_supported_qwen3_vl_moe_rope_version():
        logger.warning_rank0_once("Qwen3-VL-MoE MUSA RoPE patch supports Transformers 5.2.x only; skipped.")
        return 0, False, False

    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    Qwen3VLMoeTextRotaryEmbedding = modeling_qwen3_vl_moe.Qwen3VLMoeTextRotaryEmbedding

    patched_modules = 0
    for module in model.modules():
        if isinstance(module, Qwen3VLMoeTextRotaryEmbedding):
            if not all(
                hasattr(module, name)
                for name in ("inv_freq", "mrope_section", "attention_scaling", "apply_interleaved_mrope")
            ):
                continue
            forward = torch.no_grad()(modeling_qwen3_vl_moe.dynamic_rope_update(_qwen3_vl_moe_rope_forward))
            module.forward = MethodType(forward, module)
            patched_modules += 1

    fused_rope_patched = use_fused_rope and patched_modules > 0 and hasattr(torch, "rope")
    if fused_rope_patched:
        modeling_qwen3_vl_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_musa

    broadcast_mul_patched = use_bmm_workaround and patched_modules > 0
    return patched_modules, broadcast_mul_patched, fused_rope_patched


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
