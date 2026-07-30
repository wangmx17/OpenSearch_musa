# Copyright 2025 LMSYS and the LlamaFactory team.
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

"""OpenSearch-owned MUSA fused RoPE runtime patch.

The installed Transformers package remains untouched. The model patcher swaps
the Qwen3-VL-MoE module-level RoPE function for this implementation at runtime.
"""

import torch

from ...extras import logging


logger = logging.get_logger(__name__)

MUSA_ROPE_FREQ_CIS_ATTR = "_opensearch_musa_rope_freq_cis"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rope_text_musa(x: torch.Tensor, freq_cis: torch.Tensor, unsqueeze_dim: int) -> torch.Tensor:
    if unsqueeze_dim == 1:
        # [batch, heads, seq, dim] -> [seq, batch, heads, dim]
        x_input = x.permute(2, 0, 1, 3)
        x_output = torch.rope(
            x_input,
            freq_cis,
            rotary_interleaved=False,
            batch_first=False,
            multi_latent_attention=False,
        )
        return x_output.permute(1, 2, 0, 3)

    if unsqueeze_dim == 2:
        # [batch, seq, heads, dim] -> [seq, batch, heads, dim]
        x_input = x.transpose(0, 1)
        x_output = torch.rope(
            x_input,
            freq_cis,
            rotary_interleaved=False,
            batch_first=False,
            multi_latent_attention=False,
        )
        return x_output.transpose(0, 1)

    raise ValueError(f"Unsupported unsqueeze_dim={unsqueeze_dim}.")


def apply_rotary_pos_emb_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fused MUSA RoPE when the exact frequency tensor is available.

    Qwen3-VL-MoE uses grouped-query attention, so Q and K intentionally have
    different head counts. They are validated and rotated independently.
    """
    freq_cis = getattr(cos, MUSA_ROPE_FREQ_CIS_ATTR, None)
    fast_path = (
        hasattr(torch, "rope")
        and q.device.type == "musa"
        and k.device == q.device
        and cos.device == q.device
        and sin.device == q.device
        and q.ndim == 4
        and k.ndim == 4
        and freq_cis is not None
        and freq_cis.device == q.device
        and freq_cis.ndim == 2
        and q.shape[-1] % 2 == 0
        and k.shape[-1] == q.shape[-1]
        and unsqueeze_dim in (1, 2)
    )
    if fast_path:
        seq_dim = 2 if unsqueeze_dim == 1 else 1
        fast_path = (
            q.shape[0] == k.shape[0]
            and q.shape[seq_dim] == k.shape[seq_dim] == freq_cis.shape[0]
            and q.shape[-1] == k.shape[-1] == freq_cis.shape[-1]
        )

    if not fast_path:
        return apply_rotary_pos_emb_eager(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    try:
        q_embed = _rope_text_musa(q, freq_cis, unsqueeze_dim)
        k_embed = _rope_text_musa(k, freq_cis, unsqueeze_dim)
        return q_embed.to(dtype=q.dtype), k_embed.to(dtype=k.dtype)
    except Exception as err:
        logger.warning_rank0_once(f"MUSA fused RoPE fell back to eager: {err}")
        return apply_rotary_pos_emb_eager(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


__all__ = ["MUSA_ROPE_FREQ_CIS_ATTR", "apply_rotary_pos_emb_eager", "apply_rotary_pos_emb_musa"]
