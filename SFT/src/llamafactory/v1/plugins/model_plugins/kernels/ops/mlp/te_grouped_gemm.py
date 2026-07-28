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

"""Transformer Engine grouped GEMM support for Qwen3-VL-MoE training on MUSA."""

import importlib.util
import types

import torch

from ......accelerator.helper import DeviceType
from ......utils import logging
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


logger = logging.get_logger(__name__)


def _is_te_grouped_gemm_available() -> bool:
    return importlib.util.find_spec("transformer_engine") is not None


def _te_grouped_gemm_api():
    # The installed MUSA TE build patches CUDA-named compatibility APIs while
    # importing. Finish torch_musa initialization first so that patching does
    # not race lazy additions to sys.modules.
    import torch_musa  # noqa: F401

    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
    from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace

    return general_grouped_gemm, get_multi_stream_cublas_workspace


def _validate_grouped_linear_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> None:
    if input.ndim != 2 or weight.ndim != 3:
        raise ValueError(f"Expected 2D input and 3D weight, got {input.shape=} and {weight.shape=}.")
    if input.size(-1) != weight.size(-1):
        raise ValueError(f"Grouped GEMM K dimensions do not match: {input.size(-1)} != {weight.size(-1)}.")
    if weight.size(0) != tokens_per_expert.numel():
        raise ValueError(
            f"Expected one token count per expert, got {tokens_per_expert.numel()} counts for {weight.size(0)} experts."
        )
    if input.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != input.dtype:
        raise TypeError(f"TE grouped GEMM requires matching fp16/bf16 tensors, got {input.dtype=} and {weight.dtype=}.")


def _split_sizes(tokens_per_expert: torch.Tensor, total_tokens: int) -> list[int]:
    split_sizes = tokens_per_expert.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(size < 0 for size in split_sizes):
        raise ValueError(f"Grouped GEMM token counts must be non-negative, got {split_sizes}.")
    if sum(split_sizes) != total_tokens:
        raise ValueError(f"Grouped GEMM token counts sum to {sum(split_sizes)}, expected {total_tokens}.")
    return split_sizes


def _expert_token_counts(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    r"""Count routes on CPU to avoid the incorrect torch-musa 2.7.x bincount kernel."""
    expert_ids_cpu = expert_ids.detach().to(device="cpu", dtype=torch.int64)
    return torch.bincount(expert_ids_cpu, minlength=num_experts)


def _te_grouped_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    split_sizes: list[int],
) -> torch.Tensor:
    general_grouped_gemm, get_workspaces = _te_grouped_gemm_api()
    input_list = list(torch.split(input, split_sizes, dim=0))
    weight_list = list(weight.unbind(dim=0))
    output = torch.empty((input.size(0), weight.size(1)), device=input.device, dtype=input.dtype)
    general_grouped_gemm(
        weight_list,
        input_list,
        [output],
        input.dtype,
        get_workspaces(),
        m_splits=split_sizes,
        single_output=True,
    )
    return output


def _te_grouped_input_grad(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    split_sizes: list[int],
) -> torch.Tensor:
    general_grouped_gemm, get_workspaces = _te_grouped_gemm_api()
    grad_output_list = list(torch.split(grad_output, split_sizes, dim=0))
    weight_list = list(weight.unbind(dim=0))
    grad_input = torch.empty(
        (grad_output.size(0), weight.size(2)), device=grad_output.device, dtype=grad_output.dtype
    )
    general_grouped_gemm(
        weight_list,
        grad_output_list,
        list(torch.split(grad_input, split_sizes, dim=0)),
        grad_output.dtype,
        get_workspaces(),
        layout="NN",
        m_splits=split_sizes,
        grad=True,
    )
    return grad_input


def _te_grouped_weight_grad(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    split_sizes: list[int],
    weight: torch.Tensor,
) -> torch.Tensor:
    general_grouped_gemm, get_workspaces = _te_grouped_gemm_api()
    input_list = list(torch.split(input, split_sizes, dim=0))
    grad_output_list = list(torch.split(grad_output, split_sizes, dim=0))
    grad_weight = torch.empty_like(weight)
    general_grouped_gemm(
        input_list,
        grad_output_list,
        list(grad_weight.unbind(dim=0)),
        grad_output.dtype,
        get_workspaces(),
        layout="NT",
        m_splits=split_sizes,
        grad=True,
    )
    return grad_weight


class _TeGroupedLinear(torch.autograd.Function):
    r"""Autograd bridge around Transformer Engine's low-level grouped GEMM API."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        _validate_grouped_linear_inputs(input, weight, tokens_per_expert)
        split_sizes = _split_sizes(tokens_per_expert, input.size(0))
        input = input.contiguous()
        weight = weight.contiguous()
        ctx.save_for_backward(input, weight)
        ctx.split_sizes = split_sizes
        return _te_grouped_forward(input, weight, split_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = _te_grouped_input_grad(grad_output, weight, ctx.split_sizes)
        if ctx.needs_input_grad[1]:
            grad_weight = _te_grouped_weight_grad(input, grad_output, ctx.split_sizes, weight)

        return grad_input, grad_weight, None


def te_grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _TeGroupedLinear.apply(input, weight, tokens_per_expert)


def te_grouped_gemm_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    r"""Qwen3-VL-MoE experts forward using Transformer Engine grouped GEMM."""
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    selected_hidden_states = hidden_states[token_idx]

    perm = torch.argsort(expert_ids)
    inv_perm = torch.argsort(perm)
    sample_weights_grouped = sample_weights[perm]
    selected_hidden_states_grouped = selected_hidden_states[perm]
    # Transformer Engine consumes host m_splits. Count on CPU as well because
    # torch-musa 2.7.x can silently undercount production-sized int64 inputs.
    tokens_per_expert = _expert_token_counts(expert_ids, self.num_experts)

    gate_up_out = te_grouped_linear(
        selected_hidden_states_grouped,
        self.gate_up_proj,
        tokens_per_expert,
    )
    gated_out = self._apply_gate(gate_up_out)
    out_per_sample_grouped = te_grouped_linear(
        gated_out,
        self.down_proj,
        tokens_per_expert,
    )
    out_per_sample_grouped = out_per_sample_grouped * sample_weights_grouped.unsqueeze(-1)

    out_per_sample = out_per_sample_grouped[inv_perm]
    return out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


@register_kernel
class TeGroupedGemmKernel(BaseKernel):
    _kernel_id = "te_grouped_gemm"
    _device = DeviceType.MUSA

    @classmethod
    def check_deps(cls) -> bool:
        return super().check_deps() and _is_te_grouped_gemm_available()

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        if model is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")
        if not cls.check_deps():
            raise RuntimeError("A MUSA-compatible Transformer Engine build is required for te_grouped_gemm.")
        if getattr(model.config, "model_type", None) != "qwen3_vl_moe":
            raise ValueError("te_grouped_gemm currently supports only Qwen3-VL-MoE models.")

        patched_experts = 0
        for module in model.modules():
            if module.__class__.__name__ != "Qwen3VLMoeTextExperts":
                continue
            if hasattr(module, "_te_grouped_gemm_original_forward"):
                continue
            module._te_grouped_gemm_original_forward = module.forward
            module.forward = types.MethodType(te_grouped_gemm_experts_forward, module)
            patched_experts += 1

        if not patched_experts:
            raise RuntimeError("No Qwen3VLMoeTextExperts modules were found to patch.")

        logger.info_rank0(f"Applied Transformer Engine grouped GEMM to {patched_experts} Qwen3-VL-MoE expert modules.")
        return model
