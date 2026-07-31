"""MATE grouped GEMM support for Qwen3-VL-MoE training on MUSA.

Mirrors the TE grouped-GEMM path:
  - forward:  ragged_m_moe_gemm_16bit
  - dX:       ragged_m_moe_gemm_16bit with transposed weights
  - dW:       ragged_k_moe_gemm_16bit (K-grouped / weight-grad)

Unlike the previous MATE integration, weight gradients also go through MATE
grouped GEMM instead of per-expert eager matmul.
"""

from __future__ import annotations

import importlib.util
import os
import types

import torch

from .......model.model_utils.musa_fused_swiglu import apply_swiglu_musa
from ......accelerator.helper import DeviceType
from ......utils import logging
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


logger = logging.get_logger(__name__)


def _is_mate_grouped_gemm_available() -> bool:
    return importlib.util.find_spec("mate") is not None and importlib.util.find_spec("torch_musa") is not None


def _mate_ragged_m_api():
    import torch_musa  # noqa: F401, I001
    import mate.gemm

    return mate.gemm.ragged_m_moe_gemm_16bit


def _mate_ragged_k_api():
    import torch_musa  # noqa: F401
    import mate.gemm

    return mate.gemm.ragged_k_moe_gemm_16bit


def _expert_token_counts(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    expert_ids_cpu = expert_ids.detach().to(device="cpu", dtype=torch.int64)
    return torch.bincount(expert_ids_cpu, minlength=num_experts)


def _split_sizes(tokens_per_expert: torch.Tensor, total_tokens: int) -> list[int]:
    split_sizes = tokens_per_expert.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(size < 0 for size in split_sizes):
        raise ValueError(f"Grouped GEMM token counts must be non-negative, got {split_sizes}.")
    if sum(split_sizes) != total_tokens:
        raise ValueError(f"Grouped GEMM token counts sum to {sum(split_sizes)}, expected {total_tokens}.")
    return split_sizes


def _validate_grouped_linear_inputs(input: torch.Tensor, weight: torch.Tensor, tokens_per_expert: torch.Tensor) -> None:
    if input.ndim != 2 or weight.ndim != 3:
        raise ValueError(f"Expected 2D input and 3D weight, got {input.shape=} and {weight.shape=}.")
    if input.size(-1) != weight.size(-1):
        raise ValueError(f"Grouped GEMM K dimensions do not match: {input.size(-1)} != {weight.size(-1)}.")
    if weight.size(0) != tokens_per_expert.numel():
        raise ValueError(
            f"Expected one token count per expert, got {tokens_per_expert.numel()} counts for {weight.size(0)} experts."
        )
    if input.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != input.dtype:
        raise TypeError(
            f"MATE grouped GEMM requires matching fp16/bf16 tensors, got {input.dtype=} and {weight.dtype=}."
        )


def _eager_grouped_forward(input: torch.Tensor, weight: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
    outputs = []
    start = 0
    for expert_idx, token_count in enumerate(split_sizes):
        end = start + token_count
        if token_count:
            outputs.append(torch.nn.functional.linear(input[start:end], weight[expert_idx]))
        start = end

    if outputs:
        return torch.cat(outputs, dim=0)
    return input.new_empty((0, weight.size(1)))


def _eager_grouped_weight_grad(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    split_sizes: list[int],
    weight: torch.Tensor,
) -> torch.Tensor:
    grad_weight = torch.empty_like(weight)
    start = 0
    for expert_idx, token_count in enumerate(split_sizes):
        end = start + token_count
        if token_count:
            grad_weight[expert_idx] = grad_output[start:end].transpose(0, 1).matmul(input[start:end])
        else:
            grad_weight[expert_idx].zero_()
        start = end
    return grad_weight


def _should_use_mate_grouped_linear(input: torch.Tensor, weight: torch.Tensor, split_sizes: list[int]) -> bool:
    """Whether to call MATE kernels.

    Defaults match TE behavior (always use the vendor kernel on MUSA when there
    is work to do). Thresholds remain overridable for debugging.
    """
    if input.device.type != "musa":
        return False
    if input.numel() == 0 or not any(split_sizes):
        return False

    # Default 1: effectively always-on for real training shapes. Set higher via
    # env only when intentionally forcing an eager fallback.
    min_tokens = int(os.getenv("OPENSEARCH_MATE_GROUPED_GEMM_MIN_TOKENS", "1"))
    min_k = int(os.getenv("OPENSEARCH_MATE_GROUPED_GEMM_MIN_K", "1"))
    min_n = int(os.getenv("OPENSEARCH_MATE_GROUPED_GEMM_MIN_N", "1"))
    return input.size(0) >= min_tokens and input.size(1) >= min_k and weight.size(1) >= min_n


def _should_use_mate_weight_grad(input: torch.Tensor, grad_output: torch.Tensor, split_sizes: list[int]) -> bool:
    """ragged_k kernels are sensitive to tiny / poorly-aligned shapes."""
    if input.device.type != "musa":
        return False
    if input.numel() == 0 or grad_output.numel() == 0 or not any(split_sizes):
        return False

    # Production MoE dims are large; keep a conservative floor so unit-sized
    # smoke tensors can still fall back to eager without killing the device.
    min_tokens = int(os.getenv("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_TOKENS", "64"))
    min_k = int(os.getenv("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_K", "64"))
    min_n = int(os.getenv("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_N", "64"))
    return input.size(0) >= min_tokens and input.size(1) >= min_k and grad_output.size(1) >= min_n


def _mate_grouped_forward(input: torch.Tensor, weight: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
    if not _should_use_mate_grouped_linear(input, weight, split_sizes):
        return _eager_grouped_forward(input, weight, split_sizes)

    ragged_m_moe_gemm_16bit = _mate_ragged_m_api()
    output = torch.empty((input.size(0), weight.size(1)), device=input.device, dtype=input.dtype)
    tokens_per_expert = torch.tensor(split_sizes, device=input.device, dtype=torch.int32)
    ragged_m_moe_gemm_16bit(
        input.contiguous(),
        weight.contiguous(),
        tokens_per_expert,
        output,
        gemm_mode="per_expert",
        major_a_mode="K",
        major_b_mode="K",
    )
    return output


def _mate_grouped_input_grad(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    split_sizes: list[int],
) -> torch.Tensor:
    # dX = dY @ W  <=>  ragged_m(dY, W^T) with W: [E, N, K] -> W^T: [E, K, N]
    return _mate_grouped_forward(grad_output, weight.transpose(1, 2).contiguous(), split_sizes)


def _mate_grouped_weight_grad(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    split_sizes: list[int],
    weight: torch.Tensor,
) -> torch.Tensor:
    """dW[e] = dY[e]^T @ X[e] via MATE K-grouped GEMM (same role as TE layout=NT)."""
    if not _should_use_mate_weight_grad(input, grad_output, split_sizes):
        return _eager_grouped_weight_grad(input, grad_output, split_sizes, weight)

    ragged_k_moe_gemm_16bit = _mate_ragged_k_api()
    # ragged_k expects:
    #   a: (total_tokens, N), b: (total_tokens, K), out: (E, N, K)  [fp32 or bf16]
    # Prefer fp32 accumulation for numerical parity with TE / eager refs.
    grad_weight = torch.zeros(
        (weight.size(0), weight.size(1), weight.size(2)),
        device=weight.device,
        dtype=torch.float32,
    )
    tokens_per_expert = torch.tensor(split_sizes, device=input.device, dtype=torch.int32)
    ragged_k_moe_gemm_16bit(
        grad_output.contiguous(),
        input.contiguous(),
        tokens_per_expert,
        grad_weight,
        gemm_mode="per_expert",
        major_a_mode="M",
        major_b_mode="N",
    )
    return grad_weight.to(dtype=weight.dtype)


class _MateGroupedLinear(torch.autograd.Function):
    """Autograd bridge around MATE ragged_m / ragged_k grouped GEMM APIs."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        _validate_grouped_linear_inputs(input, weight, tokens_per_expert)
        split_sizes = _split_sizes(tokens_per_expert, input.size(0))
        input = input.contiguous()
        weight = weight.contiguous()
        ctx.save_for_backward(input, weight)
        ctx.split_sizes = split_sizes
        return _mate_grouped_forward(input, weight, split_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = _mate_grouped_input_grad(grad_output, weight, ctx.split_sizes)
        if ctx.needs_input_grad[1]:
            grad_weight = _mate_grouped_weight_grad(input, grad_output, ctx.split_sizes, weight)

        return grad_input, grad_weight, None


def mate_grouped_linear(input: torch.Tensor, weight: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
    return _MateGroupedLinear.apply(input, weight, tokens_per_expert)


def mate_grouped_gemm_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    r"""Qwen3-VL-MoE experts forward using MATE grouped GEMM (fwd + full bwd)."""
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
    tokens_per_expert = _expert_token_counts(expert_ids, self.num_experts)

    gate_up_out = mate_grouped_linear(
        selected_hidden_states_grouped,
        self.gate_up_proj,
        tokens_per_expert,
    )
    gated_out = apply_swiglu_musa(
        gate_up_out,
        self.act_fn,
        hidden_act=getattr(self.config, "hidden_act", None),
    )
    out_per_sample_grouped = mate_grouped_linear(
        gated_out,
        self.down_proj,
        tokens_per_expert,
    )
    out_per_sample_grouped = out_per_sample_grouped * sample_weights_grouped.unsqueeze(-1)

    out_per_sample = out_per_sample_grouped[inv_perm]
    return out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


@register_kernel
class MateGroupedGemmKernel(BaseKernel):
    _kernel_id = "mate_grouped_gemm"
    _device = DeviceType.MUSA

    @classmethod
    def check_deps(cls) -> bool:
        return super().check_deps() and _is_mate_grouped_gemm_available()

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        if model is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")
        if not cls.check_deps():
            raise RuntimeError("A MUSA-compatible MATE build is required for mate_grouped_gemm.")
        if getattr(model.config, "model_type", None) != "qwen3_vl_moe":
            raise ValueError("mate_grouped_gemm currently supports only Qwen3-VL-MoE models.")

        patched_experts = 0
        for module in model.modules():
            if module.__class__.__name__ != "Qwen3VLMoeTextExperts":
                continue
            if hasattr(module, "_mate_grouped_gemm_original_forward"):
                continue
            module._mate_grouped_gemm_original_forward = module.forward
            module.forward = types.MethodType(mate_grouped_gemm_experts_forward, module)
            patched_experts += 1

        if not patched_experts:
            raise RuntimeError("No Qwen3VLMoeTextExperts modules were found to patch.")

        logger.info_rank0(
            f"Applied MATE grouped GEMM (fwd+dX+dW) to {patched_experts} Qwen3-VL-MoE expert modules."
        )
        return model
