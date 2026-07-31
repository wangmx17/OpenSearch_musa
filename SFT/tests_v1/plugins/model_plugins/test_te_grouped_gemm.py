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

import importlib.util
from types import SimpleNamespace

import pytest
import torch


TE_AVAILABLE = (
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and importlib.util.find_spec("transformer_engine") is not None
)

pytestmark = pytest.mark.skipif(not TE_AVAILABLE, reason="Transformer Engine on MUSA is required")


def _eager_grouped_linear(input, weight, token_counts):
    outputs = []
    start = 0
    for expert_idx, token_count in enumerate(token_counts):
        end = start + token_count
        outputs.append(torch.nn.functional.linear(input[start:end], weight[expert_idx]))
        start = end
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("token_counts", ([0, 3, 5, 1], [2, 0, 0, 7]))
def test_te_grouped_linear_forward_backward(token_counts):
    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.te_grouped_gemm import te_grouped_linear

    torch.manual_seed(0)
    device = torch.device("musa")
    dtype = torch.bfloat16
    counts = torch.tensor(token_counts, device=device, dtype=torch.int32)
    input = torch.randn(sum(token_counts), 128, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(4, 96, 128, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(sum(token_counts), 96, device=device, dtype=dtype)

    actual = te_grouped_linear(input, weight, counts)
    actual_grads = torch.autograd.grad(actual, (input, weight), grad_output)

    ref_input = input.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    expected = _eager_grouped_linear(ref_input, ref_weight, token_counts)
    expected_grads = torch.autograd.grad(expected, (ref_input, ref_weight), grad_output)

    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(actual_grads[0], expected_grads[0], atol=0.01, rtol=0.01)
    torch.testing.assert_close(actual_grads[1], expected_grads[1], atol=0.01, rtol=0.01)


def test_te_grouped_experts_avoid_musa_bincount(monkeypatch: pytest.MonkeyPatch):
    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.te_grouped_gemm import (
        _expert_token_counts,
        te_grouped_gemm_experts_forward,
    )

    class Experts:
        num_experts = 128
        act_fn = staticmethod(torch.nn.functional.silu)
        config = SimpleNamespace(hidden_act="silu")

        def __init__(self):
            self.gate_up_proj = 0.02 * torch.randn(128, 64, 64, device="musa", dtype=torch.bfloat16)
            self.down_proj = 0.02 * torch.randn(128, 64, 32, device="musa", dtype=torch.bfloat16)

        def _apply_gate(self, gate_up_out):
            gate, up = gate_up_out.chunk(2, dim=-1)
            return self.act_fn(gate) * up

    torch.manual_seed(20260726)
    monkeypatch.setenv("OPENSEARCH_MUSA_FUSED_SWIGLU", "1")
    original_swish_glu = torch.nn.functional.swish_glu
    swish_glu_calls = 0

    def counted_swish_glu(*args, **kwargs):
        nonlocal swish_glu_calls
        swish_glu_calls += 1
        return original_swish_glu(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "swish_glu", counted_swish_glu)
    hidden_states = torch.randn(1024, 64, device="musa", dtype=torch.bfloat16)
    scores = torch.randn(1024, 128, device="musa", dtype=torch.float32)
    values, indices = torch.sort(scores, dim=-1, descending=True, stable=True)
    top_k_index = indices[:, :8]
    top_k_weights = torch.softmax(values[:, :8], dim=-1)

    counts = _expert_token_counts(top_k_index.reshape(-1), 128)
    assert counts.device.type == "cpu"
    assert counts.sum().item() == top_k_index.numel()

    experts = Experts()
    actual = te_grouped_gemm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
    assert swish_glu_calls == 1

    expected = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=experts.num_experts).permute(2, 1, 0)
    for expert_idx in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
        expert_idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_state = experts.act_fn(gate) * up
        current_state = torch.nn.functional.linear(current_state, experts.down_proj[expert_idx])
        current_state = current_state * top_k_weights[token_idx, top_k_pos, None]
        expected.index_add_(0, token_idx, current_state.to(expected.dtype))

    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.05)
