import pytest
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().float().cpu().flatten()
    reference = reference.detach().float().cpu().flatten()
    error = actual - reference
    return {
        "max_abs": error.abs().max().item(),
        "mean_abs": error.abs().mean().item(),
        "nrmse": (error.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)).item(),
        "cosine": F.cosine_similarity(actual, reference, dim=0).item(),
    }


def _run_fa2(q, k, v, grad_output):
    q, k, v = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    output.backward(grad_output)
    return output.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def _run_sdpa_math(q, k, v, grad_output):
    q, k, v = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2).contiguous()
    output.backward(grad_output)
    return output.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.parametrize("sequence_length", [256, 512])
def test_qwen3_gqa_fa2_against_sdpa_math_training_step(sequence_length):
    torch.manual_seed(20260716 + sequence_length)
    device = torch.device("musa:0")
    shape_q = (1, sequence_length, 32, 128)
    shape_kv = (1, sequence_length, 4, 128)
    q = torch.randn(shape_q, device=device, dtype=torch.bfloat16) * 0.1
    k = torch.randn(shape_kv, device=device, dtype=torch.bfloat16) * 0.1
    v = torch.randn(shape_kv, device=device, dtype=torch.bfloat16) * 0.1
    grad_output = torch.randn(shape_q, device=device, dtype=torch.bfloat16) / q.numel()

    fa2 = _run_fa2(q, k, v, grad_output)
    math = _run_sdpa_math(q, k, v, grad_output)
    names = ("output", "q_grad", "k_grad", "v_grad")

    failures = []
    for name, actual, reference in zip(names, fa2, math):
        metric = _metrics(actual, reference)
        print(
            f"seq={sequence_length} tensor={name} max_abs={metric['max_abs']:.8g} "
            f"mean_abs={metric['mean_abs']:.8g} nrmse={metric['nrmse']:.8g} "
            f"cosine={metric['cosine']:.10f}"
        )
        if metric["nrmse"] > 0.02 or metric["cosine"] < 0.999:
            failures.append((name, metric))

    # One SGD step is affine in the gradients, so gradient agreement directly
    # verifies update agreement without introducing an optimizer implementation.
    learning_rate = 1e-2
    for name, original, fa_grad, math_grad in zip(("q", "k", "v"), (q, k, v), fa2[1:], math[1:]):
        metric = _metrics(original.float() - learning_rate * fa_grad.float(), original.float() - learning_rate * math_grad.float())
        print(f"seq={sequence_length} tensor={name}_sgd_update nrmse={metric['nrmse']:.8g} cosine={metric['cosine']:.10f}")
        # The update delta is tiny relative to the original tensor, which makes
        # cosine on the full post-update tensor numerically uninformative.
        if metric["nrmse"] > 1e-5:
            failures.append((f"{name}_sgd_update", metric))

    assert not failures, f"FA2 precision regression: {failures}"
