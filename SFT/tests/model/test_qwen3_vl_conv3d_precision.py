import pytest
import torch
import torch.nn.functional as F


def _error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().double().cpu().flatten()
    reference = reference.detach().double().cpu().flatten()
    error = actual - reference
    return {
        "max_abs": error.abs().max().item(),
        "mean_abs": error.abs().mean().item(),
        "nrmse": error.square().mean().sqrt().div(reference.square().mean().sqrt().clamp_min(1e-12)).item(),
        "cosine": F.cosine_similarity(actual, reference, dim=0).item(),
    }


def _linear_patch_reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    patches = (
        x.unfold(2, 2, 2)
        .unfold(3, 16, 16)
        .unfold(4, 16, 16)
        .permute(0, 2, 3, 4, 1, 5, 6, 7)
        .contiguous()
    )
    output = F.linear(patches.flatten(-4), weight.flatten(1), bias)
    return output.permute(0, 4, 1, 2, 3).contiguous()


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
@pytest.mark.parametrize(
    ("dtype", "max_nrmse", "min_cosine"),
    [(torch.float32, 1e-3, 0.999999), (torch.bfloat16, 1e-2, 0.9999)],
)
def test_qwen3_vl_conv3d_against_cpu_float64(dtype, max_nrmse, min_cosine):
    """Check Qwen3-VL's real Conv3D shape against a CPU FP64 oracle, including gradients."""
    torch.manual_seed(20260716)
    device = torch.device("musa:0")
    conv = torch.nn.Conv3d(
        3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True
    ).to(device=device, dtype=torch.float32)
    x = torch.randn(1, 3, 2, 32, 32, device=device, dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn(1, 1152, 1, 2, 2, device=device, dtype=torch.float32)

    conv.zero_grad(set_to_none=True)
    with torch.autocast(device_type="musa", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        actual_output = conv(x)
    actual_output.backward(grad_output.to(actual_output.dtype))
    actual = {
        "output": actual_output,
        "input_grad": x.grad,
        "weight_grad": conv.weight.grad,
        "bias_grad": conv.bias.grad,
    }

    x64 = x.detach().double().cpu().requires_grad_(True)
    weight64 = conv.weight.detach().double().cpu().requires_grad_(True)
    bias64 = conv.bias.detach().double().cpu().requires_grad_(True)
    reference_output = F.conv3d(x64, weight64, bias64, stride=(2, 16, 16))
    reference_output.backward(grad_output.double().cpu())
    reference = {
        "output": reference_output,
        "input_grad": x64.grad,
        "weight_grad": weight64.grad,
        "bias_grad": bias64.grad,
    }

    failures = []
    for name in actual:
        metrics = _error_metrics(actual[name], reference[name])
        print(
            f"dtype={dtype} tensor={name} max_abs={metrics['max_abs']:.8g} "
            f"mean_abs={metrics['mean_abs']:.8g} nrmse={metrics['nrmse']:.8g} "
            f"cosine={metrics['cosine']:.10f}"
        )
        if metrics["nrmse"] > max_nrmse or metrics["cosine"] < min_cosine:
            failures.append((name, metrics))

    assert not failures, f"Conv3D precision checks failed: {failures}"


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA is required")
def test_qwen3_vl_fp32_conv3d_input_gradient_against_musa_linear():
    """A second oracle isolates the Conv3D backward-input kernel on MUSA."""
    torch.manual_seed(20260716)
    device = torch.device("musa:0")
    conv = torch.nn.Conv3d(3, 1152, (2, 16, 16), stride=(2, 16, 16)).to(device)
    x = torch.randn(1, 3, 2, 32, 32, device=device, requires_grad=True)
    grad_output = torch.randn(1, 1152, 1, 2, 2, device=device)

    conv(x).backward(grad_output)
    conv_input_grad = x.grad.detach().clone()
    conv.zero_grad(set_to_none=True)
    x.grad = None

    _linear_patch_reference(x, conv.weight, conv.bias).backward(grad_output)
    metrics = _error_metrics(conv_input_grad, x.grad)
    print(
        f"fp32 Conv3D-vs-linear input_grad max_abs={metrics['max_abs']:.8g} "
        f"mean_abs={metrics['mean_abs']:.8g} nrmse={metrics['nrmse']:.8g} "
        f"cosine={metrics['cosine']:.10f}"
    )
    assert metrics["nrmse"] < 1e-3 and metrics["cosine"] > 0.999999
