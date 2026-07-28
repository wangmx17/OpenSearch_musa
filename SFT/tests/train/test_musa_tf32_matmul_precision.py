import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch


pytest.importorskip("torch_musa")


_CHILD_PROGRAM = textwrap.dedent(
    r"""
    import json
    import os

    import torch
    import torch_musa  # noqa: F401


    allow_tf32 = os.environ["MUSA_TEST_ALLOW_TF32"] == "1"
    torch.backends.mudnn.allow_tf32 = allow_tf32

    sequence_length = 8048
    head_dim = 128
    inv_freq = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    position_ids = (
        torch.arange(sequence_length, dtype=torch.long)
        .view(1, 1, -1)
        .expand(3, 1, -1)
    )
    lhs = inv_freq[None, None, :, None].expand(3, 1, -1, 1)
    rhs = position_ids[:, :, None, :].float()

    expected_matmul = lhs @ rhs
    expected_mul = lhs * rhs
    lhs_musa = lhs.to("musa")
    rhs_musa = rhs.to("musa")
    actual_matmul = (lhs_musa @ rhs_musa).cpu()
    actual_mul = (lhs_musa * rhs_musa).cpu()
    torch.musa.synchronize()

    matmul_abs_error = (actual_matmul - expected_matmul).abs()
    mul_abs_error = (actual_mul - expected_mul).abs()
    result = {
        "requested_allow_tf32": allow_tf32,
        "effective_allow_tf32": torch.backends.mudnn.allow_tf32,
        "matmul_mismatch_count": int(torch.count_nonzero(matmul_abs_error).item()),
        "matmul_max_abs_error": float(matmul_abs_error.max().item()),
        "mul_mismatch_count": int(torch.count_nonzero(mul_abs_error).item()),
        "mul_max_abs_error": float(mul_abs_error.max().item()),
        "position_2049_expected": float(expected_matmul[0, 0, 0, 2049].item()),
        "position_2049_actual": float(actual_matmul[0, 0, 0, 2049].item()),
    }
    print("MUSA_TF32_RESULT=" + json.dumps(result, sort_keys=True))
    """
)


def _run_in_fresh_process(allow_tf32: bool) -> dict:
    env = os.environ.copy()
    # A force-enable override would make the Python False setting ineffective.
    env.pop("TORCH_ALLOW_TF32_MUBLAS_OVERRIDE", None)
    env["MUSA_TEST_ALLOW_TF32"] = "1" if allow_tf32 else "0"
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    prefix = "MUSA_TF32_RESULT="
    for line in completed.stdout.splitlines():
        if line.startswith(prefix):
            return json.loads(line.removeprefix(prefix))

    raise AssertionError(
        "The TF32 child process did not emit a result.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is required")
def test_allow_tf32_changes_long_rope_matmul_but_not_broadcast_mul() -> None:
    """TF32 must affect the RoPE-shaped matmul, not its elementwise workaround."""
    fp32 = _run_in_fresh_process(allow_tf32=False)
    tf32 = _run_in_fresh_process(allow_tf32=True)

    assert fp32["effective_allow_tf32"] is False
    assert tf32["effective_allow_tf32"] is True

    assert fp32["matmul_mismatch_count"] == 0, fp32
    assert fp32["matmul_max_abs_error"] == 0.0, fp32
    assert tf32["matmul_mismatch_count"] > 0, tf32
    assert tf32["matmul_max_abs_error"] > 0.0, tf32

    assert fp32["mul_mismatch_count"] == 0, fp32
    assert tf32["mul_mismatch_count"] == 0, tf32
    assert fp32["mul_max_abs_error"] == 0.0, fp32
    assert tf32["mul_max_abs_error"] == 0.0, tf32

    assert fp32["position_2049_actual"] == fp32["position_2049_expected"], fp32
    assert tf32["position_2049_actual"] != tf32["position_2049_expected"], tf32
