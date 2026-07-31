#!/usr/bin/env python3
"""Verify MATE grouped GEMM uses ragged_m (fwd/dX) + ragged_k (dW), matching TE roles.

Run inside a MUSA pod / container with mate + torch_musa installed:
  cd .../OpenSearch_vl_musa/SFT
  PYTHONPATH=src python3 scripts/bench_te_mate/verify_mate_full_groupgemm.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback


def main() -> int:
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_GEMM_MIN_TOKENS", "1")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_GEMM_MIN_K", "1")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_GEMM_MIN_N", "1")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_TOKENS", "64")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_K", "64")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_N", "64")

    print("=== deps ===")
    for name in ("torch", "torch_musa", "mate"):
        spec = importlib.util.find_spec(name)
        print(f"{name}: {'OK' if spec else 'MISSING'}")
        if not spec:
            return 1

    import torch
    import torch_musa  # noqa: F401
    import mate.gemm

    if not torch.musa.is_available():
        print("ERROR: musa not available")
        return 1

    # Ensure SFT src is importable
    here = os.path.abspath(os.path.dirname(__file__))
    sft_root = os.path.abspath(os.path.join(here, "..", ".."))
    src = os.path.join(sft_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp import mate_grouped_gemm as m

    print("=== API presence ===")
    print("ragged_m:", hasattr(mate.gemm, "ragged_m_moe_gemm_16bit"))
    print("ragged_k:", hasattr(mate.gemm, "ragged_k_moe_gemm_16bit"))
    assert hasattr(mate.gemm, "ragged_k_moe_gemm_16bit"), "ragged_k API missing"

    device = torch.device("musa:0")
    torch.manual_seed(0)
    counts = [128, 96, 160]
    e, total_m = len(counts), sum(counts)
    k, n = 256, 128
    x = torch.randn(total_m, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(e, n, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    tpe = torch.tensor(counts, device=device, dtype=torch.int32)

    # Instrument MATE APIs + eager fallback
    calls = {"ragged_m": 0, "ragged_k": 0, "eager_wgrad": 0}
    real_m = mate.gemm.ragged_m_moe_gemm_16bit
    real_k = mate.gemm.ragged_k_moe_gemm_16bit
    real_eager = m._eager_grouped_weight_grad

    def wrap_m(*args, **kwargs):
        calls["ragged_m"] += 1
        return real_m(*args, **kwargs)

    def wrap_k(*args, **kwargs):
        calls["ragged_k"] += 1
        return real_k(*args, **kwargs)

    def wrap_eager(*args, **kwargs):
        calls["eager_wgrad"] += 1
        return real_eager(*args, **kwargs)

    mate.gemm.ragged_m_moe_gemm_16bit = wrap_m
    mate.gemm.ragged_k_moe_gemm_16bit = wrap_k
    m._eager_grouped_weight_grad = wrap_eager
    # Clear cached API getters by rebinding module-level helpers used at call time
    # (helpers import mate.gemm each call, so wraps above are enough)

    try:
        y = m.mate_grouped_linear(x, w, tpe)
        loss = y.float().pow(2).mean()
        loss.backward()
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        mate.gemm.ragged_m_moe_gemm_16bit = real_m
        mate.gemm.ragged_k_moe_gemm_16bit = real_k
        m._eager_grouped_weight_grad = real_eager

    print("=== path counters ===")
    print(calls)
    # fwd + dX each call ragged_m once => >=2; dW calls ragged_k once
    if calls["ragged_m"] < 2:
        print("FAIL: expected ragged_m for fwd and dX")
        return 1
    if calls["ragged_k"] != 1:
        print("FAIL: expected exactly one ragged_k call for dW")
        return 1
    if calls["eager_wgrad"] != 0:
        print("FAIL: dW fell back to eager")
        return 1

    # Numeric check vs eager
    def eager_fwd(inp, weight, split):
        outs = []
        s = 0
        for i, c in enumerate(split):
            if c:
                outs.append(torch.nn.functional.linear(inp[s : s + c], weight[i]))
            s += c
        return torch.cat(outs, dim=0)

    with torch.no_grad():
        y_ref = eager_fwd(x.detach(), w.detach(), counts)
        max_fwd = (y.float() - y_ref.float()).abs().max().item()
        print(f"fwd maxdiff={max_fwd:.6e}")
        if max_fwd > 5e-2:
            print("FAIL: forward mismatch")
            return 1

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    y2 = eager_fwd(x2, w2, counts)
    y2.float().pow(2).mean().backward()
    dx_diff = (x.grad.float() - x2.grad.float()).abs().max().item()
    dw_diff = (w.grad.float() - w2.grad.float()).abs().max().item()
    print(f"dX maxdiff={dx_diff:.6e}")
    print(f"dW maxdiff={dw_diff:.6e}")
    if dx_diff > 1e-1 or dw_diff > 1e-1:
        print("FAIL: gradient mismatch vs eager")
        return 1

    # Kernel registration message / apply path smoke (no full model)
    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.mate_grouped_gemm import MateGroupedGemmKernel

    assert MateGroupedGemmKernel._kernel_id == "mate_grouped_gemm"
    assert MateGroupedGemmKernel.check_deps()
    print("=== MateGroupedGemmKernel.check_deps: OK ===")
    print("PASS: MATE full grouped GEMM (fwd+dX+dW via ragged_m/ragged_k) verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
