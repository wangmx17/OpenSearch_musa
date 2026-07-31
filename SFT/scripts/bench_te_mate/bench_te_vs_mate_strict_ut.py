#!/usr/bin/env python3
"""Strict controlled-variable TE vs MATE grouped-linear UT.

Follows the methodology of JD_qwen3_30b_vl_TE_MATE/grouped_gemm_te_mate_ut.py:
  - fixed seed, identical tensors for both backends
  - uniform tokens_per_expert (no random routing)
  - precision vs eager
  - forward_only (no_grad) and forward_backward, 10-iter mean
  - plus audited kernel call counts to prove the right backend is used

Uses the rewritten MATE path under the full_groupgemm experiment tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def sync(torch):
    torch.musa.synchronize()


def make_tokens_per_expert(total_tokens: int, num_experts: int, device, torch):
    base = total_tokens // num_experts
    rem = total_tokens % num_experts
    counts = torch.full((num_experts,), base, device=device, dtype=torch.int32)
    if rem:
        counts[:rem] += 1
    return counts


def eager_grouped_linear(x, w, counts, torch):
    chunks = counts.detach().cpu().tolist()
    outs = []
    start = 0
    for expert_idx, token_count in enumerate(chunks):
        end = start + token_count
        if token_count:
            outs.append(x[start:end].matmul(w[expert_idx].transpose(0, 1)))
        start = end
    return torch.cat(outs, dim=0) if outs else x.new_empty((0, w.size(1)))


def tensor_stats(a, b, torch):
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-6)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "all_finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def time_iters(label, fn, iters, torch):
    times = []
    for _ in range(iters):
        sync(torch)
        t0 = time.perf_counter()
        fn()
        sync(torch)
        times.append(time.perf_counter() - t0)
    return {
        "label": label,
        "iters": iters,
        "times_sec": times,
        "avg_sec": sum(times) / len(times),
        "avg_ms": 1000.0 * sum(times) / len(times),
        "min_sec": min(times),
        "max_sec": max(times),
    }


def patch_cuda_stream(torch):
    if not hasattr(torch._C, "_cuda_getCurrentRawStream"):

        def _cuda_getCurrentRawStream(device_index=None):
            try:
                return torch.musa.current_stream().musa_stream
            except Exception:
                return 0

        torch._C._cuda_getCurrentRawStream = _cuda_getCurrentRawStream


class KernelAudit:
    def __init__(self):
        self.reset()

    def reset(self):
        self.te_gemm = 0
        self.mate_ragged_m = 0
        self.mate_ragged_k = 0
        self.mate_eager_wgrad = 0

    def install(self, te_mod, mate_mod, mate_gemm_mod, te_ext):
        self.te_mod = te_mod
        self.mate_mod = mate_mod
        self.mate_gemm_mod = mate_gemm_mod
        self.te_ext = te_ext
        self.real_te = te_ext.general_grouped_gemm
        self.real_m = mate_gemm_mod.ragged_m_moe_gemm_16bit
        self.real_k = mate_gemm_mod.ragged_k_moe_gemm_16bit
        self.real_eager = mate_mod._eager_grouped_weight_grad
        self.real_te_api = te_mod._te_grouped_gemm_api
        audit = self

        def te_wrap(*a, **k):
            audit.te_gemm += 1
            return self.real_te(*a, **k)

        def m_wrap(*a, **k):
            audit.mate_ragged_m += 1
            return self.real_m(*a, **k)

        def k_wrap(*a, **k):
            audit.mate_ragged_k += 1
            return self.real_k(*a, **k)

        def eager_wrap(*a, **k):
            audit.mate_eager_wgrad += 1
            return self.real_eager(*a, **k)

        te_ext.general_grouped_gemm = te_wrap
        mate_gemm_mod.ragged_m_moe_gemm_16bit = m_wrap
        mate_gemm_mod.ragged_k_moe_gemm_16bit = k_wrap
        mate_mod._eager_grouped_weight_grad = eager_wrap

        def te_api_wrap():
            _, get_ws = self.real_te_api()
            return te_ext.general_grouped_gemm, get_ws

        te_mod._te_grouped_gemm_api = te_api_wrap

    def uninstall(self):
        self.te_ext.general_grouped_gemm = self.real_te
        self.mate_gemm_mod.ragged_m_moe_gemm_16bit = self.real_m
        self.mate_gemm_mod.ragged_k_moe_gemm_16bit = self.real_k
        self.mate_mod._eager_grouped_weight_grad = self.real_eager
        self.te_mod._te_grouped_gemm_api = self.real_te_api

    def snap(self):
        return {
            "te_gemm": self.te_gemm,
            "mate_ragged_m": self.mate_ragged_m,
            "mate_ragged_k": self.mate_ragged_k,
            "mate_eager_wgrad": self.mate_eager_wgrad,
        }


def run_backend(name, backend_fn, x, w, counts, warmup, iters, torch, audit, expect):
    # precision vs eager on identical tensors
    x_b = x.detach().clone().requires_grad_(True)
    w_b = w.detach().clone().requires_grad_(True)
    x_e = x.detach().clone().requires_grad_(True)
    w_e = w.detach().clone().requires_grad_(True)

    audit.reset()
    out_b = backend_fn(x_b, w_b, counts)
    out_b.float().sum().backward()
    sync(torch)
    calls = audit.snap()
    for k, want in expect.items():
        got = calls[k]
        if want == 0 and got != 0:
            raise RuntimeError(f"{name}: expected {k}=0 got {got}; calls={calls}")
        if want > 0 and got < want:
            raise RuntimeError(f"{name}: expected {k}>={want} got {got}; calls={calls}")

    out_e = eager_grouped_linear(x_e, w_e, counts, torch)
    out_e.float().sum().backward()
    sync(torch)

    precision = {
        "output": tensor_stats(out_b, out_e, torch),
        "grad_input": tensor_stats(x_b.grad, x_e.grad, torch),
        "grad_weight": tensor_stats(w_b.grad, w_e.grad, torch),
    }

    # warmup
    x_run = x.detach().clone().requires_grad_(True)
    w_run = w.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        x_run.grad = None
        w_run.grad = None
        y = backend_fn(x_run, w_run, counts)
        y.float().sum().backward()
    sync(torch)

    def forward_only():
        with torch.no_grad():
            y = backend_fn(x, w, counts)
            _ = float(y.flatten()[0].float().item())

    def forward_backward():
        x_run.grad = None
        w_run.grad = None
        y = backend_fn(x_run, w_run, counts)
        y.float().sum().backward()

    # also isolate pure backward after one forward (same inputs)
    def backward_only_setup():
        nonlocal y_hold
        x_run.grad = None
        w_run.grad = None
        y_hold = backend_fn(x_run, w_run, counts)
        sync(torch)

    y_hold = None

    timing = {
        "forward_only": time_iters("forward_only", forward_only, iters, torch),
        "forward_backward": time_iters("forward_backward", forward_backward, iters, torch),
    }

    # pure bwd mean
    bwd_times = []
    for _ in range(iters):
        backward_only_setup()
        loss = y_hold.float().sum()
        sync(torch)
        t0 = time.perf_counter()
        loss.backward()
        sync(torch)
        bwd_times.append(time.perf_counter() - t0)
    timing["backward_only"] = {
        "label": "backward_only",
        "iters": iters,
        "times_sec": bwd_times,
        "avg_sec": sum(bwd_times) / len(bwd_times),
        "avg_ms": 1000.0 * sum(bwd_times) / len(bwd_times),
        "min_sec": min(bwd_times),
        "max_sec": max(bwd_times),
    }

    return {
        "backend": name,
        "kernel_calls_one_fwd_bwd": calls,
        "precision_vs_eager": precision,
        "timing": timing,
    }


def main():
    # force MATE full path on production dims
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_GEMM_MIN_TOKENS", "1")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_GEMM_MIN_K", "1")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_GEMM_MIN_N", "1")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_TOKENS", "64")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_K", "64")
    os.environ.setdefault("OPENSEARCH_MATE_GROUPED_WGRAD_MIN_N", "64")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--sft-root", default=None)
    args = parser.parse_args()

    sft = Path(args.sft_root) if args.sft_root else Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(sft / "src"))

    import torch
    import torch_musa  # noqa: F401
    import mate.gemm as mate_gemm_mod
    import transformer_engine.pytorch.cpp_extensions as te_ext

    patch_cuda_stream(torch)
    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp import te_grouped_gemm as te_mod
    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp import mate_grouped_gemm as mate_mod
    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.te_grouped_gemm import te_grouped_linear
    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.mate_grouped_gemm import mate_grouped_linear

    audit = KernelAudit()
    audit.install(te_mod, mate_mod, mate_gemm_mod, te_ext)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    cases = [
        {"label": "gate_up", "in_features": 2048, "out_features": 1536},
        {"label": "down", "in_features": 768, "out_features": 2048},
    ]

    te_expect = {"te_gemm": 3, "mate_ragged_m": 0, "mate_ragged_k": 0, "mate_eager_wgrad": 0}
    mate_expect = {"te_gemm": 0, "mate_ragged_m": 2, "mate_ragged_k": 1, "mate_eager_wgrad": 0}

    results = []
    try:
        for case in cases:
            # identical inputs for TE and MATE
            counts = make_tokens_per_expert(args.tokens, args.experts, device, torch)
            x = torch.randn(args.tokens, case["in_features"], device=device, dtype=dtype)
            w = torch.randn(args.experts, case["out_features"], case["in_features"], device=device, dtype=dtype)

            print(
                f"\n### {case['label']} tokens={args.tokens} experts={args.experts} "
                f"K={case['in_features']} N={case['out_features']} "
                f"tpe=[{int(counts.min())},{int(counts.max())}] (uniform)"
            )

            te_res = run_backend("te", te_grouped_linear, x, w, counts, args.warmup, args.iters, torch, audit, te_expect)
            mate_res = run_backend(
                "mate", mate_grouped_linear, x, w, counts, args.warmup, args.iters, torch, audit, mate_expect
            )

            for res in (te_res, mate_res):
                t = res["timing"]
                print(
                    f"  {res['backend']:4s} calls={res['kernel_calls_one_fwd_bwd']}  "
                    f"fwd={t['forward_only']['avg_ms']:.3f} ms  "
                    f"bwd={t['backward_only']['avg_ms']:.3f} ms  "
                    f"fwd+bwd={t['forward_backward']['avg_ms']:.3f} ms"
                )

            tf, mf = te_res["timing"]["forward_only"]["avg_ms"], mate_res["timing"]["forward_only"]["avg_ms"]
            tb, mb = te_res["timing"]["backward_only"]["avg_ms"], mate_res["timing"]["backward_only"]["avg_ms"]
            tt, mt = te_res["timing"]["forward_backward"]["avg_ms"], mate_res["timing"]["forward_backward"]["avg_ms"]
            print(f"  >> fwd:     {'TE' if tf < mf else 'MATE'} faster  MATE/TE={mf/tf:.3f}x")
            print(f"  >> bwd:     {'TE' if tb < mb else 'MATE'} faster  MATE/TE={mb/tb:.3f}x")
            print(f"  >> fwd+bwd: {'TE' if tt < mt else 'MATE'} faster  MATE/TE={mt/tt:.3f}x")

            results.append(
                {
                    "case": case["label"],
                    "shape": {
                        "tokens": args.tokens,
                        "experts": args.experts,
                        "in_features": case["in_features"],
                        "out_features": case["out_features"],
                        "tokens_per_expert": int(counts[0].item()),
                    },
                    "te": te_res,
                    "mate": mate_res,
                }
            )
    finally:
        audit.uninstall()

    print("\n" + "=" * 88)
    print("STRICT CONTROLLED SUMMARY (identical tensors, uniform tpe, seed={})".format(args.seed))
    print(
        f"{'case':8s} {'TE_fwd':>9s} {'MATE_fwd':>9s} {'TE_bwd':>9s} {'MATE_bwd':>9s} "
        f"{'TE_f+b':>9s} {'MATE_f+b':>9s}"
    )
    for r in results:
        te, mate = r["te"]["timing"], r["mate"]["timing"]
        print(
            f"{r['case']:8s} {te['forward_only']['avg_ms']:9.3f} {mate['forward_only']['avg_ms']:9.3f} "
            f"{te['backward_only']['avg_ms']:9.3f} {mate['backward_only']['avg_ms']:9.3f} "
            f"{te['forward_backward']['avg_ms']:9.3f} {mate['forward_backward']['avg_ms']:9.3f}"
        )

    out = {
        "method": "strict_controlled_identical_inputs_uniform_tpe",
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "sft_root": str(sft),
        "results": results,
    }
    out_path = sft / "scripts" / "bench_te_mate" / "bench_te_vs_mate_strict_result.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
