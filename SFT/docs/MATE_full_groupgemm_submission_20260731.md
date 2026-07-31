# MATE full grouped GEMM 提交说明

## 提交范围

本次提交面向 `musa_dev_zman_mate` 分支，核心目标是让 Qwen3-VL-MoE expert MLP 可以通过
YAML 开关选择 MATE grouped GEMM，并覆盖 forward、input gradient 和 weight gradient。

## 核心修改

- 更新 `mate_grouped_gemm.py`：
  - forward 继续使用 `mate.gemm.ragged_m_moe_gemm_16bit`；
  - dX 使用 `ragged_m` 加转置 expert weight；
  - dW 新增 `mate.gemm.ragged_k_moe_gemm_16bit`；
  - 保留小 shape / 非 MUSA 设备下的 eager fallback；
  - apply 日志更新为 `Applied MATE grouped GEMM (fwd+dX+dW) ...`。
- 新增 MATE 训练 YAML：
  - `examples/agentic_full/qwen3_vl_full_sft_30_3b_mate_groupgemm.yaml`
  - 与 TE 默认配置保持一致，仅通过 `v1_kernel_ids: mate_grouped_gemm` 切换。
- 新增验证脚本：
  - `scripts/verify_mate_full_groupgemm.py`
  - `scripts/bench_te_vs_mate_strict_ut.py`
- 新增 MATE 启动 wrapper：
  - `train_30b_mate_groupgemm.sh`
  - `launch_mate_groupgemm.sh`
  - wrapper 只固定默认 MATE yaml，不提交具体集群 hostfile。
- 新增/更新文档：
  - `docs/MATE_group_gemm_notes.md`
  - `docs/MATE_installation_and_TE_comparison.md`
  - `docs/MATE_full_groupgemm_submission_20260731.md`

## 验证方式

路径和数值验证：

```bash
PYTHONPATH=src python3 scripts/verify_mate_full_groupgemm.py
```

严格 TE/MATE 对照：

```bash
PYTHONPATH=src python3 scripts/bench_te_vs_mate_strict_ut.py
```

训练 YAML 开关：

```yaml
v1_kernel_ids: mate_grouped_gemm
```

## Review 结论

- MATE 与 TE 的 expert forward 逻辑保持同构：token expand、按 expert 排序、grouped linear、
  gate/down、router weight、inverse permutation、top-k reduce。
- grouped linear 输入约束为 `input=[M,K]`、`weight=[E,N,K]`、`counts=[E]`，与 TE/MATE 两个
  backend 的接口一致。
- dW 已从逐 expert eager matmul 升级为 `ragged_k`，并通过 UT 计数检查避免静默 fallback。
- 本次不提交具体集群 `hostfile.txt`、运行日志、benchmark JSON 结果等环境产物。
