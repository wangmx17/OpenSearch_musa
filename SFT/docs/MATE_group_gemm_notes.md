# MATE grouped GEMM 接入说明

本文说明 Qwen3-VL-MoE expert MLP 的 MATE grouped GEMM 接入原理、配置方式和验证方法。
TE grouped GEMM 是原有能力；MATE 作为独立 kernel 接入，通过同一个 YAML 字段进行切换。

## 1. 接入目标

Qwen3-VL-MoE 的 expert MLP 会根据 router 结果把 token 分配到不同 expert。每个 expert
都要执行 gate/up projection 和 down projection，因此一次 MoE 层包含多组 token 数可能不同
的 GEMM。

MATE grouped GEMM 的目标是把这些按 expert 分组的矩阵乘交给 MUSA kernel 执行，减少逐
expert eager matmul 的调度开销，并与既有 TE grouped GEMM 路径保持相同的调用语义。

## 2. 代码位置

```text
src/llamafactory/v1/plugins/model_plugins/kernels/ops/mlp/mate_grouped_gemm.py
src/llamafactory/v1/plugins/model_plugins/kernels/ops/mlp/te_grouped_gemm.py
```

MATE kernel ID：

```yaml
v1_kernel_ids: mate_grouped_gemm
```

TE kernel ID：

```yaml
v1_kernel_ids: te_grouped_gemm
```

## 3. 训练侧接线

训练 YAML 开启 `mate_grouped_gemm` 后，kernel registry 会在模型加载阶段执行
`MateGroupedGemmKernel.apply`，遍历模型中的 `Qwen3VLMoeTextExperts` 模块，并将其
`forward` patch 成 MATE 版本。

MATE forward 的主要步骤：

1. 从 `top_k_index` 取得 token 对应的 expert ID，从 `top_k_weights` 取得 router 权重。
2. 按 expert ID 排序 token，使同一个 expert 的 token 连续排列。
3. 统计 `tokens_per_expert`，长度为 `num_experts`。
4. 调用 `mate_grouped_linear` 计算 gate/up projection。
5. 执行 gate activation。
6. 再调用 `mate_grouped_linear` 计算 down projection。
7. 乘 router 权重，恢复 token 原顺序，并对 top-k expert 输出求和。

核心 grouped linear 输入形状：

```text
input:             [total_routed_tokens, K]
weight:            [num_experts, N, K]
tokens_per_expert: [num_experts]
output:            [total_routed_tokens, N]
```

## 4. MATE 与 TE 的计算对应

当前 MATE 实现对齐 TE 的 full grouped GEMM 路径：

```text
forward: ragged_m_moe_gemm_16bit(X, W, counts)
dX:      ragged_m_moe_gemm_16bit(dY, W.transpose(1, 2), counts)
dW:      ragged_k_moe_gemm_16bit(dY, X, counts)
```

其中：

- `ragged_m_moe_gemm_16bit` 用于 forward 和 input gradient。
- `ragged_k_moe_gemm_16bit` 用于 weight gradient。
- dW 不再默认走逐 expert eager matmul；只有在过小 shape 或非 MUSA 设备等条件下才回退。

## 5. 小 shape 回退

生产训练 shape 默认会走 MATE。为了避免极小 shape 触发底层 ragged-k 不稳定路径，dW 保留
可配置阈值：

```bash
OPENSEARCH_MATE_GROUPED_GEMM_MIN_TOKENS=1
OPENSEARCH_MATE_GROUPED_GEMM_MIN_K=1
OPENSEARCH_MATE_GROUPED_GEMM_MIN_N=1
OPENSEARCH_MATE_GROUPED_WGRAD_MIN_TOKENS=64
OPENSEARCH_MATE_GROUPED_WGRAD_MIN_K=64
OPENSEARCH_MATE_GROUPED_WGRAD_MIN_N=64
```

这些环境变量只影响 MATE kernel 是否回退 eager，不改变 YAML 的 kernel 选择语义。

## 6. YAML 切换方式

不需要新增单独的 MATE YAML。复现实验时，在现有训练 YAML 中把 kernel ID 改成 MATE 即可：

```yaml
experts_implementation: eager
v1_kernel_ids: mate_grouped_gemm
```

`experts_implementation: eager` 保持 Transformers 侧专家实现选择；真正切换 TE/MATE 的是
`v1_kernel_ids`。

## 7. 验证

路径和数值验证：

```bash
PYTHONPATH=src python3 scripts/bench_te_mate/verify_mate_full_groupgemm.py
```

期望：

```text
ragged_m >= 2
ragged_k == 1
eager_wgrad == 0
```

严格 TE/MATE 对照 UT：

```bash
PYTHONPATH=src python3 scripts/bench_te_mate/bench_te_vs_mate_strict_ut.py
```

该 UT 使用固定 seed、相同输入张量、均匀 `tokens_per_expert`，分别检查 TE 与 MATE 的
kernel 调用计数、数值误差和 forward/backward 平均耗时。
