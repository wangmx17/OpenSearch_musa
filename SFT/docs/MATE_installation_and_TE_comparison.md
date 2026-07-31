# MATE 安装、原理与 TE 对比

本文记录 MATE grouped GEMM 在 JD Qwen3-VL / MUSA 训练环境中的安装依赖、计算原理和 TE
对比方式。文档只保留可提交到仓库的通用信息，不包含机器账号、密码、access key 或临时
集群运维细节。

## 1. MATE 是什么

MATE 是 MUSA AI Tensor Engine，用于在 MUSA 上提供高性能算子和 JIT kernel。本项目使用
`mate.gemm` 中的 MoE ragged grouped GEMM API 来加速 Qwen3-VL-MoE expert MLP。

本次验证过的关键能力：

```text
mate.gemm.ragged_m_moe_gemm_16bit
mate.gemm.ragged_k_moe_gemm_16bit
```

其中 `ragged_m` 用于 forward / dX，`ragged_k` 用于 dW。

## 2. TE 与 MATE 的关系

TE 指 Transformer Engine。本项目中 TE 与 MATE 都只接管 Qwen3-VL-MoE expert MLP 的
grouped linear 计算，不替代 attention、embedding、optimizer 或完整模型。

| 维度 | TE | MATE |
| --- | --- | --- |
| 项目文件 | `te_grouped_gemm.py` | `mate_grouped_gemm.py` |
| YAML ID | `te_grouped_gemm` | `mate_grouped_gemm` |
| 底层 API | `general_grouped_gemm` | `ragged_m_moe_gemm_16bit` / `ragged_k_moe_gemm_16bit` |
| forward | grouped GEMM | `ragged_m` |
| dX | grouped GEMM | `ragged_m(dY, W^T)` |
| dW | grouped GEMM | `ragged_k(dY, X)` |

两条路径通过 `v1_kernel_ids` 二选一：

```yaml
v1_kernel_ids: te_grouped_gemm
```

或：

```yaml
v1_kernel_ids: mate_grouped_gemm
```

## 3. MoE grouped GEMM 原理

MoE expert MLP 中，每个 token 会被 router 分配给 top-k expert。为了把多个 expert 的线性层
合并执行，代码先按 expert ID 对 token 排序，再统计每个 expert 的 token 数：

```text
X:      [M, K]
W:      [E, N, K]
counts: [E]
Y:      [M, N]
```

数学上等价于对每个 expert 分别执行：

```python
Y_e = X_e @ W_e.T
```

PyTorch eager 可以通过循环 `torch.matmul` 或 `torch.nn.functional.linear` 实现同样计算。
MATE / TE 的价值在于把多组 expert GEMM 组织成 grouped/ragged grouped GEMM，减少大量小
GEMM 的调度开销。

## 4. MATE full backward

当前 MATE 接入覆盖完整 grouped linear autograd：

```text
forward: Y  = X @ W.T
dX:      dX = dY @ W
dW:      dW = dY.T @ X
```

对应实现：

```text
forward -> ragged_m_moe_gemm_16bit
dX      -> ragged_m_moe_gemm_16bit with transposed weights
dW      -> ragged_k_moe_gemm_16bit
```

过小 shape 下 dW 可回退 eager，以规避底层 kernel 对极小输入的不稳定行为。

## 5. 安装依赖

运行 MATE grouped GEMM 通常需要：

- `torch`
- `torch_musa`
- `mate`
- `mate-mubin`
- `tilelang`
- `apache-tvm-ffi`
- `torch_c_dlpack_ext`
- MUSA 编译工具链，例如 `mtcc`

安装时建议对 MATE、TVM、tilelang 相关 wheel 使用 `--no-deps`，避免 pip 从公网拉取不匹配的
PyTorch 或 CUDA 依赖。

安装后检查示例：

```bash
python3 - <<'PY'
import mate
import mate.gemm
import torch
import torch_musa

print("mate", getattr(mate, "__version__", "<unknown>"))
print("musa_available", torch.musa.is_available())
print("ragged_m", hasattr(mate.gemm, "ragged_m_moe_gemm_16bit"))
print("ragged_k", hasattr(mate.gemm, "ragged_k_moe_gemm_16bit"))
PY
```

## 6. 验证建议

1. 先确认 `import mate`、`import mate.gemm`、`import torch_musa` 正常。
2. 确认 `ragged_m_moe_gemm_16bit` 和 `ragged_k_moe_gemm_16bit` 均存在。
3. 执行 `scripts/verify_mate_full_groupgemm.py` 验证 MATE fwd/dX/dW 调用计数和数值正确性。
4. 执行 `scripts/bench_te_vs_mate_strict_ut.py` 做 TE/MATE 同输入严格对比。
5. 最后通过 `v1_kernel_ids: mate_grouped_gemm` 的训练 YAML 做端到端训练验证。

## 7. 注意事项

- `experts_implementation: eager` 与 `v1_kernel_ids` 是不同层面的开关；TE/MATE 切换依赖
  `v1_kernel_ids`。
- 缺少 MATE 或 `torch_musa` 时，`mate_grouped_gemm` 应直接报依赖错误，不应静默走 TE。
- 单算子 UT 只证明 grouped GEMM 路径和数值，不等价于端到端吞吐结论。
- 文档和脚本中不要提交私有集群账号、密码、access key 或临时下载凭据。
