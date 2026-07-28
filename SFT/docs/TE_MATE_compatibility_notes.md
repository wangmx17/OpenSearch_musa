# MATE grouped GEMM 接入说明

本文只说明本分支新增的 MATE grouped GEMM 接入原理、配置方式、验证结果和
安装包获取方式。TE grouped GEMM 是原有能力，TE 的接入与使用说明请参考项目
已有 TE 文档，本文不重复展开。

## 1. 接入目标

Qwen3-VL-MoE 的 expert MLP 会根据 router 的结果，把 token 分配到不同 expert。
每个 expert 都要执行一次矩阵乘法，因此一次 MoE 层实际上包含多个不同 token
数量的 GEMM。MATE 的 grouped GEMM 接口可以把这些按 expert 分组的矩阵乘法
交给 MUSA kernel 执行，减少 Python 层逐 expert 调度的开销。

本分支新增独立的 `mate_grouped_gemm` kernel，不替换或修改原有
`te_grouped_gemm` 实现。两条路径通过同一个 YAML 参数选择，便于在相同数据、
模型和训练超参数下进行性能与数值对比。

## 2. MATE 接入原理

代码位于：

```text
src/llamafactory/v1/plugins/model_plugins/kernels/ops/mlp/mate_grouped_gemm.py
```

一次 expert 前向的主要步骤如下：

1. 从 `top_k_index` 取得每个 token 被路由到的 expert ID，从 `top_k_weights`
   取得对应的 router 权重。
2. 按 expert ID 对 token 排序，把属于同一个 expert 的 token 放在连续区域，
   同时统计 `tokens_per_expert`。
3. 将排序后的 token、expert 权重和每个 expert 的 token 数量传给
   `mate.gemm.ragged_m_moe_gemm_16bit`，执行 gate/up projection 和 down
   projection 的 grouped GEMM。
4. 将 token 恢复到原始顺序，乘以 router 权重，并合并每个 token 的 top-k
   expert 输出。

`ragged_m_moe_gemm_16bit` 的核心输入可以概括为：

```text
连续排列的 token + [num_experts, N, K] 的 expert 权重
                  + 每个 expert 的 token 数量
                  -> 连续排列的 grouped GEMM 输出
```

### 反向计算说明

当前接入使用自定义 `torch.autograd.Function` 保持训练图完整：

- forward 使用 MATE 的 `ragged_m_moe_gemm_16bit`；
- `grad_input` 复用 MATE grouped GEMM，使用转置后的 expert 权重计算输入梯度；
- `grad_weight` 当前按 expert 切分后调用矩阵乘法计算，并不是声明 MATE 提供了
  一个完整的自动求导 wrapper。

因此，当前版本是“MATE 加速 forward 和 input-gradient，weight-gradient
保留明确的逐 expert 计算”的接入方案。评估训练性能时应使用完整的
forward + backward + optimizer step，而不能只测 forward。

### 小规模输入回退

代码会在以下情况回退到 eager grouped 计算：

- 当前设备不是 MUSA；
- 输入为空或所有 expert 都没有 token；
- token 数、输入 K 维度或输出 N 维度低于当前实现的阈值。

默认阈值可以通过以下环境变量调整：

```bash
OPENSEARCH_MATE_GROUPED_GEMM_MIN_TOKENS=128
OPENSEARCH_MATE_GROUPED_GEMM_MIN_K=128
OPENSEARCH_MATE_GROUPED_GEMM_MIN_N=64
```

这些阈值只影响 MATE kernel 是否启用，不改变 YAML 中的 kernel 选择语义。

## 3. YAML 配置切换

原有 TE 路径：

```yaml
v1_kernel_ids: te_grouped_gemm
```

MATE 路径：

```yaml
v1_kernel_ids: mate_grouped_gemm
```

切换只修改训练 YAML 的 `v1_kernel_ids`，不需要修改 Python 代码或启动脚本。
MATE kernel 会在模型加载阶段注册，并只对 `Qwen3VLMoeTextExperts` 模块应用
替代 forward。

如果环境缺少 MATE 或 `torch_musa`，选择 `mate_grouped_gemm` 时应直接报出依赖
错误，而不是静默地把 MATE 配置当成 TE 使用。

## 4. 已验证版本

本分支在 JD MUSA 测试环境中验证的 MATE 包为：

```text
mate 0.2.1+mu437
```

对应发布包为：

```text
mate_0.2.1.PH1.tar.gz
```

发布目录使用 MUSA SDK 4.3.7：

```text
musa/external/4.3.7/deb/others/
```

MATE 的 JIT 运行还需要同一发布目录中的配套 `tvm.tar.gz`。本次安装包中
验证到的关键配套包包括：

```text
apache_tvm_ffi-0.1.9.post3.dev0+musa.1...whl
torch_c_dlpack_ext-0.1.5-*.whl
```

具体 wheel 文件名可能随构建日期变化，应以解包后的实际文件名为准。

## 5. 获取与安装

### 方式一：wget

```bash
mkdir -p /tmp/mate_install
cd /tmp/mate_install
wget https://sh-moss.mthreads.com/sw-release/musa/external/4.3.7/deb/others/mate_0.2.1.PH1.tar.gz
wget https://sh-moss.mthreads.com/sw-release/musa/external/4.3.7/deb/others/tvm.tar.gz
```

`mate_0.2.1.PH1.tar.gz` 是发布归档，内部包含 wheel，不能直接把 tar.gz
当作标准 Python 源码包安装：

```bash
tar -xzf mate_0.2.1.PH1.tar.gz
python3 -m pip install --no-cache-dir ./mate-0.2.1+mu437-py3-none-any.whl
```

安装配套 JIT 依赖：

```bash
mkdir -p /tmp/mate_tvm_install
tar -xzf tvm.tar.gz -C /tmp/mate_tvm_install
python3 -m pip install --no-cache-dir \
  /tmp/mate_tvm_install/torch_c_dlpack_ext-*.whl \
  /tmp/mate_tvm_install/apache_tvm_ffi-*.whl
```

### 方式二：mc

如果使用 MOSS 官方 `mc` 入口，先按官方凭据管理方式配置临时 alias；不要把
access key、secret key 写入代码仓库、脚本或本文档：

```bash
mc alias set sh-moss https://sh-moss.mthreads.com '<ACCESS_KEY>' '<SECRET_KEY>'
mc cp sh-moss/sw-release/musa/external/4.3.7/deb/others/mate_0.2.1.PH1.tar.gz ./
mc cp sh-moss/sw-release/musa/external/4.3.7/deb/others/tvm.tar.gz ./
```

### 安装后检查

```bash
python3 - <<'PY'
import importlib.util
import mate
import mate.gemm

print('mate_spec:', importlib.util.find_spec('mate').origin)
print('mate_version:', getattr(mate, '__version__', '<no __version__>'))
print('has_ragged_m_moe_gemm:', hasattr(mate.gemm, 'ragged_m_moe_gemm_16bit'))
PY
```

预期至少应满足：

```text
mate_version: 0.2.1+mu437
has_ragged_m_moe_gemm: True
```

## 6. 验证建议

安装成功不等于训练路径已经可用，建议按以下顺序验证：

1. 先执行 `import mate`、`import mate.gemm`；
2. 使用小张量验证 MATE grouped GEMM 输出是 finite，并与 baseline 做误差比较；
3. 分别使用 `te_grouped_gemm` 和 `mate_grouped_gemm` 的 YAML 跑相同的短训练；
4. 同时检查 loss、梯度是否 finite、step 时间和完整 forward/backward 时间；
5. 再进行更长训练和收敛曲线对比。

本分支已在双机 MUSA 测试环境中验证：TE YAML 和 MATE YAML 均能进入正常训练
流程；MATE 版本记录为 `0.2.1+mu437`。
