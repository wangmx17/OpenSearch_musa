# MUSA 上的 Transformer Engine Grouped GEMM 接入说明

本文说明 Qwen3-VL-MoE 训练中 `te_grouped_gemm` 的目录设计、接入原理、数值正确性保护以及性能提升来源。阅读本文不要求了解 Transformer Engine（下文简称 TE）的内部实现，只需要知道矩阵乘法和反向传播的基本概念即可。

## 1. 结论先行

### 1.1 当前文件放在 `ops/mlp/` 是否合理

当前实现位于：

```text
src/llamafactory/v1/plugins/model_plugins/kernels/ops/mlp/te_grouped_gemm.py
```

**结论：当前路径合理，现阶段建议保留，不需要为了目录命名移动代码。**

原因如下：

1. 这个文件虽然调用的底层原语叫 grouped GEMM，但它对模型做的事情并不是替换任意矩阵乘法，而是替换完整的 `Qwen3VLMoeTextExperts.forward`。
2. 一个 MoE expert 本质上就是一个带门控激活的 MLP：先执行 `gate_up_proj`，再执行 SwiGLU 类门控，最后执行 `down_proj`。
3. 仓库现有目录是按“模型算子语义”分类，而不是按第三方库分类。例如 `npu_swiglu.py` 和 `npu_fused_moe.py` 都放在 `ops/mlp/` 下。
4. `te_grouped_gemm.py` 与 `npu_fused_moe.py` 优化的是同一类模型热点，把它们放在一起便于查找和横向比较。

名称中强调 `te_grouped_gemm` 也有意义：它明确说明实现依赖 TE，并且核心加速来自 grouped GEMM，而不是一个包含路由、通信和激活的全融合 MoE kernel。

### 1.2 什么时候应该调整目录

如果后续实现扩大到多个模型、多个后端，建议再做如下拆分：

```text
ops/
├── moe/
│   └── qwen3_vl_te_experts.py       # Qwen3-VL-MoE 的路由、重排和 forward patch
└── common/                          # 或 kernels/backends/transformer_engine/
    └── te_grouped_linear.py         # 与具体模型无关的 grouped linear/autograd 封装
```

适合拆分的信号包括：

- 第二个模型开始复用 `te_grouped_linear`；
- CUDA、MUSA 等多个设备开始共享同一套 TE 封装；
- 路由重排、激活融合或 expert parallel 逻辑明显增多；
- 文件同时承担过多模型适配和后端兼容职责。

目前实现只支持 MUSA 上的 Qwen3-VL-MoE，代码规模有限，而且已经完成双机训练验证。现在移动文件只会带来 import、测试路径和 review 噪声，没有实际收益。

## 2. 为什么需要人为接入 grouped GEMM

### 2.1 普通 MLP 在做什么

最简单的线性层可以写成：

```text
Y = X · Wᵀ
```

其中：

- `X` 是一批 token 的隐藏状态；
- `W` 是模型权重；
- `Y` 是线性变换结果。

普通稠密模型的所有 token 使用同一份 `W`，所以一次较大的矩阵乘法就能处理很多 token，硬件利用率通常比较高。

### 2.2 MoE 为什么会产生大量小矩阵乘法

MoE（Mixture of Experts，混合专家）会为每个 token 选择少数几个 expert。当前 Qwen3-VL-MoE 有 128 个 expert，每个 token 选择 top-k expert；本次训练配置中 top-k 为 8。

假设 token 0 被分给 expert 2 和 expert 9，token 1 被分给 expert 2 和 expert 17，那么不同 expert 收到的 token 数并不相同：

```text
expert 0:  0 个 token
expert 1:  3 个 token
expert 2: 67 个 token
...
expert 127: 5 个 token
```

每个 expert 都有自己的权重，因此数学上需要完成一组不同形状的矩阵乘法：

```text
Y₀ = X₀ · W₀ᵀ
Y₁ = X₁ · W₁ᵀ
...
Y₁₂₇ = X₁₂₇ · W₁₂₇ᵀ
```

这里每个 `X_e` 的行数都不同，甚至可能为 0。

### 2.3 当前 Torch 原生路径为什么不够用

当前 MUSA 环境使用的 Torch 版本没有可直接用于该模型的原生 grouped GEMM API。eager 实现只能围绕 expert 逐个组织计算，形成很多小 GEMM 和大量 Python/框架调度。

小 GEMM 的问题不是乘法本身很慢，而是每次计算太少，kernel launch、参数准备和调度成本占比很高。128 个 expert、48 个 MoE experts 模块再叠加前向、反向和梯度累积后，这类固定开销会被放大。

TE 已经在当前 MUSA 环境提供底层 `general_grouped_gemm`，因此本次接入选择复用环境中已经编译好的实现，而不是修改 Torch 或自行编译新算子。

## 3. Grouped GEMM 的核心思想

Grouped GEMM 并没有改变数学公式。它只是把“很多个独立的小矩阵乘法”作为一组交给同一个优化后端统一调度。

```text
eager：
Python/框架 -> GEMM expert 0
            -> GEMM expert 1
            -> GEMM expert 2
            -> ...
            -> GEMM expert 127

grouped GEMM：
Python/框架 -> grouped GEMM([expert 0, expert 1, ..., expert 127])
                         -> 后端统一安排整组计算
```

它主要减少了：

- Python 循环和框架 dispatcher 次数；
- 大量小 kernel 的独立 launch 开销；
- 每个 expert 重复准备工作区和元数据的成本；
- 小任务之间无法被后端统一调度造成的硬件空闲。

它不会减少模型参数量，也不会把 128 个 expert 合并成同一份权重。

## 4. 一次 MoE experts 前向是如何完成的

实现入口是 `te_grouped_gemm_experts_forward`。整体数据流如下：

```text
hidden_states [T, H]
        │
        ├─ 根据 top-k 索引复制/选择 token
        ▼
selected_hidden_states [T × K, H]
        │
        ├─ 按 expert id 排序
        ▼
同一 expert 的 token 连续存放
        │
        ├─ CPU bincount 得到每个 expert 的 token 数 m_splits
        ▼
TE grouped GEMM：gate_up_proj
        │
        ├─ self._apply_gate（门控激活）
        ▼
TE grouped GEMM：down_proj
        │
        ├─ 乘 routing weight
        ├─ 使用逆排列恢复原 token/top-k 顺序
        ▼
对同一 token 的 K 个 expert 输出求和
        │
        ▼
最终输出 [T, H]
```

下面逐步解释。

### 4.1 展开 top-k 路由

原始隐藏状态中每个 token 只有一行，但每个 token 会发送给 K 个 expert。因此代码先构造 `token_idx`，把隐藏状态展开成 `T × K` 行：

```python
selected_hidden_states = hidden_states[token_idx]
```

同时把 `top_k_index` 和 `top_k_weights` 展平成一维。每一行展开后的隐藏状态都有一个目标 expert 和一个路由权重。

### 4.2 按 expert 排序

Grouped GEMM 需要同一个 expert 的输入连续存放。代码使用：

```python
perm = torch.argsort(expert_ids)
inv_perm = torch.argsort(perm)
```

- `perm` 把输入按 expert id 分组；
- `inv_perm` 在计算完成后恢复原来的 token/top-k 顺序。

排序之后，输入看起来类似：

```text
[expert 0 的全部 token]
[expert 1 的全部 token]
...
[expert 127 的全部 token]
```

### 4.3 生成 `m_splits`

`m_splits[e]` 表示 expert `e` 收到多少行输入。TE 使用这个列表判断连续输入的分组边界。

本次实现特意在 CPU 上统计：

```python
expert_ids_cpu = expert_ids.detach().to(device="cpu", dtype=torch.int64)
tokens_per_expert = torch.bincount(expert_ids_cpu, minlength=num_experts)
```

这不是随意把计算搬到 CPU。TE 的当前接口本来就需要 host 侧的 `m_splits`，而且当前 torch-musa 的 `bincount` 存在正确性问题，详见第 7 节。

### 4.4 两次 grouped linear

Qwen3-VL-MoE expert 的结构可以简化为：

```text
gate_up = Linear(hidden)
middle  = activation(gate_up)
output  = Linear(middle)
```

因此每层执行两次 grouped linear：

1. 所有 expert 的 `gate_up_proj` 作为一组执行；
2. 门控激活完成后，所有 expert 的 `down_proj` 再作为一组执行。

模型权重仍保持原来的堆叠 Parameter 形式：

```text
weight.shape = [num_experts, output_dim, input_dim]
```

`weight.unbind(0)` 和 `torch.split(input, split_sizes)` 主要生成张量 view/描述信息，不会把模型拆成 128 套新 Parameter。这一点对 checkpoint、优化器和 DeepSpeed ZeRO-3 很重要。

### 4.5 恢复顺序并合并 expert 输出

Grouped GEMM 输出仍按 expert 分组。代码先乘每条路由的 `top_k_weights`，再使用 `inv_perm` 恢复顺序，最后把同一 token 的 K 个 expert 输出求和：

```python
out_per_sample = out_per_sample_grouped[inv_perm]
output = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
```

因此优化前后执行的是同一个 MoE 数学过程。

## 5. 反向传播为什么需要自定义 autograd

Torch 不知道 TE 低层 C++ 扩展内部如何计算，所以无法自动推导它的梯度。本次实现通过 `torch.autograd.Function` 明确提供 forward 和 backward。

对于：

```text
Y = X · Wᵀ
```

反向传播需要：

```text
dX = dY · W
dW = dYᵀ · X
```

`_TeGroupedLinear` 在 forward 中保存 `input`、`weight` 和分组大小；backward 中再调用 TE grouped GEMM 分别计算：

- `_te_grouped_input_grad`：输入梯度 `dX`；
- `_te_grouped_weight_grad`：权重梯度 `dW`。

代码中的 `layout="NN"`、`layout="NT"` 是 TE 对输入转置关系的描述。它们不改变梯度公式，只是告诉底层库如何解释内存中的矩阵。

第三个输入 `tokens_per_expert` 只是分组元数据，不参与求导，所以 backward 返回 `None`。

这种方式也能配合 gradient checkpointing：checkpoint 重算 forward 时会再次走相同的分组、排序和 TE grouped GEMM，之后仍由自定义 backward 计算梯度。

## 6. 为什么选择 TE 低层 API，而不是高层 `GroupedLinear`

本次使用：

```python
transformer_engine.pytorch.cpp_extensions.general_grouped_gemm
```

没有直接实例化 TE 高层 `GroupedLinear`，主要有两个原因：

1. 当前 MUSA TE 的高层模块导入路径依赖环境中的 `musa_patch.mem_utils`，在现有镜像中不可用；低层 grouped GEMM 扩展本身可以正常工作。
2. 高层模块通常希望自己持有和管理线性层权重，而当前 Qwen3-VL-MoE 已有堆叠权重 Parameter。重建模块会影响参数名称、checkpoint 加载、优化器引用和 ZeRO-3 参数分片，风险远高于只替换 forward。

低层接入保留了模型原有 Parameter，只替换计算路径。因此：

- 无需转换 checkpoint；
- DeepSpeed 仍然看到原来的参数对象；
- 优化器仍然更新原来的权重；
- 可以用 eager 实现直接做前向和梯度对比；
- 不需要修改已安装的 TE 源码。

另外，`_te_grouped_gemm_api()` 会先导入 `torch_musa`，再导入 TE。当前镜像中的 TE 会在导入期间安装 CUDA 命名兼容补丁；先完成 torch_musa 初始化可以避免二者同时修改延迟加载模块造成的冷启动导入问题。

## 7. 本次 NaN 的真正原因与修复

### 7.1 发现的问题

在与真实路由规模一致的固定输入上，共有 8192 条 token-to-expert 路由。当前 torch-musa 2.7.1 的 MUSA `torch.bincount` 只统计出 8177：

```text
实际路由条数：8192
MUSA bincount 之和：8177
缺失：15
```

这不是普通的 BF16 舍入误差，而是整数计数结果错误。

### 7.2 错误计数为什么会导致 NaN

Grouped GEMM 依靠 `m_splits` 划定每个 expert 的输入范围。如果所有分组之和只有 8177，而实际输出缓冲区有 8192 行，那么最后 15 行没有被任何分组覆盖。

输出由 `torch.empty` 创建，未覆盖区域不是自动清零，而是保留显存中的旧数据。后续门控、路由加权、前向传播和反向传播会继续使用这些未初始化值，最终可能把 NaN 扩散到 loss。

之前的 MATE 接入没有对分组总数做强校验，因此错误被静默传入 grouped GEMM。

### 7.3 当前实现的两层保护

第一层是在 CPU 上执行整数计数：

```python
return torch.bincount(expert_ids_cpu, minlength=num_experts)
```

第二层是在进入 TE 前检查分组边界：

```python
if sum(split_sizes) != total_tokens:
    raise ValueError(...)
```

还会检查：

- 是否存在负数分组；
- expert 数量和 count 数量是否一致；
- 输入与权重的 K 维是否一致；
- 输入和权重是否为相同的 FP16/BF16 类型。

这样即使以后再次遇到路由计数异常，也会在 grouped GEMM 前明确报错，而不是继续计算并产生难以定位的 NaN。

### 7.4 0-token expert

路由不均衡时，某些 expert 可能没有任何 token。实现和测试都允许 `m_splits` 中出现 0，TE 可以跳过这些空分组。0-token expert 不是错误，也不应该为了规避它而伪造输入。

## 8. 为什么性能提升会很大

性能提升不是来自改变精度、减少 expert 或少算了某些 token，而是来自更高效地完成同样的计算。

### 8.1 eager 的固定开销被重复了很多次

Qwen3-VL-MoE 当前有 128 个 expert 和 48 个被替换的 experts 模块。每个 expert 至少有 `gate_up_proj` 和 `down_proj` 两次线性计算。

如果按 expert 逐个调度，单次前向在最朴素的理解下就可能涉及：

```text
48 层 × 128 experts × 2 个线性层
```

反向还要计算输入梯度和权重梯度，训练中又有 8 次梯度累积。即使框架已经做了部分优化，仍会产生数量很大的小任务。

### 8.2 每个 expert 收到的 token 通常不多

MoE 的优点是每个 token 只访问少数 expert，但这也意味着单个 expert 的 GEMM 行数 `M` 较小且不均匀。小 GEMM 很难填满计算单元，独立 launch 的开销占比会很高。

Grouped GEMM 让后端同时看到所有 expert 的形状和地址，可以针对整组任务统一调度。计算量没有减少，但用于等待和调度的时间显著下降。

### 8.3 权重布局保持连续且无需重建模块

当前 expert 权重原本就是 `[E, N, K]` 的堆叠张量。实现通过 unbind 得到每个 expert 的 view，并直接交给 TE，没有在每次 forward 中重新拼接 128 份权重，也没有在 host 上重建模型结构。

### 8.4 优化覆盖了训练的主要热点

本次替换了 48 个 Qwen3-VL-MoE experts 模块，并且 forward、dX、dW 都走 TE grouped GEMM。不是只有推理前向加速，训练反向中的主要线性计算同样受益。

### 8.5 额外开销为什么仍然值得

当前方案仍有以下开销：

- 两次 `argsort`；
- token 重排和逆重排；
- expert id 从 MUSA 拷到 CPU；
- CPU `bincount`；
- 构造 split/list 元数据。

但 TE 本来就需要 host `m_splits`，因此 CPU 计数没有引入第二次不必要的同步。对于 30B MoE 模型，expert 的两组大规模线性计算仍是主成本，节省的大量小 GEMM 调度开销明显超过上述路由整理成本。

隔离测试中，真实权重、5312 条路由输入并包含 gradient checkpoint 的单个 experts 模块，预热后的 forward+backward 约为 0.099 秒。这个数字不能直接等同于端到端训练吞吐，但能说明 TE 前向和反向没有退回逐 expert 的慢路径。

双机正式训练首个 optimizer step 约 282 秒，loss 和 grad norm 均为有限值。性能对比应使用相同数据、相同序列长度、相同缓存状态和相同采样步骤继续测量，不应只比较包含模型初始化或 checkpoint 保存的总 runtime。

## 9. Kernel 是如何被配置、发现和应用的

### 9.1 YAML 显式选择

训练配置中使用：

```yaml
experts_implementation: eager
v1_kernel_ids: te_grouped_gemm
```

`experts_implementation: eager` 与 `te_grouped_gemm` 不冲突：

1. Transformers 先用 eager 方式创建标准模型和标准参数；
2. 权重加载、ZeRO-3 初始化和模型 patch 正常完成；
3. LLaMA-Factory kernel plugin 随后把 experts 实例的 `forward` 替换为 TE 版本。

保留 eager 作为基础实现反而降低了接入风险，因为不依赖当前 Torch 尚不支持的原生 `grouped_mm`。

### 9.2 精确选择优先于旧布尔开关

`ModelArguments` 新增 `v1_kernel_ids`，可以只指定需要的 kernel：

```yaml
v1_kernel_ids: te_grouped_gemm
```

模型加载时：

```python
selected_v1_kernels = model_args.v1_kernel_ids or model_args.use_v1_kernels
```

显式 ID 优先于 `use_v1_kernels`，避免 `auto` 模式意外启用当前设备上的其他实验 kernel。

### 9.3 自动发现与设备过滤

Kernel 系统会递归扫描 `kernels/ops/**/*.py`。导入 `te_grouped_gemm.py` 后，`@register_kernel` 注册：

```python
_kernel_id = "te_grouped_gemm"
_device = DeviceType.MUSA
```

注册表只在当前 accelerator 类型为 MUSA 时保留它。`check_deps()` 还会检查 `transformer_engine` 是否存在。

### 9.4 只 patch 目标模型

应用阶段有多重限制：

- 必须是 `model_type == "qwen3_vl_moe"`；
- 只替换类名为 `Qwen3VLMoeTextExperts` 的模块；
- 重复调用时不会重复 patch；
- 找不到目标模块会立即报错；
- 原始 forward 保存在 `_te_grouped_gemm_original_forward`，便于调试和对比。

本次 30B 模型启动日志会出现：

```text
Applied Transformer Engine grouped GEMM to 48 Qwen3-VL-MoE expert modules.
```

## 10. 修改文件说明

| 文件 | 作用 |
|---|---|
| `src/llamafactory/v1/plugins/model_plugins/kernels/ops/mlp/te_grouped_gemm.py` | TE grouped linear、autograd、Qwen3-VL-MoE forward 和 kernel 注册 |
| `src/llamafactory/hparams/model_args.py` | 新增可精确指定 kernel ID 的 `v1_kernel_ids` 参数 |
| `src/llamafactory/model/loader.py` | 在模型加载完成后应用指定 kernel，并让显式 ID 优先 |
| `src/llamafactory/v1/accelerator/helper.py` | 为 v1 kernel 注册系统增加 `DeviceType.MUSA` |
| `examples/agentic_full/qwen3_vl_full_sft_30_3b_trace.yaml` | 在本次训练配置中启用 `te_grouped_gemm` |
| `tests_v1/plugins/model_plugins/test_te_grouped_gemm.py` | 验证 forward、dX、dW、0-token expert 和 MUSA bincount 回归 |
| `tests/model/test_model_args.py` | 验证 `v1_kernel_ids` 参数解析 |

## 11. 已完成的正确性验证

### 11.1 单元测试

MUSA Pod 中运行相关测试：

```text
4 passed, 2 warnings
```

测试覆盖：

- 多种分组大小；
- 0-token expert；
- forward 与 eager 对比；
- 输入梯度 `dX` 与 eager 对比；
- 权重梯度 `dW` 与 eager 对比；
- 8192 条路由的 CPU count 总数检查；
- 完整 experts forward 与 eager 对比。

### 11.2 真实权重验证

使用 checkpoint 中的真实 expert 权重进行前向对比：

```text
所有输出 finite
最大绝对误差：0.03125
平均绝对误差：约 0.0002123
```

这是 BF16 下不同合法 GEMM 调度顺序产生的正常舍入差异。

### 11.3 双机 optimizer 边界验证

双机 16 rank、8 次梯度累积的诊断任务中：

- 128 条 microbatch loss 全部 finite；
- 16 个 rank 的 pre-optimizer 张量全部 finite；
- 16 个 rank 的 post-optimizer 张量全部 finite；
- optimizer step loss 为 `1.018`；
- grad norm 为 `0.1172`。

关闭精度调试后的正式训练首步得到相同的 `loss=1.018` 和 `grad_norm=0.1172`。

## 12. 已知限制与后续优化方向

当前实现有意保持范围较小，已知限制包括：

1. 只支持 MUSA 上的 `qwen3_vl_moe`。
2. 通过类名匹配并 monkey-patch forward，对 Transformers 大版本变更比较敏感。
3. 使用的是 TE 低层扩展 API，升级 TE 后需要重新运行 forward/backward 回归测试。
4. 每层都需要把 expert id 同步到 CPU；这是当前正确性和 TE host `m_splits` 要求下的保守选择。
5. token 排序和逆排序仍有优化空间，例如未来使用经过验证的 MUSA token-permute kernel。
6. 当前只替换 expert MLP，不包含 router、token permutation、激活和通信的全融合。

建议后续优化遵循“先证明正确，再一次只替换一个边界”的原则：

1. 保留当前 TE grouped GEMM 作为正确基线；
2. 单独评估更快且正确的 expert count；
3. 单独评估 token permute/unpermute；
4. 每次变更都比较 forward、dX、dW 和 optimizer 前后 finite 状态；
5. 不直接修改 TE 安装包来掩盖模型侧问题。

## 13. 使用与回退

启用：

```yaml
experts_implementation: eager
v1_kernel_ids: te_grouped_gemm
```

确认日志中存在：

```text
Applied Transformer Engine grouped GEMM to 48 Qwen3-VL-MoE expert modules.
```

回退到原始 eager experts 时，删除或置空 `v1_kernel_ids`：

```yaml
experts_implementation: eager
v1_kernel_ids: null
```

回退不需要转换模型权重，因为本次实现从未替换 Parameter，只替换了运行时 forward 方法。

## 14. Review 建议

本次 review 建议重点关注：

1. 是否接受现阶段继续放在 `ops/mlp/`；本文建议接受。
2. forward 中排序、加权和逆排序是否与原 Qwen3-VL-MoE 语义一致。
3. backward 的 `dX`、`dW` 是否始终走正确的 TE layout。
4. CPU bincount 和分组总数强校验是否必须保留；本文建议在 torch-musa 修复并完成大规模回归前保留。
5. 是否继续保持低层 TE API，以避免重构 Parameter 和影响 ZeRO-3；本文建议保持。
6. 后续若支持第二个模型，是否按第 1.2 节拆分公共 grouped linear 与模型适配层。
