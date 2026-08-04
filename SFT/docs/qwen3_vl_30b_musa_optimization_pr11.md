# Qwen3-VL-30B-A3B MUSA 训练优化说明

本文用于说明当前 PR 中保留的 5 个优化 commit，面向第一次接触大模型训练、MUSA 和分布式训练的读者。

## 1. 文档范围

本文对应的是 `musa-dev` 到 PR 当前提交 `874b8df421d21bf0d6a596d67b46629de5776310` 的代码，也就是当前 PR 中保留的以下 5 个 commit：

| 顺序 | Commit | 主题 | 主要位置 |
| --- | --- | --- | --- |
| 1 | `f05f564` | 保护优化器选择，避免 FusedAdamW 覆盖用户配置 | `train/trainer_utils.py` |
| 2 | `439d178` | 为 MUSA fused kernel 增加失败熔断和 eager 回退 | `model/model_utils/musa_fused_*.py` |
| 3 | `d33f0c0` | 只对真正参与监督训练的 token 计算 lm-head 和 loss | `model/model_utils/musa_sparse_lm_head.py`、SFT trainer |
| 4 | `d9ee7e0` | 在 30B trace 启动脚本中默认打开 sparse lm-head | `train_30b_trace.sh` |
| 5 | `874b8df` | 将 MCCL channel 数固定为 8 | `train_30b_trace.sh` |

当前工作树中还存在上一轮回退处理留下的未提交辅助性代码和脚本。本文只解释上表 5 个已提交 commit 的内容，不把工作树中的其他文件当作 PR 组成部分。

本文是代码说明和验证指南，不是性能报告。当前 PR 的代码中没有一张可以直接引用的端到端性能结果表，因此文中不虚构加速百分比；最终收益需要在相同硬件、相同数据和相同训练配置下用 MUSA 实机测量。

## 2. 先建立几个基本概念

### 2.1 一次训练 step 在做什么

可以把一次大模型训练简单理解为下面的流程：

```text
文本/图片
   ↓ 分词与特征处理
token 的 hidden state
   ↓ Transformer 层：归一化、位置编码、注意力、MoE 专家
每个位置的 hidden state
   ↓ lm-head：把 hidden state 映射到词表
logits：每个位置对每个词的分数
   ↓ 交叉熵 loss：和正确的下一个 token 比较
loss
   ↓ 反向传播
梯度
   ↓ optimizer
更新模型参数
```

如果使用多张 MUSA 卡，卡与卡之间还要交换梯度、参数或中间结果。这些集合通信由 MCCL 负责。

几个名词可以这样理解：

- **token**：模型处理文本的基本单位，可能是一个汉字、一个词的一部分或一个特殊符号。
- **hidden state**：Transformer 对 token 的当前理解，通常是一个长度为 `hidden_size` 的向量。
- **词表（vocabulary）**：模型可以预测的所有 token 的集合。词表越大，lm-head 的输出维度越大。
- **logits**：还没有经过 softmax 的预测分数。形状通常是 `[batch, sequence, vocab_size]`。
- **label**：训练时的正确答案，即下一个 token。代码中的 `-100`（`IGNORE_INDEX`）表示该位置不计入 loss。
- **lm-head**：最后一个线性层，把一个 hidden state 投影成词表大小的 logits。
- **optimizer**：根据梯度决定参数如何更新，例如 AdamW。它不是一个普通的算子开关，而是训练算法的一部分。
- **kernel**：在 MUSA 设备上执行一类计算的底层程序。一个 Python/PyTorch 操作可能对应一个或多个 kernel。
- **fused kernel**：把多个小操作合并到一个底层 kernel 中，通常可以减少中间张量、显存读写和 kernel 启动次数。
- **eager 路径**：由普通 PyTorch 操作逐步完成的参考实现。通常更容易兼容和排错，但可能更慢。
- **MoE（Mixture of Experts）**：每个 token 只选择少数几个专家计算。模型总参数量可以很大，但单个 token 实际激活的参数较少。

### 2.2 “性能优化”和“正确性保护”不是一回事

这 5 个 commit 混合了三类工作：

1. 直接减少计算量：sparse lm-head。
2. 尝试降低单个算子或通信的开销：fused kernel、MCCL channel 配置。
3. 防止优化开关破坏原有训练语义：优化器选择保护、kernel 失败熔断。

第 3 类未必让 step time 变短，但它能避免“为了加速而改变算法”或“一个不兼容 kernel 让训练直接退出”。审查时要分别看待吞吐收益和工程安全性。

## 3. 五个 commit 的整体关系

这 5 个 commit 作用在训练链路的不同位置：

```text
Transformer 中间计算                         训练尾部                    多卡通信
┌─────────────────────────────┐       ┌──────────────────────┐      ┌─────────────┐
│ RMSNorm / RoPE / SwiGLU      │  →    │ sparse lm-head / loss │  →   │ MCCL channel │
│ fused + 失败后 eager 回退     │       │ 只选有效监督 token   │      │ 固定为 8     │
└─────────────────────────────┘       └──────────────────────┘      └─────────────┘
             ↑                                  ↑                         ↑
       运行时安全保护                       训练语义保护                通信参数实验

optimizer 选择保护贯穿反向传播后的参数更新阶段，不应被“启用 FusedAdamW”替换掉。
```

最终 30B trace 启动脚本的相关默认值大致是：MCCL channel 为 8，RMSNorm/SwiGLU/RoPE/FusedAdamW 和 sparse lm-head 默认打开。所有这些开关都可以通过环境变量覆盖；“默认打开”不等于“已经证明在所有场景都最优”。

## 4. Commit 1：保护优化器选择

### 4.1 它解决了什么问题

对应 commit：`f05f564dee4cbde69368c031b50bfee4c69aae5c`，代码见 [`trainer_utils.py`](../src/llamafactory/train/trainer_utils.py)。

训练脚本提供了：

```text
OPENSEARCH_USE_MUSA_FUSED_ADAMW=1
```

这个变量的含义是：如果用户本来选择的是标准 `adamw_torch`，并且没有启用其他特殊 optimizer 功能，可以尝试使用 MUSA 的 `FusedAdamW` 实现。

旧逻辑的问题是，看到这个环境变量为 1 后，可能直接返回 `torch_musa.optim.FusedAdamW`。这样会覆盖用户显式配置的其他 optimizer，例如：

- GaLore：通过低秩方式减少 optimizer 状态或梯度相关内存。
- APOLLO：另一种参数/梯度优化策略。
- LoRA+：对 LoRA 的不同参数使用不同学习率。
- BAdam、Adam-mini、Muon 等其他 optimizer 或训练特性。

这些不是“同一个 optimizer 的不同实现名”，而是可能改变更新公式、状态、参数分组或学习率策略的训练功能。用一个更快的 AdamW 实现替换它们，会改变训练语义。

### 4.2 现在的判断逻辑

新的代码先读取：

1. `training_args.optim` 的实际值，并兼容 enum 和字符串两种形式。
2. GaLore、APOLLO、LoRA+、BAdam、Adam-mini、Muon 等替代 optimizer 的开关。
3. `OPENSEARCH_USE_MUSA_FUSED_ADAMW` 是否打开。

只有在下面条件同时满足时才走 FusedAdamW：

```text
启用了 MUSA FusedAdamW
        且
用户选择的 optimizer 是 adamw_torch
        且
没有启用其他 optimizer 或互斥训练特性
```

如果用户选择了其他 optimizer，代码记录 warning，然后继续执行原来的 optimizer 创建逻辑，而不是强行替换。若 `torch_musa.optim.FusedAdamW` 导入失败，也会 warning 并回到原有路径。

### 4.3 为什么这算优化相关的必要保护

FusedAdamW 的目标是把 AdamW 的部分计算合并，减少 Python 调度和设备端开销；但它只能替换“同样是 AdamW 语义”的实现。这个 commit 给快速实现增加了选择边界：

```text
用户选择的算法优先
        ↓
确认可以安全替换
        ↓
才使用 MUSA FusedAdamW
```

因此它的主要收益是训练结果和配置可控，而不是直接承诺吞吐提升。

### 4.4 代码和测试重点

对应测试在 [`test_trainer_utils.py`](../tests/train/test_trainer_utils.py)，覆盖了：

- optimizer enum 和字符串形式都能识别。
- FusedAdamW 不覆盖替代 optimizer。
- FusedAdamW 导入失败时能够回退。

审查时重点确认：新开关没有改变未设置该变量时的原有行为；显式 optimizer、参数分组和特殊训练策略仍然优先。

### 4.5 回退方式

```bash
export OPENSEARCH_USE_MUSA_FUSED_ADAMW=0
```

这只关闭 MUSA FusedAdamW 尝试，不会改变用户在训练参数中选择的 optimizer。

## 5. Commit 2：fused kernel 的失败熔断与 eager 回退

### 5.1 它解决了什么问题

对应 commit：`439d178ba2e9ef8079a36485a329bfab96a5c613`。

涉及的实现和 patch 入口：

- [`musa_fused_rmsnorm.py`](../src/llamafactory/model/model_utils/musa_fused_rmsnorm.py)
- [`musa_fused_rope.py`](../src/llamafactory/model/model_utils/musa_fused_rope.py)
- [`musa_fused_swiglu.py`](../src/llamafactory/model/model_utils/musa_fused_swiglu.py)
- [`patcher.py`](../src/llamafactory/model/patcher.py)

当前训练路径尝试使用三个 MUSA fused kernel：

| 算子 | 初学者理解 | 为什么可能有收益 |
| --- | --- | --- |
| RMSNorm | 对 hidden state 做尺度归一化，保持数值范围稳定 | 归一化相关操作可合并，减少中间结果和启动次数 |
| RoPE | 给 attention 的 query/key 注入位置信息 | 位置旋转的乘法、旋转等操作可交给专用 kernel |
| SwiGLU | MoE 专家中的门控激活计算 | gate、激活函数和 up 分支乘法可以减少拆分和读写 |

普通 eager 实现通常是多步 PyTorch 操作。fused 实现通常更快，但它对设备、张量布局、维度、Transformers 版本和后端 API 的要求更严格。

### 5.2 “安装 patch”不等于“kernel 已经成功运行”

`patcher.py` 负责在满足模型类型和运行环境条件时替换目标函数。新的日志把两个阶段区分开：

```text
Installed runtime patch
    = 已经把调用入口替换成带保护的包装函数

MUSA fused ... fast path is active
    = 第一次真实调用成功，fused kernel 确实运行过
```

这个区分很重要。仅看到 patch 安装日志，不能证明设备上的 fused kernel 已经成功执行。

### 5.3 熔断器如何工作

每个 fused 模块都维护一个进程内状态，大致逻辑如下：

```text
第一次调用
   ↓
检查开关、设备、形状和后端 API
   ↓
尝试 fused kernel
   ├─ 成功：记录一次 active 日志，后续继续走 fused
   └─ 异常：记录一次 warning，把 fast path 标记为 disabled
                       ↓
                  后续调用直接走 eager
```

这叫“一次失败、当前进程熔断”。它有三个工程目的：

1. 避免每个训练 step 都重复触发同一种异常。
2. 避免日志被同一个错误刷屏。
3. 让训练在可能的情况下继续运行，只牺牲该算子的 fused 性能。

分布式训练中每个 rank 通常是独立进程，所以某个 rank 的熔断状态不会自动替其他 rank 做判断。实际验证时要查看所有 rank 的日志，不能只看 rank 0。

### 5.4 RMSNorm 的保护

`apply_rms_norm_musa` 先检查：

- 全局 fused 开关是否关闭。
- 输入是否在 MUSA 上。
- 后端是否提供 `F.rms_norm`。

满足条件才尝试 fused kernel。失败后使用 eager 公式：先以 float32 计算均方和倒数平方根，再乘以权重，最后转换回原数据类型。这条回退路径保留了原有计算含义，虽然不等价于已经完成数值验证。

实现中还会在第一次成功时打 active 日志，在第一次失败时只打一次 warning。

### 5.5 RoPE 的保护

RoPE（Rotary Position Embedding）可以理解为：根据 token 的位置，对 attention 的 query 和 key 向量做成对旋转，使模型知道顺序信息。

fused RoPE 在调用前会检查：

- 开关和 MUSA 设备。
- query、key、cos、sin 的设备是否一致。
- query/key 是否为预期的四维张量。
- 旋转维度是否有效且为偶数。
- `freq_cis`、序列长度和布局是否匹配。
- `unsqueeze_dim` 等布局参数是否有效。

代码还保留了 grouped-query attention（GQA）场景：query 和 key 的 head 数可以不同，但 batch、序列和旋转维度仍要分别满足要求。布局转换后调用 MUSA 的 `torch.rope`，再恢复原布局；任意检查或调用失败都回到 eager 的 `cos/sin` 旋转公式。

### 5.6 SwiGLU 的保护

SwiGLU 是 MoE 专家常见的门控前馈结构。可以把输入拆成两路：

```text
gate, up = gate_up.chunk(2, dim=-1)
output = SiLU(gate) * up
```

在满足以下条件时才使用 `F.swish_glu`：

- 开关打开。
- hidden activation 是 `silu` 或 `swish`。
- 输入在 MUSA 上。
- 后端提供 `swish_glu`。
- 当前模型是目标 Qwen3-VL-MoE 路径，且 Transformers 版本和 expert implementation 符合预期。

如果 fused 调用失败，后续该进程直接使用上面的 eager 公式。patch 也不会覆盖 Transformers 已经提供的 grouped matrix multiplication 等其他专家实现。

### 5.7 这个 commit 的收益、代价和边界

收益是：后端 kernel 在当前输入上不可用时，训练可以尝试安全回退；可用时减少算子拆分、显存读写或启动开销。

代价是：

- 第一次失败后，该进程会永久关闭对应 fast path，除非重新启动进程。
- eager 和 fused 仍需做数值一致性验证，不能只看“没有崩溃”。
- 过宽松的形状检查可能让后端收到不支持的输入，过严格则可能错过优化机会。

### 5.8 回退方式

```bash
export OPENSEARCH_MUSA_FUSED_RMSNORM=0
export OPENSEARCH_MUSA_FUSED_ROPE=0
export OPENSEARCH_MUSA_FUSED_SWIGLU=0
```

其中 `OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND` 是另外的 RoPE/BMM 兼容性开关，不应和 fused RoPE 开关混为一谈。

## 6. Commit 3：sparse supervised-token lm-head 和 loss

### 6.1 它解决了什么问题

对应 commit：`d33f0c040af455ac91d7abbf3eaeff08551d79d3`。

主要实现见 [`musa_sparse_lm_head.py`](../src/llamafactory/model/model_utils/musa_sparse_lm_head.py)，训练接入见 [`trainer.py`](../src/llamafactory/train/sft/trainer.py)，patch 入口见 [`patcher.py`](../src/llamafactory/model/patcher.py)。

语言模型最后要对每个位置预测一个词表分数。假设：

```text
hidden state: [batch, sequence, hidden_size]
词表大小：   vocab_size
```

普通 dense lm-head 会做：

```text
[batch, sequence, hidden_size]
        × [hidden_size, vocab_size]
        → [batch, sequence, vocab_size]
```

如果序列中有很多位置只是输入上下文、图片占位或 padding，它们的 label 会是 `-100`，最终不会参与 loss。但 dense 路径仍然先为这些位置生成了完整的词表 logits。词表很大时，这个投影会消耗很多计算、显存带宽和临时显存。

### 6.2 为什么可以只计算部分位置

SFT（监督微调）真正用于训练的目标 token 通常只有一部分。新的路径先找出 label 不等于 `-100` 的位置，只保留这些位置对应的 hidden state：

```text
原 hidden：      [batch, sequence, hidden_size]
有效位置筛选后： [valid_tokens, hidden_size]

原 lm-head 输出： [batch, sequence, vocab_size]
稀疏 lm-head：    [valid_tokens, vocab_size]
```

这里的 `valid_tokens` 是真正有监督目标的 token 数。如果有效 token 数远小于整个序列长度，最后的矩阵乘法就会明显变小。

### 6.3 因果语言模型的 shift

语言模型通常用当前位置预测下一个 token：

```text
输入：  我 喜 欢 MUSA
目标：     喜 欢 MUSA <结束>
```

因此代码会对齐：

- hidden state 去掉最后一个位置：`hidden_states[..., :-1, :]`。
- labels 去掉第一个位置：`labels[..., 1:]`。

之后才根据 `label != -100` 选择有效位置。这个顺序很关键；如果 shift 错一位，loss 可能仍然是一个正常数字，但模型学到的是错误的对应关系。

### 6.4 loss 的计算

筛选后的 logits 是二维张量 `[valid_tokens, vocab_size]`，筛选后的 labels 是 `[valid_tokens]`。新的 `sparse_causal_lm_loss`：

1. 把 logits 转成 float 计算交叉熵，降低低精度下 loss 计算不稳定的风险。
2. 使用 `ignore_index=-100` 的语义。
3. 没有 `num_items_in_batch` 时使用原有 reduction 逻辑。
4. 有 `num_items_in_batch` 时先求有效项 loss 的总和，再按该数量归一化，和 Transformers 的 `ForCausalLMLoss` 约定保持一致。

正确性上要关注的不只是 loss 数值，还要比较 dense 和 sparse 两条路径对模型参数产生的梯度。因为训练真正依赖的是梯度，而不是只在日志里显示的 loss。

### 6.5 为什么不能所有场景都使用稀疏输出

训练 loss 场景只需要有效位置的 logits；但下面场景可能需要完整输出：

- `return_outputs=True` 的训练流程。
- evaluation，需要按完整序列计算指标。
- generation，需要最后一个位置的完整词表分数来选下一个 token。
- 用户传入自定义 `compute_loss_func`。
- 使用 `label_smoother` 等会读取完整 logits 的逻辑。
- 某些外部代码直接检查模型输出形状。

所以实现没有永久改变模型的 `lm_head`，而是采用“只在安全上下文中启用”的设计：

```text
trainer 确认：纯训练 loss、没有自定义 loss、没有 label smoother
        ↓
进入 sparse 上下文
        ↓
model.forward 临时记录 labels
        ↓
lm-head 只投影有效 hidden state
        ↓
loss wrapper 使用筛选后的 labels
        ↓
finally 恢复临时状态
```

如果上下文条件不满足，或者输入形状不符合预期，代码调用原始 dense 路径。这个设计把性能优化限制在已知安全边界内，也避免 eval/generation 被误伤。

### 6.6 空有效 token 的处理

如果一整个 batch 的目标 label 都是 `-100`，就没有可以投影的 token。选择函数返回 `None`，调用方回到 dense 或原有逻辑，而不是在 MUSA 上为了判断一个布尔值强制做不必要的 host/device 同步。

这类边界情况必须测试，因为多模态样本、padding 或特殊 batch 组合可能出现没有有效监督 token 的输入。

### 6.7 patch 的限制条件和可维护性

patch 只针对：

- `model_type == qwen3_vl_moe`。
- 存在可调用的 `lm_head`。
- 原始 loss 函数名称是预期的 `ForCausalLMLoss`。
- sparse 开关打开。

同时设置幂等标记，避免初始化流程重复 patch。训练上下文使用 `ContextVar`，避免把 labels 永久写入模型对象，并在 `finally` 中恢复旧状态。

### 6.8 这个优化什么时候最有价值

粗略地说，收益和下面两个因素相关：

```text
可减少的投影工作量 ≈ 被忽略的位置数量 × 词表投影成本
```

因此，监督 token 占比越低、词表越大、lm-head 越成为瓶颈，潜在收益越大。如果几乎每个位置都有监督 label，sparse 选择和 gather 本身也有成本，收益可能很小，甚至需要实测才能判断。

### 6.9 代码和测试重点

测试见 [`test_musa_sparse_lm_head.py`](../tests/model/model_utils/test_musa_sparse_lm_head.py)，重点包括：

- dense 和 sparse loss 数值及梯度一致。
- 所有 label 都为 `-100` 时返回空选择。
- `num_items_in_batch` 为整数时的归一化。
- 不在纯 loss 上下文时保留 dense 输出。
- 自定义 loss 时跳过 sparse patch。

实际验收还应补充：训练与 eval/generation 的输出形状、长序列、多模态占位 token、不同 batch 的有效 token 比例和端到端 loss 曲线。

### 6.10 回退方式

```bash
export OPENSEARCH_MUSA_SPARSE_LM_HEAD=0
```

关闭后恢复 dense lm-head/loss 路径，不需要修改模型代码。

## 7. Commit 4：在 30B trace profile 中默认打开 sparse lm-head

对应 commit：`d9ee7e05ad6325d65fe56edc7815b7e2876ca6c6`，只修改 [`train_30b_trace.sh`](../train_30b_trace.sh)。

Commit 3 已经实现了 sparse lm-head，但启动脚本默认值还是 0。这个 commit 将：

```bash
export OPENSEARCH_MUSA_SPARSE_LM_HEAD="${OPENSEARCH_MUSA_SPARSE_LM_HEAD:-1}"
```

并注明该 profile 已在 Kubernetes 测试负载 `jd-qwen-vl-30b-a3b-test4` 中验证，仍可设置为 0 回退。这里的“test4”是该 K8s 负载名称的一部分，不是泛指某个编号为 test4 的测试步骤。

这不是新的算法实现，而是“把已经存在的实验开关纳入 30B trace 默认配置”。`${VAR:-1}` 的含义是：

- 外部没有设置变量时，使用 1。
- 外部明确设置为 0 时，保留 0，不会被脚本覆盖。

因此配置升级和实现 commit 分开是有意义的：代码可以先合入并保持关闭，经过目标场景验证后，再单独提升默认开关。

审查时要区分两件事：

1. “Kubernetes 测试负载 `jd-qwen-vl-30b-a3b-test4` 能运行且结果正确”是开启默认值的最低依据。
2. “sparse lm-head 在完整训练中带来多少吞吐收益”仍需同配置 dense/sparse A/B 测量。

## 8. Commit 5：将 MCCL channel 数固定为 8

对应 commit：`874b8df421d21bf0d6a596d67b46629de5776310`，只修改 [`train_30b_trace.sh`](../train_30b_trace.sh)。

### 8.1 MCCL 和 channel 是什么

多张 MUSA 卡训练时，一张卡产生的数据经常需要和其他卡交换，例如梯度同步。MCCL 是负责这类集合通信的 MUSA 通信库。

可以把通信 channel 简化理解成并行搬运数据的工作通道：

```text
待通信数据
   ├─ channel 1
   ├─ channel 2
   ├─ ...
   └─ channel N
```

channel 太少，可能无法充分利用链路；channel 太多，会增加调度、资源竞争和管理开销，还可能和计算 kernel 争用资源。最优值取决于卡数、拓扑、消息大小、MCCL 版本和训练并发情况，不能只从配置文件推导出来。

### 8.2 具体改动

旧配置只设置：

```bash
MCCL_MAX_NCHANNELS=14
```

新的 30B trace profile 设置：

```bash
export MCCL_MIN_NCHANNELS="${MCCL_MIN_NCHANNELS:-8}"
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-8}"
```

当外部没有覆盖变量时，min 和 max 都是 8，MCCL 会被要求使用 8 个 channel，而不是在上限 14 内自行选择。已有的 `MCCL_PROTOS=2`、`MCCL_ALGOS=1` 和 20 MiB `MCCL_BUFFSIZE` 没有在此 commit 中改变。

### 8.3 为什么固定值可能帮助 trace

trace profile 的目标是让同一套实验更容易比较。自动选择可能因环境、消息大小或版本差异选择不同 channel 数；固定为 8 可以减少一个变量，使不同运行之间更可比。

但固定值也会降低适应性。如果实际拓扑或消息大小不适合 8，通信可能变慢。因此该 commit 应被理解为“针对 30B trace 的候选配置”，不是对所有任务的通用最优设置。

### 8.4 如何验证是否真的生效

不能只看启动脚本里有 `export`。建议同时检查：

1. 启动日志打印出的环境变量。
2. MCCL 初始化日志中的实际 channel 数。
3. 所有 rank 是否使用相同配置。
4. warmup 后的通信时间、通信暴露时间和总 step time。

至少做固定 channel 8 与对照配置的 A/B：数据、batch、sequence length、卡数、通信拓扑、MCCL 版本和 warmup 必须相同。报告 `step time`、`tokens/s/GPU`、显存峰值、loss/梯度稳定性，并重复多次；单次最短 step 不能证明收益。

### 8.5 回退方式

当前脚本会在未设置时把 min 默认成 8，所以只设置 max=14 并不能完全恢复“旧的自动下限行为”。如果需要一个明确的固定 14 对照，可以在启动前设置：

```bash
export MCCL_MIN_NCHANNELS=14
export MCCL_MAX_NCHANNELS=14
```

如果要复现旧脚本的自动选择语义，需要另行调整启动脚本，使 `MCCL_MIN_NCHANNELS` 不被默认写成 8；不要把“未设置环境变量”误认为已经回到旧行为。

## 9. 最终 profile 中各开关的含义

以 PR HEAD 的 `SFT/train_30b_trace.sh` 为准，关键默认值如下：

| 开关 | 默认值 | 作用 | 关闭后的主要变化 |
| --- | ---: | --- | --- |
| `OPENSEARCH_USE_MUSA_FUSED_ADAMW` | `1` | 在安全条件下尝试 FusedAdamW | 使用原有 optimizer 实现 |
| `OPENSEARCH_MUSA_FUSED_RMSNORM` | `1` | 尝试 fused RMSNorm | eager RMSNorm |
| `OPENSEARCH_MUSA_FUSED_ROPE` | `1` | 尝试 fused RoPE | eager RoPE |
| `OPENSEARCH_MUSA_FUSED_SWIGLU` | `1` | 尝试 fused SwiGLU | eager SwiGLU |
| `OPENSEARCH_MUSA_SPARSE_LM_HEAD` | `1` | loss-only 训练场景只投影有效 token | dense lm-head/loss |
| `MCCL_MIN_NCHANNELS` | `8` | 通信 channel 下限 | 取外部指定值 |
| `MCCL_MAX_NCHANNELS` | `8` | 通信 channel 上限 | 取外部指定值 |

这些开关可以组合成隔离实验。例如：

```bash
# 只关闭 sparse lm-head，观察其独立影响
OPENSEARCH_MUSA_SPARSE_LM_HEAD=0 bash SFT/train_30b_trace.sh

# 保留其他优化，只比较 MCCL channel
MCCL_MIN_NCHANNELS=14 MCCL_MAX_NCHANNELS=14 bash SFT/train_30b_trace.sh
```

实际命令还需要沿用项目原有的环境、节点和启动参数；上面的例子只展示开关的覆盖方式。

## 10. 推荐的验收顺序

为了让初学者也能定位问题，建议按从局部到整体的顺序验收：

### 10.1 先做正确性对照

使用相同输入和随机种子，分别打开和关闭单个优化：

- dense 与 sparse lm-head：比较 loss、参数梯度和输出形状。
- eager 与 fused RMSNorm/RoPE/SwiGLU：比较输出、反向梯度和 NaN/Inf。
- FusedAdamW 与原 AdamW：确认只比较同一 optimizer 语义，不要拿它和 GaLore 等不同算法直接比较。
- channel 8 与对照 channel：确认 loss、梯度和训练稳定性不变。

允许低精度存在合理的微小误差，但要设定明确的误差阈值，并记录 dtype、设备、软件版本和输入形状。

### 10.2 再做边界和回退测试

至少覆盖：

- CPU 或非 MUSA 输入是否仍走 eager。
- fused API 不存在或参数不支持时，是否只 warning 一次并继续训练。
- GQA 的 query/key head 数不同时，RoPE 是否正确。
- 所有 label 都是 `-100` 时，sparse 路径是否安全。
- `return_outputs`、eval、generation、自定义 loss、label smoother 是否保持 dense 行为。
- 多 rank 是否都打印了实际 fast path 和 MCCL 配置。

### 10.3 最后测性能

性能测试至少应记录：

| 指标 | 说明 |
| --- | --- |
| warmup 后 step time 的中位数/P95 | 避免只看首次或偶然最短 step |
| tokens/s/GPU | 比只看 step time 更能跨序列长度比较 |
| forward、backward、optimizer、通信时间 | 判断瓶颈是否真的位于被优化的位置 |
| peak memory | sparse/fused 是否减少显存，以及是否引入临时峰值 |
| loss、梯度范数、NaN/Inf | 证明性能没有以训练稳定性为代价 |
| MCCL 实际 channel 和通信暴露 | 证明配置真的生效 |

每次只改变一个变量，并保留原始日志和环境变量。否则即使 step time 变化，也无法知道是 sparse lm-head、fused kernel、MCCL channel 还是其他环境差异造成的。

## 11. 这 5 个 commit 没有做什么

为避免在 review 或后续 PR 描述中产生误解，当前 5 个 commit 不等于：

- 已证明所有 batch、所有 MUSA 版本上的 fused kernel 都比 eager 快。
- 已证明 MCCL channel=8 对所有拓扑都是最优。
- 已改变 ZeRO/FSDP、通信重叠或并行策略。
- 已改变 Qwen3-VL 的模型结构或 MoE 路由算法。
- 已完成完整的生产稳定性、长时间训练和多节点性能验收。

它们提供的是一组有边界、有回退开关的优化路径，并把 30B trace profile 的两个实验选择（sparse lm-head、MCCL channel=8）显式化。最终是否保留默认开启，应以目标 MUSA 环境上的正确性、稳定性和可重复性能数据为准。

## 12. Review checklist

- [ ] 文档中的 commit hash 与 PR 当前历史一致。
- [ ] FusedAdamW 不会覆盖显式 optimizer 或 GaLore/APOLLO/LoRA+ 等替代功能。
- [ ] RMSNorm、RoPE、SwiGLU 的 fast path 均有条件检查、一次性 warning 和 eager fallback。
- [ ] “patch 已安装”和“第一次真实 fused 调用成功”日志含义没有混淆。
- [ ] sparse lm-head 只在纯训练 loss 场景启用，dense 输出场景没有被改变。
- [ ] causal shift、`-100` mask、空有效 token 和 `num_items_in_batch` 语义正确。
- [ ] sparse 与 dense 的 loss、梯度、eval/generation、custom loss 行为有对照证据。
- [ ] MCCL 实际 channel 配置在所有 rank 生效，并有固定 8 与对照组的实测数据。
- [ ] 所有性能结论都带有硬件、软件、数据、warmup、重复次数和指标定义。
- [ ] 需要回退时使用环境变量，不把工作树中的辅助性验证脚本误当作 PR 生产代码。
