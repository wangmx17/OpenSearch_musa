# Qwen3-VL-30B-A3B MUSA ZeRO-3 通信重叠与参数驻留优化说明

本文说明本次 PR 对 [`ds_z3_config_change.json`](../examples/deepspeed/ds_z3_config_change.json) 的 ZeRO-3 配置调整。内容面向第一次接触大模型训练、DeepSpeed ZeRO-3 和分布式通信的读者。

## 1. 本次改动的结论先行

这次改动的核心目标是：在不增加单卡 micro-batch、也不改变模型计算结果的前提下，用更多参数驻留空间换取更少的参数搬运，并让反向传播中的梯度通信尽量和计算同时进行。

可以把它概括成两件事：

1. **少搬几次。** 增大参数驻留阈值和复用窗口，尽量避免刚取回的参数过早释放、随后又重新 AllGather。
2. **搬运时继续计算。** 将 `overlap_comm` 从 `false` 改为 `true`，尝试把 ReduceScatter 梯度通信隐藏在反向计算后面。

test5 的 `exp_59` Trace 已经观察到真实的通信/计算重叠，证明机制确实生效；但它和关闭 overlap 的实验不是严格性能配对，因此本文不宣称一个确定的加速百分比。正式生产结论仍需用 micro-batch=1、GA=8、关闭 profiler 的 A/B/A 实验确认。

## 2. 初学者需要先理解的概念

### 2.1 为什么 30B 模型需要 ZeRO-3

训练模型时，显存里不只有模型参数，还要保存梯度、优化器状态、激活值和通信临时缓冲区。30B 级模型即使使用 BF16，完整训练状态也很难放进一张卡。

ZeRO 会把训练状态分散到多张卡：

| ZeRO 阶段 | 分片的主要内容 |
| --- | --- |
| Stage 1 | 优化器状态 |
| Stage 2 | 优化器状态和梯度 |
| Stage 3 | 优化器状态、梯度和模型参数 |

本项目使用 Stage 3。假设有 32 个 rank，可以把它简单理解为：平时每个 rank 只保留参数的一部分；某一层真正要计算时，再临时把这层需要的完整参数取回来。

### 2.2 AllGather 和 ReduceScatter 是什么

ZeRO-3 训练中有两类非常重要的集合通信：

```text
前向/反向需要某层参数：
32 个 rank 各持有一部分参数
          ↓ AllGather
每个 rank 暂时拿到该层计算所需的完整参数

反向产生完整梯度：
每个 rank 算出本地梯度
          ↓ ReduceScatter
先汇总梯度，再把结果分片给不同 rank 保存
```

- **AllGather（AG）**：把散落在不同 rank 的参数片段收集起来。
- **ReduceScatter（RS）**：先聚合各 rank 的梯度，再把聚合结果分片。

通信本身不改变数学公式，但会占用设备、链路和时间。如果计算必须停下来等待通信，一次 step 就会被拉长。

### 2.3 “通信重叠”为什么可能更快

关闭重叠时，流程容易近似串行：

```text
反向计算 A ──────┐
                 └→ 等待 ReduceScatter A ─→ 反向计算 B
```

打开重叠后，已经计算完成的梯度可以在通信 stream 上开始归约，同时主计算 stream 继续计算别的层：

```text
计算 stream：反向计算 A ─→ 反向计算 B ─→ 反向计算 C
通信 stream：             ReduceScatter A ─→ ReduceScatter B
```

如果两条时间线真正重合，通信中的一部分就不会单独增加 wall time。这里的关键不是配置文件里出现了 `true`，而是 Trace 中必须看到计算 kernel 和通信 kernel 在不同 stream 上真实并发，并确认等待点没有只是被挪到 step 尾部。

### 2.4 “用显存换性能”具体指什么

参数被 AllGather 后，可以立刻释放，也可以保留一段时间。保留时间更长的好处是下次复用时可能不用再次通信；代价是更多参数同时驻留在每张卡上。

这与缓存很像：

```text
小缓存：占空间少，但经常重新取数据
大缓存：占空间多，但命中时可以少搬数据
```

本次的 persistence、max-live 和 reuse-distance 调整都属于这类取舍。它们是“上限或策略阈值”，不是启动时立刻预分配同样大小的一块显存。

## 3. 暂存区中的配置差异

| 配置项 | 修改前 | 修改后 | 主要作用 |
| --- | ---: | ---: | --- |
| `overlap_comm` | `false` | `true` | 尝试让梯度归约和反向计算重叠 |
| `allgather_bucket_size` | `1e8` | `5e8` | 提高单次 AllGather 的元素上限 |
| `stage3_prefetch_bucket_size` | `auto` | `3,774,873` | 固定已验证的提前取参窗口 |
| `stage3_param_persistence_threshold` | `auto` | `21,233,664` | 让更多较小参数保持不分片驻留 |
| `stage3_max_live_parameters` | `1e8` | `1e9` | 提高单卡可同时驻留参数元素的上限 |
| `stage3_max_reuse_distance` | `1e8` | `1e9` | 参数很快会复用时，允许更长时间不释放 |

这些大小的单位是 **parameter elements（参数元素个数）**，不是字节。实际显存还与 dtype、参数分片状态、通信 buffer、allocator、激活值和优化器状态有关，不能简单地把一个数乘以 2 就当成整次训练的显存峰值。

配置中 `3,774,873` 和 `21,233,664` 以数字字符串保存，是为了把实验中已经解析出的整数显式冻结下来。exp54、exp55 和 exp59 的运行时 resolved 配置已经确认了这些实际整数值。

## 4. 每个优化点的原理和边界

### 4.1 `overlap_comm: false -> true`

#### 它控制什么

DeepSpeed 对该字段的定义是：尝试让梯度 reduction 与 backward computation 重叠。在 ZeRO-3 中，本项目关注的主要是 ReduceScatter 与反向计算的并发。

#### 为什么可能更快

关闭 overlap 的 `exp_55` Trace 中，ReduceScatter 每步出现 492 次，并且几乎没有和非通信 kernel 重叠；三个 Trace step 中，RS 暴露时间占 wall 的约 `24.83% / 40.52% / 35.19%`。这说明“等待梯度通信”当时位于关键路径上，存在明确的优化空间。

打开 overlap 后，`exp_59` 的 RS 有 `85.4%–92.0%` 与计算重叠。也就是说，大部分 RS 已经被后续计算覆盖，而不是让计算 stream 完全停下来等待。

#### 为什么不能直接说加速了多少

`exp_55` 和 `exp_59` 的 `max_steps`、学习率调度、gather-on-save 和 profiler 条件不完全一致，不是只切换 `overlap_comm` 的严格配对实验。因此：

- 可以说 Trace 证明了重叠机制已生效；
- 不可以把两次任务的 step 差值全部归因于该开关；
- 不可以据此直接给出 GA=8 正式训练的加速百分比。

#### 风险

- 通信需要额外的 in-flight buffer，可能提高显存峰值。
- stream、event 和 bucket 生命周期处理不正确可能导致数据被过早复用。
- MUSA/MCCL 或 DeepSpeed 版本变化后，可能出现 hang、watchdog timeout 或等待点迁移。
- “主机等待时间很长”不等于这些时间都能被删除；exp59 后续审计发现相关 event wait 主要承担有界反压，不能直接去掉。

### 4.2 `stage3_param_persistence_threshold: auto -> 21,233,664`

#### 它控制什么

小于该阈值的参数 tensor 不再被 ZeRO-3 分片。修改前的 `auto` 在当前模型中解析为 `20,480`，修改后显式提高到 `21,233,664`。

#### 为什么可能更快

模型中有很多 bias、norm、路由或其他相对较小的参数。单个参数不大，但如果每次使用都要发起一次很小的 AllGather，固定通信延迟和 Python/调度开销可能累积得很明显。

提高 threshold 后，更多这类参数会直接驻留在每个 rank：

```text
旧路径：释放小参数 → 下次使用前再次 AllGather
新路径：小参数继续驻留 → 下次直接使用
```

#### 代价与证据边界

这些参数从“分片保存”变成“每个 rank 都保留”，所以每卡显存会增加。前期静态 manifest 估算显示，如果把当前阈值降到 `1,048,576` 或 `262,144`，每 rank 约可少驻留 `2.42 GiB` 或 `2.61 GiB`；这是容量估算，不等于 exp59 已经采集到的运行时峰值差。

exp55 Trace 中 AllGather 每步约 979 次，但 steps 4/5 的近似暴露时间只有 `2.392s / 2.701s`，约 `87%–89%` 已被计算隐藏；`fetch_sub_module` 每步 2459 次，inclusive 时间约 `247–248ms`。因此当前高 persistence 作为冻结基线保留，但不能继续断言“阈值越大越快”，也不再盲目提高到 `1e9`。

### 4.3 `stage3_max_live_parameters: 1e8 -> 1e9`

#### 它控制什么

这是每张卡在释放参数前允许同时保持 live 状态的参数元素上限。上限较小时，DeepSpeed 为控制显存会更早释放参数；上限较大时，可以容纳更多已经取回或即将使用的参数。

#### 为什么可能更快

更大的 live window 能减少“预取了参数，却因为上限过小又被迫释放”的情况，也为 prefetch 和 reuse 策略留出空间。

#### 风险

`1e9` 是允许上限，不是固定分配量，但它放宽了 DeepSpeed 的显存约束。真实峰值必须采集全部 32 个 rank；只看 rank0 或只看 allocator allocated 都不足以证明安全。

### 4.4 `stage3_max_reuse_distance: 1e8 -> 1e9`

#### 它控制什么

DeepSpeed 会估算一个参数距离下次使用还有多远。如果复用距离在阈值内，就尽量暂不释放。提高阈值意味着系统愿意为了未来复用而把参数保留更久。

#### 为什么可能更快

Qwen3-VL MoE 和多模态结构中可能存在跨模块复用或密集的参数访问。如果一个参数刚被释放不久又要使用，就会产生重复 AllGather。更大的 reuse distance 试图避免这种“刚扔掉又捡回来”的通信。

#### 与 max-live 的关系

reuse-distance 表示“希望保留多久”，max-live 表示“最多能同时保留多少”。只提高 reuse-distance 而不给足 live 上限，保留策略可能仍被容量上限打断；所以当前冻结配置同时把两者设为 `1e9`。

这两个字段组合后更难单独归因。当前提交保留的是用户已接受并经过短训/Trace 运行的组合基线，不声称已经完成每个字段的独立 A/B。

### 4.5 `stage3_prefetch_bucket_size: auto -> 3,774,873`

#### 它控制什么

这是 ZeRO-3 在真正使用参数前，最多提前获取多少参数元素。预取的目标是把下一层参数的 AllGather 放进当前层计算时间里。

#### 本次为什么写成显式值

在当前模型上，原 `auto` 已经解析为 `3,774,873`。本次把解析结果显式写入配置，主要作用是冻结实验基线，避免 hidden size、Transformers/DeepSpeed 自动规则或配置解析变化后，实际窗口悄悄改变。

因此，这一行本身不是“把 prefetch 变大”的性能优化，而是可复现性改进。

#### 已知负结果

前期 test5 配对筛查曾只把 prefetch 从 `3,774,873` 加倍到 `7,549,746`。A0 与 P2 连续 5 个稳态点同速，收益为 0%，32 卡物理峰值反而增加约 100 MiB；该候选已判定 NO-GO。因此当前提交明确保留原 resolved 值，不继续扩大预取窗口。

### 4.6 `allgather_bucket_size: 1e8 -> 5e8`

#### 它控制什么

DeepSpeed 通用 ZeRO 配置把它定义为单次 AllGather 的参数元素上限。更大的 bucket 可能减少小批次通信和调度开销，但也可能增大临时 buffer、单次通信延迟和显存峰值。

#### 当前版本中的谨慎解释

现有 DeepSpeed 0.19.3 源码审计记录显示，当前 ZeRO-3 optimizer 的前向参数预取主路径主要由 prefetch、persistence、max-live 和 reuse-distance 控制，没有足够证据证明 `allgather_bucket_size` 会独立缩短本训练路径的 step time。把它从 `1e8` 提到 `5e8` 更接近 DeepSpeed 默认值，也可能影响参数聚合或保存相关路径，但不能仅凭 JSON 字段推断收益。

因此该字段必须通过运行时 resolved 配置、Trace 中真实消息大小，以及 save/checkpoint 显存峰值共同验收，不能把“5 倍 bucket 上限”写成“5 倍通信性能”。

## 5. test5 的现有验证证据

### 5.1 实验环境

- Kubernetes 负载：`his-test/deployment/jd-qwen-vl-30b-a3b-test5`
- 拓扑：4 Pods × 8 GPU，world size 32
- 模型：Qwen3-VL-30B-A3B
- checkpoint：C100，text 48/48、vision 27/27
- micro-batch：1
- Trace 实验：`exp_59`
- GA：1，仅用于机制和短训筛查
- global batch：32

### 5.2 exp59 Trace 结果

| 指标 | 三个 Trace step | 可以说明什么 |
| --- | ---: | --- |
| step wall | `42.317 / 45.755 / 39.591s`，均值 `42.55s` | GA1 + profiler 下的时间量级 |
| GPU active | 均值 `98.16%` | 不是整体 GPU 饥饿 |
| communication union | 均值 `35.77s` | 通信仍是重要组成 |
| compute union | 均值 `37.79s` | 计算与通信都很重 |
| comm/compute overlap | 均值 `31.79s` | 通信和计算真实并发 |
| exposed communication | 均值 `3.98s`，占 wall `9.45%` | 仍有未隐藏通信，但远小于通信累计时间 |
| RS overlap | `85.4%–92.0%` | `overlap_comm=true` 的主要机制已生效 |
| RS 尾部 drain | 约 `105ms/step` | 不支持继续盲目增大 reduce bucket |

三个 step 的通信暴露情况：

| Trace step | AG exposed | RS exposed | 全通信 exposed |
| --- | ---: | ---: | ---: |
| `ProfilerStep#2` | `4.053s` | `1.212s` | `5.266s / 12.44%` |
| `ProfilerStep#3` | `0.876s` | `1.505s` | `2.383s / 5.21%` |
| `ProfilerStep#4` | `1.346s` | `2.033s` | `3.380s / 8.54%` |

注意：kernel 累计时间可以大于 wall time，因为不同 stream 能同时运行；inclusive CPU 等待也可能覆盖 GPU 工作。性能分析必须看时间区间的 union 和 overlap，不能把各项累计时间直接相加。

### 5.3 正确性和稳定性短训

exp59 共记录 18 步 loss/gradient norm，全部 finite；四个 Pod 均未出现 OOM、MCCL/ProcessGroup timeout、watchdog 或 restart。任务结束后，精确匹配 `logs/exp_59/runtime_trace.yaml` 的进程为 0。

这证明当前组合至少通过了短训健康检查，但不等于通过长时间生产稳定性测试。特别是 exp59 没有完整的 32-rank allocator/物理显存工件，不能据此宣布显存门已通过。

### 5.4 当前证据不能证明什么

- 不能证明 GA=8、global batch=256 下有同样的重叠比例和 step 收益。
- 不能证明每个 bucket/persistence/live/reuse 字段各自贡献了多少性能。
- 不能证明所有 32 个 rank 的长尾峰值都满足生产显存门。
- 不能证明 checkpoint save/load/resume 路径不会因更大聚合或驻留窗口出现峰值问题。
- 不能把 profiler 下三个 step 的均值当作正式训练吞吐基线。

## 6. 为什么这次不修改 micro-batch 和训练语义

`per_device_train_batch_size=1` 是当前正式训练硬约束。改变 micro-batch 会同时改变 global batch、梯度累积工作量、样本吞吐口径、学习率调度和可能的收敛行为，不能算作同一训练配方下的底层优化。

GA1 的 optimizer step 只包含 1 个 micro-batch，而正式 GA8 的 optimizer step 包含 8 个。GA1 step 看起来接近 GA8 的八分之一，主要是工作量不同，不代表每个样本更快。因此本 PR 只调整 ZeRO-3 调度配置，不改训练 batch 语义。

## 7. 正式验收方案

正式验证应恢复：

```text
micro-batch = 1
GA = 8
world size = 32
global batch = 256
profiler = off
```

建议至少执行 A/B/A，并在资源允许时补一轮候选：

```text
A0：修改前 ZeRO-3 配置
B0：本 PR 配置
A1：再次运行修改前配置
B1：再次运行本 PR 配置
```

每次运行至少 30 个 optimizer steps，前 5 步只做 warmup，不进入统计；候选配置至少运行两次。除 ZeRO-3 配置外，模型、checkpoint policy、数据顺序、seed/data_seed、LR schedule、MCCL 参数、软件版本和节点拓扑必须一致。

最低验收指标：

| 类别 | 必须记录的内容 |
| --- | --- |
| 性能 | 稳态 step P50/P95、tokens/s/GPU、样本数和有效 token 数 |
| Trace | AG/RS 次数、真实消息大小、通信 union、计算 union、overlap、exposed communication |
| 显存 | 全 32 rank allocated/reserved/物理峰值、长尾和 checkpoint 保存峰值 |
| 数值 | loss、gradient norm、NaN/Inf、相同 seed 下的合理轨迹 |
| 稳定性 | OOM、hang、watchdog、MCCL/ProcessGroup timeout、rank 退出和 Pod restart |
| 可恢复性 | save、load、resume，以及 16-bit 权重聚合保存 |

生产 GO 至少要求：

- step P50 或归一化吞吐有可重复收益，不能只看最短 step；
- 按当前优化计划，候选的 step P50 或有效 token/GPU/s 提升应达到 `3%` 门槛；
- P95 不明显恶化；
- 所有 rank 显存留有安全余量；
- loss/gradient norm 正常，无 NaN/Inf；
- 无 hang、timeout、OOM 或 checkpoint 回归。

## 8. 回退方式

如果出现性能回退、显存峰值过高、MCCL timeout、hang 或 checkpoint 问题，可以把本文件对应配置恢复为：

```json
{
  "overlap_comm": false,
  "allgather_bucket_size": 1e8,
  "stage3_prefetch_bucket_size": "auto",
  "stage3_param_persistence_threshold": "auto",
  "stage3_max_live_parameters": 1e8,
  "stage3_max_reuse_distance": 1e8
}
```

回退时不要修改 `stage`、`contiguous_gradients`、`reduce_bucket_size` 或 batch 配置，否则会引入新的实验变量。回退后仍应检查运行时 resolved config，确认 `auto` 在当前软件版本中解析成预期数值。

## 9. Review checklist

- [ ] PR 只包含 `ds_z3_config_change.json` 和本文档，没有实验脚本、Trace、日志或 checkpoint-policy 工作树代码。
- [ ] 运行时 resolved 配置与本文表格一致，字符串整数被正确解析。
- [ ] 所有 rank 实际启用了同一份 `overlap_comm=true` 配置。
- [ ] Trace 证明 RS 与反向计算真实重叠，而不是只移动了等待点。
- [ ] 没有把 exp55/exp59 当作严格性能 A/B，也没有虚构加速百分比。
- [ ] prefetch 2× 的 NO-GO 结论得到保留，没有继续扩大窗口。
- [ ] 记录全部 32 rank 显存峰值，而不是只看 rank0。
- [ ] GA8 正式 A/B/A 固定 micro-batch=1、数据、seed、LR、MCCL 和软件版本。
- [ ] loss、gradient norm、OOM、timeout、watchdog、save/load/resume 全部通过。
- [ ] 回退只恢复本 PR 的六个 ZeRO-3 字段，不混入其他训练配方修改。

## 10. 参考资料

- [DeepSpeed ZeRO-3 官方文档](https://deepspeed.readthedocs.io/en/stable/zero3.html)：ZeRO-3、`overlap_comm`、prefetch、persistence、max-live、reuse-distance 和 bucket 字段的官方定义。
- 本 PR 的数据结论来自 `jd-qwen-vl-30b-a3b-test5` 的 exp55/exp59 运行记录和 Trace 离线分析；实验工件不作为生产代码提交。
