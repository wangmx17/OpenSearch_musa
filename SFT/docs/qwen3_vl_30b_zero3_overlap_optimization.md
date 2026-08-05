# Qwen3-VL-30B-A3B MUSA ZeRO-3 通信重叠与参数驻留优化说明

本文说明本次 PR 对 [`ds_z3_config_change.json`](../examples/deepspeed/ds_z3_config_change.json) 的 ZeRO-3 配置调整。内容面向第一次接触大模型训练、DeepSpeed ZeRO-3 和分布式通信的读者。

## 1. 本次改动的结论先行

这次改动的核心目标是：在不增加单卡 micro-batch、也不改变模型计算结果的前提下，用更多参数驻留空间换取更少的参数搬运，并让反向传播中的梯度通信尽量和计算同时进行。

可以把它概括成两件事：

1. **少搬几次。** 增大参数驻留阈值和复用窗口，尽量避免刚取回的参数过早释放、随后又重新 AllGather。
2. **搬运时继续计算。** 将 `overlap_comm` 从 `false` 改为 `true`，尝试把 ReduceScatter 梯度通信隐藏在反向计算后面。

test5 的 `exp_59` Trace 已经观察到真实的通信/计算重叠，证明 overlap 机制确实生效；它和关闭 overlap 的实验不是严格性能配对，因此不能据此给 overlap 单独归因。第一阶段的 trace-off GA1 A/B/A 和候选因果 Trace 证明 `live=1e10/reuse=2e10` 可以减少重复大参数 AllGather。随后在相同正式 GA8 配方下，`exp_71` 将驻留窗口进一步提高到 `stage3_max_live_parameters=2e10`、`stage3_max_reuse_distance=4e10`，完成了10步性能与稳定性短测。GA8 仍需继续长跑，本文不把短测外推为收敛或保存恢复结论。

## 2. 需要先理解的概念

### 2.1 为什么 30B 模型需要 ZeRO-3

训练模型时，显存里不只有模型参数，还要保存梯度、优化器状态、激活值和通信临时缓冲区。30B 级模型即使使用 BF16，完整训练状态也很难放进一张卡。

ZeRO 会把训练状态分散到多张卡：

| ZeRO 阶段 | 分片的主要内容             |
| --------- | -------------------------- |
| Stage 1   | 优化器状态                 |
| Stage 2   | 优化器状态和梯度           |
| Stage 3   | 优化器状态、梯度和模型参数 |

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

| 配置项                                 |    修改前 |         修改后 | 主要作用                                                    |
| -------------------------------------- | --------: | -------------: | ----------------------------------------------------------- |
| `overlap_comm`                       | `false` |       `true` | 尝试让梯度归约和反向计算重叠                                |
| `allgather_bucket_size`              |   `1e8` |        `5e8` | 提高单次 AllGather 的元素上限                               |
| `stage3_prefetch_bucket_size`        |  `auto` |  `3,774,873` | 固定已验证的提前取参窗口                                    |
| `stage3_param_persistence_threshold` |  `auto` | `21,233,664` | 让更多较小参数保持不分片驻留                                |
| `stage3_max_live_parameters`         |   `1e8` |       `2e10` | 把同时处于 live 状态的参数元素上限提高到200亿               |
| `stage3_max_reuse_distance`          |   `1e8` |       `4e10` | 参数会在未来约400亿参数元素的遍历距离内复用时，允许暂不释放 |

这些大小的单位是 **parameter elements（参数元素个数）**，不是字节。实际显存还与 dtype、参数分片状态、通信 buffer、allocator、激活值和优化器状态有关，不能简单地把一个数乘以 2 就当成整次训练的显存峰值。`2e10` 和 `4e10` 是 JSON 科学计数法，分别等于 `20,000,000,000` 和 `40,000,000,000`。

表中的“修改前/修改后”表示相对 PR base 的最终差异。最新 GA8 增量验证以 PR#12 的 `live=1e10/reuse=2e10` 为历史基线，候选只把这一对耦合预算改成 `live=2e10/reuse=4e10`；其余 DeepSpeed、模型、数据、seed、micro-batch 和 GA 均冻结。

配置中 `3,774,873` 和 `21,233,664` 以数字字符串保存，是为了把实验中已经解析出的整数显式冻结下来。exp54、exp55 和 exp59 的运行时 resolved 配置已经确认了这些实际整数值。

## 4. 每个优化点的原理和边界

### 4.1 `overlap_comm: false -> true`

#### 它控制什么

DeepSpeed 对该字段的定义是：尝试让梯度 reduction 与 backward computation 重叠。在 ZeRO-3 中，本项目关注的主要是 ReduceScatter 与反向计算的并发。

#### 为什么可能更快

关闭 overlap 的 `exp_55` Trace 中，ReduceScatter 每步出现 492 次，并且几乎没有和非通信 kernel 重叠；三个 Trace step 中，RS 暴露时间占 wall 的约 `24.83% / 40.52% / 35.19%`。这说明“等待梯度通信”当时位于关键路径上，存在明确的优化空间。

打开 overlap 后，`exp_59` 的 RS 有 `85.4%–92.0%` 与计算重叠。也就是说，大部分 RS 已经被后续计算覆盖，而不是让计算 stream 完全停下来等待。

#### 风险

- 通信需要额外的 in-flight buffer，可能提高显存峰值。
- stream、event 和 bucket 生命周期处理不正确可能导致数据被过早复用。

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

### 4.3 `stage3_max_live_parameters: 1e8 -> 2e10`

#### 它控制什么

这是每张卡在释放参数前允许同时保持 live 状态的参数元素上限。上限较小时，DeepSpeed 为控制显存会更早释放参数；上限较大时，可以容纳更多已经 AllGather 回来的完整参数，为后续复用保留空间。

它是调度上限，不是启动时直接申请一块 `2e10 × dtype_size` 的连续显存。最终物理峰值由实际同时驻留参数、分片状态、通信 buffer、激活、优化器状态和 allocator 共同决定。

#### 为什么选择 `2e10`

exp59 基线 Trace 每个 profiler step 都包含两类主要大参数 AllGather：

- `95 × 402,653,184` elements；
- `95 × 201,326,592` elements。

两类消息合计约对应每个 text layer `603,979,776` 个大参数元素，并在 forward 与 checkpoint recompute/backward 路径重复出现。`live=1e9` 只能容纳约一个这样的 layer 组合；`live=1e10` 和 `2e10` 的理论容量直觉分别约为16个和33个 layer 组合，可以让更多 forward 尾部参数跨过重计算边界继续驻留。

这里的层数只是用 Trace 消息尺寸计算出的容量直觉，不代表 DeepSpeed 会机械地固定保留相同数量的层。`1e10` 先由 trace-off GA1 A/B/A 和 exp64 因果 Trace 验证；继续提高到 `2e10` 后，GA8 `exp_71` 的 step2–10 Mean 从 PR#12 历史 `exp_65` 的 `287.11s` 降到 `265.33s`，P50/P95 从 `287/293s` 降到 `265/272s`。四节点观察到的物理显存最高值为 `78,585 MiB`，且用户确认该余量可接受。

#### 为什么不继续增大

本轮只验证并固化到 `2e10`。继续增大可能遇到驻留收益饱和，同时进一步压缩激活、通信 buffer、checkpoint 保存和动态 shape 的安全余量；没有新的配对性能证据时不再自动上调。

### 4.4 `stage3_max_reuse_distance: 1e8 -> 4e10`

#### 它控制什么

DeepSpeed 会沿未来的 module/parameter trace 累加参数元素，估算当前参数距离下次使用还有多远。如果复用距离在阈值内，并且 live 容量仍允许，就尽量暂不释放。提高阈值意味着系统愿意为了未来复用把参数保留更久。

#### 为什么选择 `4e10`

reuse-distance 衡量的是“从当前使用点到下一次使用点之间遍历了多少参数元素”，不是当前实际驻留量。对于 activation checkpointing，参数可能在 forward 尾部使用后，经过其他模块遍历，再在 recompute/backward 中复用；因此其复用距离可以明显大于同一时刻需要驻留的参数量。

本轮把 `reuse=4e10` 与 `live=2e10` 作为一个耦合候选：

```text
reuse=4e10：允许尾部参数跨更长的未来遍历距离继续保留
live=2e10：限制任何时刻真正允许驻留的总量，承担显存压力阀作用
```

`4e10 = 2 × 2e10` 是本模型、当前 checkpoint policy 和当前访问序列下的实验配比，不是 DeepSpeed 的通用公式。只提高 reuse-distance 而保持较小 live 上限，保留意图仍会被容量门打断；只提高 live 而 reuse-distance 太短，参数仍可能在到达复用点前被释放。因此两项继续作为一个 Stage3 residency budget 共同验证，不对单个字段拆分宣称收益。

#### 因果证据

候选 exp64 相对 exp59 的三个可配对 Trace step 中，两类目标大 AllGather 都严格从每步 `95` 次降到 `80` 次，各减少15次；`_allgather_base` 从 `977` 次降到 `946` 次。三步 wall 全部同向改善，均值提升 `6.25%`，AllGather exposed 均值下降 `23.15%`，并且目标大 AllGather 没有迁移到 optimizer。

第三个 step 的 ReduceScatter exposed 存在上升，因此该证据只说明“更大 live/reuse 窗口减少了重复参数获取并改善端到端 wall”，不外推为“所有通信等待都改善”。

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

## 5. 现有验证证据

### 5.1 实验环境

- 拓扑：4 Pods × 8 GPU，world size 32
- 模型：Qwen3-VL-30B-A3B
- checkpoint：C100，text 48/48、vision 27/27
- micro-batch：1
- overlap/驻留基线 Trace：`exp_59`，GA=1、global batch=32
- 驻留预算 trace-off A/B/A：`exp_61/exp_62/exp_63`，GA=1、global batch=32
- 驻留候选因果 Trace：`exp_64`，GA=1、global batch=32
- 正式早期健康检查：`exp_65`，GA=8、global batch=256、profiler off

### 5.2 exp59 Trace 结果

| 指标                  |                                三个 Trace step | 可以说明什么                           |
| --------------------- | ---------------------------------------------: | -------------------------------------- |
| step wall             | `42.317 / 45.755 / 39.591s`，均值 `42.55s` | GA1 + profiler 下的时间量级            |
| GPU active            |                                 均值`98.16%` | 不是整体 GPU 饥饿                      |
| communication union   |                                 均值`35.77s` | 通信仍是重要组成                       |
| compute union         |                                 均值`37.79s` | 计算与通信都很重                       |
| comm/compute overlap  |                                 均值`31.79s` | 通信和计算真实并发                     |
| exposed communication |               均值`3.98s`，占 wall `9.45%` | 仍有未隐藏通信，但远小于通信累计时间   |
| RS overlap            |                               `85.4%–92.0%` | `overlap_comm=true` 的主要机制已生效 |
| RS 尾部 drain         |                               约`105ms/step` | 不支持继续盲目增大 reduce bucket       |

三个 step 的通信暴露情况：

| Trace step         | AG exposed | RS exposed |      全通信 exposed |
| ------------------ | ---------: | ---------: | ------------------: |
| `ProfilerStep#2` | `4.053s` | `1.212s` | `5.266s / 12.44%` |
| `ProfilerStep#3` | `0.876s` | `1.505s` |  `2.383s / 5.21%` |
| `ProfilerStep#4` | `1.346s` | `2.033s` |  `3.380s / 8.54%` |

注意：kernel 累计时间可以大于 wall time，因为不同 stream 能同时运行；inclusive CPU 等待也可能覆盖 GPU 工作。性能分析必须看时间区间的 union 和 overlap，不能把各项累计时间直接相加。

### 5.3 GA1 trace-off A/B/A：端到端短筛

三臂固定 C100、world size 32、micro-batch=1、GA=1、数据、seed、checkpoint policy、`overlap_comm=true` 和其余 DeepSpeed 配置，只改变耦合的 live/reuse 驻留预算。前2步预热，统计 optimizer steps 3–10：

| Arm | EXP   | live / reuse    | steps 3–10 Mean |       P50 |         P95 | 全节点物理显存峰值 |
| --- | ----- | --------------- | ---------------: | --------: | ----------: | -----------------: |
| A0  | exp61 | `1e9 / 1e9`   |      `40.500s` | `40.0s` |   `44.6s` |     `39,413 MiB` |
| B0  | exp62 | `1e10 / 2e10` |      `37.625s` | `37.0s` |   `41.6s` |     `56,546 MiB` |
| A1  | exp63 | `1e9 / 1e9`   |      `40.375s` | `40.0s` | 约`44.0s` |     `39,413 MiB` |

- B0 相对 A0/A1 的 P50 均提升 `7.5%`，Mean 分别提升 `7.1%/6.8%`；P95 没有恶化。
- A0/A1 P50 漂移为 `0%`、Mean 漂移约 `0.3%`，基线没有明显漂移。
- B0 global worst 增加 `17,133 MiB`，仍低于 `73,728 MiB` 硬门并保留约 `17.2 GiB` 设备余量。
- 三臂 loss/gradient norm 全部 finite；未发现 OOM、hang、watchdog、MCCL/ProcessGroup timeout 或 Pod restart。

因此该耦合候选通过 `GA1 SCREEN GO`。GA1 optimizer step 的工作量不同于 GA8；这里证明的是同语义短筛收益和显存安全性，不是正式收敛结论。

### 5.4 exp64 因果 Trace：为什么会变快

exp64 使用 B0 驻留预算补抓 rank0 Trace，与 exp59 的 `ProfilerStep#2–4` 可按 shape 和 collective 序列配对：

| 指标                      |                exp59 基线 |                exp64 候选 |                          变化 |
| ------------------------- | ------------------------: | ------------------------: | ----------------------------: |
| step wall                 | `42.317/45.755/39.591s` | `39.685/43.243/36.755s` | 三步均改善，Mean提升`6.25%` |
| 402,653,184-element AG    |               `95/step` |               `80/step` |                  `-15/step` |
| 201,326,592-element AG    |               `95/step` |               `80/step` |                  `-15/step` |
| `_allgather_base`       |              `977/step` |              `946/step` |                  `-31/step` |
| AG exposed Mean           |                `2.092s` |                `1.607s` |                   `-23.15%` |
| ReduceScatter / AllReduce |        `492/2 per step` |        `492/2 per step` |            collective语义不变 |

EventSync、stream wait 和 DeepSpeed record wait 均未增加，两类目标大 AG 在 optimizer 内均为0，说明收益没有简单迁移成 optimizer 尾部等待。三个 step 的全 collective exposed 均值下降 `11.76%`，但第三个 step 的 RS exposed 从 `2.033s` 增至 `3.246s`，所以通信重叠仍有 step 级波动。

该 Trace 完整解析 `9,241,856` 个事件，文件 size 和 SHA256 在远端/本地一致。因果结论是：`live=1e10/reuse=2e10` 让部分尾部参数跨复用边界继续驻留，精确减少了两类大参数的重复 AllGather，并带来超过3%的 wall 收益。

### 5.5 exp65 正式 GA8 早期健康检查

exp65 固定 micro-batch=1、GA=8、world size 32、global batch=256、profiler off，并保留生产 checkpoint 保存语义。前5步结果：

- step2–5 增量约为 `286/295/290/281s`，Mean `288.0s`、P50 `288.0s`；
- 相对历史 exp54 step2–9 Mean `315.75s` 方向性改善约 `8.8%`；
- loss 为 `1.020/1.026/1.010/1.023/1.017`，gradient norm 为 `13.88/13.44/13.81/13.69/13.50`，全部 finite；
- 四节点物理显存峰值为 `58,869/56,793/57,809/59,568 MiB`，global worst `59,568 MiB`，距 `73,728 MiB` 门限仍有 `14,160 MiB`；
- 四个 Pod 均 Running/Ready、restart=0，未发现 OOM、MCCL/ProcessGroup timeout、watchdog 或 fatal，任务继续长跑。

exp65 与 exp54 不是同轮配对 A/B，数据阶段和运行时状态可能不同，因此 `8.8%` 只作为正式配方下的早期方向性佐证；当前可归因的性能结论仍以 GA1 A/B/A 和 exp64 因果 Trace 为准。

### 5.6 exp71 GA8 驻留窗口增量验证

exp71 保持 exp65 的 C100、world size 32、micro-batch 1、GA8、global batch 256、`overlap_comm=true` 及其余 ZeRO-3 配置，只将 `live/reuse` 从 `1e10/2e10` 提高到 `2e10/4e10`。统计 step2–10：

| EXP   | live / reuse    | Mean      | P50    | P95    |
| ----- | --------------- | --------: | -----: | -----: |
| exp65 | `1e10 / 2e10` | `287.11s` | `287s` | `293s` |
| exp71 | `2e10 / 4e10` | `265.33s` | `265s` | `272s` |

Mean step time 下降 `7.59%`，超过 `3%` 门槛；10步 loss/gradient norm 均 finite，四个 Pod 零重启，未发现 OOM、MCCL/ProcessGroup timeout、watchdog 或 fatal。四节点观察到的最高物理显存为 `78,585 MiB`，按用户确认作为本轮可接受峰值。

### 5.7 当前证据仍不能证明什么

- 不能把 live/reuse 耦合候选的收益拆分归因到其中单个字段。
- 不能把 exp65 与历史 exp54 的差值当作严格 GA8 A/B 收益。
- 不能证明 GA8 长跑、动态 shape 长尾和 checkpoint save/load/resume 已全部通过。
- 不能把 profiler 下的 kernel 累计时间直接当作 wall time，也不能声称所有通信等待均改善。

## 6. 为什么这次不修改 micro-batch 和训练语义

`per_device_train_batch_size=1` 是当前正式训练硬约束。改变 micro-batch 会同时改变 global batch、梯度累积工作量、样本吞吐口径、学习率调度和可能的收敛行为，不能算作同一训练配方下的底层优化。

GA1 的 optimizer step 只包含 1 个 micro-batch，而正式 GA8 的 optimizer step 包含 8 个。GA1 step 看起来接近 GA8 的八分之一，主要是工作量不同，不代表每个样本更快。因此本 PR 只调整 ZeRO-3 调度配置，不改训练 batch 语义。

## 7. 正式验收方案

正式验证固定为：

```text
micro-batch = 1
GA = 8
world size = 32
global batch = 256
profiler = off
```

当前 GA1 A0/B0/A1、因果 Trace 和 exp71 GA8 10-step 增量验证已完成。exp71 仍在继续运行；最终生产验收仍建议在相同节点和数据条件下执行更长的 GA8 A/B/A：

```text
A0：修改前 ZeRO-3 配置
B0：本 PR 配置
A1：再次运行修改前配置
B1：再次运行本 PR 配置
```

每次运行至少 30 个 optimizer steps，前 5 步只做 warmup，不进入统计；候选配置至少运行两次。除 ZeRO-3 配置外，模型、checkpoint policy、数据顺序、seed/data_seed、LR schedule、MCCL 参数、软件版本和节点拓扑必须一致。

最低验收指标：

| 类别     | 必须记录的内容                                                                   |
| -------- | -------------------------------------------------------------------------------- |
| 性能     | 稳态 step P50/P95、tokens/s/GPU、样本数和有效 token 数                           |
| Trace    | AG/RS 次数、真实消息大小、通信 union、计算 union、overlap、exposed communication |
| 显存     | 全 32 rank allocated/reserved/物理峰值、长尾和 checkpoint 保存峰值               |
| 数值     | loss、gradient norm、NaN/Inf、相同 seed 下的合理轨迹                             |
| 稳定性   | OOM、hang、watchdog、MCCL/ProcessGroup timeout、rank 退出和 Pod restart          |
| 可恢复性 | save、load、resume，以及 16-bit 权重聚合保存                                     |

生产 GO 至少要求：

- step P50 或归一化吞吐有可重复收益，不能只看最短 step；
- 按当前优化计划，候选的 step P50 或有效 token/GPU/s 提升应达到 `3%` 门槛；
- P95 不明显恶化；
- 所有 rank 显存留有安全余量；
- loss/gradient norm 正常，无 NaN/Inf；
- 无 hang、timeout、OOM 或 checkpoint 回归。

## 8. 回退方式

如果只需要回退本轮 `2e10/4e10` 驻留预算，保持 overlap、bucket、prefetch 和 persistence 不变，恢复到 PR#12 上一版已验证配置：

```json
{
  "stage3_max_live_parameters": 1e10,
  "stage3_max_reuse_distance": 2e10
}
```

如果需要完整回退整个 PR 的六项 ZeRO-3 调整，再恢复为 PR base：

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

回退时不要修改 `stage`、`contiguous_gradients`、`reduce_bucket_size` 或 batch 配置，否则会引入新的实验变量。优先使用两项最小回退，便于判断问题是否来自驻留窗口；完整回退后仍应检查运行时 resolved config，确认 `auto` 在当前软件版本中解析成预期数值。

## 9. Review checklist

- [ ] PR 只包含 `ds_z3_config_change.json` 和本文档，没有实验脚本、Trace、日志或 checkpoint-policy 工作树代码。
- [ ] 运行时 resolved 配置与本文表格一致，科学计数法和字符串整数均被正确解析。
- [ ] 所有 rank 实际启用了同一份 `overlap_comm=true` 配置。
- [ ] Trace 证明 RS 与反向计算真实重叠，而不是只移动了等待点。
- [ ] 没有把 exp55/exp59 或 exp54/exp65 当作严格性能 A/B；可归因收益只引用 exp61/62/63 与 exp59/64。
- [ ] exp61/62/63 的 GA1 A/B/A 满足 P50、P95、基线漂移、数值和全节点显存门。
- [ ] exp64 中两类大 AllGather 均为 `95 -> 80`，且等待没有迁移到 optimizer。
- [ ] exp71 的 `2e10/4e10` 配置完成 GA8 10-step 验证，step2–10 性能超过 `3%` 门槛且数值、显存和四节点状态正常。
- [ ] prefetch 2× 的 NO-GO 结论得到保留，没有继续扩大窗口。
- [ ] 记录全部 32 rank 显存峰值，而不是只看 rank0。
- [ ] GA8 正式 A/B/A 固定 micro-batch=1、数据、seed、LR、MCCL 和软件版本。
- [ ] GA8 长跑的 loss、gradient norm、OOM、timeout、watchdog、save/load/resume 全部通过。
- [ ] 回退优先只恢复 live/reuse；需要完整回退时才恢复本 PR 的六个 ZeRO-3 字段，不混入其他训练配方修改。

## 10. 参考资料

- [DeepSpeed ZeRO-3 官方文档](https://deepspeed.readthedocs.io/en/stable/zero3.html)：ZeRO-3、`overlap_comm`、prefetch、persistence、max-live、reuse-distance 和 bucket 字段的官方定义。
- 本 PR 的数据结论来自 `jd-qwen-vl-30b-a3b-test5` 的 exp55、exp59、exp61–exp65、exp69 和 exp71 运行记录与 Trace 离线分析；实验工件不作为生产代码提交。
