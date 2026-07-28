# MUSA 多机 MCCL 健康检查与 VMM 卡死复现

本文说明 30B 四机训练新增的启动前 MCCL 健康检查、VMM 的基本原理，以及如何在独占的空闲 GPU 上复现和区分 allocator/MCCL 卡死。

## 1. 本次增加了什么

| 文件 | 作用 |
|---|---|
| `scripts/musa_mccl_health_check.py` | 调用原生 `all_reduce_perf` 做小型 MCCL 启动前检查 |
| `launch_train_30b_trace.sh` | 在任何训练 rank 启动前同步执行健康检查，失败就不 dispatch 训练 |
| `scripts/repro_musa_vmm_mccl_hang.py` | 分 rank 记录 allocator、collective 和 ZeRO-like partition 的阶段时间线 |
| `scripts/run_musa_vmm_mccl_repro.sh` | 通过 OpenMPI 在 8/16/32 rank 上启动复现器，并提供进程级总超时 |
| `tests/train/test_musa_vmm_allocator.py` | 单卡、全新子进程中的 expandable-segments 分配回归测试 |

MCCL 检查和 VMM 复现器没有放进模型 kernel 或 `ops/mlp/`。它们属于分布式运行环境诊断，必须发生在模型加载、Transformer Engine patch 和 DeepSpeed ZeRO 初始化之前。

## 2. 下一次训练如何使用健康检查

`launch_train_30b_trace.sh` 默认启用健康检查：

```bash
bash launch_train_30b_trace.sh hostfile.txt
```

执行顺序为：

```text
读取并校验本次 hostfile
        ↓
同步执行小型 MCCL all-reduce
        ↓
检查返回码、输出数据行和数值正确性
        ↓
PASS：为实验分配 exp_id，并向各节点 dispatch 训练
FAIL：退出，不启动任何 torchrun/模型加载
```

检查日志默认位于本次 launcher 日志目录下的 `mccl_preflight/`：

```text
all_reduce_perf.log  # 原始 MCCL tests 输出
summary.json         # 节点、耗时、带宽、正确性和失败原因
```

只有临时排障时才建议显式跳过：

```bash
OPENSEARCH_MCCL_PREFLIGHT=0 bash launch_train_30b_trace.sh hostfile.txt
```

如果测试 binary 移动了，可以指定：

```bash
MCCL_PREFLIGHT_TEST_BIN=/new/path/all_reduce_perf \
  bash launch_train_30b_trace.sh hostfile.txt
```

### 2.1 为什么这个检查足够小

参考实现来自 pod 共享目录：

```text
/home/jd/liang.geng/mccl-test-master
```

健康检查直接复用已经编译并链接 MCCL/MUSA/OpenMPI 的：

```text
/home/jd/liang.geng/mccl-test-master/build/all_reduce_perf
```

原参考脚本会扫描很大的 buffer，适合带宽压测，不适合作为训练前检查。本次固定使用：

```text
4 个 MPI 进程（每节点 1 个）
每进程 8 张 MUSA 卡
总计 32 个 MCCL rank
buffer 固定 1 MiB
1 次 warmup + 2 次正式迭代
开启结果检查（-c 1）
MCCL stream timeout 60 秒
MPI timeout 90 秒
外层进程级 timeout 120 秒
```

这能覆盖四台机器、32 张卡、MCCL communicator 初始化和一次真实跨机 collective，同时避免大 buffer 长时间占用显存和网络。

它是 communicator/fabric 的快速检查，不会完全模拟训练的“每张卡一个 Python/torchrun 进程”。如果原生 preflight 通过但 ZeRO init 仍卡住，应继续使用第 4～6 节的 32-Python-process 组合复现器，而不能据此排除 VMM、进程模型或 stream 交互问题。

### 2.2 成功和失败如何判定

默认硬检查：

- `mpirun/all_reduce_perf` 必须正常退出；
- 必须产生有效性能数据行；
- 每一行的 `#wrong` 必须为 0；
- 汇总必须包含 `# Out of bounds values : 0 OK`；
- MPI/MCCL 初始化或 collective 超过总超时会失败。

1 MiB、2 次迭代的结果容易受冷启动影响，所以默认只报告带宽，不把带宽作为硬门槛。如果积累了稳定基线，可以显式设置：

```bash
MCCL_PREFLIGHT_MIN_BUSBW_GBPS=<集群实测阈值> \
  bash launch_train_30b_trace.sh hostfile.txt
```

不要直接复用大 buffer 双机 netcheck 的 `170 GB/s` 阈值；消息大小、rank 数和拓扑不同，数字不能直接比较。

### 2.3 hostfile 必须来自当前 Running Pods

不要复用旧 IP。示例：

```bash
kubectl get pod -n his-test \
  -l app=jd-qwen3-vl-30b-a3b \
  --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.status.hostIP}{" slots=8\n"}{end}' \
  | sort -V > hostfile.txt
```

健康检查会把它转换成每节点 `slots=1` 的临时 MPI hostfile，再由每个 `all_reduce_perf` 进程管理本机 8 张卡。训练 launcher 仍按原 hostfile 的每行一台机器进行 dispatch。

## 3. 什么是 VMM

VMM 是 Virtual Memory Management，即“虚拟内存管理”。这里说的是 GPU/MUSA 设备内存的虚拟地址管理，不是把显存换到硬盘，也不会凭空增加物理显存。

可以把它理解成“先规划地址，再逐步交付房间”：

1. allocator 先预留一段很大的连续虚拟地址。此时只是保留门牌号，不一定已经占用等量物理显存。
2. Tensor 真正需要更多空间时，再申请若干物理显存块。
3. 驱动把这些物理块映射到之前预留的虚拟地址，对应 `map`。
4. 驱动为当前设备或允许的本机 peer 设置访问权限，对应 `setAccess`。
5. 一部分空间不再需要时，可以解除映射；虚拟地址范围仍可留给以后继续扩展。

`PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True` 利用的就是这种能力。普通 allocator 在尺寸不断变化时容易留下许多无法拼成大块的碎片；expandable segment 可以保留一段连续地址并按需增加物理 backing，从而减少“空闲显存总量够，但没有合适连续块”的情况。

它有三个容易混淆的点：

- VMM 管理的是每张 GPU 的本地虚拟地址和物理显存映射，不会因为四机训练就把远端 GPU 映射成本地显存。
- `expandable_segments` 解决的是 allocator 的扩展和碎片问题，不是 MCCL 通信算法。
- allocator 调用发生在 device/stream 已有工作的上下文中，因此驱动、设备同步或 MCCL 的慢状态可能让 VMM API 变慢；反过来，一个 rank 的 VMM 映射变慢也会让其他 rank 等在 collective 上。

此前 native stack 停在：

```text
muMemSetAccess
→ MUSACachingAllocator::ExpandableSegment::mapAndSetAccess
→ DeepSpeed ZeRO _partition_param
```

它能证明“采样时这个 rank 卡在设置 VMM 映射访问权限”，但不能仅凭这一帧证明最早的触发一定来自 allocator。四机从 16 rank 增加到 32 rank 后，跨机 collective 更多、尾延迟概率更高；任意一个 rank 变慢，其他 31 个 rank 都可能停在当前或下一次 broadcast。因此必须把 allocator 和 MCCL 分开测，再测组合路径。

## 4. 能否用单元测试稳定复现

可以写测试，但要区分两件事：

- 普通单元测试可以稳定验证 allocator 基本功能、数值正确性和“小规模分配是否在超时内完成”。
- 此次四机卡死依赖 driver、设备状态、32-rank collective 时序和异常退出后的上下文，不能通过 mock 的普通 UT 保证每次重现。它本质上是硬件相关的压力/集成测试。

因此本次提供三层测试。

### 4.1 单卡 VMM 回归测试

测试必须在全新子进程中设置 allocator 配置，因为 `PYTORCH_MUSA_ALLOC_CONF` 要在 Torch allocator 初始化前生效：

```bash
cd /path/to/OpenSearch_vl_musa/SFT
RUN_SLOW=1 pytest -q tests/train/test_musa_vmm_allocator.py
```

它反复执行不同大小的 allocate、写入、同步、释放和 `empty_cache`，外部 pytest 进程设置 90 秒超时。它可以发现单卡 VMM 硬错误、silent corruption 或本地 map/unmap 卡死，但不能覆盖 MCCL。

### 4.2 原生 MCCL 启动前检查

第 2 节的 `all_reduce_perf` 只回答：当前 32 卡 communicator 能否初始化、collective 能否完成、结果是否正确。

它使用 MUSA/MCCL 的 buffer，但不经过 PyTorch caching allocator，因此不能单独验证 `expandable_segments`。

### 4.3 PyTorch VMM + MCCL 组合复现器

复现器提供四种模式：

| 模式 | 做什么 | 主要回答的问题 |
|---|---|---|
| `allocator` | 反复变尺寸分配、写入、同步、释放和清 cache | 单卡/本机 VMM 是否自己就会卡 |
| `mccl` | 预分配后反复 broadcast 与 all-reduce，并校验首尾值 | MCCL/fabric 在相同 rank 数下是否会卡或 WA |
| `combined` | 分配完整参数→broadcast→同步→clone 本 rank shard→保留 shard→释放完整参数 | 接近 ZeRO-3 `_partition_param` 的同步组合路径 |
| `overlap` | async broadcast 未完成时继续分配临时 buffer，再 wait/partition | 是否存在更敏感的 stream/VMM/MCCL 交叠问题 |

每个 rank 都独立、实时追加：

```text
rank_<rank>_<host>_<pid>.jsonl
```

事件包括：

```text
alloc_begin / alloc_done
process_group_init_begin / process_group_init_done
broadcast_begin / broadcast_done
partition_begin / partition_done
empty_cache_begin / empty_cache_done
iteration_done / probe_pass / probe_fail
```

如果发生卡死，比较 32 个文件的最后一行，就能先判断停在分配、MCCL 初始化、broadcast、同步还是 partition。进程 30 秒无返回时还会周期性输出 Python stack；要确认 native 栈是否在 `muMemSetAccess`，仍应在专用复现环境中配合 `gdb/pstack`。

## 5. 如何运行组合复现

这个脚本会占用目标 hostfile 中的 GPU，必须使用独占的空闲 pod，并显式确认：

```bash
cd /path/to/OpenSearch_vl_musa/SFT

MUSA_VMM_REPRO_ALLOW=1 \
  bash scripts/run_musa_vmm_mccl_repro.sh hostfile.txt \
  --mode combined --iterations 20
```

默认行为：

- 每节点 8 个 Python 进程，每进程绑定一个 local rank；
- 四机为 32 rank，双机为 16 rank；
- allocator 配置为生产使用的 `expandable_segments:True,garbage_collection_threshold:0.8`；
- 进程组 timeout 60 秒，faulthandler 30 秒输出一次栈，外层总 timeout 600 秒；
- 输出写到 `logs/musa_vmm_mccl_repro_<timestamp>/`；
- 外层 timeout 只终止本次 mpirun 的进程组，不按模糊进程名清理训练。

建议依次执行下面的隔离矩阵，每次只改变一个条件：

```bash
# 1. 纯 allocator
MUSA_VMM_REPRO_ALLOW=1 \
  bash scripts/run_musa_vmm_mccl_repro.sh hostfile.txt --mode allocator

# 2. 纯 MCCL
MUSA_VMM_REPRO_ALLOW=1 \
  bash scripts/run_musa_vmm_mccl_repro.sh hostfile.txt --mode mccl

# 3. 同步组合路径
MUSA_VMM_REPRO_ALLOW=1 \
  bash scripts/run_musa_vmm_mccl_repro.sh hostfile.txt --mode combined

# 4. 交叠路径
MUSA_VMM_REPRO_ALLOW=1 \
  bash scripts/run_musa_vmm_mccl_repro.sh hostfile.txt --mode overlap
```

仅用于定位的 allocator A/B：

```bash
PYTORCH_MUSA_ALLOC_CONF=expandable_segments:False \
MUSA_VMM_REPRO_ALLOW=1 \
  bash scripts/run_musa_vmm_mccl_repro.sh hostfile.txt --mode combined
```

`expandable_segments:False` 不是生产修复方案。当前 30B 任务中它曾让正常约 3 分钟的 ZeRO init 退化到几十分钟，只能作为“是否经过 VMM expandable map 路径”的诊断对照。

## 6. 如何解释结果

| 结果 | 更支持的方向 |
|---|---|
| 单节点 `allocator` 已经卡住 | 本机 driver/GPU/VMM allocator |
| allocator 正常，16/32-rank `mccl` 卡住或结果错误 | MCCL、RDMA/fabric、坏节点或拓扑配置 |
| allocator 和 mccl 单项都正常，只有 `combined/overlap` 卡住 | VMM 与 MCCL/stream/device state 的交互 |
| 16 rank 稳定、32 rank 不稳定 | 四机规模放大的 collective 尾延迟或交互问题；不能仅据此归因某一节点 |
| clean rollout 稳定，异常中止后容易复现 | stale device/driver/MCCL context 是重要触发条件 |
| `expandable=False` 不进 `mapAndSetAccess` 但极慢 | 绕开了 VMM 路径，不等于 MCCL 或集群已恢复正常 |

最有价值的证据不是“总 timeout 了”，而是：

1. 哪个 rank 最后停在哪个 `*_begin`；
2. 其他 rank 当时停在哪个 collective 序号/阶段；
3. native stack 是否命中 `muMemSetAccess/mapAndSetAccess`；
4. 相同节点在 allocator-only 和 MCCL-only 中是否正常；
5. 16 与 32 rank、fresh 与异常退出后、expandable true 与 false 的差异。

## 7. 安全边界

- MCCL 启动前检查只能在本次训练 rank 尚未 dispatch 时运行。
- VMM/MCCL 复现器只能在专用、空闲 GPU 上运行，不能与训练并行。
- 如果外层 timeout 触发，应先确认本次 MPI/Python probe 已全部退出，再启动训练。
- 不要通过 `pkill python` 之类的模糊命令清理；它可能误杀其他任务。
- 异常 kill 某个 rank 后立即重启的故障注入更接近 stale-context 场景，但只能人工在隔离环境执行，不应放进普通 pytest 或训练前检查。

## 8. 参考资料

- [PyTorch CUDA memory management：expandable_segments](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.rst)
- [PyTorch CUDACachingAllocator 的 ExpandableSegment/VMM 实现](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)
- [MooreThreads torch_musa releases](https://github.com/MooreThreads/torch_musa/releases)

前两项是 CUDA 后端的上游原理与实现参考；本文把相同的 allocator/VMM 心智模型映射到当前 MUSA 栈。实际故障结论仍以本环境的 torch_musa/MUSA native stack 和复现实验为准。
