# MUSA 四机 ZeRO Init 卡死重复复现与取证

本文说明四机训练 ZeRO Init 卡死诊断代码的设计原理、使用方法、测试结果和当前结论。相关的 MCCL/VMM 分层诊断见 [MUSA 多机 MCCL 健康检查与 VMM 卡死复现](mccl-health-check-musa.md)。

## 1. 目标与观察边界

这套工具重复启动真实的四机训练入口，并为每个 rank 记录从参数解析到 `Trainer.on_train_begin` 的阶段事件。`on_train_begin` 触发时，模型构建以及 DeepSpeed/ZeRO engine 初始化已经完成，因此它可以回答：

- 32 个 rank 是否都完成 ZeRO Init；
- 卡死发生在 tokenizer、数据集、模型加载、Trainer 构建还是 ZeRO Init 之前；
- 哪些 rank、节点和进程停在不同阶段；
- 同一节点组合连续启动时，问题是否具有概率性和可重复性。

探针到达 `on_train_begin` 后会等待外部监督器释放并直接退出，不执行第一个训练 step。因此当前工具不能证明长时间训练稳定，也不能覆盖发生在 forward、backward、optimizer step 或后续 checkpoint 保存中的 hang。若要观察这些阶段，应另加 `on_step_begin`、`on_step_end` 和 loss 心跳，不能把本工具的“通过”解释为整段训练通过。

## 2. 修改原理

### 2.1 按 rank 记录阶段时间线

`src/llamafactory/train/zero_init_probe.py` 在探针显式开启后，将事件写入共享目录：

```text
training_args_ready
tokenizer_load_begin / tokenizer_load_done
dataset_load_begin / dataset_load_done
model_load_begin / model_load_done
trainer_init_begin / trainer_init_done
trainer_train_begin
zero_init_done
```

每条记录包含 attempt id、rank/local rank/node rank、hostname、PID、时间戳和进程启动后的耗时。时间线使用每 rank 独立的 JSONL 文件；`latest/` 和 `done/` 标记采用临时文件加原子替换，避免监督器读到写了一半的 JSON。

探针默认关闭，不改变正常训练行为。`SIGUSR1` faulthandler 只在探针开启时注册，方便 hang 现场对存活进程采集 Python 栈。

### 2.2 在 ZeRO Init 完成点建立屏障

`ZeroInitProbeCallback.on_train_begin` 写入 `zero_init_done`。重复启动模式会让每个 rank 在该点：

1. 周期性写 heartbeat；
2. 等待监督器创建 `RELEASE` 文件；
3. 看到 release 后记录 `release_seen` 并正常退出。

这样可避免先完成的 rank 继续训练或退出，导致其他 rank 出现二次 collective 错误并掩盖最初阶段。rank 侧还有独立 hold timeout，防止监督器消失后永久等待。

### 2.3 外部监督器负责重复启动和判定

`scripts/repeat_zero_init_repro.py` 每轮执行以下流程：

```text
显式授权 + 进程锁
        ↓
检查所有目标节点无训练进程且 GPU 空闲
        ↓
使用独立 master port 异步启动真实四机 launcher
        ↓
轮询所有 rank 的 latest/done/heartbeat
        ↓
32/32 done：写 RELEASE，等待全部进程退出，再进入下一轮
未按时完成：保存现场并立即停止后续轮次
```

成功条件不是 launcher 返回，而是预期的所有 rank 都写出 `done/rank_*.json`。监督器同时区分：

- `launch_failed`：launcher 非零退出，或启动宽限期内没有任何 rank 探针；
- `early_exit`：所有节点 launcher 已退出，但并非所有 rank 完成初始化；
- `hang`：总超时内完成数不足；
- `release_failed`：全部 rank 已完成，但 release 后进程没有按时退出；
- `passed`：全部 rank 完成且释放后退出。

发生异常时，脚本记录 rank 最后阶段，并采集各节点的进程、GPU、网络、InfiniBand 状态和 kernel tail。它不会自动 `pkill`、重启 Pod 或隔离节点，以便保留现场。

### 2.4 启动脚本只在诊断模式增加取证

`launch_train_30b_trace.sh` 和 `train_30b_trace.sh` 透传探针参数，并在探针开启时增加：

- `MCCL_DEBUG_SUBSYS=INIT,COLL,NET,ENV` 和按 host/PID 分文件的 MCCL 日志；
- PyTorch Flight Recorder 兼容环境变量；
- 每节点 launcher PID/PGID 和退出码；
- `OPENSEARCH_DIAGNOSTIC_SKIP_FINAL_SAVE=1`，避免诊断轮次退出时保存大模型。

这些行为均由环境变量显式控制，默认训练路径不启用探针，也不跳过最终保存。

### 2.5 MCCL 健康检查兼容交错日志

多 local device 共用 stdout 时，`MCCL INFO` 可能插入 `all_reduce_perf` 的性能数据行，使单行列解析失败。`scripts/musa_mccl_health_check.py` 现在额外解析最终原子输出的 `Avg bus bandwidth`：

- 完整性能行存在时仍优先使用逐行带宽；
- 性能行被日志打断时，允许用最终平均带宽作为 liveness/带宽观测；
- 返回码、外部超时、`Out of bounds values` 和 `#wrong` 正确性检查仍然保留；
- 平均带宽也必须是有限正数，并受显式最小带宽阈值约束。

这只修复日志解析误判，不会把 MCCL 错误或数值错误判为通过。

## 3. 主要文件

| 文件 | 作用 |
|---|---|
| `scripts/repeat_zero_init_repro.py` | 重复启动、超时判定、空闲检查、release 和现场采集 |
| `src/llamafactory/train/zero_init_probe.py` | rank 事件、完成标记、heartbeat 和 hold/release 回调 |
| `src/llamafactory/train/tuner.py` | 在训练参数完成后接入探针 callback |
| `src/llamafactory/train/sft/workflow.py` | 记录 SFT 初始化阶段，并支持诊断模式跳过最终保存 |
| `launch_train_30b_trace.sh` | 向远端节点透传诊断环境变量 |
| `train_30b_trace.sh` | 配置 MCCL/Flight Recorder、记录 node launcher 信息和退出码 |
| `scripts/musa_mccl_health_check.py` | 兼容 MCCL INFO 与性能输出交错 |
| `tests/scripts/test_repeat_zero_init_repro.py` | 监督器和 hostfile 解析 UT |
| `tests/train/test_zero_init_probe.py` | rank 事件、callback release 和开关 UT |
| `tests/scripts/test_musa_mccl_health_check.py` | MCCL 交错输出及带宽阈值 UT |

## 4. 测试方法

### 4.1 单元测试

在 SFT 目录执行：

```bash
PYTHONPATH=src pytest -q \
  tests/scripts/test_repeat_zero_init_repro.py \
  tests/train/test_zero_init_probe.py \
  tests/scripts/test_musa_mccl_health_check.py
```

单元测试覆盖开关默认关闭、事件文件格式、原子完成标记、release 退出、重复启动器不阻塞、hostfile 校验、launcher 返回码判断，以及 MCCL 交错输出 fallback。

2026-07-29 在 test2 Pod 使用上述命令执行，结果为 `24 passed`。

### 4.2 四机重复启动

hostfile 必须根据当次 Running Pod 动态生成，只作为环境运行输入，不提交到仓库。确认四台机器和 32 张卡均为本任务独占后执行：

```bash
OPENSEARCH_ZERO_INIT_REPRO_ALLOW=1 \
python3 scripts/repeat_zero_init_repro.py ./hostfile.runtime \
  --attempts 50 \
  --timeout-seconds 900 \
  --cooldown-seconds 15
```

关键安全参数：

- `OPENSEARCH_ZERO_INIT_REPRO_ALLOW=1`：重复占用四机的显式授权；
- `--lock-file`：不同机器池使用不同锁，防止同一池重复启动；
- `--base-port`：每轮递增，多个独立机器池应分配不重叠端口；
- `--disable-mccl-preflight`：仅用于 A/B，正常基线保留 MCCL preflight；
- `--log-root`：结果目录必须不存在，避免覆盖旧证据。

每轮目录包含 launcher 日志、各节点训练日志、rank 时间线、heartbeat、状态文件；异常轮次额外产生 `evidence/`。

## 5. 2026-07-28 至 2026-07-29 实验结果

所有统计只计算已有终态 `status.json` 的轮次。被人工中断且没有终态的轮次不计为 pass 或 hang。

### 5.1 test2 对照组

节点池为 `worker32006`、`worker32007`、`worker32009`、`worker32010`，每节点 8 卡，共 32 rank。默认 allocator 配置下连续执行 50 轮：

| 完成轮次 | 通过 | hang | `zero_init_done` | 单轮完成耗时 |
|---:|---:|---:|---:|---|
| 50 | 50 | 0 | 每轮 32/32 rank | 最快 309.425 s；中位 312.908 s；平均 314.135 s；最慢 335.895 s |

日志目录：

```text
logs/zero_init_repro_test2_20260728_202507/
logs/zero_init_repro_test2_driver_20260728_202507.log
```

test2 在这 50 次启动中没有复现 ZeRO Init hang，且耗时分布较集中，可作为当前软件版本和测试方法的稳定对照。

### 5.2 含 worker33082 的 test3 组

节点池为 `worker33082`、`worker32002`、`worker32003`、`worker32004`，每节点 8 卡，共 32 rank。

| 实验 | rank 顺序/allocator | 有终态轮次 | 结果 |
|---|---|---:|---|
| 初始重复启动 | 原顺序，默认 expandable | 2 | 第 1 轮通过，第 2 轮 hang |
| 独立重跑 | 原顺序，默认 expandable | 1 | 第 1 轮 hang |
| 调整节点顺序 | worker33082 放在最后，默认 expandable | 4 | 4 轮通过；第 5 轮被中断且无终态，不计入统计 |
| worker33082 为 rank0 | worker33082 放在最前，默认 expandable | 2 | 第 1 轮通过，第 2 轮 hang |
| allocator A/B | worker33082 为 rank0，`expandable_segments=False` | 1 | 通过 |
| 恢复默认 allocator | worker33082 为 rank0，重新启用 expandable | 1 | hang |

默认 `expandable_segments=True` 的 test3 已完成轮次合计为 10 次：6 次通过、4 次确认 hang。四次 hang 都满足：

- 900 秒内 `zero_init_done` 为 0/32；
- 32 个 rank 的最后 Python 阶段均为 `model_load_begin`；
- 说明所有 rank 都已进入模型加载，但尚无 rank 完成模型加载及后续 ZeRO Init。

作为补充隔离实验，同一 test3 节点池上的小型 `allocator` 20/200 次、`mccl` 200 次、`combined` 200 次和 `overlap` 200 次均通过。这说明小张量 allocator/MCCL 压测不足以复现，触发条件更接近真实 30B 模型加载、显存映射规模或 ZeRO 参数分区路径。

### 5.3 当前结论

证据支持以下结论：

1. 问题不是每次四机启动都会发生：test2 连续 50 次稳定，test3 也存在成功轮次。
2. 问题与 test3 节点组合以及默认 expandable allocator 路径具有较强相关性：默认配置在多个独立实验中共确认 4 次超时，关闭 expandable 的单次 A/B 通过，恢复后再次超时。
3. 调整 worker33082 的 rank 位置不能消除问题，因此它不是简单的 rank0 特例。
4. 当前证据仍不能把根因唯一归到 worker33082 硬件。四次 Python 探针都只定位到 32 个 rank 同时停在 `load_model` 内部；单次 `expandable_segments=False` 通过的样本量也不足以证明 allocator 是唯一根因。
5. 要确认 worker33082，需要保持其他三台机器和软件环境不变，只替换这一台做足量 A/B；同时在 hang 现场保存各 rank native stack、MCCL Flight Recorder、driver/kernel 日志，确认最早停住的设备和调用栈是否稳定落在 `muMemSetAccess` / `mapAndSetAccess`。

## 6. 结果文件与判读

```text
attempt_NNN/
  launcher.log
  node_logs/
  probe/
    rank_*.jsonl
    latest/rank_*.json
    done/rank_*.json
    heartbeat/rank_*.json
    nodes/node_*.json
  status.json
  evidence/                 # 仅异常轮次
```

先查看 `status.json` 和 `evidence/rank_summary.json`，再按 rank 比较 JSONL 的最后事件：

- rank 最后事件不一致：优先排查 CPU 控制流、异常退出、数据加载或 rank 间分支差异；
- 所有 rank 都停在同一阶段：进入该阶段内部采集 native stack，不能仅凭 Python 边界判断 allocator、MCCL 或硬件；
- 所有 rank enqueue/start 同一 collective 但均未完成：更支持 MCCL、网络、stream 依赖或设备 hang；
- 某 rank 未 enqueue、其他 rank 在等待：未进入 collective 的 rank 更接近第一原因。

异常后不要立即覆盖日志或广泛清理进程。先保存 `evidence/`、node log、MCCL/Flight Recorder 文件和 native stack，再按记录的 PID/PGID 做精确清理。
