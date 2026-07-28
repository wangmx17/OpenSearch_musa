# TE / MATE 兼容性说明

本文记录的是当前 `Qwen3-VL-30B-A3B` 的 grouped GEMM 切换过程中，
TE 与 MATE 在摩尔线程 MUSA 环境里遇到的兼容性问题、修改原因和排查方向。

## 背景

当前训练代码同时支持两条 MoE grouped GEMM 路径：

- `te_grouped_gemm`
- `mate_grouped_gemm`

目标是通过同一套 SFT 配置，对比 TE 和 MATE 的性能与精度。

当前环境中安装并验证过的 MATE 版本是 `mate 0.2.1+mu437`。

## 历史现象与本次复现

此前的排查记录中，直接加载 TE 路径时曾报告过导入阶段不稳定，典型表现包括：

- `dictionary changed size during iteration`
- `_cuda_getDevice`

这些错误被认为发生在 `transformer_engine` 导入 / patch 早期，而不是训练主循环里。

本次在 `his-test/jd-qwen-vl-30b-a3b-test3` 的 `10.121.32.2` 和
`10.121.32.3` 上，使用未包含 TE 稳定性补丁的基线代码进行了双机 8 卡、
3-step 的真实 TE 训练。训练完成且未出现上述两类错误；因此当前证据不足以
证明必须修改 TE 才能保证该环境正常启动。

## 原因判断

结合日志和代码路径，问题更像是 **TE 在 MUSA 环境下初始化时与 `torch_musa` 的兼容/补丁顺序冲突**，而不是 MoE 计算公式本身错误。

简单理解就是：

1. TE 在导入时会做一些兼容性 patch
2. 当前 MUSA 环境也会对相关设备接口做初始化和注册
3. 两边如果顺序不合适，就可能在导入阶段触发异常

因此，这类问题通常不是“模型代码写错了”，而是“运行时初始化顺序不稳”。

## TE 修改的处理结论

此前对 TE 的修改，目的不是改算法，而是：

- 让原本的 TE baseline 在当前环境里更稳定地启动
- 避免导入阶段的偶发失败影响 TE vs MATE 对比

这些改动在本分支最终不保留，TE 继续使用基线实现。若后续在同一环境
再次稳定复现上述错误，应先保留完整错误日志和复现条件，再单独提交
TE 稳定性修复，而不要和 MATE kernel 开关混在一起。

历史补丁的改动重点曾包括：

- 先完成 `torch_musa` 初始化
- 再导入 `transformer_engine`
- 如果出现 partial import，则清理残留模块后重试

## MATE 是怎么接入的

MATE 侧新增的是独立 kernel：

- `mate_grouped_gemm`

它通过 `v1_kernel_ids` 切换，不影响原有训练主流程。

当前对比方式是：

- `v1_kernel_ids: te_grouped_gemm`
- `v1_kernel_ids: mate_grouped_gemm`

## 如何快速定位这个问题

如果别人遇到类似错误，可以优先检查：

1. `transformer_engine` 是否能在当前 MUSA 环境正常导入
2. `torch_musa` 是否已经先完成初始化
3. 日志里是否出现在训练前的 import / patch 阶段报错
4. 当前选择的是 `te_grouped_gemm` 还是 `mate_grouped_gemm`

## 结论

本分支的核心改动是新增 `mate_grouped_gemm` 并保留 YAML 切换能力，
不是修改 TE。当前双机验证表明，基线 TE 和 MATE 都可以通过
`v1_kernel_ids` 在训练 YAML 中选择并进入正常训练流程。
