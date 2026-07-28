# TE / MATE 兼容性说明

本文记录的是当前 `Qwen3-VL-30B-A3B` 的 grouped GEMM 切换过程中，
TE 与 MATE 在摩尔线程 MUSA 环境里遇到的兼容性问题、修改原因和排查方向。

## 背景

当前训练代码同时支持两条 MoE grouped GEMM 路径：

- `te_grouped_gemm`
- `mate_grouped_gemm`

目标是通过同一套 SFT 配置，对比 TE 和 MATE 的性能与精度。

当前环境中安装并验证过的 MATE 版本是 `mate 0.2.1+mu437`。

## 当时遇到的问题

在当前环境中直接加载 TE 路径时，`transformer_engine` 的导入阶段出现过不稳定现象。
典型表现包括：

- `dictionary changed size during iteration`
- `_cuda_getDevice`

这些错误出现在 `transformer_engine` 导入 / patch 早期，而不是训练主循环里。

## 原因判断

结合日志和代码路径，问题更像是 **TE 在 MUSA 环境下初始化时与 `torch_musa` 的兼容/补丁顺序冲突**，而不是 MoE 计算公式本身错误。

简单理解就是：

1. TE 在导入时会做一些兼容性 patch
2. 当前 MUSA 环境也会对相关设备接口做初始化和注册
3. 两边如果顺序不合适，就可能在导入阶段触发异常

因此，这类问题通常不是“模型代码写错了”，而是“运行时初始化顺序不稳”。

## 为什么会改 TE

当时对 TE 的修改，目的不是改算法，而是：

- 让原本的 TE baseline 在当前环境里更稳定地启动
- 避免导入阶段的偶发失败影响 TE vs MATE 对比

所以改动重点是：

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

这次修改的核心不是“TE 算法有问题”，而是 **当前环境里 TE 的导入兼容性不够稳**。
MATE 则是新增的一条可切换路径，用于和 TE 做同条件对比。
