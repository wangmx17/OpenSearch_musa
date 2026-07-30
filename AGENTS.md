# 环境访问工具边界

本项目只允许使用以下两种远程环境访问方式：

1. `bastion-k8s`
2. `remote_container_workspace` skill

二者对应彼此不通的两套环境。互斥边界以“目标环境”为单位：访问同一个环境时只能选择其中一种方式，禁止同时启用或混用连接信息。一次用户请求可以明确包含多个环境；此时必须把每个环境作为独立阶段依次处理，不得并行操作不同环境。

## 选择规则

- 用户明确指定 `bastion-k8s` 时，只能使用 `bastion-k8s` 提供的能力。
- 用户明确指定远程工作空间、`remote_container_workspace`、SSH 主机或主机上的 Docker 容器时，只能使用 `remote_container_workspace` skill，并完整遵循其 `SKILL.md`。
- 用户给出 Kubernetes namespace、workload、deployment、pod 等集群目标，并且没有指定其他访问方式时，优先选择 `bastion-k8s`。
- 用户给出 SSH 用户、主机 IP、容器名或容器工作目录，并且没有指定其他访问方式时，优先选择 `remote_container_workspace`。
- 如果目标信息同时符合两种方式，或无法可靠判断所属环境，必须先向用户确认；不得通过尝试连接两边来猜测。
- 如果一次请求包含多个环境，应先为每个环境分别确定唯一访问方式。用户已经明确环境与访问方式的对应关系时，无需再次确认。

## 互斥与切换

- 使用 `bastion-k8s` 时，不得加载或调用 `remote_container_workspace` skill，也不得从本机直接 SSH 到 Pod、节点或目标主机来绕过堡垒机架构。
- 使用 `remote_container_workspace` 时，不得调用任何 `bastion-k8s` 工具，也不得借用 Kubernetes 中发现的地址、namespace、Pod 或节点信息作为远程工作空间的连接依据。
- 一种方式连接失败时，不得自动降级到另一种方式。应报告失败原因，并等待用户明确授权切换。
- 用户要求在同一任务中访问多个环境，或明确要求切换环境时，应先结束当前环境阶段并说明下一阶段使用的访问方式；进入新环境后重新验证目标，不得把上一环境的路径、身份、配置、进程状态或检查结果当作新环境事实。
- 不同环境不得同时执行远程操作。完成一个环境的远程命令后，才能开始另一个环境的连接和命令。
- 同一回复中的事实必须标明来自哪个环境和哪一种访问方式。不得把不同环境的结果合并成同一机器列表、容器列表或无来源区分的诊断结论。
- 用户明确要求跨环境比较时，可以在各环境独立采集完成后进行离线对比；对比项必须保留环境来源，不得使用一个环境的工具直接验证另一个环境。

## 目标范围

- 只访问用户指定或可由当前任务上下文唯一确定的 namespace、workload、Pod、SSH 主机、容器和工作目录。
- 连接后先执行最小化的只读验证，确认实际目标与用户要求一致，再开展后续工作。
- 不得因发现其他主机、Pod、容器、挂载目录或凭据而扩大检查范围。
- 需要跨主机、跨 namespace、跨 workload 或跨容器时，必须有用户明确授权或任务本身的清晰范围依据。

## 常用环境清单

本节只保存稳定的环境路由信息，不保存密码、令牌、动态 Pod 名、Pod IP、节点分配或运行状态。用户当次指令与本节不一致时，以用户当次明确指令为准，并重新验证目标。

### `jd_starvla_test`

- 访问方式：`bastion-k8s`
- Kubernetes namespace：`his-test`
- Workload：`deployment/jd-starvla-test`
- 项目工作目录：`/home/jd/liang.geng/OpenSearch_vl_musa/SFT/`
- Pod 选择：使用稳定标签 `app=jd-starvla-test` 动态查询当前 Running Pod。
- 禁止缓存或复用历史 Pod 名、Pod IP、节点名和副本状态；每次任务均须重新查询。

### `jd-qwen3-vl-30b-a3b`

- 访问方式：`bastion-k8s`
- Kubernetes namespace：`his-test`
- Workload：`deployment/jd-qwen3-v-30b-a3b`
- 项目工作目录：`/home/jd/jd-jiexuan/OpenSearch_vl_musa/SFT`
- Pod 选择：使用稳定标签 `app=jd-qwen3-vl-30b-a3b` 动态查询当前 Running Pod。
- 禁止缓存或复用历史 Pod 名、Pod IP、节点名和副本状态；每次任务均须重新查询。

### `jd-qwen-vl-30b-a3b-test2`

- 访问方式：`bastion-k8s`
- Kubernetes namespace：`his-test`
- Workload：`deployment/jd-qwen3-v-30b-a3b-test2`
- 项目工作目录：`/home/jd/gl_dev_test1/OpenSearch_vl_musa/SFT`
- Pod 选择：使用稳定标签 `app=jd-qwen3-vl-30b-a3b-test2` 动态查询当前 Running Pod。
- 禁止缓存或复用历史 Pod 名、Pod IP、节点名和副本状态；每次任务均须重新查询。

### `jd-qwen-vl-30b-a3b-test3`

- 访问方式：`bastion-k8s`
- Kubernetes namespace：`his-test`
- Workload：`deployment/jd-qwen3-v-30b-a3b-test3`
- 项目工作目录：`/home/jd/gl_dev_test1/OpenSearch_vl_musa/SFT`
- Pod 选择：使用稳定标签 `app=jd-qwen3-vl-30b-a3b-test3` 动态查询当前 Running Pod。
- 禁止缓存或复用历史 Pod 名、Pod IP、节点名和副本状态；每次任务均须重新查询。


### `jd-qwen-vl-30b-a3b-test4`

- 访问方式：`bastion-k8s`
- Kubernetes namespace：`his-test`
- Workload：`deployment/jd-qwen3-v-30b-a3b-test4`
- 项目工作目录：`/home/jd/gl_dev_test1/OpenSearch_vl_musa/SFT`
- Pod 选择：使用稳定标签 `app=jd-qwen3-vl-30b-a3b-test4` 动态查询当前 Running Pod。
- 禁止缓存或复用历史 Pod 名、Pod IP、节点名和副本状态；每次任务均须重新查询。


### `jd-qwen-vl-30b-a3b-test5`

- 访问方式：`bastion-k8s`
- Kubernetes namespace：`his-test`
- Workload：`deployment/jd-qwen3-v-30b-a3b-test5`
- 项目工作目录：`/home/jd/gl_dev_test1/OpenSearch_vl_musa/SFT`
- Pod 选择：使用稳定标签 `app=jd-qwen3-vl-30b-a3b-test5` 动态查询当前 Running Pod。
- 禁止缓存或复用历史 Pod 名、Pod IP、节点名和副本状态；每次任务均须重新查询。

### `jd-qwen-remote`

- 访问方式：`remote_container_workspace` skill
- SSH host：`10.20.35.31` `10.20.35.29`
- SSH user：`mccxadmin`
- 常用容器：
  - `jd_qwen3_vl-520`
- 项目工作目录：`/data/share/liang.geng/jd_test/OpenSearch_VL/SFT`
- 进入指定容器前，必须使用该 skill 的最小只读验证确认容器和工作目录均存在。
- 同一任务访问两个容器时，按用户指定范围依次验证和执行，不得把一个容器的环境、进程或依赖状态当作另一个容器的事实。
- 认证优先使用 SSH key/agent；本清单不得添加 SSH 密码。

### `jd-qwen-remote-sdk5.1.0`

- 访问方式：`remote_container_workspace` skill
- SSH host：`10.20.35.29`
- SSH user：`mccxadmin`
- 常用容器：
  - `jd_dev`
- 项目工作目录：`/data/share/liang.geng/jd_test/OpenSearch_VL/SFT`
- 进入指定容器前，必须使用该 skill 的最小只读验证确认容器和工作目录均存在。
- 同一任务访问两个容器时，按用户指定范围依次验证和执行，不得把一个容器的环境、进程或依赖状态当作另一个容器的事实。
- 认证优先使用 SSH key/agent；本清单不得添加 SSH 密码。

## 凭据安全

- 优先使用 SSH key 或 SSH agent。
- 用户提供的密码只允许用于当前连接；不得写入仓库、脚本、配置文件、命令历史、日志、报告或长期环境变量。
- 密码认证只能通过短生命周期的进程环境变量或交互式提示完成，并在操作结束后清理临时 askpass 文件和相关环境变量。
- SSH 首次连接使用 `StrictHostKeyChecking=accept-new`；禁止使用 `StrictHostKeyChecking=no`。
- 回复中不得复述密码、令牌或其他敏感凭据。

## 操作安全

- 环境盘点、状态查询和故障诊断默认采用只读命令。
- 未经用户明确要求，不得执行删除、覆盖、kill/pkill、重启、扩缩容、rollout、修改容器、安装/卸载依赖、修改训练配置或启动/停止训练任务等操作。
- 用户授权变更时，只执行完成请求所需的最小范围变更，并在执行前再次核对目标环境、主机、容器或 Kubernetes 资源。
- 临时诊断文件必须使用唯一名称，不得覆盖已有文件，并在任务结束时清理；清理失败必须向用户说明。

## 结果报告

- 报告实际使用的访问方式，以及已验证的目标标识，例如 namespace/workload/Pod，或 SSH host/container/workdir。
- 只把命令成功返回且未被截断的数据作为事实；失败、超时或截断必须明确说明。
- 不得使用旧任务或另一访问方式缓存的状态冒充当前查询结果。
- 如果目标不可达、工具不可用或权限不足，应停留在当前访问方式内诊断并报告，不得擅自切换环境。
