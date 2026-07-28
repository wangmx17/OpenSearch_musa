# bastion-k8s MCP 使用教程

**使用场景：本地可以直连堡垒机，通过堡垒机 进入K8s 集群，AI 辅助分析代码，文件传输等等.**

**本机 Cursor → SSH 堡垒机 → kubectl / ssh → Pod / 节点**（笔记本不直连 Pod）。

:::
建议配套 [**k8s-multi-node-deploy**](https://sh-code.mthreads.com/andy.wang/agent-compose/-/tree/master/skill/env/k8s-multi-node-deploy?ref_type=heads) skill 效果更佳~~~，无法访问联系我 $\color{#0089FF}{@Andy Wang}$ 
:::

## 1. 安装 Node.js

MCP 需要运行在Node.js 环境上，**Node.js >= 18**。官网 [nodejs.org](https://nodejs.org) 下载 LTS，安装后验证：

```bash
node -v
```

## 2. MCP 配置（Cursor为例）

配置文件：`.cursor/mcp.json`，或 Cursor → Settings → MCP → Edit Config。

:::
**需要先验证本地机器通过 VPN 能登录到 10.121.120.7 才行（找四季青运维）**
:::
```json
{
  "mcpServers": {
    "bastion-k8s": {
      "command": "npx",
      "args": ["-y", "bastion-k8s-mcp@3.0.1"],
      "env": {
        "BASTION_HOST": "10.121.120.7",
        "BASTION_USER": "mccxadmin",
        "BASTION_PASSWORD": "mccxadminjd",
        "BASTION_PORT": "22",
        "ALLOWED_NAMESPACES": "his-test"
      }
    }
  }
}

```

`BASTION_HOST` + `BASTION_USER` 填你**本机能 SSH 直连堡垒机**的地址和账号即可（MCP 会用同样的方式登录）。

保存后重启 MCP 服务。验证连通：

> 用 bastion-k8s 跑 `kubectl cluster-info --request-timeout=5s`

---

## 3. 怎么用

对 Agent **用自然语言**说清：**命名空间、负载/Pod 名、文件路径**；**需要时加「只读，不要改」,可减少AI 操作带来的不确定性风险，比方做破坏性操作**。

| 你想做的事 | 告诉 Agent |
| --- | --- |
| 找 Pod | 命名空间 + Deployment/负载名 |
| 看日志 | Pod 名 + 日志**文件路径**（训练日志多在挂载盘，不是 stdout） |
| 拷到本地 | 远程路径 + 本地目录（Agent 读取后写入本机） |
| 排查故障 | 日志 / dmesg / core 路径 + 「只读分析」 |

输出太大被截断时，让 Agent 用 `grep`、`tail` 缩小范围。

---

## 4. 示例

1.  MCP + SKILL  --> 让从零重新创建一个负载，然后拉128个节点
    
    ![d36c358b1c6ee6cc73b978f300978c27.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65A66aVv7Oke/img/8045491c-9f79-49bb-9ee8-63e90c726336.png)
    
2.  帮进行 多机 mccl-test 测试（替换新mccl 测试）
    
    （上下文信息已经有了对应16机信息)
    
    ![a24515a4bcdf008990bcf01b899b89d9.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65A66aVv7Oke/img/c6dbb6fb-bfe1-4f25-b351-fa670f2523dd.png)
    
3.  批量配置 128机环境，并帮拉起训练做对比验证![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65A66aVv7Oke/img/bc0d5e32-fc06-41d2-8387-0cfbe101f46d.png)
    
4.  负载任务
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65A66aVv7Oke/img/4f8dc2c2-6445-44f8-b1d9-8aaf27d37ccc.png)

## 常见问题

*   **MCP 起不来**：检查 `node -v`、堡垒机地址账号、本机能否 `ssh user@10.121.120.7`
    
*   **k8s\_logs 为空**：日志在文件里，让 Agent 读挂载路径，不要用 container stdout
    
*   **多个节点都 Killed**：先找第一个报 MUSA/SIGABRT 的 rank，其余可能是连锁退出
    

# 补充：

## Codex使用

本地vscode+codex也可用，

将上述[《bastion-k8s MCP 使用教程》](https://alidocs.dingtalk.com/i/nodes/dxXB52LJqwlKlO2nuKjLqAzkWqjMp697?utm_scene=team_space&iframeQuery=anchorId%3Duu_mr4ns3h4hf8ttet6vur)的配置拷贝给codex插件，直接让其根据该配置，自动本地配好MCP server，

配置好后大概如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65A66aVv7Oke/img/4286d40d-ca12-4bbd-94bf-beff93baefa7.png)

也可以打开 ~/.config.toml 增加配置如下

```shell
[mcp_servers.bastion-k8s]
command = 'C:\Users\liang.geng\AppData\Local\OpenAI\Codex\runtimes\cua_node\ecfc0d9aa02807e3\bin\node.exe'
args = ['C:\Users\liang.geng\.codex\mcp\bastion-k8s\node_modules\bastion-k8s-mcp\dist\index.js']

[mcp_servers.bastion-k8s.env]
ALLOWED_NAMESPACES = "his-test"
BASTION_HOST = "10.121.120.7"
BASTION_PASSWORD = "mccxadminjd"
BASTION_PORT = "22"
BASTION_USER = "mccxadmin"
```

> 使用时整个vscode窗口只能当作一个类似codex桌面版app，建议配合terminal使用