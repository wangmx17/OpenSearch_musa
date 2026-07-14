# 训练环境问题修复记录

## 最终可用配置

经过多次尝试，基于公开镜像：`pytorch2.8-transformers5.8-transformer_engine2.5-deepep2.0-megatron-cuda13.0-gpu-py3.12-ubuntu24.04`

启动命令：

```bash
export http_proxy=http://11.137.114.83:8443
export https_proxy=http://11.137.114.83:8443
bash /mnt/workspace/zhaoji/codes/OpenSearch-VL-main/SFT/train_4x8.sh
```

具体可以我的运行实例：https://joybuilder-console.jdcloud.com/workspace/ws-7eoinqo4tl/cn-north-1/training/job/detail?jobId=job-syla6nb3kd

---

## 修改 0：从 Ray 迁移到 torchrun 多节点启动（4x8=32 卡），适配我们的集群

### 0.1 新增启动脚本 `train_4x8.sh`

基于 `/mnt/workspace/zhaoji/codes/LLama-Factory-main-caoyuhang/example4.sh` 改写，去掉 Ray，改用 `FORCE_TORCHRUN=1` 多节点分布式启动。

主要模块：

| 模块 | 说明 |
|------|------|
| 依赖安装 | `pip install -e ".[metrics,deepspeed]"`, `qwen-vl-utils`, `flash-attn`, `swanlab`, `deepspeed` 等 |
| 分布式环境 | 继承集群注入的 `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT` |
| NCCL | IB GID 自动检测、HCA 自动探测、socket ifname 自适应、超时 7200s |
| PYTHONPATH | 设为 `OpenSearch-VL-main/SFT/src`，确保 `llamafactory` 包可被发现 |
| 启动命令 | `FORCE_TORCHRUN=1 python -m llamafactory.cli train <yaml>` |

---

### 0.2 新增训练配置 `examples/agentic_full/qwen3_vl_full_sft_30_3b.yaml`

基于同目录下的 `qwen3_vl_full_sft_30_3b_ray.yaml` 改写。主要差异：

| 配置项 | Ray 版 (`_ray.yaml`) | 非 Ray 版 (`.yaml`) |
|--------|---------------------|---------------------|
| `model_name_or_path` | `Qwen/Qwen3-VL-30B-A3B-Instruct` (HF hub) | `/mnt/workspace/zhaoji/models/Qwen3-VL/Qwen3-VL-30B-A3B-Instruct` (本地) |
| `flash_attn` | 未设置 | `fa2` |
| `ray_*` 相关配置 | 有（256 workers, PACK 策略等） | **全部删除** |
| `output_dir` | `saves/qwen3_vl_30b_a3b/full/sft_data_v1` | `saves/qwen3_vl_30b_a3b/full/sft_agent` |
| `save_steps` | 400 | 200 |
| `report_to` | `tensorboard` | `none` |
| `save_total_limit` | 未设置 | 10 |
| `gradient_accumulation_steps` | 1（256 GPU 下有效 bs=256） | 8（32 GPU x bs1 x ga8 = 有效 bs 256，保持一致） |
| `ddp_timeout` | 600 | 180000000 |
| `resume_from_checkpoint` | true | 未设置 |

---

### 0.3 数据准备

数据位于 `/mnt/workspace/zhaoji/data/Search-VL/`，通过软链接接入 `SFT/data/` 目录：

```bash
RAW_DATA=/mnt/workspace/zhaoji/data/Search-VL
SFT_DATA=/mnt/workspace/zhaoji/codes/OpenSearch-VL-main/SFT/data

mkdir -p $SFT_DATA/{new_fvqa,palace,WebQA,new_livevqa,wikiart,wiki_zh,wiki_en}
ln -s $RAW_DATA/fvqa/fvqa_llama_factory_clean.json    $SFT_DATA/new_fvqa/
ln -s $RAW_DATA/palace/palace_llama_factory_filtered.json $SFT_DATA/palace/
# ... 其余数据集类似
```

---

## 修改 1：修复 torchaudio 导入不兼容问题

**文件**: `src/llamafactory/data/mm_plugin.py`

**问题**: `torchaudio 2.11.0`（PyPI 版本）与 NVIDIA 自定义构建的 PyTorch（`torch 2.8.0a0+34c6371d24.nv25.8`）ABI 不兼容，导致 `import torchaudio` 时报错：
```
OSError: /usr/local/lib/python3.12/dist-packages/torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl
```

**修改内容**: 将第 29 行的 `import torchaudio` 改为 try/except 可选导入，并在实际使用处（`_regularize_audios` 方法）增加了缺失时的错误提示。

```python
# 修改前
import torchaudio

# 修改后
try:
    import torchaudio
except (OSError, ImportError):
    torchaudio = None
```

在 `_regularize_audios` 方法中增加判断：
```python
if torchaudio is None:
    raise ImportError(
        "torchaudio is required for audio processing but failed to load. "
        "Install a version compatible with your PyTorch build."
    )
```

---

## 修改 2：修复数据集图片路径找不到问题

**文件**: 7 个数据集 JSON 文件

**问题**: 数据集 JSON 中的图片路径为相对于各自子目录的路径（如 `images/fvqa_train_2305.jpg`），但 LLaMA-Factory 从 `dataset_dir`（即 `data/`）解析路径，导致找不到文件：
```
FileNotFoundError: [Errno 2] No such file or directory: 'images/fvqa_train_2305.jpg'
```

**修改内容**: 为所有数据集 JSON 文件中的图片路径添加了各自子目录前缀：

| 数据集文件 | 路径前缀 | 示例变化 |
|---|---|---|
| `data/new_fvqa/fvqa_llama_factory_clean.json` | `new_fvqa/` | `images/xxx.jpg` → `new_fvqa/images/xxx.jpg` |
| `data/palace/palace_llama_factory_filtered.json` | `palace/` | `images/xxx.png` → `palace/images/xxx.png` |
| `data/WebQA/webqa_llama_factory_filtered.json` | `WebQA/` | `images/xxx.png` → `WebQA/images/xxx.png` |
| `data/new_livevqa/livevqa_llama_factory_filtered.json` | `new_livevqa/` | `images/xxx.jpg` → `new_livevqa/images/xxx.jpg` |
| `data/wikiart/wikiart_llama_factory_filtered.json` | `wikiart/` | `images/xxx.jpg` → `wikiart/images/xxx.jpg` |
| `data/wiki_zh/wiki_zh_llama_factory_filtered.json` | `wiki_zh/` | `images/xxx.jpg` → `wiki_zh/images/xxx.jpg` |
| `data/wiki_en/wiki_en_llama_factory_filtered.json` | `wiki_en/` | `images/xxx.jpg` → `wiki_en/images/xxx.jpg` |

---

## 修改 3：修复 DeepSpeed NVTX 兼容性问题

**文件**: `src/llamafactory/launcher.py`

**问题**: DeepSpeed 0.19.1 内部调用 `nvtx.Domain.push_range()`，但 NVIDIA PyTorch 构建中的 `nvtx` 包已移除该方法，导致模型加载时报错：
```
AttributeError: 'Domain' object has no attribute 'push_range'
```

**修改内容**: 在 `launcher.py` 的 `__main__` 入口处，于 DeepSpeed 实际使用前将其 NVTX 的 `_range_push` 和 `_range_pop` 替换为空操作：

```python
# 在 from llamafactory.train.tuner import run_exp 之前添加
import deepspeed.utils.nvtx as _ds_nvtx
_ds_nvtx._range_push = lambda accelerator, msg: None
_ds_nvtx._range_pop = lambda accelerator: None
```

此修改不影响训练功能，仅跳过 NVTX 性能标记（profiling marker），对训练结果无任何影响。

---

## 修改 4：修复评估时 tokenizer 加载报错 `'list' object has no attribute 'keys'`

**文件**: `saves/qwen3_vl_30b_a3b/full/sft_agent/checkpoint-1144/tokenizer_config.json`

**问题**: 训练时使用的 transformers 5.2.0 将 `extra_special_tokens` 保存为 **list** 格式：

```json
"extra_special_tokens": ["<|im_start|>", "<|im_end|>", "<|object_ref_start|>", ...]
```

而评估环境的 transformers 5.7.0（以及 5.x 较新版本）在 `tokenization_utils_base.py` 中调用 `special_tokens.keys()`，期望 `extra_special_tokens` 为 **dict** 格式，导致加载时一直报错：

```
AttributeError: 'list' object has no attribute 'keys'
```

**修改内容**: 将 `tokenizer_config.json` 中的 `extra_special_tokens` 从 list 转为 dict，以 token 名（去掉 `<|` 和 `|>`）为 key，token 字符串为 value：

```json
"extra_special_tokens": {
  "im_start": "<|im_start|>",
  "im_end": "<|im_end|>",
  "object_ref_start": "<|object_ref_start|>",
  "object_ref_end": "<|object_ref_end|>",
  "box_start": "<|box_start|>",
  "box_end": "<|box_end|>",
  "quad_start": "<|quad_start|>",
  "quad_end": "<|quad_end|>",
  "vision_start": "<|vision_start|>",
  "vision_end": "<|vision_end|>",
  "vision_pad": "<|vision_pad|>",
  "image_pad": "<|image_pad|>",
  "video_pad": "<|video_pad|>"
}
```

**注意**: 所有 checkpoint 目录下的 `tokenizer_config.json` 都需要同样修改。可用以下脚本批量修复：

```python
import json, glob
for path in glob.glob("saves/**/tokenizer_config.json", recursive=True):
    with open(path) as f:
        cfg = json.load(f)
    if isinstance(cfg.get("extra_special_tokens"), list):
        cfg["extra_special_tokens"] = {
            t.replace("<|", "").replace("|>", ""): t
            for t in cfg["extra_special_tokens"]
        }
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"Fixed: {path}")
```

---

## 修改 5：评测时 judge 服务地址和模型名不匹配

**文件**: `Understanding_Qwen3-VL_Style_v8/core/api_judge.py`、`Understanding_Qwen3-VL_Style_v8/1.sh`

**问题**: 评测的 LLM judge 阶段分数异常低（5.3%），实际模型推理结果是正确的，但 judge 打分几乎全部失败。原因有两个：

1. **judge 服务地址失效**: `api_judge.py` 中写死的 `host = "6.178.54.230"` 已不可达（ConnectTimeout / No route to host），所有端口 9000-9007 均无法连接。需要改为当前可用的 `6.178.4.225`。

2. **模型名路径前缀不匹配**: 即使地址改对后，`1.sh` 中的 `EVAL_MODEL="/mnt/workspace/zhuyongfu/model/openai/gpt-oss-20b"` 与远端 vLLM 服务注册的模型 ID `/mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b` 不一致，返回 HTTP 404。虽然本地有 `ln -s /mnt/workspace /mnt/public/users` 软链接，但 model 名是作为**字符串**通过 HTTP 请求发送给远端服务做字符串匹配的，本地 symlink 不影响远端匹配。

**修改内容**:

`api_judge.py` 第 81 行：
```python
# 修改前
host = "6.178.54.230"
# 修改后
host = "6.178.4.225"
```

`1.sh` 第 18 行：
```bash
# 修改前
EVAL_MODEL="/mnt/workspace/zhuyongfu/model/openai/gpt-oss-20b"
# 修改后
EVAL_MODEL="/mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b"
```

**排查方法**: 可通过 `curl http://<host>:<port>/v1/models` 查询远端服务实际注册的模型 ID。


任务运行链接：https://joybuilder-console.jdcloud.com/workspace/ws-7eoinqo4tl/cn-north-1/training/job/detail?jobId=job-7fls6atgdm