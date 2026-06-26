## 下载：
数据集：https://huggingface.co/datasets/OpenSearch-VL/Search-VL-SFT-36K
model：https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
评测模型：https://huggingface.co/openai/gpt-oss-20b

##修改0 环境兼容性适配
### 0.1 /OpenSearch-VL-main/SFT/src/llamafactory/ 依赖 Python 3.11+
当前运行环境为 Python 3.10.12
修改了'/OpenSearch-VL-main/SFT/pyproject.toml'第11行：
```bash
#修改前
requires-python = ">=3.11.0"
#修改后
requires-python = ">=3.10.0"
```

### 0.2 对部分标准库导入做了向下兼容处理：
'/OpenSearch-VL-main/SFT/src/llamafactory/data/data_utils.py'
```bash
#修改前
from enum import StrEnum, unique
#修改后
from enum import unique
from strenum import StrEnum
```

'/OpenSearch-VL-main/SFT/src/llamafactory/data/mm_plugin.py'
```bash
#修改前
from typing import TYPE_CHECKING, BinaryIO, Literal, NotRequired, Optional, TypedDict, Union
#修改后
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union
from typing_extensions import NotRequired
```

'/OpenSearch-VL-main/SFT/src/llamafactory/extras/constants.py'
```bash
#修改前
from enum import StrEnum, unique
#修改后
from enum import unique
from strenum import StrEnum
```

'/OpenSearch-VL-main/SFT/src/llamafactory/hparams/model_args.py'
```bash
#修改前
from typing import Any, Literal, Self
#修改后
from typing import Any, Literal
from typing_extensions import Self
```

## 1  数据准备并修复数据集图片路径找不到问题
数据集位于'OpenSearch-VL-Search-VL-SFT-36K'，7个数据集的images.zip分别解压：
```python
python -c "import zipfile; zipfile.ZipFile('images.zip', 'r').extractall('./images')"
```

创建软链接：
```bash
bash setup_data_links.sh
```

