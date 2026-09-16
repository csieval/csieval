# CSI Eval — 统一 CSI 模型评测框架

> 用于 5G/6G 无线系统中神经网络 CSI（信道状态信息）模型的统一评测框架。支持两个互补任务——
> **CSI 反馈压缩** 和 **CSI 频域预测** ——使用统一的 API 和完全一致的报告格式。

[![PyPI version](https://badge.fury.io/py/csi-eval.svg)](https://pypi.org/project/csi-eval/)
[![Python >= 3.9](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/csi-eval/)
[![PyTorch >= 2.0](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pypi.org/project/csi-eval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/csieval/csieval/blob/main/LICENSE)

---

## 目录

- [架构概览](#架构概览)
- [快速上手](#快速上手)
- [两大评测任务](#两大评测任务)
  - [任务一：CSI 反馈压缩](#任务一csi-反馈压缩-taskfeedback)
    - [反馈任务 — `model_bundle` 目录结构](#反馈任务--model_bundle-目录结构)
  - [任务二：CSI 频域预测](#任务二csi-频域预测-taskprediction)
    - [预测任务 — `model_bundle` 目录结构](#预测任务--model_bundle-目录结构)
- [统一报告接口](#统一报告接口)
- [目录结构](#目录结构)
- [自定义模型（Model Bundle）](#自定义模型model-bundle)
  - [统一目录结构](#统一目录结构)
  - [任务一：CSI 反馈压缩模型要求](#任务一csi-反馈压缩模型要求)
  - [任务二：CSI 频域预测模型要求](#任务二csi-频域预测模型要求)
  - [模型代码（`my_model.py`）](#模型代码my_modelpy)
- [数据集与参考模型](#数据集与参考模型)
- [依赖说明](#依赖说明)
- [API 参考](#api-参考)

---

## 架构概览

```
csi_eval/                          # 统一评测门面
├── __init__.py                   # CSIEvaluator：任务路由器
├── evaluator.py                  # _run_feedback / _run_prediction 分发
├── feedback_eval/                # CSI 反馈压缩评测
│   ├── core/                    # EvalReport、Evaluator、Config
│   ├── reports/                 # HTML/Markdown/JSON 报告（共享样式）
│   ├── runners/                 # 指标计算 Runner
│   ├── metrics/                 # 任务性能、鲁棒性、效能指标
│   ├── tasks/                   # eigenvector_feedback 任务定义
│   └── models/                  # 占位符 / 插件模型注册表
└── pre_eval/                    # CSI 频域预测评测
    ├── evaluator.py              # CSIPreEvaluator → 生成 EvalReport
    ├── metrics/                 # 任务指标、存储、计算、鲁棒性
    ├── model_registry.py        # 从 Bundle 加载模型
    └── data_loader.py          # Scenario1 / Scenario2 数据加载
```

两个子包均生成相同的 `EvalReport` 对象，共享 `html_report.py` /
`markdown_report.py` 渲染器。`report.save()` 接口对两个任务完全一致。

---

## 快速上手

### 方式 A — 使用统一门面（推荐）

```python
from csi_eval import CSIEvaluator

# CSI 反馈压缩
report = CSIEvaluator(
    task="feedback",
    out_dir="results/feedback",
    device="cuda",
    data="data_feedback/2_6GHz",
    model_bundle="ev_csinet_bundle/",
).run()

report.save("json")
report.save("html")
report.save("markdown")

# CSI 频域预测
report = CSIEvaluator(
    task="prediction",
    out_dir="results/prediction",
    device="cuda",
    data="data_pre",
    model_bundle="wifo_bundle",
).run()

report.save("json")
report.save("html")
report.save("markdown")
```

### 方式 B — 直接使用子包

```python
# 反馈压缩任务
from csi_eval.feedback_eval import Evaluator
report = Evaluator(
    task="eigenvector_feedback",
    model_bundle="ev_csinet_bundle/",
    data="data_feedback/2_6GHz",
    output_dir="results/feedback",
).run()

# 频域预测任务
from csi_eval.pre_eval import CSIPreEvaluator, EvalConfig
evaluator = CSIPreEvaluator(
    output_dir="results/prediction",
    device="cuda",
    max_samples=1000,
)
evaluator.add_model("wifo", checkpoint="wifo_bundle/best.pth")
evaluator.run_all()

evaluator.report.save("json")
evaluator.report.save("html")
evaluator.report.save("markdown")
```

---

## 两大评测任务

### 任务一：CSI 反馈压缩（`task="feedback"`）

评测将高维 CSI 压缩为低码率反馈比特流的模型（如 CsiNet、EV-CsiNet）。
评估分布内准确率、存储开销、推理延迟，以及对噪声 / 量化 / 跨场景泛化的鲁棒性。

**报告章节：** 任务性能 → 部署与存储 → 计算效能 → 鲁棒与泛化（SNR 扫描 → 量化扫描 → 跨场景 OOD）

#### 反馈任务 — `model_bundle` 目录结构

```
ev_csinet_bundle/                 # Bundle 目录
├── best.pt                       # 模型权重（必填）
└── model_meta.json              # 模型元信息（必填）
```

**`model_meta.json` 字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_name` | string | 模型显示名称 |
| `task` | string | 任务类型，必须为 `"eigenvector_feedback"` |
| `input_shape` | list | 输入张量形状 `[batch, 2, Nt, K]` |
| `output_shape` | list | 输出张量形状（与输入相同） |
| `default_checkpoint` | string | 检查点文件名（如 `"best.pt"`） |
| `description` | string | 模型描述 |
| `model_kwargs` | dict | 架构参数（`nt`, `n_subbands`） |

**最小配置示例：**

```json
{
    "model_name": "EVCsiNet",
    "task": "eigenvector_feedback",
    "input_shape": [2, 32, 13],
    "output_shape": [2, 32, 13],
    "default_checkpoint": "best.pt",
    "description": "Eigenvector-based CSI Feedback Network",
    "model_kwargs": {
        "nt": 32,
        "n_subbands": 13
    }
}
```

> **注意：** `embed_dim`、`nhead`、`num_layers` 等架构参数会从检查点的 `model_info` 字段自动检测。
> 只需在 `model_kwargs` 中指定 `nt` 和 `n_subbands`。

### 任务二：CSI 频域预测（`task="prediction"`）

评测从部分观测预测被遮挡频域 CSI 的模型（如 WiFo、CsiNet）。
评估填充准确率、存储、推理延迟，以及跨 Mask / SNR / 跨场景泛化的鲁棒性。

**报告章节：** 任务性能 → 部署与存储 → 计算效能 → 鲁棒与泛化（Cross-Mask Δ → Cross Ratio Δ → SNR 扫描 → S1→S2）

#### 预测任务 — `model_bundle` 目录结构

```
wifo_bundle/                      # Bundle 目录
├── best.pth                      # 模型权重（必填）
└── model_meta.json             # 模型元信息（必填）
```

**`model_meta.json` 字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_class` | string | 模型类名（如 `"WiFo"`） |
| `module_path` | string | 模型类的导入路径（如 `"wifo.WiFo"`） |
| `build_params` | dict | 模型架构参数 |
| `training_info` | dict | 训练配置元信息 |

**`build_params` 参数说明（WiFo 示例）：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `nt_port` | int | 发射天线数 |
| `nr` | int | 接收天线数 |
| `nf` | int | 子载波数 |
| `patch_size` | int | Patch 嵌入尺寸 |
| `d_model` | int | 模型隐藏层维度 |
| `depth` | int | Transformer 层数 |
| `num_heads` | int | 注意力头数 |
| `mlp_ratio` | float | MLP 扩展比率 |
| `dropout` | float | Dropout 比率 |

**最小配置示例：**

```json
{
    "model_class": "WiFo",
    "module_path": "wifo.WiFo",
    "build_params": {
        "nt_port": 256,
        "nr": 8,
        "nf": 52,
        "patch_size": 4,
        "d_model": 128,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.1
    },
    "training_info": {
        "task": "joint",
        "mask_mode": "comb",
        "mask_ratio": 0.5
    }
}
```

---

## 统一报告接口

两个任务均生成 `EvalReport` 对象，`.save()` 方法支持：

| 格式 | 输出文件 | 说明 |
|------|----------|------|
| `"json"` | `report.json` | 结构化指标 + 子结果 |
| `"html"` | `report.html` | 交互式 HTML，含 Plotly 图表、渐变标题栏 |
| `"markdown"` | `report.md` | Markdown 表格，GitHub 兼容 |

```python
report = CSIEvaluator(task="prediction", ...).run()

# 保存全部三种格式
report.save("json")
report.save("html")
report.save("markdown")
```

### HTML 报告特性

- 渐变 Hero 标题栏，含模型名称和元信息 Chip
- 颜色编码指标表格（↑ 好 / ↓ 差）
- Plotly 交互图表（SNR、量化鲁棒性曲线）
- 响应式设计，支持移动端

---

## 目录结构

```
your_project/
├── csi_eval/                    # 本包
│   ├── feedback_eval/           # 反馈压缩评测
│   └── pre_eval/              # 频域预测评测
│
├── data_feedback/              # task="feedback" 的数据
│   └── 2_6GHz/               # 含 DATA_HtestI.npy
│
├── data_pre/                   # task="prediction" 的数据
│   ├── generated_scenario_1_*/  # 分布内（S1）数据
│   └── generated_scenario_2_*/  # 跨场景（S2）数据
│
├── ev_csinet_bundle/          # 反馈任务的模型 Bundle
│   ├── best.pt                #   └── 模型权重
│   └── model_meta.json         #   └── 元信息
│
├── wifo_bundle/               # 预测任务的模型 Bundle
│   ├── best.pth               #   └── 模型权重
│   └── model_meta.json        #   └── 元信息
│
└── results/
    ├── feedback/
    │   ├── report.html
    │   ├── report.md
    │   └── report.json
    └── prediction/
        ├── report.html
        ├── report.md
        └── report.json
```

---

## 自定义模型（Model Bundle）

两个任务使用**统一的 Bundle 格式**，用户只需提供模型权重和元信息即可评测。

### 统一目录结构

```
my_bundle/                          # Bundle 目录（任意命名）
├── best.pt / best.pth             # 模型权重（必填，文件名可自定义）
└── model_meta.json                 # 模型元信息（必填）
```

> **注意：** 如果模型类不在评估包内置列表中，还需放置模型代码文件（如 `my_model.py`），详见下文「模型代码」章节。

### 统一 `model_meta.json` 结构

两个任务的 `model_meta.json` 核心结构相同，评估器会根据 `task` 字段自动适配：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task` | string | 是 | 任务类型：`"eigenvector_feedback"` 或 `"joint"` |
| `model_class` | string | 是 | 模型类名（如 `"EVCsiNet"`, `"WiFo"`） |
| `module_path` | string | 是 | 模型类的导入路径 |
| `build_params` | dict | 是 | 模型架构参数（见各任务详细说明） |
| `training_info` | dict | 否 | 训练配置元信息 |

### 任务一：CSI 反馈压缩模型要求

**接口要求：**

```python
class MyFeedbackModel(nn.Module):
    def __init__(self, nt: int, n_subbands: int, **kwargs):
        """必选参数：nt（发射天线数）、n_subbands（子载波数）"""
        super().__init__()
        self.nt = nt
        self.n_subbands = n_subbands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x — [B, 2, Nt, K]，复数的实部/虚部堆叠
        输出: [B, 2, Nt, K]，重建的 CSI 反馈
        """
        return x  # 替换为你的模型前向传播

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """可选：返回压缩后的码字，用于计算压缩比"""
        raise NotImplementedError
```

**`model_meta.json` 示例：**

```json
{
    "task": "eigenvector_feedback",
    "model_class": "EVCsiNet",
    "module_path": "ev_csinet_bundle.model.MyEVCsiNet",
    "build_params": {
        "nt": 32,
        "n_subbands": 13
    },
    "training_info": {
        "dataset": "2_6GHz",
        "compression_dim": 104
    }
}
```

### 任务二：CSI 频域预测模型要求

**接口要求：**

```python
class MyPredictionModel(nn.Module):
    def __init__(self, nt_port: int, nr: int, nf: int, **kwargs):
        """
        必选参数：
        - nt_port: 发射天线数（通常 256）
        - nr: 接收天线数（通常 8）
        - nf: 子载波数（通常 52）
        """
        super().__init__()

    def forward(self, H_known: torch.Tensor,
                mask: torch.Tensor = None,
                task: str = "joint") -> torch.Tensor:
        """
        支持三种签名，自动适配：
        - (H_known)              ← 最简
        - (H_known, mask)        ← 带 Mask
        - (H_known, mask, task)  ← 完整

        输入: H_known — [B, 2, Nt_port, Nr, Nf]
        输出: [B, 2, Nt_port, Nr, Nf]，预测的完整 CSI
        """
        return H_known  # 替换为你的模型前向传播
```

**`model_meta.json` 示例：**

```json
{
    "task": "joint",
    "model_class": "WiFo",
    "module_path": "wifo_bundle.model.WiFo",
    "build_params": {
        "nt_port": 256,
        "nr": 8,
        "nf": 52,
        "patch_size": 4,
        "d_model": 128,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.1
    },
    "training_info": {
        "task": "joint",
        "mask_mode": "comb",
        "mask_ratio": 0.5
    }
}
```

### 模型代码（`my_model.py`）

如果模型类不在评估包内置列表中，需在 Bundle 目录放置模型代码文件：

```
my_bundle/
├── best.pth
├── model_meta.json
└── my_model.py      # 模型代码（评估包无内置时必填）
```

**`my_model.py` 示例（反馈压缩）：**

```python
import torch
import torch.nn as nn

class MyFeedbackNet(nn.Module):
    def __init__(self, nt: int, n_subbands: int, compression_dim: int = 128):
        super().__init__()
        self.nt = nt
        self.n_subbands = n_subbands
        self.compression_dim = compression_dim

        in_dim = 2 * nt * n_subbands
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, compression_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compression_dim, 512),
            nn.ReLU(),
            nn.Linear(512, in_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        code = self.encoder(x_flat)
        out = self.decoder(code)
        return out.reshape(x.shape)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.encoder(x.reshape(b, -1))
```

**`my_model.py` 示例（频域预测）：**

```python
import torch
import torch.nn as nn

class MyPredictionNet(nn.Module):
    def __init__(self, nt_port: int, nr: int, nf: int,
                 d_model: int = 128, depth: int = 6):
        super().__init__()
        self.nt = nt_port
        self.nr = nr
        self.nf = nf

        self.proj = nn.Linear(2 * nr * nf, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=depth,
        )
        self.head = nn.Linear(d_model, 2 * nr * nf)

    def forward(self, H_known: torch.Tensor) -> torch.Tensor:
        b, c, nt, nr, nf = H_known.shape
        x = H_known.permute(0, 2, 3, 4, 1).reshape(b * nt, nr * nf * c)
        x = self.proj(x)
        x = self.transformer(x)
        x = self.head(x)
        out = x.reshape(b, nt, nr, nf, c).permute(0, 4, 1, 2, 3)
        return out
```

> **提示：** `module_path` 应指向 Bundle 内模型文件的模块路径，例如 `my_bundle.model.MyModel`。

### 完整示例：从零评测自定义模型

```python
from csi_eval import CSIEvaluator

# 1. 创建 Bundle 目录结构
# my_feedback_bundle/
#   ├── best.pt
#   ├── model_meta.json
#   └── my_model.py

# 2. 运行评估
report = CSIEvaluator(
    task="feedback",
    model_bundle="my_feedback_bundle/",
    data="data_feedback/2_6GHz",
    out_dir="results/my_model",
).run()

# 3. 保存报告
report.save("html")
```

---

## 数据集与参考模型

所有数据集和参考预训练模型均托管于 Hugging Face：

**[YSSAie/CSIEval](https://huggingface.co/datasets/YSSAie/CSIEval)** — 29.9 GB

数据集仓库包含以下内容：

| 目录 | 说明 |
|------|------|
| `data_feedback/` | CSI 反馈压缩测试数据集（如 `2_6GHz/`） |
| `data_pre/` | CSI 频域预测数据集（Scenario 1 & 2） |
| `ev_csinet_bundle/` | 预训练的 EV-CsiNet 模型包，用于反馈任务 |
| `wifo_bundle/` | 预训练的 WiFo 模型包，用于预测任务 |

### 使用 `huggingface-cli` 下载

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载整个仓库
huggingface-cli download --repo-type dataset YSSAie/CSIEval --local-dir ./CSIEval

# 下载指定文件/目录
huggingface-cli download --repo-type dataset YSSAie/CSIEval data_pre/ --local-dir ./CSIEval
huggingface-cli download --repo-type dataset YSSAie/CSIEval wifo_bundle/ --local-dir ./CSIEval
```

### 使用 Python 下载

```python
from huggingface_hub import snapshot_download

# 下载整个仓库
snapshot_download(repo_id="YSSAie/CSIEval", local_dir="./CSIEval")

# 下载指定文件夹
snapshot_download(
    repo_id="YSSAie/CSIEval",
    allow_patterns=["data_pre/*", "wifo_bundle/*"],
    local_dir="./CSIEval"
)
```

---

## 依赖说明

```txt
# 必需
torch>=2.0.0
numpy>=1.24.0

# 可选 — YAML 配置文件
pyyaml>=6.0

# 可选 — 交互式 HTML 报告（Plotly 图表）
plotly>=5.0.0

# 可选 — DataFrame 导出
pandas>=2.0.0

# 可选 — 鲁棒性曲线绘制
matplotlib>=3.5.0
```

安装：`pip install -r requirements.txt`

---

## API 参考

### `CSIEvaluator`（统一门面）

```python
from csi_eval import CSIEvaluator

report = CSIEvaluator(
    task="feedback" | "prediction",   # 必填
    model_bundle="path/to/bundle",   # 必填：bundle 目录（含 best.pth + model_meta.json）
    data="path/to/data",              # 必填：数据目录
    out_dir="./results",              # 报告输出目录，默认 "./results"
    device="cuda",                   # 推理设备，默认 "cuda"
    max_samples=15000,               # 最大评估样本数，默认 15000
).run()

report.save()   # 同时保存 json / html / markdown
report.save("html")  # 只保存 HTML
```

### `EvalReport`（两个任务共享）

```python
report["metric_name"]              # 单个指标值
report["category/metric_name"]     # 按类别查询指标
report.filter("task_performance")  # 按类别生成子报告
report.save(fmt, output_dir=...)  # 保存报告
report.print_summary()           # 打印摘要到 stdout
```

### `CSIPreEvaluator`（预测任务）

```python
from csi_eval.pre_eval import CSIPreEvaluator, EvalConfig

evaluator = CSIPreEvaluator(
    output_dir="results/prediction",
    device="cuda",
    max_samples=15000,
    eval_cfg=EvalConfig(),
)
evaluator.add_model("model_name", checkpoint="path/to/best.pth")
evaluator.run_all()

evaluator.report.save("html")   # 与 feedback 完全相同的 EvalReport 接口
```
