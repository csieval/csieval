# CSI Eval — Unified CSI Model Evaluation Framework

> A unified evaluation framework for neural-network-based CSI (Channel State Information)
> models in 5G/6G wireless systems. Supports two complementary tasks —
> **CSI Feedback Compression** and **CSI Frequency-Domain Prediction** — under one
> consistent API with identical report formats.

[![Python >= 3.9](https://img.shields.io/badge/python-3.9+-blue.svg)](#)
[![PyTorch >= 2.0](https://img.shields.io/badge/pytorch-2.0+-red.svg)](#)

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Two Evaluation Tasks](#two-evaluation-tasks)
  - [Task 1 — CSI Feedback Compression](#task-1--csi-feedback-compression-taskfeedback)
    - [Feedback Task — `model_bundle` Structure](#feedback-task--model_bundle-structure)
  - [Task 2 — CSI Frequency-Domain Prediction](#task-2--csi-frequency-domain-prediction-taskprediction)
    - [Prediction Task — `model_bundle` Structure](#prediction-task--model_bundle-structure)
- [Unified Report Interface](#unified-report-interface)
- [Directory Structure](#directory-structure)
- [Custom Models (Model Bundle)](#custom-models-model-bundle)
  - [Unified Directory Structure](#unified-directory-structure)
  - [Task 1: CSI Feedback Compression Model Requirements](#task-1-csi-feedback-compression-model-requirements)
  - [Task 2: CSI Frequency-Domain Prediction Model Requirements](#task-2-csi-frequency-domain-prediction-model-requirements)
  - [Model Code (`my_model.py`)](#model-code-my_modelpy)
- [Datasets & Reference Models](#datasets--reference-models)
- [Requirements](#requirements)
- [API Reference](#api-reference)

---

## Architecture

```
csi_eval/                          # Unified evaluation facade
├── __init__.py                    # CSIEvaluator: task router
├── evaluator.py                   # _run_feedback / _run_prediction dispatch
├── feedback_eval/                 # CSI Feedback Compression evaluation
│   ├── core/                      # EvalReport, Evaluator, Config
│   ├── reports/                   # HTML / Markdown / JSON writers (shared styles)
│   ├── runners/                  # Metric computation runners
│   ├── metrics/                  # Task performance, robustness, efficiency
│   ├── tasks/                    # eigenvector_feedback task definition
│   └── models/                   # Placeholder / plugin model registry
└── pre_eval/                     # CSI Frequency-Domain Prediction evaluation
    ├── evaluator.py               # CSIPreEvaluator → produces EvalReport
    ├── metrics/                  # Task metrics, storage, compute, robustness
    ├── model_registry.py         # Model loading from bundle
    └── data_loader.py            # Scenario 1 / Scenario 2 data loading
```

Both sub-packages produce the same `EvalReport` object and share the same
`html_report.py` / `markdown_report.py` writers. The `report.save()` API is
identical for both tasks.

---

## Quick Start

```python
from csi_eval import CSIEvaluator

# CSI Feedback Compression
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

# CSI Frequency-Domain Prediction
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

---

## Two Evaluation Tasks

### Task 1 — CSI Feedback Compression (`task="feedback"`)

Evaluates models that compress high-dimensional CSI into a low-rate feedback bitstream
(e.g. CsiNet, EV-CsiNet). Assesses in-distribution accuracy, storage cost,
inference latency, and robustness to noise / quantization / cross-scenario generalization.

**Report sections:** Task Performance → Deployment & Storage → Computation Efficiency →
Robustness & Generalization (SNR sweep → Quantization sweep → Cross-scenario OOD).

#### Feedback Task — `model_bundle` Structure

```
ev_csinet_bundle/                 # Bundle directory
├── best.pt                       # Model weights (required)
└── model_meta.json               # Model metadata (required)
```

**`model_meta.json` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Display name of the model |
| `task` | string | Must be `"eigenvector_feedback"` |
| `input_shape` | list | Input tensor shape `[batch, 2, Nt, K]` |
| `output_shape` | list | Output tensor shape (same as input) |
| `default_checkpoint` | string | Checkpoint filename (e.g. `"best.pt"`) |
| `description` | string | Human-readable description |
| `model_kwargs` | dict | Architecture parameters (`nt`, `n_subbands`) |

**Minimal example:**

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

> **Note:** Architecture parameters like `embed_dim`, `nhead`, `num_layers`, etc. are auto-detected
> from the checkpoint's `model_info` field. Only `nt` and `n_subbands` need to be specified in
> `model_kwargs`.

### Task 2 — CSI Frequency-Domain Prediction (`task="prediction"`)

Evaluates models that predict masked frequency-domain CSI from partial observations
(e.g. WiFo, CsiNet). Assesses imputation accuracy, storage, inference latency, and
cross-mask / SNR / cross-scenario generalization.

**Report sections:** Task Performance → Deployment & Storage → Computation Efficiency →
Robustness & Generalization (Cross-Mask Δ → Cross Ratio Δ → SNR sweep → S1→S2).

#### Prediction Task — `model_bundle` Structure

```
wifo_bundle/                      # Bundle directory
├── best.pth                      # Model weights (required)
└── model_meta.json             # Model metadata (required)
```

**`model_meta.json` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_class` | string | Class name of the model (e.g. `"WiFo"`) |
| `module_path` | string | Import path to the model class (e.g. `"wifo.WiFo"`) |
| `build_params` | dict | Model architecture parameters |
| `training_info` | dict | Training configuration metadata |

**`build_params` (WiFo example):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `nt_port` | int | Number of transmit antennas |
| `nr` | int | Number of receive antennas |
| `nf` | int | Number of subcarriers |
| `patch_size` | int | Patch embedding size |
| `d_model` | int | Model hidden dimension |
| `depth` | int | Number of transformer layers |
| `num_heads` | int | Number of attention heads |
| `mlp_ratio` | float | MLP expansion ratio |
| `dropout` | float | Dropout rate |

**Minimal example:**

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

## Unified Report Interface

Both tasks produce a `EvalReport` object. The `.save()` method accepts:

| Format | File | Description |
|--------|------|-------------|
| `"json"` | `report.json` | Structured metrics + sub-results |
| `"html"` | `report.html` | Interactive HTML with Plotly charts, gradient hero |
| `"markdown"` | `report.md` | Markdown table, GitHub-compatible |

```python
report = CSIEvaluator(task="prediction", ...).run()

# Save all three formats
report.save("json")
report.save("html")
report.save("markdown")
```

### HTML Report Features

- Gradient hero header with model name and metadata chips
- Color-coded metric tables (↑ good / ↓ bad)
- Plotly interactive charts for SNR and quantization robustness curves
- Responsive design, works on mobile

---

## Directory Structure

```
your_project/
├── csi_eval/                    # This package
│   ├── feedback_eval/           # Feedback compression evaluator
│   └── pre_eval/               # Frequency-domain prediction evaluator
│
├── data_feedback/              # For task="feedback"
│   └── 2_6GHz/                # Dataset (contains DATA_HtestI.npy)
│
├── data_pre/                   # For task="prediction"
│   ├── generated_scenario_1_*/  # In-distribution (S1) data
│   └── generated_scenario_2_*/  # Cross-scenario (S2) data
│
├── ev_csinet_bundle/          # Model bundle — feedback task
│   ├── best.pt                #   └── model weights
│   └── model_meta.json         #   └── metadata
│
├── wifo_bundle/               # Model bundle — prediction task
│   ├── best.pth               #   └── model weights
│   └── model_meta.json        #   └── metadata
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

## Custom Models (Model Bundle)

Both tasks use a **unified Bundle format**. Users only need to provide model weights and metadata to evaluate.

### Unified Directory Structure

```
my_bundle/                          # Bundle directory (any name)
├── best.pt / best.pth             # Model weights (required; filename can be customized)
└── model_meta.json                 # Model metadata (required)
```

> **Note:** If your model class is not in the built-in list, you also need to place the model code file (e.g., `my_model.py`) in the bundle directory. See the "Model Code" section below.

### Unified `model_meta.json` Structure

Both tasks share the same core `model_meta.json` structure. The evaluator automatically adapts based on the `task` field:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | string | Yes | Task type: `"eigenvector_feedback"` or `"joint"` |
| `model_class` | string | Yes | Model class name (e.g., `"EVCsiNet"`, `"WiFo"`) |
| `module_path` | string | Yes | Import path to the model class |
| `build_params` | dict | Yes | Model architecture parameters (see task-specific details) |
| `training_info` | dict | No | Training configuration metadata |

### Task 1: CSI Feedback Compression Model Requirements

**Interface requirements:**

```python
class MyFeedbackModel(nn.Module):
    def __init__(self, nt: int, n_subbands: int, **kwargs):
        """Required params: nt (transmit antennas), n_subbands (subcarriers)"""
        super().__init__()
        self.nt = nt
        self.n_subbands = n_subbands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x — [B, 2, Nt, K], real/imaginary stacked complex
        Output: [B, 2, Nt, K], reconstructed CSI feedback
        """
        return x  # Replace with your model forward pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Optional: return compressed code for compression ratio calculation"""
        raise NotImplementedError
```

**`model_meta.json` example:**

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

### Task 2: CSI Frequency-Domain Prediction Model Requirements

**Interface requirements:**

```python
class MyPredictionModel(nn.Module):
    def __init__(self, nt_port: int, nr: int, nf: int, **kwargs):
        """
        Required params:
        - nt_port: Number of transmit antennas (typically 256)
        - nr: Number of receive antennas (typically 8)
        - nf: Number of subcarriers (typically 52)
        """
        super().__init__()

    def forward(self, H_known: torch.Tensor,
                mask: torch.Tensor = None,
                task: str = "joint") -> torch.Tensor:
        """
        Supports three signatures, auto-adapted:
        - (H_known)              ← Minimal
        - (H_known, mask)        ← With mask
        - (H_known, mask, task)  ← Full

        Input: H_known — [B, 2, Nt_port, Nr, Nf]
        Output: [B, 2, Nt_port, Nr, Nf], predicted full CSI
        """
        return H_known  # Replace with your model forward pass
```

**`model_meta.json` example:**

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

### Model Code (`my_model.py`)

If your model class is not in the evaluator's built-in list, place the model code file in the bundle directory:

```
my_bundle/
├── best.pth
├── model_meta.json
└── my_model.py      # Model code (required when not built-in)
```

**`my_model.py` example (feedback compression):**

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

**`my_model.py` example (frequency-domain prediction):**

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

> **Tip:** `module_path` should point to the module path within the bundle, e.g., `my_bundle.model.MyModel`.

### Complete Example: Evaluating a Custom Model from Scratch

```python
from csi_eval import CSIEvaluator

# 1. Create Bundle directory structure
# my_feedback_bundle/
#   ├── best.pt
#   ├── model_meta.json
#   └── my_model.py

# 2. Run evaluation
report = CSIEvaluator(
    task="feedback",
    model_bundle="my_feedback_bundle/",
    data="data_feedback/2_6GHz",
    out_dir="results/my_model",
).run()

# 3. Save report
report.save("html")
```

---

## Datasets & Reference Models

All datasets and reference pre-trained models are available on Hugging Face:

**[YSSAie/CSIEval](https://huggingface.co/datasets/YSSAie/CSIEval)** — 29.9 GB

The dataset repository includes:

| Directory | Description |
|-----------|-------------|
| `data_feedback/` | CSI Feedback Compression test datasets (e.g., `2_6GHz/`) |
| `data_pre/` | CSI Frequency-Domain Prediction datasets (Scenario 1 & 2) |
| `ev_csinet_bundle/` | Pre-trained EV-CsiNet model bundle for feedback task |
| `wifo_bundle/` | Pre-trained WiFo model bundle for prediction task |

### Download with `huggingface-cli`

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download entire repository
huggingface-cli download --repo-type dataset YSSAie/CSIEval --local-dir ./CSIEval

# Download specific files/directories
huggingface-cli download --repo-type dataset YSSAie/CSIEval data_pre/ --local-dir ./CSIEval
huggingface-cli download --repo-type dataset YSSAie/CSIEval wifo_bundle/ --local-dir ./CSIEval
```

### Download with Python

```python
from huggingface_hub import snapshot_download

# Download entire repository
snapshot_download(repo_id="YSSAie/CSIEval", local_dir="./CSIEval")

# Download specific folders
snapshot_download(
    repo_id="YSSAie/CSIEval",
    allow_patterns=["data_pre/*", "wifo_bundle/*"],
    local_dir="./CSIEval"
)
```

---

## Requirements

```txt
# Required
torch>=2.0.0
numpy>=1.24.0

# Optional — YAML config files
pyyaml>=6.0

# Optional — Interactive HTML reports (Plotly charts)
plotly>=5.0.0

# Optional — DataFrame export
pandas>=2.0.0

# Optional — Robustness curve plotting
matplotlib>=3.5.0
```

Install with: `pip install -r requirements.txt`

---

## API Reference

### `CSIEvaluator` (unified facade)

```python
from csi_eval import CSIEvaluator

report = CSIEvaluator(
    task="feedback" | "prediction",  # required
    out_dir="results/...",             # required
    device="cuda" | "cpu",             # default: "cuda"
    data="path/to/data",              # dataset path
    model_bundle="path/to/bundle",     # model bundle directory
).run()

report.save("json" | "html" | "markdown")
```

### `EvalReport` (shared by both tasks)

```python
report["metric_name"]              # single metric value
report["category/metric_name"]     # metric by category
report.filter("task_performance")  # sub-report by category
report.save(fmt, output_dir=...)   # save report
report.print_summary()            # print to stdout
```

### `CSIPreEvaluator` (prediction task)

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

evaluator.report.save("html")   # same EvalReport interface
```
