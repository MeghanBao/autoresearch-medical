# autoresearch-medical

> Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
> This fork replaces language model training with **medical image classification**
> on MedMNIST+, exploring automated DL research for radiology applications.

## What's different from the original?

|  | karpathy/autoresearch | This fork |
|---|---|---|
| Task | GPT language model pretraining | Medical image classification |
| Dataset | FineWeb (text) | MedMNIST+ (12 medical imaging datasets) |
| Model | GPT (Transformer decoder) | CNN / ViT / Foundation Model probe |
| Metric | val_bpb (bits per byte) | AUC + Accuracy |
| GPU | H100 required | Any GPU (Colab T4 works) |

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/MeghanBao/autoresearch-medical.git
cd autoresearch-medical
git checkout medical

# 2. Install dependencies (requires Python ≥ 3.10)
pip install -e .

# 3. Download ChestMNIST and verify the data pipeline
python prepare.py

# 4. Train for 5 minutes
python train.py

# Switch dataset or resolution via environment variables
DATASET=pathmnist IMAGE_SIZE=224 python train.py
```

## How it works

The repo has three key files:

### `prepare.py` — data & evaluation (fixed, do not modify)

Downloads MedMNIST+ datasets via the `medmnist` package and exposes:

- `get_dataloaders(dataset_name, image_size, batch_size)` → `(train, val, test)` loaders
- `evaluate(model, dataloader, num_classes)` → `{"auc": ..., "accuracy": ...}`

The evaluation is fixed so results are comparable across all experiments.

### `train.py` — the experiment (agent edits this)

Implements a ResNet-18 baseline with four clearly marked editable zones:

```
# === MODEL ARCHITECTURE (agent can modify) ===
# === DATA AUGMENTATION (agent can modify) ===
# === OPTIMIZER & SCHEDULER (agent can modify) ===
# === HYPERPARAMETERS (agent can modify) ===
# === TRAINING LOOP (agent can modify) ===
```

Training runs for a **fixed 5-minute wall-clock budget**. After training, it logs
`val_auc` and `val_acc`.

### `program.md` — agent instructions

Tells the AI research agent exactly what to do: read `train.py`, form one hypothesis,
make one change, run the experiment, keep if `val_auc` improved, revert if not, repeat.

## Supported Datasets

All 12 MedMNIST+ datasets are supported. Set `DATASET=<name>`:

| Dataset | Task | Classes | Modality |
|---|---|---|---|
| `chestmnist` | Multi-label classification | 14 | Chest X-ray |
| `pathmnist` | Multi-class classification | 9 | Colon pathology |
| `dermamnist` | Multi-class classification | 7 | Dermatoscopy |
| `octmnist` | Multi-class classification | 4 | Retinal OCT |
| `pneumoniamnist` | Binary classification | 2 | Chest X-ray |
| `retinamnist` | Ordinal regression | 5 | Fundus camera |
| `breastmnist` | Binary classification | 2 | Breast ultrasound |
| `bloodmnist` | Multi-class classification | 8 | Blood cell microscopy |
| `tissuemnist` | Multi-class classification | 8 | Kidney cortex microscopy |
| `organamnist` | Multi-class classification | 11 | Abdominal CT |
| `organcmnist` | Multi-class classification | 11 | Abdominal CT |
| `organsmnist` | Multi-class classification | 11 | Abdominal CT |

## Experiment Results

| Iteration | Dataset | val_auc | val_acc | test_auc | Steps | Epochs | Status | Description |
|---|---|---|---|---|---|---|---|---|
| 0 | ChestMNIST 28×28 | 0.7716 | 0.5376 | 0.7715 | 6,143 | 6 | ✅ baseline | ResNet-18, from scratch, AdamW, lr=1e-3, batch=128, T4 GPU, 5min budget |

> Baseline trained on Google Colab T4 GPU (16GB VRAM, 440MB used). Training budget: 5 minutes wall clock.
> Reference: xAILab Bamberg reports ResNet-18 (from scratch) AUC ~0.76-0.78 on ChestMNIST 28×28 (Doerrich et al., Scientific Reports 2025).

## References

- Doerrich et al. (2025), "Rethinking model prototyping through the MedMNIST+ dataset collection", *Scientific Reports*
- Di Salvo et al. (2025), "MedMNIST-C: Comprehensive benchmark and improved classifier robustness by training on randomized image corruptions", *ADSMI @ MICCAI*
- Yang et al. (2023), "MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification", *Scientific Data*
- Karpathy (2026), [autoresearch: AI agents running research automatically](https://github.com/karpathy/autoresearch)

## Acknowledgments

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — original framework and design
- [MedMNIST](https://medmnist.com/) — standardised medical imaging benchmark
- [xAILab Bamberg](https://www.uni-bamberg.de/xai/) (Prof. Christian Ledig) — MedMNIST+ research

## License

MIT
