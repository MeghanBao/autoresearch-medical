"""
One-time data preparation for autoresearch-medical experiments.
Downloads MedMNIST+ datasets and provides DataLoader utilities.

Usage:
    python prepare.py                          # download ChestMNIST (28x28)
    DATASET=pathmnist python prepare.py        # download PathMNIST
    IMAGE_SIZE=224 python prepare.py           # download at 224x224 resolution

Environment variables:
    DATASET     MedMNIST dataset name (default: chestmnist)
    IMAGE_SIZE  Image resolution 28 or 224 (default: 28)
"""

import logging
import os

import medmnist
import numpy as np
import torch
from medmnist import INFO
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATASET: str = os.environ.get("DATASET", "chestmnist")
IMAGE_SIZE: int = int(os.environ.get("IMAGE_SIZE", "28"))
TRAIN_TIME_MINUTES: int = 5
SEED: int = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def get_dataloaders(
    dataset_name: str = DATASET,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 64,
    train_transform: transforms.Compose | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download (if needed) and return train/val/test DataLoaders for a MedMNIST+ dataset.

    Args:
        dataset_name:    MedMNIST dataset identifier (e.g. ``"chestmnist"``).
        image_size:      Spatial resolution; 28 or 224.
        batch_size:      Number of samples per mini-batch.
        train_transform: Optional custom transform for the training split. When
                         ``None`` a default normalisation transform is used.
                         Eval transforms are always the default (no augmentation).
        num_workers:     Number of DataLoader worker processes. Defaults to 0
                         (main-process loading) for cross-platform compatibility.
                         Increase on Linux/macOS for faster data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if dataset_name not in INFO:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Valid options: {sorted(INFO.keys())}"
        )
    if image_size not in (28, 224):
        raise ValueError(f"image_size must be 28 or 224, got {image_size}")

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    # Normalisation parameters depend on number of channels (greyscale vs RGB)
    n_channels: int = info["n_channels"]
    norm_mean = [0.5] * n_channels
    norm_std = [0.5] * n_channels

    # Default eval transform (never augmented)
    if image_size == 28:
        eval_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )
    else:
        eval_tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

    # Train transform: use caller-supplied override or fall back to eval_tf
    if train_transform is not None:
        train_tf = train_transform
    else:
        train_tf = eval_tf

    logger.info(
        "Preparing dataset=%s image_size=%d batch_size=%d n_channels=%d",
        dataset_name,
        image_size,
        batch_size,
        n_channels,
    )

    size_kwarg = {"size": image_size} if image_size != 28 else {}

    train_ds = DataClass(split="train", transform=train_tf, download=True, **size_kwarg)
    val_ds = DataClass(split="val", transform=eval_tf, download=True, **size_kwarg)
    test_ds = DataClass(split="test", transform=eval_tf, download=True, **size_kwarg)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "Dataset ready: train=%d val=%d test=%d num_classes=%d task=%s",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        len(info["label"]),
        info["task"],
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Evaluate *model* on *dataloader* and return AUC and accuracy.

    Supports both multi-label (sigmoid) and multi-class (softmax) outputs.
    AUC is computed as macro-average one-vs-rest using sklearn.

    Args:
        model:       PyTorch model in eval mode.
        dataloader:  DataLoader yielding (images, labels) batches.
        num_classes: Number of output classes/labels.
        device:      Target device; inferred from model parameters if None.

    Returns:
        Dict with keys ``"auc"`` and ``"accuracy"``.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(
            labels.cpu().numpy()
        )  # .cpu() required for pinned-memory tensors

    if not all_logits:
        return {"auc": float("nan"), "accuracy": float("nan")}

    logits_np = np.concatenate(all_logits, axis=0)  # (N, C)
    labels_np = np.concatenate(all_labels, axis=0)  # (N, C) or (N, 1)

    # Determine task type from label shape
    is_multilabel = labels_np.ndim == 2 and labels_np.shape[1] > 1

    if is_multilabel:
        probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
        preds = (probs >= 0.5).astype(int)
        flat_labels = labels_np.astype(int)
        flat_preds = preds
    else:
        labels_np = labels_np.squeeze(1).astype(int)  # (N,)
        exp_logits = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # stable softmax
        preds = np.argmax(probs, axis=1)
        flat_labels = labels_np
        flat_preds = preds

    # AUC — macro-average one-vs-rest
    try:
        if is_multilabel:
            auc = roc_auc_score(flat_labels, probs, average="macro")
        else:
            auc = roc_auc_score(
                flat_labels,
                probs,
                average="macro",
                multi_class="ovr",
            )
    except ValueError:
        # Fallback when some classes have no positive samples in this split
        auc = float("nan")

    # Accuracy
    # Multi-label: exact-match (per-sample all labels must match).
    # Multi-class: standard per-sample accuracy.
    if is_multilabel:
        acc = float((flat_preds == flat_labels).all(axis=1).mean())
    else:
        acc = accuracy_score(flat_labels, flat_preds)

    return {"auc": float(auc), "accuracy": float(acc)}


# ---------------------------------------------------------------------------
# Dataset info helpers
# ---------------------------------------------------------------------------


def get_num_classes(dataset_name: str = DATASET) -> int:
    """Return the number of output labels/classes for *dataset_name*."""
    return len(INFO[dataset_name]["label"])


def is_multilabel(dataset_name: str = DATASET) -> bool:
    """Return True if *dataset_name* is a multi-label classification task."""
    return INFO[dataset_name]["task"] == "multi-label, binary-class"


# ---------------------------------------------------------------------------
# Main — smoke-test the pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "Running prepare.py smoke-test for dataset=%s image_size=%d",
        DATASET,
        IMAGE_SIZE,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=DATASET,
        image_size=IMAGE_SIZE,
        batch_size=64,
    )

    images, labels = next(iter(train_loader))
    logger.info(
        "Sample batch — images: %s  labels: %s",
        tuple(images.shape),
        tuple(labels.shape),
    )
    logger.info(
        "num_classes=%d  multilabel=%s",
        get_num_classes(DATASET),
        is_multilabel(DATASET),
    )
    logger.info("Done. Ready to train.")
