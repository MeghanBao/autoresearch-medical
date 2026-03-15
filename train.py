"""
Medical image classification training script. Single-GPU, single-file.
Fixed 5-minute wall-clock training budget.

Usage:
    python train.py
    DATASET=pathmnist IMAGE_SIZE=224 python train.py

Environment variables (passed through to prepare.py):
    DATASET     MedMNIST dataset name (default: chestmnist)
    IMAGE_SIZE  Image resolution 28 or 224 (default: 28)
"""

import logging
import time

import torch
import torch.nn as nn
from torchvision import models

from prepare import (
    DATASET,
    IMAGE_SIZE,
    TRAIN_TIME_MINUTES,
    evaluate,
    get_dataloaders,
    get_n_channels,
    get_num_classes,
    is_multilabel,
)

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
# === HYPERPARAMETERS (agent can modify) ===
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
PRETRAINED: bool = True  # use ImageNet pretrained weights

# ---------------------------------------------------------------------------
# === MODEL ARCHITECTURE (agent can modify) ===
# ---------------------------------------------------------------------------


def build_model(
    num_classes: int,
    n_channels: int,
    image_size: int,
    pretrained: bool = PRETRAINED,
) -> nn.Module:
    """
    Build ResNet-18 with a modified first and last layer for *num_classes* outputs.

    Handles both greyscale (1-channel) and RGB (3-channel) datasets at 28×28 and
    224×224 resolutions. The correct adaptation is determined by *n_channels* and
    *image_size*, not by assuming all datasets share the same modality.

    Adaptations applied:
    - Greyscale (n_channels=1): conv1 in_channels changed to 1.
    - 28×28 resolution: conv1 kernel/stride reduced to 3×3/1 and maxpool removed
      to prevent the feature map collapsing on tiny images.
    - WARNING: any modification to conv1 discards the pretrained conv1 weights.

    Args:
        num_classes: Number of output logits (labels for multi-label, classes for multi-class).
        n_channels:  Number of input image channels (1 = greyscale, 3 = RGB).
        image_size:  Spatial resolution of input images (28 or 224).
        pretrained:  If True, load ImageNet pretrained weights.

    Returns:
        A ``torchvision`` ResNet-18 model with adapted ``conv1`` and ``fc`` layers.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    modify_conv1 = (n_channels != 3) or (image_size == 28)

    if modify_conv1:
        if pretrained:
            logger.warning(
                "conv1 replaced (n_channels=%d, image_size=%d) — pretrained conv1 weights discarded",
                n_channels,
                image_size,
            )
        # stride=1 + no maxpool prevents over-downsampling at 28×28;
        # for 224×224 greyscale we keep the original stride=2 / maxpool.
        if image_size == 28:
            model.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.maxpool = nn.Identity()
        else:
            model.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# === DATA AUGMENTATION (agent can modify) ===
# ---------------------------------------------------------------------------
# Pass a custom torchvision transform via the train_transform argument of
# get_dataloaders() below.  The eval transform is always fixed (no augmentation).
# Make sure Normalize uses the correct number of channels for the dataset:
#   greyscale datasets (chestmnist, pneumoniamnist, breastmnist, octmnist): mean/std length 1
#   RGB datasets (pathmnist, dermamnist, bloodmnist, tissuemnist, organ*):  mean/std length 3
#
# Example (uncomment and pass as train_transform=TRAIN_TRANSFORM):
#
# from torchvision import transforms
# TRAIN_TRANSFORM = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5]),   # adjust channels as needed
# ])
TRAIN_TRANSFORM = None  # set to a transforms.Compose to override default


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)
logger.info("Dataset: %s  image_size: %d", DATASET, IMAGE_SIZE)

num_classes = get_num_classes(DATASET)
n_channels = get_n_channels(DATASET)
multilabel = is_multilabel(DATASET)
logger.info(
    "num_classes=%d  n_channels=%d  multilabel=%s", num_classes, n_channels, multilabel
)

train_loader, val_loader, test_loader = get_dataloaders(
    dataset_name=DATASET,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    train_transform=TRAIN_TRANSFORM,
    num_workers=0,  # 0 = main-process loading; safe on all platforms
)

# ---------------------------------------------------------------------------
# === OPTIMIZER & SCHEDULER (agent can modify) ===
# ---------------------------------------------------------------------------

model = build_model(
    num_classes=num_classes,
    n_channels=n_channels,
    image_size=IMAGE_SIZE,
    pretrained=PRETRAINED,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Cosine LR schedule with warm restarts every epoch.
# T_0=len(train_loader) means one full cosine period = one epoch.
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=len(train_loader),
    eta_min=LEARNING_RATE * 0.01,
)

criterion: nn.Module = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
logger.info("Loss: %s", criterion.__class__.__name__)

# ---------------------------------------------------------------------------
# === TRAINING LOOP (agent can modify) ===
# ---------------------------------------------------------------------------

TIME_BUDGET_SECONDS = TRAIN_TIME_MINUTES * 60

logger.info("Time budget: %ds (%d min)", TIME_BUDGET_SECONDS, TRAIN_TIME_MINUTES)

t_train_start = time.time()
step = 0
epoch = 0

while True:
    model.train()
    epoch += 1
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)

        if multilabel:
            labels = labels.float().to(device, non_blocking=True)
        else:
            labels = labels.squeeze(1).long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        step += 1
        elapsed = time.time() - t_train_start

        if step % 50 == 0:
            pct = 100 * elapsed / TIME_BUDGET_SECONDS
            logger.info(
                "epoch=%d step=%d loss=%.4f lr=%.2e elapsed=%.0fs (%.1f%%)",
                epoch,
                step,
                loss.item(),
                scheduler.get_last_lr()[0],
                elapsed,
                pct,
            )

        if elapsed >= TIME_BUDGET_SECONDS:
            break

    if time.time() - t_train_start >= TIME_BUDGET_SECONDS:
        break

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

logger.info("Training complete. Running final evaluation...")

# Record training end time BEFORE evaluation so train_time excludes eval overhead
t_train_end = time.time()
train_time = t_train_end - t_train_start

# val: used by agent to decide keep/revert
# test: held-out final number; do NOT use to guide experiment decisions
val_metrics = evaluate(model, val_loader, num_classes=num_classes, device=device)
test_metrics = evaluate(model, test_loader, num_classes=num_classes, device=device)

total_time = time.time() - t_start

logger.info("---")
logger.info("val_auc:          %.6f", val_metrics["auc"])
logger.info("val_acc:          %.6f", val_metrics["accuracy"])
logger.info("test_auc:         %.6f", test_metrics["auc"])
logger.info("test_acc:         %.6f", test_metrics["accuracy"])
logger.info("training_seconds: %.1f", train_time)
logger.info("total_seconds:    %.1f", total_time)
logger.info("num_steps:        %d", step)
logger.info("num_epochs:       %d", epoch)
if torch.cuda.is_available():
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    logger.info("peak_vram_mb:     %.1f", peak_vram_mb)
