from pathlib import Path

import numpy as np
import torch
from PIL import Image


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension wrt sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(dim=0)
    union = gts + (1.0 - gt_sorted).cumsum(dim=0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def flatten_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten predictions and labels while dropping ignored pixels."""
    c = probs.shape[1]
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, c)
    labels = labels.view(-1)

    valid = labels != ignore_index
    if not valid.any():
        return probs.new_empty((0, c)), labels.new_empty((0,), dtype=labels.dtype)

    return probs[valid], labels[valid]


def lovasz_softmax_flat(
    probs: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss on flattened tensors."""
    if probs.numel() == 0:
        return probs.sum() * 0.0

    num_classes = probs.shape[1]
    losses = []

    for class_idx in range(num_classes):
        fg = (labels == class_idx).float()

        if classes == "present" and fg.sum() == 0:
            continue

        class_pred = probs[:, class_idx]
        errors = (fg - class_pred).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        fg_sorted = fg[permutation]

        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))

    if not losses:
        return probs.sum() * 0.0

    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = 255, classes: str = "present"):
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs_flat, labels_flat = flatten_probs(probs, labels, ignore_index=self.ignore_index)
        return lovasz_softmax_flat(probs_flat, labels_flat, classes=self.classes)


class CombinedCrossEntropyLovaszLoss(torch.nn.Module):
    def __init__(
        self,
        ce_lambda: float,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
        lovasz_classes: str = "present",
    ):
        super().__init__()
        if not 0.0 <= ce_lambda <= 1.0:
            raise ValueError("ce_lambda must be between 0 and 1.")

        self.ce_lambda = ce_lambda
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index, classes=lovasz_classes)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, labels)
        lovasz_loss = self.lovasz(logits, labels)
        return self.ce_lambda * ce_loss + (1.0 - self.ce_lambda) * lovasz_loss


def compute_inverse_frequency_class_weights(
    mask_paths: list[Path],
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute normalized inverse-frequency CE weights from class-index masks."""
    counts = np.zeros(num_classes, dtype=np.int64)

    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("L"))
        valid = mask != ignore_index
        if not valid.any():
            continue

        class_counts = np.bincount(mask[valid].ravel(), minlength=num_classes)
        counts += class_counts[:num_classes]

    present = counts > 0
    if not present.any():
        raise ValueError("Cannot compute class weights because no valid mask pixels were found.")

    weights = np.zeros(num_classes, dtype=np.float32)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)
