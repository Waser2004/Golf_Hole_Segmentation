from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from golf_hole_segmentation.inference_app.core import (
    MODEL_FCN,
    MODEL_SEGFORMER,
    MODEL_UNET,
    SEGFORMER_VARIANTS,
)


@dataclass(slots=True)
class CheckpointInfo:
    path: Path
    model_guess: str | None = None
    segformer_variant_guess: str | None = None

    @property
    def label(self) -> str:
        guessed = self.model_guess or "unknown"
        if guessed == MODEL_SEGFORMER and self.segformer_variant_guess:
            guessed = f"{guessed}-{self.segformer_variant_guess}"
        return f"{self.path.name} [{guessed}]"


def _guess_model_from_name(filename: str):
    lower = filename.lower()
    if "segformer" in lower:
        return MODEL_SEGFORMER
    if "unet" in lower:
        return MODEL_UNET
    if "fcn" in lower or "resnet50" in lower:
        return MODEL_FCN
    return None


def _guess_segformer_variant_from_name(filename: str):
    upper = filename.upper()
    for variant in SEGFORMER_VARIANTS:
        if variant in upper:
            return variant
    return None


def discover_checkpoints(root: Path) -> list[CheckpointInfo]:
    if not root.exists():
        return []

    files = sorted([path for path in root.rglob("*") if path.is_file()])
    checkpoint_paths = [path for path in files if path.suffix.lower() in {".pth", ".pt", ".ckpt"}]

    return [
        CheckpointInfo(
            path=path,
            model_guess=_guess_model_from_name(path.name),
            segformer_variant_guess=_guess_segformer_variant_from_name(path.name),
        )
        for path in checkpoint_paths
    ]
