from __future__ import annotations

from pathlib import Path

from PIL import Image

from golf_hole_segmentation.inference_app.core import InferenceResult


def save_inference_outputs(
    result: InferenceResult,
    output_dir: Path,
    image_path: Path,
    checkpoint_path: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_checkpoint_name = checkpoint_path.stem.replace(" ", "_")
    image_stem = image_path.stem

    mask_path = output_dir / f"{image_stem}_{safe_checkpoint_name}_pred_mask.png"
    blended_path = output_dir / f"{image_stem}_{safe_checkpoint_name}_blended.png"

    Image.fromarray(result.predicted_mask).save(mask_path)
    Image.fromarray(result.blended_image).save(blended_path)

    return mask_path, blended_path
