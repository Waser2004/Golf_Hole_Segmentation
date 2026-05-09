import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from golf_hole_segmentation.data.converter import DataConverter
from golf_hole_segmentation.utils.paths import (
    DATASET_CSV,
    GOLF_HOLE_MASKS_DIR,
    GOLF_HOLES_DIR,
    REFERENCE_IMAGES_DIR,
)


def generate_masks(
    dataset_csv: Path = DATASET_CSV,
    reference_images_dir: Path = REFERENCE_IMAGES_DIR,
    output_images_dir: Path = GOLF_HOLES_DIR,
    output_masks_dir: Path = GOLF_HOLE_MASKS_DIR,
    only_par: int | None = None,
):
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    converter = DataConverter(
        grid_size=(768, 768),
        box_size=1,
        dataset_csv=dataset_csv,
        reference_images_dir=reference_images_dir,
        output_images_dir=output_images_dir,
    )
    converter.convert_all(only_par=only_par)

    for index, image_array in converter.converted_data.items():
        image = Image.fromarray(image_array.astype("uint8").squeeze(), mode="L")
        image.save(output_masks_dir / f"{index}.png")

    whiten_ignored_pixels(output_images_dir, output_masks_dir)


def whiten_ignored_pixels(images_dir: Path, masks_dir: Path):
    for mask_path in masks_dir.glob("*.png"):
        image_path = images_dir / mask_path.name
        if not image_path.exists():
            continue

        mask_array = np.array(Image.open(mask_path))
        image_array = np.array(Image.open(image_path).convert("RGB"))
        image_array = np.where(mask_array[:, :, np.newaxis] == 255, 255, image_array)
        Image.fromarray(image_array.astype("uint8"), mode="RGB").save(image_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate segmentation masks from annotation CSV data.")
    parser.add_argument("--dataset-csv", type=Path, default=DATASET_CSV)
    parser.add_argument("--reference-images-dir", type=Path, default=REFERENCE_IMAGES_DIR)
    parser.add_argument("--output-images-dir", type=Path, default=GOLF_HOLES_DIR)
    parser.add_argument("--output-masks-dir", type=Path, default=GOLF_HOLE_MASKS_DIR)
    parser.add_argument("--only-par", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_masks(
        dataset_csv=args.dataset_csv,
        reference_images_dir=args.reference_images_dir,
        output_images_dir=args.output_images_dir,
        output_masks_dir=args.output_masks_dir,
        only_par=args.only_par,
    )


if __name__ == "__main__":
    main()
