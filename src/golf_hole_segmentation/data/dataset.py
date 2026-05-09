from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ModuleNotFoundError:
    A = None
    ToTensorV2 = None


CLASS_COLORS = {
    0: [0, 0, 0],        # Background / unlabeled
    1: [50, 205, 50],    # Green
    2: [104, 155, 64],   # Tee
    3: [33, 153, 50],    # Fairway
    4: [20, 101, 33],    # Semi Rough
    5: [17, 76, 25],     # High Rough
    6: [210, 180, 140],  # Bunker
    7: [240, 230, 140],  # Waste Area
    8: [17, 48, 25],     # Wald
    9: [70, 130, 180],   # Wasser
    10: [255, 255, 255], # Out
    11: [128, 128, 128], # Weg
    12: [226, 114, 91],  # Haus
    255: [0, 0, 0],      # Ignore index / Background / unlabeled
}


class GolfHoleSegmentationDataset(Dataset):
    IGNORE_INDEX = 255

    def __init__(self, images, masks=None, train=True, augment=None):
        if A is None or ToTensorV2 is None:
            raise ImportError(
                "GolfHoleSegmentationDataset requires albumentations. "
                "Install project dependencies with: python -m pip install -r requirements.txt"
            )

        self.images = [Path(image) for image in images]
        self.train = train
        self.colors = CLASS_COLORS

        if train:
            if masks is None:
                raise ValueError("Training mode requires mask paths.")
            self.masks_by_name = {Path(mask).name: Path(mask) for mask in masks}
        else:
            self.masks_by_name = {}

        if augment is None:
            augment = train

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if augment:
            self.aug = A.Compose(
                [
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                        scale=(0.90, 1.10),
                        rotate=(-20, 20),
                        interpolation=1,
                        mask_interpolation=0,
                        fill=0,
                        fill_mask=self.IGNORE_INDEX,
                        p=0.7,
                    ),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        else:
            self.aug = A.Compose(
                [
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))

        if not self.train:
            out = self.aug(image=image)
            return out["image"], None

        try:
            mask_path = self.masks_by_name[image_path.name]
        except KeyError as exc:
            raise FileNotFoundError(f"No mask found for image {image_path.name}") from exc

        mask = np.array(Image.open(mask_path).convert("L"))
        out = self.aug(image=image, mask=mask)
        return out["image"], out["mask"]

    def blend(self, image, mask, alpha=0.5):
        """Blend an RGB image with a class-index mask for quick visual checks."""
        mask_rgb = np.zeros_like(image)

        for class_idx, color in self.colors.items():
            mask_rgb[mask == class_idx] = np.array(color)

        return (alpha * image + (1 - alpha) * mask_rgb).astype(np.uint8)
