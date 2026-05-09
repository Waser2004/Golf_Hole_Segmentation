from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from golf_hole_segmentation.data.dataset import CLASS_COLORS

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ModuleNotFoundError:
    A = None
    ToTensorV2 = None


DatasetMode = Literal["mapped", "original"]
KAGGLE_DATASET_HANDLE = "jacotaco/danish-golf-courses-orthophotos"


DANISH_CLASS_COLORS = {
    0: [0, 0, 0],  # Background
    1: CLASS_COLORS[3],  # Fairway
    2: CLASS_COLORS[1],  # Green
    3: CLASS_COLORS[2],  # Tee
    4: CLASS_COLORS[6],  # Bunker
    5: CLASS_COLORS[9],  # Water
    255: CLASS_COLORS[255],  # Ignore index
}


DANISH_TO_GOLF_HOLE_CLASSES = {
    0: 0,  # Background
    1: 3,  # Fairway
    2: 1,  # Green
    3: 2,  # Tee
    4: 6,  # Bunker
    5: 9,  # Water
    255: 255,  # Ignore index
}


class DanishGolfCoursesOrthophotosDataset(Dataset):
    """Kaggle Danish Golf Courses Orthophotos semantic segmentation dataset.

    The Kaggle class masks use ids 0..5 for background, fairway, green, tee,
    bunker, and water. Use ``mode="mapped"`` to remap those ids to this
    project's 13-class golf-hole mask schema. Use ``mode="original"`` to keep
    the Kaggle ids unchanged.
    """

    IGNORE_INDEX = 255

    def __init__(
        self,
        images,
        masks=None,
        train: bool = True,
        augment: bool | None = None,
        mode: DatasetMode = "mapped",
    ):
        if A is None or ToTensorV2 is None:
            raise ImportError(
                "DanishGolfCoursesOrthophotosDataset requires albumentations. "
                "Install project dependencies with: python -m pip install -r requirements.txt"
            )
        if mode not in {"mapped", "original"}:
            raise ValueError("mode must be either 'mapped' or 'original'.")

        self.images = [Path(image) for image in images]
        self.train = train
        self.mode = mode
        self.colors = CLASS_COLORS if mode == "mapped" else DANISH_CLASS_COLORS

        if train:
            if masks is None:
                raise ValueError("Training mode requires mask paths.")
            self.masks_by_stem = {Path(mask).stem: Path(mask) for mask in masks}
        else:
            self.masks_by_stem = {}

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

    @classmethod
    def from_kaggle(
        cls,
        train: bool = True,
        augment: bool | None = None,
        mode: DatasetMode = "mapped",
        force_download: bool = False,
    ):
        root = download_kaggle_dataset(force_download=force_download)
        images_dir, masks_dir = find_kaggle_image_mask_dirs(root)

        image_paths = sorted(_image_paths(images_dir))
        mask_paths = sorted(_image_paths(masks_dir)) if train else None

        if not image_paths:
            raise FileNotFoundError(f"No images found in {images_dir}")
        if train and not mask_paths:
            raise FileNotFoundError(f"No class masks found in {masks_dir}")

        return cls(
            image_paths,
            mask_paths,
            train=train,
            augment=augment,
            mode=mode,
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
            mask_path = self.masks_by_stem[image_path.stem]
        except KeyError as exc:
            raise FileNotFoundError(f"No class mask found for image {image_path.name}") from exc

        mask = np.array(Image.open(mask_path).convert("L"))
        if self.mode == "mapped":
            mask = _remap_mask(mask, DANISH_TO_GOLF_HOLE_CLASSES)

        out = self.aug(image=image, mask=mask)
        return out["image"], out["mask"]

    def blend(self, image, mask, alpha=0.5):
        """Blend an RGB image with a class-index mask for quick visual checks."""
        mask_rgb = np.zeros_like(image)

        for class_idx, color in self.colors.items():
            mask_rgb[mask == class_idx] = np.array(color)

        return (alpha * image + (1 - alpha) * mask_rgb).astype(np.uint8)


def _image_paths(directory: Path):
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if not directory.exists():
        return []
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]


def download_kaggle_dataset(force_download: bool = False) -> Path:
    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Loading the Danish Kaggle dataset requires kagglehub. "
            "Install project dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    return Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET_HANDLE,
            force_download=force_download,
        )
    )


def find_kaggle_image_mask_dirs(root: Path) -> tuple[Path, Path]:
    image_candidates = _find_dirs(root, {"orthophotos", "orthophoto", "images", "image"})
    mask_candidates = _find_dirs(root, {"class masks", "class_masks", "masks", "mask"})
    if not image_candidates or not mask_candidates:
        image_candidates, mask_candidates = _infer_image_mask_dirs(root)

    if not image_candidates:
        raise FileNotFoundError(
            f"No orthophotos/images directory found in Kaggle dataset cache: {root}"
        )
    if not mask_candidates:
        raise FileNotFoundError(f"No class masks directory found in Kaggle dataset cache: {root}")

    return image_candidates[0], mask_candidates[0]


def _find_dirs(root: Path, names: set[str]) -> list[Path]:
    normalized_names = {_normalize_dir_name(name) for name in names}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and _normalize_dir_name(path.name) in normalized_names
    )


def _normalize_dir_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", " ")
    parts = normalized.split(maxsplit=1)
    if len(parts) == 2 and parts[0].rstrip(".").isdigit():
        normalized = parts[1]
    return normalized


def _infer_image_mask_dirs(root: Path) -> tuple[list[Path], list[Path]]:
    directories_with_images = [
        path
        for path in root.rglob("*")
        if path.is_dir() and _image_paths(path)
    ]
    image_dirs = []
    mask_dirs = []

    for directory in directories_with_images:
        name = _normalize_dir_name(directory.name)
        if "class" in name and "mask" in name:
            mask_dirs.append(directory)
        elif "mask" not in name:
            image_dirs.append(directory)

    return sorted(image_dirs), sorted(mask_dirs)


def _remap_mask(mask: np.ndarray, class_mapping: dict[int, int]) -> np.ndarray:
    remapped = np.full(
        mask.shape,
        fill_value=DanishGolfCoursesOrthophotosDataset.IGNORE_INDEX,
        dtype=np.uint8,
    )
    for source_idx, target_idx in class_mapping.items():
        remapped[mask == source_idx] = target_idx
    return remapped
