from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

DATASET_CSV = DOCS_DIR / "data" / "dataset.csv"
GOLF_HOLES_DIR = DATA_DIR / "golf_holes"
GOLF_HOLE_MASKS_DIR = DATA_DIR / "golf_hole_masks"
REFERENCE_IMAGES_DIR = DATA_DIR / "reference_imgs"
UNLABELED_GOLF_HOLES_DIR = DATA_DIR / "unlabeled_golf_holes"
