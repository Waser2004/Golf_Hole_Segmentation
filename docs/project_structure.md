# Project Structure

The repository now separates code, documentation, and data artifacts:

- `src/golf_hole_segmentation/dataset.py` contains the PyTorch dataset and visualization blend helper.
- `src/golf_hole_segmentation/data_converter.py` converts annotation CSV rows into segmentation arrays.
- `src/golf_hole_segmentation/mask_generation.py` is the CLI wrapper for generating masks and cleaned input images.
- `src/golf_hole_segmentation/inference.py` loads the existing FCN-ResNet50 architecture and visualizes predictions.
- `src/golf_hole_segmentation/data_collection/` contains the legacy Tkinter annotation UI, now package-relative.
- `docs/data/dataset.csv` stores annotations.
- `docs/notebooks/Golf_Hole_Segmentation_Training.ipynb` keeps the original training notebook outside the package.
- `data/golf_holes`, `data/golf_hole_masks`, and `data/reference_imgs` contain image assets.

The default paths are centralized in `src/golf_hole_segmentation/paths.py`.

