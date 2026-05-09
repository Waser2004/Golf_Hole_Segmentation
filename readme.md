# Golf Hole Segmentation

This project builds on the [Golf Hole GANs](https://github.com/Waser2004/Golf_hole_GANs) data to prepare golf hole segmentation masks and run a Unet/SegFormer semantic segmentation model.

## Inference GUI

Run the inference workbench:
```bash
python3 src/golf_hole_segmentation/inference.py
```

The GUI lets you:
- select model architecture (`fcn_resnet50`, `unet`, `segformer`)
- select a checkpoint from `checkpoints/` (or browse manually)
- select an input image through the file explorer
- run inference and save `pred_mask` and `blended` outputs to `inference_outputs/`

## Layout

- `src/golf_hole_segmentation/` - importable source package.
- `src/golf_hole_segmentation/data_collection/` - Tkinter annotation/data collection tool and its icons.
- `docs/data/dataset.csv` - saved annotation data.
- `docs/notebooks/` - exploratory and training notebooks.
- `data/` - generated/curated image data used by the converters and dataset class.
- `checkpoints/` - local model weights. This folder is ignored by git.

## Run on RunPod

1. Upload zip to pod
```bash
scp -P <PORT> -i <SSH_KEY> "D:\2. Programmieren\2. Python\Golf_hole_segmentation_for_zip.zip" root@<RUNPOD_IP>:/workspace/
```

2. Install unzip
```bash
ssh -p <PORT> -i <SSH_KEY> root@<RUNPOD_IP>
apt update && apt install unzip -y
```
3. Unzip zip file
```bash
cd /workspace
unzip Golf_hole_segmentation_for_zip.zip
```
4. Install dependencies
```bash
cd Golf_hole_segmentation_for_zip
pip install -r requirements.txt
```
5. Run script
```bash
cd src
python -m golf_hole_segmentation.train --dataset danish --danish-mode mapped --model segformer --variant B1 --epochs 100
```
6. Download checkpoints
```bash
scp -P <PORT> -i <SSH_KEY> root@<RUNPOD_IP>:/workspace/Golf_hole_segmentation/checkpoints/last_unet_lovasz_danish_mapped.pth "C:\Users\nicow\Downloads"
```