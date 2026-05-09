from __future__ import annotations

import os
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.models.segmentation import fcn_resnet50

from golf_hole_segmentation.models.segformer import SegFormer
from golf_hole_segmentation.models.unet import UNet
from golf_hole_segmentation.data.dataset import GolfHoleSegmentationDataset


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_FCN = "fcn_resnet50"
MODEL_UNET = "unet"
MODEL_SEGFORMER = "segformer"
MODEL_OPTIONS = (MODEL_FCN, MODEL_UNET, MODEL_SEGFORMER)
SEGFORMER_VARIANTS = ("B0", "B1", "B2", "B3", "B4", "B5")


@dataclass(slots=True)
class InferenceRequest:
    model_type: str
    checkpoint_path: Path
    image_path: Path
    device: torch.device
    num_classes: int = 13
    segformer_variant: str = "B2"
    unsafe_load: bool = False
    blend_alpha: float = 0.5


@dataclass(slots=True)
class InferenceResult:
    original_image: np.ndarray
    predicted_mask: np.ndarray
    blended_image: np.ndarray


@contextmanager
def _portable_path_class_loading():
    import pathlib

    original_posix = pathlib.PosixPath
    original_windows = pathlib.WindowsPath

    if os.name == "nt":
        pathlib.PosixPath = pathlib.PurePosixPath
    else:
        pathlib.WindowsPath = pathlib.PureWindowsPath

    try:
        yield
    finally:
        pathlib.PosixPath = original_posix
        pathlib.WindowsPath = original_windows


def _checkpoint_safe_globals():
    import pathlib

    safe_items = [
        pathlib.Path,
        pathlib.PurePath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
    ]

    if os.name == "nt":
        safe_items.append((pathlib.WindowsPath, "pathlib.WindowsPath"))
        safe_items.append((pathlib.PurePosixPath, "pathlib.PosixPath"))
    else:
        safe_items.append((pathlib.PosixPath, "pathlib.PosixPath"))
        safe_items.append((pathlib.PureWindowsPath, "pathlib.WindowsPath"))

    return safe_items


def load_checkpoint_payload(checkpoint_path: Path, device: torch.device, unsafe_load: bool):
    if unsafe_load:
        with _portable_path_class_loading():
            return torch.load(checkpoint_path, map_location=device, weights_only=False)

    from torch.serialization import safe_globals

    try:
        with _portable_path_class_loading(), safe_globals(_checkpoint_safe_globals()):
            return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            "Safe checkpoint loading failed. Enable 'Unsafe load' only for trusted checkpoints."
        ) from exc


def extract_state_dict(payload):
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            return payload["model_state_dict"]
        if "state_dict" in payload:
            return payload["state_dict"]
    return payload


def extract_training_args(payload) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("args"), dict):
        return payload["args"]
    return {}


def _strip_prefix_from_state_dict(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(key.startswith(prefix) for key in keys):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _build_fcn(num_classes: int):
    try:
        return fcn_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
    except TypeError:
        return fcn_resnet50(pretrained=False, num_classes=num_classes)


def _build_unet(num_classes: int, in_channels: int, depth: int, base_channels: int):
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=depth,
        base_channels=base_channels,
    )


def _build_segformer(num_classes: int, in_channels: int, variant: str):
    variant = variant.upper()
    if variant not in SEGFORMER_VARIANTS:
        raise ValueError(f"Unknown SegFormer variant '{variant}'.")
    return SegFormer(variant=variant, in_channels=in_channels, num_classes=num_classes)


def _resolve_hyper_parameters(request: InferenceRequest, training_args: dict):
    num_classes = int(training_args.get("num_classes", request.num_classes))
    in_channels = int(training_args.get("in_channels", 3))
    depth = int(training_args.get("depth", 5))
    base_channels = int(training_args.get("base_channels", 64))
    segformer_variant = str(training_args.get("variant", request.segformer_variant)).upper()
    return num_classes, in_channels, depth, base_channels, segformer_variant


def _build_model_for_request(request: InferenceRequest, training_args: dict):
    num_classes, in_channels, depth, base_channels, segformer_variant = _resolve_hyper_parameters(
        request, training_args
    )

    if request.model_type == MODEL_FCN:
        model = _build_fcn(num_classes=num_classes)
    elif request.model_type == MODEL_UNET:
        model = _build_unet(
            num_classes=num_classes,
            in_channels=in_channels,
            depth=depth,
            base_channels=base_channels,
        )
    elif request.model_type == MODEL_SEGFORMER:
        model = _build_segformer(
            num_classes=num_classes,
            in_channels=in_channels,
            variant=segformer_variant,
        )
    else:
        allowed = ", ".join(MODEL_OPTIONS)
        raise ValueError(f"Unknown model '{request.model_type}'. Allowed values: {allowed}")

    return model


def _to_vis_uint8(img_chw: torch.Tensor):
    x = img_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    x = np.clip(x * STD + MEAN, 0, 1)
    return (x * 255).astype(np.uint8)


def _predict_logits(model: torch.nn.Module, input_tensor: torch.Tensor):
    output = model(input_tensor)
    if isinstance(output, dict) and "out" in output:
        return output["out"]
    return output


def run_inference(request: InferenceRequest) -> InferenceResult:
    dataset = GolfHoleSegmentationDataset([request.image_path], train=False, augment=False)
    image_tensor, _ = dataset[0]
    input_tensor = image_tensor.unsqueeze(0).to(request.device)

    payload = load_checkpoint_payload(request.checkpoint_path, request.device, request.unsafe_load)
    training_args = extract_training_args(payload)
    state_dict = extract_state_dict(payload)
    state_dict = _strip_prefix_from_state_dict(state_dict, "module.")

    model = _build_model_for_request(request, training_args)
    model = model.to(request.device)
    model.eval()

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not match selected model configuration. "
            "Pick a different model/checkpoint pair."
        ) from exc

    with torch.no_grad():
        logits = _predict_logits(model, input_tensor)

    predicted_mask = torch.argmax(logits.squeeze(0), dim=0).detach().cpu().numpy().astype(np.uint8)

    original_image = _to_vis_uint8(image_tensor)
    blended = dataset.blend(original_image, predicted_mask, alpha=request.blend_alpha)

    return InferenceResult(
        original_image=original_image,
        predicted_mask=predicted_mask,
        blended_image=blended,
    )
