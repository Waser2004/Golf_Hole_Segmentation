import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from golf_hole_segmentation.data.danish_golf_courses import (
    DANISH_TO_GOLF_HOLE_CLASSES,
    DanishGolfCoursesOrthophotosDataset,
    download_kaggle_dataset,
    find_kaggle_image_mask_dirs,
)
from golf_hole_segmentation.models.segformer import SegFormer
from golf_hole_segmentation.models.unet import UNet
from golf_hole_segmentation.data.dataset import GolfHoleSegmentationDataset
from golf_hole_segmentation.training.losses import (
    CombinedCrossEntropyLovaszLoss,
)
from golf_hole_segmentation.utils.paths import CHECKPOINTS_DIR, GOLF_HOLE_MASKS_DIR, GOLF_HOLES_DIR
from golf_hole_segmentation.data.splits import EVAL_HOLE_ID_RANGES, split_pairs_by_hole_id


DANISH_MAPPED_LOSS_CLASSES = {1, 2, 3, 6, 9}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MaskTransform = Callable[[np.ndarray], np.ndarray]


def mean_iou(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    class_ids: set[int] | None = None,
) -> float:
    preds = torch.argmax(logits, dim=1)
    num_classes = logits.shape[1]
    ious = []

    classes = sorted(class_ids) if class_ids is not None else range(num_classes)
    for cls in classes:
        if cls == ignore_index:
            continue

        valid = labels != ignore_index
        pred_mask = (preds == cls) & valid
        true_mask = (labels == cls) & valid
        intersection = (pred_mask & true_mask).sum().item()
        union = (pred_mask | true_mask).sum().item()

        if union > 0:
            ious.append(intersection / union)

    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    criterion: torch.nn.Module,
    device: torch.device,
    ignore_index: int,
    loss_class_ids: set[int] | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    seen_samples = 0
    total_batches = len(loader)
    epoch_start = time.time()

    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device=device, dtype=torch.long)
        masks = mask_labels_for_loss(masks, ignore_index=ignore_index, class_ids=loss_class_ids)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        seen_samples += images.size(0)
        avg_loss = running_loss / seen_samples
        progress = batch_idx / total_batches
        filled = int(30 * progress)
        bar = "#" * filled + "-" * (30 - filled)
        elapsed = time.time() - epoch_start
        print(
            f"\r  Train [{bar}] {progress * 100:6.2f}% "
            f"({batch_idx}/{total_batches}) loss={avg_loss:.4f} elapsed={elapsed:5.1f}s",
            end="",
            flush=True,
        )

    print()
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    ignore_index: int,
    loss_class_ids: set[int] | None = None,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    total_miou = 0.0
    count_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device=device, dtype=torch.long)
        masks = mask_labels_for_loss(masks, ignore_index=ignore_index, class_ids=loss_class_ids)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        total_miou += mean_iou(
            logits,
            masks,
            ignore_index=ignore_index,
            class_ids=loss_class_ids,
        )
        count_batches += 1

    avg_loss = running_loss / len(loader.dataset)
    mean_iou_score = (total_miou / count_batches) if count_batches > 0 else 0.0
    return avg_loss, mean_iou_score


def mask_labels_for_loss(
    labels: torch.Tensor,
    ignore_index: int,
    class_ids: set[int] | None,
) -> torch.Tensor:
    if class_ids is None:
        return labels

    keep = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in class_ids:
        keep |= labels == class_id
    return torch.where(keep, labels, torch.full_like(labels, ignore_index))


def default_run_name(args: argparse.Namespace) -> str:
    model_name = args.model
    variant = args.variant
    mode_suffix = ""

    if args.training_mode == "finetune":
        mode_suffix = "_finetune"
    elif args.dataset == "danish":
        mode_suffix = f"_danish_{args.danish_mode}"

    if model_name == "unet":
        return f"unet_lovasz{mode_suffix}"
    return f"segformer_{variant.lower()}_lovasz{mode_suffix}"


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "unet":
        return UNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            depth=args.depth,
            base_channels=args.base_channels,
        )

    return SegFormer(
        variant=args.variant,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )


def build_optimizer(
    model: torch.nn.Module,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if encoder_lr <= 0.0:
        raise ValueError("--encoder-lr must be positive.")
    if decoder_lr <= 0.0:
        raise ValueError("--decoder-lr must be positive.")

    if isinstance(model, SegFormer):
        encoder_modules = [model.patch_embeds, model.blocks, model.norms]
        decoder_modules = [model.decoder_mlps, model.linear_fuse, model.dropout, model.classifier]
    elif isinstance(model, UNet):
        encoder_modules = [model.down_blocks, model.bottleneck]
        decoder_modules = [model.up_transpose, model.up_blocks, model.final_conv]
    else:
        raise TypeError(f"Unsupported model type for parameter grouping: {type(model).__name__}")

    encoder_params = [param for module in encoder_modules for param in module.parameters()]
    decoder_params = [param for module in decoder_modules for param in module.parameters()]
    grouped_param_ids = {id(param) for param in encoder_params + decoder_params}
    missing_params = [param for param in model.parameters() if id(param) not in grouped_param_ids]
    if missing_params:
        raise ValueError(
            "Some model parameters were not assigned to encoder or decoder optimizer groups."
        )

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr, "name": "encoder"},
            {"params": decoder_params, "lr": decoder_lr, "name": "decoder"},
        ],
        weight_decay=weight_decay,
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_iters: int,
    warmup_ratio: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if warmup_iters < 0:
        raise ValueError("--lr-warmup-iters must be non-negative.")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("--lr-warmup-ratio must be between 0 and 1.")
    if min_lr < 0.0:
        raise ValueError("--min-lr must be non-negative.")

    base_lrs = [group["lr"] for group in optimizer.param_groups]
    if any(base_lr <= 0.0 for base_lr in base_lrs):
        raise ValueError("All optimizer learning rates must be positive when using LR decay.")
    min_lr_factors = [min(min_lr / base_lr, 1.0) for base_lr in base_lrs]
    warmup_iters = min(warmup_iters, max(0, total_steps - 1))

    def lr_factor(step: int, min_lr_factor: float) -> float:
        current_step = min(step, total_steps)
        if warmup_iters > 0 and current_step < warmup_iters:
            warmup_progress = current_step / warmup_iters
            return warmup_ratio + (1.0 - warmup_ratio) * warmup_progress

        decay_steps = max(1, total_steps - warmup_iters)
        decay_progress = (current_step - warmup_iters) / decay_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(max(decay_progress, 0.0), 1.0)))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_factor

    lr_lambdas = [
        lambda step, min_lr_factor=min_lr_factor: lr_factor(step, min_lr_factor)
        for min_lr_factor in min_lr_factors
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)


def parse_args(default_model: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train golf hole segmentation models with CE + Lovasz loss."
    )
    parser.add_argument(
        "--model",
        choices=["unet", "segformer"],
        default=default_model or "segformer",
    )
    parser.add_argument("--training-mode", choices=["train", "finetune"], default="train")
    parser.add_argument("--dataset", choices=["golf", "danish"], default="golf")
    parser.add_argument("--images-dir", type=Path, default=GOLF_HOLES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=GOLF_HOLE_MASKS_DIR)
    parser.add_argument(
        "--danish-mode",
        choices=["mapped", "original"],
        default="mapped",
        help="mapped uses this project's 13-class ids; original keeps Kaggle ids 0..5.",
    )
    parser.add_argument(
        "--force-kaggle-download",
        action="store_true",
        help="Force KaggleHub to refresh the Danish dataset cache before training.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--finetune-from",
        type=Path,
        help="Checkpoint to initialize from when --training-mode finetune is used.",
    )

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--lr",
        type=float,
        help="Compatibility fallback used for both parameter groups if group-specific LRs are omitted.",
    )
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--lr-scheduler",
        choices=["cosine", "none"],
        default="cosine",
        help="Learning-rate decay strategy.",
    )
    parser.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=1500,
        help="Linear warmup iterations before cosine decay.",
    )
    parser.add_argument(
        "--lr-warmup-ratio",
        type=float,
        default=1e-6,
        help="Initial warmup LR as a fraction of each parameter group's base LR.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Final learning rate floor for cosine decay.",
    )
    parser.add_argument(
        "--ce-lambda",
        type=float,
        default=0.5,
        help="Weight for weighted cross entropy in lambda * CE + (1 - lambda) * Lovasz.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--depth", type=int, default=5, help="U-Net depth.")
    parser.add_argument("--base-channels", type=int, default=64, help="U-Net base channel count.")
    parser.add_argument("--variant", default="B2", choices=["B0", "B1", "B2", "B3", "B4", "B5"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.training_mode == "finetune":
        args.dataset = "golf"
        if args.finetune_from is None:
            raise ValueError("--training-mode finetune requires --finetune-from.")
    if args.num_classes is None:
        args.num_classes = 6 if args.dataset == "danish" and args.danish_mode == "original" else 13
    if args.epochs is None:
        args.epochs = 30 if args.model == "unet" else 100
    if args.encoder_lr is None:
        args.encoder_lr = args.lr if args.lr is not None else 1e-4
    if args.decoder_lr is None:
        args.decoder_lr = args.lr if args.lr is not None else 1e-3
    if args.run_name is None:
        args.run_name = default_run_name(args)
    return args


def image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_pairs_random(
    image_paths_: list[Path],
    mask_paths_: list[Path],
    val_fraction: float,
    seed: int,
) -> tuple[tuple[list[Path], list[Path]], tuple[list[Path], list[Path]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")

    masks_by_stem = {path.stem: path for path in mask_paths_}
    pairs = []
    for image_path in image_paths_:
        try:
            pairs.append((image_path, masks_by_stem[image_path.stem]))
        except KeyError as exc:
            raise FileNotFoundError(f"No mask found for image {image_path.name}") from exc

    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_count = max(1, round(len(pairs) * val_fraction))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    if not train_pairs:
        raise ValueError("Validation split consumed all samples; reduce --val-fraction.")

    train_images, train_masks = zip(*train_pairs)
    val_images, val_masks = zip(*val_pairs)
    return (list(train_images), list(train_masks)), (list(val_images), list(val_masks))


def remap_danish_mask(mask: np.ndarray) -> np.ndarray:
    remapped = np.full(mask.shape, GolfHoleSegmentationDataset.IGNORE_INDEX, dtype=np.uint8)
    for source_idx, target_idx in DANISH_TO_GOLF_HOLE_CLASSES.items():
        remapped[mask == source_idx] = target_idx
    return remapped


def ignore_classes_except(mask: np.ndarray, class_ids: set[int]) -> np.ndarray:
    return np.where(
        np.isin(mask, list(class_ids)),
        mask,
        GolfHoleSegmentationDataset.IGNORE_INDEX,
    ).astype(np.uint8)


def compute_class_weights(
    mask_paths_: list[Path],
    num_classes: int,
    ignore_index: int,
    transform: MaskTransform | None = None,
) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)

    for mask_path in mask_paths_:
        mask = np.array(Image.open(mask_path).convert("L"))
        if transform is not None:
            mask = transform(mask)

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


def build_datasets(
    args: argparse.Namespace,
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    list[Path],
    int,
    MaskTransform | None,
    set[int] | None,
    str,
]:
    if args.dataset == "golf":
        train_label = "golf"
        images = image_paths(args.images_dir)
        masks = image_paths(args.masks_dir)
        if not images:
            raise FileNotFoundError(f"No images found in {args.images_dir}")
        if not masks:
            raise FileNotFoundError(f"No masks found in {args.masks_dir}")

        (train_images, train_masks), (val_images, val_masks) = split_pairs_by_hole_id(
            image_paths=images,
            mask_paths=masks,
        )
        train_dataset = GolfHoleSegmentationDataset(
            train_images,
            train_masks,
            train=True,
            augment=True,
        )
        val_dataset = GolfHoleSegmentationDataset(val_images, val_masks, train=True, augment=False)
        return (
            train_dataset,
            val_dataset,
            train_masks,
            GolfHoleSegmentationDataset.IGNORE_INDEX,
            None,
            None,
            train_label,
        )

    kaggle_root = download_kaggle_dataset(force_download=args.force_kaggle_download)
    images_dir, masks_dir = find_kaggle_image_mask_dirs(kaggle_root)
    images = image_paths(images_dir)
    masks = image_paths(masks_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    if not masks:
        raise FileNotFoundError(f"No class masks found in {masks_dir}")

    (train_images, train_masks), (val_images, val_masks) = split_pairs_random(
        images,
        masks,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_dataset = DanishGolfCoursesOrthophotosDataset(
        train_images,
        train_masks,
        train=True,
        augment=True,
        mode=args.danish_mode,
    )
    val_dataset = DanishGolfCoursesOrthophotosDataset(
        val_images,
        val_masks,
        train=True,
        augment=False,
        mode=args.danish_mode,
    )

    mask_transform = None
    loss_class_ids = None
    if args.danish_mode == "mapped":
        mask_transform = lambda mask: ignore_classes_except(
            remap_danish_mask(mask),
            DANISH_MAPPED_LOSS_CLASSES,
        )
        loss_class_ids = DANISH_MAPPED_LOSS_CLASSES

    return (
        train_dataset,
        val_dataset,
        train_masks,
        DanishGolfCoursesOrthophotosDataset.IGNORE_INDEX,
        mask_transform,
        loss_class_ids,
        f"danish/{args.danish_mode}",
    )


def load_finetune_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Finetune checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)


def main(default_model: str | None = None) -> None:
    args = parse_args(default_model=default_model)

    if not 0.0 <= args.ce_lambda <= 1.0:
        raise ValueError("--ce-lambda must be between 0 and 1.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    (
        train_dataset,
        val_dataset,
        train_masks,
        ignore_index,
        mask_transform,
        loss_class_ids,
        dataset_label,
    ) = build_datasets(args)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(args).to(device)
    if args.training_mode == "finetune":
        load_finetune_checkpoint(model, args.finetune_from, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params:,}")

    class_weights = compute_class_weights(
        train_masks,
        num_classes=args.num_classes,
        ignore_index=ignore_index,
        transform=mask_transform,
    ).to(device)
    criterion = CombinedCrossEntropyLovaszLoss(
        ce_lambda=args.ce_lambda,
        class_weights=class_weights,
        ignore_index=ignore_index,
    )
    optimizer = build_optimizer(
        model,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_iters=args.lr_warmup_iters,
            warmup_ratio=args.lr_warmup_ratio,
            min_lr=args.min_lr,
        )

    best_val_loss = float("inf")
    last_path = args.checkpoint_dir / f"last_{args.run_name}.pth"
    best_path = args.checkpoint_dir / f"best_{args.run_name}.pth"

    print(f"Device: {device}")
    print(f"Training mode: {args.training_mode}")
    print(f"Dataset: {dataset_label}")
    print(f"Model: {args.model}")
    if args.model == "segformer":
        print(f"Model variant: {args.variant}")
    if args.dataset == "golf":
        eval_ranges = ", ".join(f"{start:02d}-{end:02d}" for start, end in EVAL_HOLE_ID_RANGES)
        print(f"Eval hole IDs: {eval_ranges}")
    else:
        print(f"Validation fraction: {args.val_fraction:.3f}")
        if args.danish_mode == "mapped":
            print(f"Loss classes: {sorted(DANISH_MAPPED_LOSS_CLASSES)}")
            print("Background and unavailable mapped classes are ignored in the loss.")
    if args.training_mode == "finetune":
        print(f"Finetune checkpoint: {args.finetune_from}")
    print(f"Num classes: {args.num_classes}")
    print(f"CE lambda: {args.ce_lambda:.3f}")
    print(
        "Optimizer: AdamW | "
        f"encoder_lr={args.encoder_lr:g} | decoder_lr={args.decoder_lr:g} | "
        f"weight_decay={args.weight_decay:g}"
    )
    if scheduler is None:
        print("LR scheduler: none")
    else:
        effective_warmup_iters = min(args.lr_warmup_iters, max(0, total_steps - 1))
        print(
            "LR scheduler: cosine "
            f"(steps={total_steps}, warmup_iters={effective_warmup_iters}, "
            f"warmup_ratio={args.lr_warmup_ratio:g}, min_lr={args.min_lr:g})"
        )
    print(f"Class weights: {[round(weight, 4) for weight in class_weights.tolist()]}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            ignore_index=ignore_index,
            loss_class_ids=loss_class_ids,
        )
        val_loss, val_miou = evaluate(
            model,
            val_loader,
            criterion,
            device,
            ignore_index=ignore_index,
            loss_class_ids=loss_class_ids,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_miou,
            "args": vars(args),
        }
        torch.save(checkpoint, last_path)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(checkpoint, best_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_miou:.4f}{' | best' if is_best else ''}"
        )

    print(f"Saved last checkpoint: {last_path}")
    print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
