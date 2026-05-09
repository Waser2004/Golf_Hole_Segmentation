import argparse
import copy
import csv
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golf_hole_segmentation.training.losses import CombinedCrossEntropyLovaszLoss
from golf_hole_segmentation.training.trainer import (
    build_optimizer,
    build_datasets,
    build_model,
    compute_class_weights,
    load_finetune_checkpoint,
    mask_labels_for_loss,
)
from golf_hole_segmentation.utils.paths import CHECKPOINTS_DIR, GOLF_HOLE_MASKS_DIR, GOLF_HOLES_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two short learning-rate range tests for SegFormer: decoder first, "
            "then encoder with the decoder LR fixed. This does not save a checkpoint."
        )
    )
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
        help="Force KaggleHub to refresh the Danish dataset cache before loading data.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Maximum number of training batches to run.",
    )
    parser.add_argument("--decoder-min-lr", type=float, default=1e-7)
    parser.add_argument("--decoder-max-lr", type=float, default=1e-2)
    parser.add_argument("--encoder-min-lr", type=float, default=1e-8)
    parser.add_argument("--encoder-max-lr", type=float, default=1e-3)
    parser.add_argument(
        "--fixed-decoder-lr",
        type=float,
        help="Decoder LR to use during the encoder sweep. Defaults to the decoder sweep recommendation.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
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
    parser.add_argument("--variant", default="B2", choices=["B0", "B1", "B2", "B3", "B4", "B5"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--finetune-from",
        type=Path,
        help="Optional SegFormer checkpoint to initialize before running the LR range test.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=CHECKPOINTS_DIR / "segformer_lr_range_test.csv",
        help="CSV file for phase, step, lr, raw_loss, smoothed_loss, and fixed_decoder_lr.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path to save a PNG plot of smoothed loss vs. learning rate.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.98,
        help="Exponential smoothing factor for the displayed loss.",
    )
    parser.add_argument(
        "--diverge-factor",
        type=float,
        default=4.0,
        help="Stop early once smoothed loss is this many times higher than the best loss.",
    )

    args = parser.parse_args()
    args.model = "segformer"
    args.training_mode = "finetune" if args.finetune_from is not None else "train"
    args.depth = 5
    args.base_channels = 64
    if args.num_classes is None:
        args.num_classes = 6 if args.dataset == "danish" and args.danish_mode == "original" else 13
    if args.steps <= 1:
        raise ValueError("--steps must be greater than 1.")
    if args.decoder_min_lr <= 0.0 or args.decoder_max_lr <= 0.0:
        raise ValueError("--decoder-min-lr and --decoder-max-lr must be positive.")
    if args.encoder_min_lr <= 0.0 or args.encoder_max_lr <= 0.0:
        raise ValueError("--encoder-min-lr and --encoder-max-lr must be positive.")
    if args.decoder_min_lr >= args.decoder_max_lr:
        raise ValueError("--decoder-min-lr must be lower than --decoder-max-lr.")
    if args.encoder_min_lr >= args.encoder_max_lr:
        raise ValueError("--encoder-min-lr must be lower than --encoder-max-lr.")
    if args.fixed_decoder_lr is not None and args.fixed_decoder_lr <= 0.0:
        raise ValueError("--fixed-decoder-lr must be positive.")
    if not 0.0 <= args.ce_lambda <= 1.0:
        raise ValueError("--ce-lambda must be between 0 and 1.")
    if not 0.0 <= args.smoothing < 1.0:
        raise ValueError("--smoothing must be at least 0 and lower than 1.")
    if args.diverge_factor <= 1.0:
        raise ValueError("--diverge-factor must be greater than 1.")
    return args


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def learning_rate_for_step(
    min_lr: float,
    max_lr: float,
    step_index: int,
    total_steps: int,
) -> float:
    progress = step_index / max(1, total_steps - 1)
    return min_lr * (max_lr / min_lr) ** progress


def recommend_from_divergence(history: list[dict[str, float]]) -> tuple[float, float, float]:
    if not history:
        raise ValueError("Cannot choose a learning rate without LR test history.")

    diverged_rows = [row for row in history if row.get("diverged", 0.0) == 1.0]
    divergence_lr = diverged_rows[0]["lr"] if diverged_rows else history[-1]["lr"]
    return divergence_lr, divergence_lr / 10.0, divergence_lr / 3.0


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "phase",
                "step",
                "lr",
                "raw_loss",
                "smoothed_loss",
                "fixed_decoder_lr",
                "diverged",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_plot(
    path: Path,
    decoder_history: list[dict[str, float]],
    encoder_history: list[dict[str, float]],
    decoder_recommendation: tuple[float, float],
    encoder_recommendation: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    phases = [
        ("decoder", decoder_history, decoder_recommendation, axes[0]),
        ("encoder", encoder_history, encoder_recommendation, axes[1]),
    ]

    for title, history, recommendation, axis in phases:
        lrs = [row["lr"] for row in history]
        losses = [row["smoothed_loss"] for row in history]
        axis.plot(lrs, losses, marker="o", linewidth=1.5)
        axis.axvspan(
            recommendation[0],
            recommendation[1],
            color="tab:red",
            alpha=0.15,
            label=f"recommended {recommendation[0]:.1e}-{recommendation[1]:.1e}",
        )
        axis.set_xscale("log")
        axis.set_xlabel("Learning rate")
        axis.set_ylabel("Smoothed loss")
        axis.set_title(f"SegFormer {title} LR range test")
        axis.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def set_requires_grad(params: list[torch.nn.Parameter], requires_grad: bool) -> None:
    for param in params:
        param.requires_grad = requires_grad


def run_lr_sweep(
    *,
    phase: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    ignore_index: int,
    loss_class_ids: set[int] | None,
    min_lr: float,
    max_lr: float,
    total_steps: int,
    smoothing: float,
    diverge_factor: float,
    fixed_decoder_lr: float | None = None,
    swept_group_name: str | None = None,
) -> list[dict[str, float]]:
    model.train()
    history: list[dict[str, float]] = []
    smoothed_loss = 0.0
    best_loss = float("inf")

    for step_index, (images, masks) in enumerate(train_loader):
        if step_index >= total_steps:
            break

        lr = learning_rate_for_step(min_lr, max_lr, step_index, total_steps)
        if swept_group_name is None:
            set_learning_rate(optimizer, lr)
        else:
            for group in optimizer.param_groups:
                if group.get("name") == swept_group_name:
                    group["lr"] = lr
                elif group.get("name") == "decoder" and fixed_decoder_lr is not None:
                    group["lr"] = fixed_decoder_lr

        images = images.to(device)
        masks = masks.to(device=device, dtype=torch.long)
        masks = mask_labels_for_loss(masks, ignore_index=ignore_index, class_ids=loss_class_ids)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        raw_loss = float(loss.item())
        smoothed_loss = smoothing * smoothed_loss + (1.0 - smoothing) * raw_loss
        bias_corrected_loss = smoothed_loss / (1.0 - smoothing ** (step_index + 1))
        diverged = (
            not math.isfinite(raw_loss)
            or (step_index > 0 and bias_corrected_loss > diverge_factor * best_loss)
        )

        history.append(
            {
                "phase": phase,
                "step": step_index + 1,
                "lr": lr,
                "raw_loss": raw_loss,
                "smoothed_loss": bias_corrected_loss,
                "fixed_decoder_lr": fixed_decoder_lr if fixed_decoder_lr is not None else "",
                "diverged": 1.0 if diverged else 0.0,
            }
        )
        best_loss = min(best_loss, bias_corrected_loss)
        print(
            f"{phase.capitalize()} step {step_index + 1:03d}/{total_steps:03d} | "
            f"lr={lr:.3e} | loss={raw_loss:.4f} | smooth={bias_corrected_loss:.4f}"
        )

        if diverged:
            print(f"Stopping {phase} sweep because loss diverged.")
            break

    return history


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    (
        train_dataset,
        _val_dataset,
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

    model = build_model(args).to(device)
    if args.finetune_from is not None:
        load_finetune_checkpoint(model, args.finetune_from, device)

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
    original_state_dict = copy.deepcopy(model.state_dict())
    grouping_optimizer = build_optimizer(
        model,
        encoder_lr=args.encoder_min_lr,
        decoder_lr=args.decoder_min_lr,
        weight_decay=args.weight_decay,
    )
    encoder_params = list(grouping_optimizer.param_groups[0]["params"])
    decoder_params = list(grouping_optimizer.param_groups[1]["params"])

    total_steps = min(args.steps, len(train_loader))
    print(f"Device: {device}")
    print(f"Dataset: {dataset_label}")
    print(f"Model: segformer | variant={args.variant}")
    print(f"Train samples: {len(train_dataset)} | batch_size={args.batch_size}")
    print(
        "Decoder LR range: "
        f"{args.decoder_min_lr:g} -> {args.decoder_max_lr:g} | steps={total_steps}"
    )
    print(
        "Encoder LR range: "
        f"{args.encoder_min_lr:g} -> {args.encoder_max_lr:g} | steps={total_steps}"
    )

    set_requires_grad(encoder_params, False)
    set_requires_grad(decoder_params, True)
    decoder_optimizer = torch.optim.AdamW(
        [{"params": decoder_params, "lr": args.decoder_min_lr, "name": "decoder"}],
        weight_decay=args.weight_decay,
    )
    decoder_history = run_lr_sweep(
        phase="decoder",
        model=model,
        train_loader=train_loader,
        optimizer=decoder_optimizer,
        criterion=criterion,
        device=device,
        ignore_index=ignore_index,
        loss_class_ids=loss_class_ids,
        min_lr=args.decoder_min_lr,
        max_lr=args.decoder_max_lr,
        total_steps=total_steps,
        smoothing=args.smoothing,
        diverge_factor=args.diverge_factor,
    )
    decoder_divergence_lr, decoder_low_lr, decoder_high_lr = recommend_from_divergence(
        decoder_history
    )
    fixed_decoder_lr = args.fixed_decoder_lr or decoder_low_lr
    model.load_state_dict(original_state_dict)

    set_requires_grad(encoder_params, True)
    set_requires_grad(decoder_params, True)
    encoder_optimizer = build_optimizer(
        model,
        encoder_lr=args.encoder_min_lr,
        decoder_lr=fixed_decoder_lr,
        weight_decay=args.weight_decay,
    )
    encoder_history = run_lr_sweep(
        phase="encoder",
        model=model,
        train_loader=train_loader,
        optimizer=encoder_optimizer,
        criterion=criterion,
        device=device,
        ignore_index=ignore_index,
        loss_class_ids=loss_class_ids,
        min_lr=args.encoder_min_lr,
        max_lr=args.encoder_max_lr,
        total_steps=total_steps,
        smoothing=args.smoothing,
        diverge_factor=args.diverge_factor,
        fixed_decoder_lr=fixed_decoder_lr,
        swept_group_name="encoder",
    )
    encoder_divergence_lr, encoder_low_lr, encoder_high_lr = recommend_from_divergence(
        encoder_history
    )
    model.load_state_dict(original_state_dict)

    history = decoder_history + encoder_history
    write_history_csv(args.output_csv, history)

    if args.plot is not None:
        save_plot(
            args.plot,
            decoder_history,
            encoder_history,
            (decoder_low_lr, decoder_high_lr),
            (encoder_low_lr, encoder_high_lr),
        )
        print(f"Saved plot: {args.plot}")

    print(f"Saved CSV: {args.output_csv}")
    print(
        "Decoder divergence LR: "
        f"{decoder_divergence_lr:.3e} | recommended range: "
        f"{decoder_low_lr:.3e} - {decoder_high_lr:.3e}"
    )
    print(f"Fixed decoder LR used for encoder sweep: {fixed_decoder_lr:.3e}")
    print(
        "Encoder divergence LR: "
        f"{encoder_divergence_lr:.3e} | recommended range: "
        f"{encoder_low_lr:.3e} - {encoder_high_lr:.3e}"
    )
    print(
        "Use the recommended ranges as starting points, "
        "then confirm with a normal short training run."
    )


if __name__ == "__main__":
    main()
