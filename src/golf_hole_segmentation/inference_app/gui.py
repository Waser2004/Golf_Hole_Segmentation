from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageTk

from golf_hole_segmentation.inference_app.checkpoints import CheckpointInfo, discover_checkpoints
from golf_hole_segmentation.inference_app.core import (
    MODEL_OPTIONS,
    MODEL_SEGFORMER,
    MODEL_UNET,
    SEGFORMER_VARIANTS,
    InferenceRequest,
    run_inference,
)
from golf_hole_segmentation.inference_app.io_utils import save_inference_outputs
from golf_hole_segmentation.utils.paths import CHECKPOINTS_DIR

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TK_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None
    TK_IMPORT_ERROR = exc

if TYPE_CHECKING:
    import tkinter as tk_types

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


class InferenceGui:
    def __init__(self, root: "tk_types.Tk"):
        self.root = root
        self.root.title("Golf Hole Segmentation Inference")
        self.root.geometry("1320x840")
        self.root.minsize(1100, 740)

        self._checkpoint_infos: list[CheckpointInfo] = []
        self._checkpoint_labels: list[str] = []
        self._preview_refs: dict[str, ImageTk.PhotoImage] = {}

        self.model_var = tk.StringVar(value=MODEL_UNET)
        self.variant_var = tk.StringVar(value="B2")
        self.checkpoint_var = tk.StringVar()
        self.image_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "inference_outputs"))
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes_var = tk.StringVar(value="13")
        self.unsafe_load_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Select model, checkpoint, and image.")

        self._build_layout()
        self.refresh_checkpoints()
        self._on_model_change()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main, text="Inference Setup", padding=10)
        controls.pack(fill="x", pady=(0, 12))

        controls.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(controls, text="Model").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        model_box = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=list(MODEL_OPTIONS),
            state="readonly",
            width=22,
        )
        model_box.grid(row=row, column=1, sticky="w", pady=4)
        model_box.bind("<<ComboboxSelected>>", lambda _event: self._on_model_change())

        ttk.Label(controls, text="SegFormer Variant").grid(
            row=row, column=2, sticky="w", padx=(16, 8), pady=4
        )
        self.variant_box = ttk.Combobox(
            controls,
            textvariable=self.variant_var,
            values=list(SEGFORMER_VARIANTS),
            state="readonly",
            width=10,
        )
        self.variant_box.grid(row=row, column=3, sticky="w", pady=4)

        ttk.Label(controls, text="Num Classes").grid(row=row, column=4, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(controls, textvariable=self.num_classes_var, width=8).grid(
            row=row, column=5, sticky="w", pady=4
        )

        row += 1
        ttk.Label(controls, text="Checkpoint").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        self.checkpoint_box = ttk.Combobox(controls, textvariable=self.checkpoint_var, state="readonly")
        self.checkpoint_box.grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)
        self.checkpoint_box.bind("<<ComboboxSelected>>", lambda _event: self._on_checkpoint_selected())

        ttk.Button(controls, text="Refresh", command=self.refresh_checkpoints).grid(
            row=row, column=4, sticky="ew", padx=(8, 4), pady=4
        )
        ttk.Button(controls, text="Browse...", command=self._browse_checkpoint).grid(
            row=row, column=5, sticky="ew", padx=(4, 0), pady=4
        )

        row += 1
        ttk.Label(controls, text="Image").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.image_var).grid(
            row=row, column=1, columnspan=4, sticky="ew", pady=4
        )
        ttk.Button(controls, text="Browse...", command=self._browse_image).grid(
            row=row, column=5, sticky="ew", padx=(4, 0), pady=4
        )

        row += 1
        ttk.Label(controls, text="Output Folder").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.output_dir_var).grid(
            row=row, column=1, columnspan=4, sticky="ew", pady=4
        )
        ttk.Button(controls, text="Browse...", command=self._browse_output_dir).grid(
            row=row, column=5, sticky="ew", padx=(4, 0), pady=4
        )

        row += 1
        ttk.Label(controls, text="Device").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=6)
        device_values = ["cpu"]
        if torch.cuda.is_available():
            device_values.insert(0, "cuda")
        ttk.Combobox(
            controls,
            textvariable=self.device_var,
            values=device_values,
            state="readonly",
            width=10,
        ).grid(row=row, column=1, sticky="w", pady=6)

        ttk.Checkbutton(controls, text="Unsafe load", variable=self.unsafe_load_var).grid(
            row=row, column=2, sticky="w", padx=(16, 0), pady=6
        )

        self.run_button = ttk.Button(controls, text="Run Inference", command=self._run_inference_clicked)
        self.run_button.grid(row=row, column=5, sticky="ew", padx=(4, 0), pady=6)

        status = ttk.Label(main, textvariable=self.status_var, foreground="#1f4f7a")
        status.pack(fill="x", pady=(0, 8))

        previews = ttk.Frame(main)
        previews.pack(fill="both", expand=True)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        previews.columnconfigure(2, weight=1)

        self.original_preview = self._build_preview_slot(previews, 0, "Original")
        self.mask_preview = self._build_preview_slot(previews, 1, "Predicted Mask")
        self.blended_preview = self._build_preview_slot(previews, 2, "Blended")

    def _build_preview_slot(self, parent: "tk_types.Frame", column: int, title: str):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=0, column=column, sticky="nsew", padx=4)

        label = ttk.Label(frame, text="No image", anchor="center")
        label.pack(fill="both", expand=True)
        return label

    def refresh_checkpoints(self):
        self._checkpoint_infos = discover_checkpoints(CHECKPOINTS_DIR)
        self._checkpoint_labels = [info.label for info in self._checkpoint_infos]
        self.checkpoint_box["values"] = self._checkpoint_labels

        if self._checkpoint_labels:
            if self.checkpoint_var.get() not in self._checkpoint_labels:
                self.checkpoint_var.set(self._checkpoint_labels[0])
                self._on_checkpoint_selected()
            self.status_var.set(f"Found {len(self._checkpoint_labels)} checkpoint(s).")
        else:
            self.checkpoint_var.set("")
            self.status_var.set(f"No checkpoints found in {CHECKPOINTS_DIR}.")

    def _on_model_change(self):
        state = "readonly" if self.model_var.get() == MODEL_SEGFORMER else "disabled"
        self.variant_box.configure(state=state)

    def _on_checkpoint_selected(self):
        info = self._get_selected_checkpoint_info()
        if info is None:
            return

        if info.model_guess:
            self.model_var.set(info.model_guess)
            self._on_model_change()
        if info.segformer_variant_guess:
            self.variant_var.set(info.segformer_variant_guess)

    def _get_selected_checkpoint_info(self):
        selected_label = self.checkpoint_var.get()
        for info in self._checkpoint_infos:
            if info.label == selected_label:
                return info
        return None

    def _resolve_checkpoint_path(self):
        info = self._get_selected_checkpoint_info()
        if info is not None:
            return info.path

        text = self.checkpoint_var.get().strip()
        if not text:
            return None
        return Path(text)

    def _browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("Checkpoint Files", "*.pth *.pt *.ckpt"), ("All Files", "*.*")],
            initialdir=str(CHECKPOINTS_DIR),
        )
        if path:
            self.checkpoint_var.set(path)

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.image_var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output directory", mustexist=False)
        if path:
            self.output_dir_var.set(path)

    def _parse_num_classes(self):
        text = self.num_classes_var.get().strip()
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError("Num classes must be an integer.") from exc
        if value < 2:
            raise ValueError("Num classes must be >= 2.")
        return value

    def _run_inference_clicked(self):
        try:
            checkpoint_path = self._resolve_checkpoint_path()
            if checkpoint_path is None:
                raise ValueError("Select a checkpoint file.")
            checkpoint_path = checkpoint_path.expanduser().resolve()
            if not checkpoint_path.exists() or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            image_text = self.image_var.get().strip()
            if not image_text:
                raise ValueError("Select an input image.")
            image_path = Path(image_text).expanduser().resolve()
            if not image_path.exists() or not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            output_dir = Path(self.output_dir_var.get().strip()).expanduser().resolve()
            num_classes = self._parse_num_classes()
            device = torch.device(self.device_var.get())

            request = InferenceRequest(
                model_type=self.model_var.get(),
                checkpoint_path=checkpoint_path,
                image_path=image_path,
                device=device,
                num_classes=num_classes,
                segformer_variant=self.variant_var.get(),
                unsafe_load=self.unsafe_load_var.get(),
            )

            self.run_button.configure(state="disabled")
            self.status_var.set("Running inference...")
            self.root.update_idletasks()

            result = run_inference(request)
            mask_path, blended_path = save_inference_outputs(
                result=result,
                output_dir=output_dir,
                image_path=image_path,
                checkpoint_path=checkpoint_path,
            )

            self._update_previews(
                result.original_image,
                result.predicted_mask,
                result.blended_image,
            )
            self.status_var.set(f"Saved: {mask_path.name} | {blended_path.name}")

        except Exception as exc:
            messagebox.showerror("Inference failed", str(exc))
            self.status_var.set(f"Inference failed: {exc}")
        finally:
            self.run_button.configure(state="normal")

    def _update_previews(self, original: np.ndarray, mask: np.ndarray, blended: np.ndarray):
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        self._set_preview(self.original_preview, original, "original")
        self._set_preview(self.mask_preview, mask_rgb, "mask")
        self._set_preview(self.blended_preview, blended, "blended")

    def _set_preview(self, widget: ttk.Label, image_array: np.ndarray, key: str):
        image = Image.fromarray(image_array.astype(np.uint8))
        image.thumbnail((410, 410), RESAMPLE_LANCZOS)
        photo = ImageTk.PhotoImage(image)
        widget.configure(image=photo, text="")
        self._preview_refs[key] = photo


def launch_inference_gui():
    if TK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Tkinter is not available in this Python environment. "
            "Install Tk support or use a Python build that includes tkinter."
        ) from TK_IMPORT_ERROR

    root = tk.Tk()
    app = InferenceGui(root)
    _ = app
    root.mainloop()
