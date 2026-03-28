from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device must be one of: auto, cuda, cpu")


def timestamp_string() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.dim() != 3 or image.shape[0] != 1:
        raise ValueError("Expected image tensor with shape [1, H, W]")
    array = ((image.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).round().byte().squeeze(0).numpy()
    return Image.fromarray(array, mode="L")


def build_image_grid(
    images: torch.Tensor,
    rows: int,
    cols: int | None = None,
    padding: int = 4,
) -> Image.Image:
    if images.dim() != 4:
        raise ValueError("Expected image tensor with shape [N, C, H, W]")
    if cols is None:
        cols = math.ceil(images.shape[0] / rows)

    _, _, height, width = images.shape
    canvas = Image.new(
        "L",
        (
            cols * width + padding * (cols + 1),
            rows * height + padding * (rows + 1),
        ),
        color=0,
    )

    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        pil_image = tensor_to_pil(image)
        x = padding + col * (width + padding)
        y = padding + row * (height + padding)
        canvas.paste(pil_image, (x, y))

    return canvas


def save_image_grid(
    images: torch.Tensor,
    path: Path,
    rows: int,
    cols: int | None = None,
    padding: int = 4,
) -> None:
    canvas = build_image_grid(images, rows=rows, cols=cols, padding=padding)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_image_grid_pair(
    left_images: torch.Tensor,
    right_images: torch.Tensor,
    path: Path,
    rows: int,
    cols: int | None = None,
    padding: int = 4,
    gap: int = 12,
) -> None:
    left_canvas = build_image_grid(left_images, rows=rows, cols=cols, padding=padding)
    right_canvas = build_image_grid(right_images, rows=rows, cols=cols, padding=padding)

    canvas = Image.new(
        "L",
        (left_canvas.width + gap + right_canvas.width, max(left_canvas.height, right_canvas.height)),
        color=0,
    )
    canvas.paste(left_canvas, (0, 0))
    canvas.paste(right_canvas, (left_canvas.width + gap, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
