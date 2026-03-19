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


def save_image_grid(
    images: torch.Tensor,
    path: Path,
    rows: int,
    cols: int | None = None,
    padding: int = 4,
) -> None:
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

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
