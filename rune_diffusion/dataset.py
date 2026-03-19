from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png"}


def _numeric_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value


def discover_common_classes(data_dir: Path) -> tuple[list[Path], list[str]]:
    artist_dirs = sorted(
        [path for path in data_dir.iterdir() if path.is_dir()],
        key=lambda path: _numeric_sort_key(path.name),
    )
    if not artist_dirs:
        raise FileNotFoundError(f"No artist folders found in {data_dir}")

    common_classes: set[str] | None = None
    for artist_dir in artist_dirs:
        classes = {path.name for path in artist_dir.iterdir() if path.is_dir()}
        common_classes = classes if common_classes is None else common_classes & classes

    if not common_classes:
        raise RuntimeError(f"No common class folders shared by all artists in {data_dir}")

    class_names = sorted(common_classes, key=_numeric_sort_key)
    return artist_dirs, class_names


def load_grayscale_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image).convert("L")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).unsqueeze(0)


def build_dataset_records(data_dir: Path) -> dict[str, Any]:
    artist_dirs, class_names = discover_common_classes(data_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    artist_names = [path.name for path in artist_dirs]
    artist_to_idx = {name: idx for idx, name in enumerate(artist_names)}

    grouped_records: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for artist_dir in artist_dirs:
        artist_idx = artist_to_idx[artist_dir.name]
        for class_name in class_names:
            class_dir = artist_dir / class_name
            class_idx = class_to_idx[class_name]
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    grouped_records[(class_idx, artist_idx)].append(
                        {
                            "path": image_path,
                            "class_idx": class_idx,
                            "artist_idx": artist_idx,
                        }
                    )

    records: list[dict[str, Any]] = []
    for key in sorted(grouped_records):
        records.extend(grouped_records[key])

    if not records:
        raise RuntimeError(f"No PNG images found in {data_dir}")

    return {
        "records": records,
        "grouped_records": grouped_records,
        "class_names": class_names,
        "artist_names": artist_names,
    }


def split_records(
    grouped_records: dict[tuple[int, int], list[dict[str, Any]]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    rng = random.Random(seed)
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for key in sorted(grouped_records):
        items = list(grouped_records[key])
        rng.shuffle(items)
        if val_fraction == 0.0 or len(items) < 2:
            train_records.extend(items)
            continue

        val_count = max(1, int(round(len(items) * val_fraction)))
        val_count = min(val_count, len(items) - 1)
        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    return train_records, val_records


class RuneDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.records[index]
        image = load_grayscale_image(Path(item["path"]), self.image_size)
        class_idx = torch.tensor(item["class_idx"], dtype=torch.long)
        artist_idx = torch.tensor(item["artist_idx"], dtype=torch.long)
        return image, class_idx, artist_idx


def build_datasets(
    data_dir: Path,
    image_size: int,
    val_fraction: float,
    seed: int,
) -> tuple[RuneDataset, RuneDataset, dict[str, Any]]:
    dataset_info = build_dataset_records(data_dir)
    train_records, val_records = split_records(dataset_info["grouped_records"], val_fraction, seed)
    metadata = {
        "class_names": dataset_info["class_names"],
        "artist_names": dataset_info["artist_names"],
        "num_classes": len(dataset_info["class_names"]),
        "num_artists": len(dataset_info["artist_names"]),
        "train_size": len(train_records),
        "val_size": len(val_records),
        "image_size": image_size,
    }
    return RuneDataset(train_records, image_size), RuneDataset(val_records, image_size), metadata

