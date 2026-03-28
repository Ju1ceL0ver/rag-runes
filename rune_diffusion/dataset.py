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


def _sorted_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: _numeric_sort_key(path.name),
    )


def _has_direct_images(root: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in root.iterdir()
    )


def _has_subdirectories(root: Path) -> bool:
    return any(path.is_dir() for path in root.iterdir())


def infer_dataset_layout(data_dir: Path) -> str:
    top_level_dirs = _sorted_dirs(data_dir)
    if not top_level_dirs:
        raise FileNotFoundError(f"No class or artist folders found in {data_dir}")

    has_direct_images = any(_has_direct_images(path) for path in top_level_dirs)
    has_nested_dirs = any(_has_subdirectories(path) for path in top_level_dirs)

    if has_direct_images:
        return "flat_class"
    if has_nested_dirs:
        return "artist_class"
    raise RuntimeError(f"Could not infer dataset layout for {data_dir}")


def discover_flat_classes(data_dir: Path) -> list[Path]:
    class_dirs = _sorted_dirs(data_dir)
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in {data_dir}")
    return class_dirs


def discover_common_classes(data_dir: Path) -> tuple[list[Path], list[str]]:
    artist_dirs = _sorted_dirs(data_dir)
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
    grouped_records: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    layout = infer_dataset_layout(data_dir)

    if layout == "flat_class":
        class_dirs = discover_flat_classes(data_dir)
        class_names = [path.name for path in class_dirs]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        artist_names = [data_dir.name]
        artist_idx = 0

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = class_to_idx[class_name]
            for image_path in sorted(class_dir.iterdir(), key=lambda path: path.name):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    record = {
                        "path": image_path,
                        "class_idx": class_idx,
                        "artist_idx": artist_idx,
                    }
                    grouped_records[(class_idx, artist_idx)].append(record)
                    records.append(record)
    else:
        artist_dirs, class_names = discover_common_classes(data_dir)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        artist_names = [path.name for path in artist_dirs]
        artist_to_idx = {name: idx for idx, name in enumerate(artist_names)}

        for artist_dir in artist_dirs:
            artist_idx = artist_to_idx[artist_dir.name]
            for class_name in class_names:
                class_dir = artist_dir / class_name
                class_idx = class_to_idx[class_name]
                for image_path in sorted(class_dir.iterdir(), key=lambda path: path.name):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        record = {
                            "path": image_path,
                            "class_idx": class_idx,
                            "artist_idx": artist_idx,
                        }
                        grouped_records[(class_idx, artist_idx)].append(record)
                        records.append(record)

    if not records:
        raise RuntimeError(f"No PNG images found in {data_dir}")

    records.sort(
        key=lambda item: (
            int(item["class_idx"]),
            int(item["artist_idx"]),
            Path(item["path"]).name,
        )
    )

    return {
        "records": records,
        "grouped_records": grouped_records,
        "class_names": class_names,
        "artist_names": artist_names,
        "layout": layout,
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
        "dataset_layout": dataset_info["layout"],
        "train_size": len(train_records),
        "val_size": len(val_records),
        "image_size": image_size,
    }
    return RuneDataset(train_records, image_size), RuneDataset(val_records, image_size), metadata


def sample_reference_images(
    data_dir: Path,
    image_size: int,
    class_indices: list[int],
    seed: int,
    artist_indices: list[int] | None = None,
) -> torch.Tensor:
    if artist_indices is not None and len(artist_indices) != len(class_indices):
        raise ValueError("artist_indices must have the same length as class_indices")

    dataset_info = build_dataset_records(data_dir)
    grouped_records = dataset_info["grouped_records"]
    records_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in dataset_info["records"]:
        records_by_class[int(record["class_idx"])].append(record)

    rng = random.Random(seed)
    images: list[torch.Tensor] = []
    for index, class_idx in enumerate(class_indices):
        candidates: list[dict[str, Any]] = []
        if artist_indices is not None:
            artist_idx = artist_indices[index]
            candidates = list(grouped_records.get((class_idx, artist_idx), []))
        if not candidates:
            candidates = records_by_class.get(class_idx, [])
        if not candidates:
            raise RuntimeError(f"No reference image found for class index {class_idx}")
        chosen = rng.choice(candidates)
        images.append(load_grayscale_image(Path(chosen["path"]), image_size))

    return torch.stack(images, dim=0)

