from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.nn import functional as F


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embedding = nn.Embedding(num_classes, num_features * 2)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma, beta = self.embedding(labels).chunk(2, dim=1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta = beta.view(-1, x.shape[1], 1, 1)
        return out * (1.0 + gamma) + beta


class GenBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(x, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        out = self.cbn1(x, labels)
        out = F.relu(out, inplace=False)
        out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = self.cbn2(out, labels)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        return out + residual


class RuneGeneratorModel(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, base_channels: int) -> None:
        super().__init__()
        channels = [
            base_channels * 8,
            base_channels * 4,
            base_channels * 2,
            base_channels,
            base_channels // 2,
        ]
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, channels[0] * 4 * 4)
        self.block1 = GenBlock(channels[0], channels[1], num_classes)
        self.block2 = GenBlock(channels[1], channels[2], num_classes)
        self.block3 = GenBlock(channels[2], channels[3], num_classes)
        self.block4 = GenBlock(channels[3], channels[4], num_classes)
        self.bn = nn.BatchNorm2d(channels[4])
        self.output = nn.Conv2d(channels[4], 1, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        out = out.view(z.shape[0], -1, 4, 4)
        out = self.block1(out, labels)
        out = self.block2(out, labels)
        out = self.block3(out, labels)
        out = self.block4(out, labels)
        out = self.bn(out)
        out = F.relu(out, inplace=False)
        out = self.output(out)
        return torch.tanh(out)


@dataclass(slots=True)
class RuneMaskSample:
    class_id: int
    class_name: str
    seed: int
    raw_image: Image.Image
    mask_image: Image.Image
    score: float
    occupancy: float


def _tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = ((array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return array


def _otsu_threshold(values: np.ndarray) -> int:
    histogram, _ = np.histogram(values.ravel(), bins=256, range=(0, 256))
    total = values.size
    sum_total = np.dot(np.arange(256), histogram)
    sum_background = 0.0
    weight_background = 0.0
    variance_max = -1.0
    threshold = 127

    for idx in range(256):
        weight_background += histogram[idx]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += idx * histogram[idx]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if variance_between > variance_max:
            variance_max = variance_between
            threshold = idx
    return threshold


def _mask_score(mask: np.ndarray, grayscale: np.ndarray) -> tuple[float, float]:
    occupancy = float(mask.mean())
    if occupancy <= 0.01 or occupancy >= 0.70:
        return -10.0, occupancy
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return -10.0, occupancy
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    aspect = min(width, height) / max(width, height)
    contrast = float(grayscale.std()) / 255.0
    score = (
        -abs(occupancy - 0.18) * 4.0
        + contrast * 0.8
        + aspect * 0.5
    )
    return score, occupancy


def _extract_mask(grayscale: np.ndarray) -> tuple[np.ndarray, float, float]:
    threshold = _otsu_threshold(grayscale)
    candidates = [
        grayscale < threshold,
        grayscale > threshold,
        grayscale < max(threshold - 12, 32),
        grayscale > min(threshold + 12, 223),
    ]

    best_mask: np.ndarray | None = None
    best_score = -999.0
    best_occupancy = 0.0
    for candidate in candidates:
        score, occupancy = _mask_score(candidate.astype(np.uint8), grayscale)
        if score > best_score:
            best_mask = candidate.astype(np.uint8)
            best_score = score
            best_occupancy = occupancy

    assert best_mask is not None
    mask_image = Image.fromarray(best_mask * 255, mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
    mask_image = mask_image.filter(ImageFilter.MinFilter(3))
    soft = mask_image.filter(ImageFilter.GaussianBlur(radius=1.2))
    final_mask = (np.asarray(soft, dtype=np.uint8) > 92).astype(np.uint8)
    score, occupancy = _mask_score(final_mask, grayscale)
    return final_mask * 255, score, occupancy


class RuneMaskGenerator:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        use_ema: bool = True,
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config") or {}
        dataset_metadata = checkpoint.get("dataset_metadata") or {}
        latent_dim = int(config.get("latent_dim", 128))
        base_channels = int(config.get("g_base_channels", 64))
        num_classes = int(dataset_metadata.get("num_classes", 0))
        class_names = list(dataset_metadata.get("class_names") or [])
        if num_classes <= 0 or len(class_names) != num_classes:
            raise RuntimeError(
                f"Invalid dataset metadata in checkpoint '{checkpoint_path}'."
            )

        self.class_names = class_names
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.image_size = int(config.get("image_size", 64))
        self.model = RuneGeneratorModel(
            latent_dim=latent_dim,
            num_classes=num_classes,
            base_channels=base_channels,
        )
        state_key = "ema_generator_state" if use_ema else "generator_state"
        state_dict = checkpoint.get(state_key)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Missing '{state_key}' in checkpoint '{checkpoint_path}'.")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def class_to_id(self, class_id: int | None = None, class_name: str | None = None) -> int:
        if class_id is not None:
            if class_id < 0 or class_id >= self.num_classes:
                raise ValueError(
                    f"class_id={class_id} is out of range 0..{self.num_classes - 1}"
                )
            return class_id
        if class_name is None:
            raise ValueError("Either class_id or class_name must be provided.")
        if class_name not in self.class_names:
            raise ValueError(f"Unknown class_name='{class_name}'.")
        return self.class_names.index(class_name)

    def generate(
        self,
        class_id: int | None = None,
        class_name: str | None = None,
        seed: int = 42,
        candidates: int = 6,
    ) -> RuneMaskSample:
        target_class_id = self.class_to_id(class_id=class_id, class_name=class_name)
        label_name = self.class_names[target_class_id]
        batch_size = max(1, candidates)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        latents = torch.randn(batch_size, self.latent_dim, generator=generator)
        labels = torch.full((batch_size,), target_class_id, dtype=torch.long)
        with torch.inference_mode():
            outputs = self.model(
                latents.to(self.device),
                labels.to(self.device),
            )

        images = _tensor_to_uint8(outputs[:, 0])
        best: RuneMaskSample | None = None
        for idx, image in enumerate(images):
            mask, score, occupancy = _extract_mask(image)
            sample = RuneMaskSample(
                class_id=target_class_id,
                class_name=label_name,
                seed=seed + idx,
                raw_image=Image.fromarray(image, mode="L"),
                mask_image=Image.fromarray(mask, mode="L"),
                score=score,
                occupancy=occupancy,
            )
            if best is None or sample.score > best.score:
                best = sample

        assert best is not None
        return best
