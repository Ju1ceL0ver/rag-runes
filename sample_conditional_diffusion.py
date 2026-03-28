from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rune_diffusion.dataset import sample_reference_images
from rune_diffusion.diffusion import GaussianDiffusion
from rune_diffusion.model import ConditionalUNet
from rune_diffusion.utils import (
    save_image_grid,
    save_image_grid_pair,
    select_device,
    set_seed,
    tensor_to_pil,
    timestamp_string,
)


@dataclass(slots=True)
class DiffusionSamplingConfig:
    checkpoint: Path | None = None
    class_name: str | None = None
    artist_name: str = "random"
    num_samples: int = 16
    guidance_scale: float = 4.0
    steps: int = 50
    seed: int = 123
    device: str = "auto"
    output_dir: Path = Path("artifacts") / "generated_runes"
    comparison_grid_side: int = 4


def generate(config: DiffusionSamplingConfig) -> dict[str, Any]:
    if config.checkpoint is None:
        raise ValueError("DiffusionSamplingConfig.checkpoint must point to a trained checkpoint")

    set_seed(config.seed)
    device = select_device(config.device)

    checkpoint = torch.load(config.checkpoint, map_location=device)
    metadata = checkpoint["dataset_metadata"]
    config_payload = checkpoint["config"]
    class_names: list[str] = metadata["class_names"]
    artist_names: list[str] = metadata.get("artist_names", [])

    if config.class_name and config.class_name not in class_names:
        available = ", ".join(class_names)
        raise ValueError(f"class_name must be one of: {available}")
    if config.artist_name != "random" and config.artist_name not in artist_names:
        available = ", ".join(artist_names)
        raise ValueError(f"artist_name must be one of: random, {available}")

    state_dict = checkpoint.get("ema_state") or checkpoint["model_state"]
    expects_artist_condition = "artist_embedding.weight" in state_dict
    model = ConditionalUNet(
        num_classes=metadata["num_classes"],
        num_artists=metadata.get("num_artists", 0) if expects_artist_condition else 0,
        base_channels=config_payload["base_channels"],
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=config_payload["timesteps"],
        beta_start=config_payload["beta_start"],
        beta_end=config_payload["beta_end"],
    ).to(device)

    output_tag = config.class_name or "random"
    output_dir = config.output_dir / f"class_{output_tag}_{timestamp_string()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.class_name:
        class_idx = class_names.index(config.class_name)
        class_labels = torch.full((config.num_samples,), class_idx, device=device, dtype=torch.long)
        artist_labels = None
        if artist_names:
            if config.artist_name == "random":
                artist_labels = torch.randint(0, len(artist_names), (config.num_samples,), device=device, dtype=torch.long)
            else:
                artist_idx = artist_names.index(config.artist_name)
                artist_labels = torch.full((config.num_samples,), artist_idx, device=device, dtype=torch.long)

        samples = diffusion.ddim_sample(
            model=model,
            shape=(config.num_samples, 1, config_payload["image_size"], config_payload["image_size"]),
            class_labels=class_labels,
            artist_labels=artist_labels,
            guidance_scale=config.guidance_scale,
            steps=config.steps,
            device=device,
        )
        save_image_grid(samples, output_dir / "grid.png", rows=1, cols=config.num_samples)

        for index, sample in enumerate(samples):
            tensor_to_pil(sample).save(output_dir / f"{config.class_name}_{index:02d}.png")

    comparison_count = config.comparison_grid_side * config.comparison_grid_side
    comparison_class_labels = torch.randint(
        low=0,
        high=metadata["num_classes"],
        size=(comparison_count,),
        device=device,
        dtype=torch.long,
    )
    comparison_artist_labels = None
    comparison_artist_indices: list[int] | None = None
    if artist_names:
        if config.artist_name == "random":
            comparison_artist_labels = torch.randint(
                0,
                len(artist_names),
                (comparison_count,),
                device=device,
                dtype=torch.long,
            )
        else:
            artist_idx = artist_names.index(config.artist_name)
            comparison_artist_labels = torch.full(
                (comparison_count,),
                artist_idx,
                device=device,
                dtype=torch.long,
            )
        comparison_artist_indices = comparison_artist_labels.cpu().tolist()

    comparison_samples = diffusion.ddim_sample(
        model=model,
        shape=(comparison_count, 1, config_payload["image_size"], config_payload["image_size"]),
        class_labels=comparison_class_labels,
        artist_labels=comparison_artist_labels,
        guidance_scale=config.guidance_scale,
        steps=config.steps,
        device=device,
    )
    reference_samples = sample_reference_images(
        data_dir=Path(config_payload["data_dir"]),
        image_size=int(config_payload["image_size"]),
        class_indices=comparison_class_labels.cpu().tolist(),
        artist_indices=comparison_artist_indices,
        seed=config.seed,
    )
    save_image_grid(
        comparison_samples,
        output_dir / "random_generated_grid_4x4.png",
        rows=config.comparison_grid_side,
        cols=config.comparison_grid_side,
    )
    save_image_grid(
        reference_samples,
        output_dir / "matching_real_grid_4x4.png",
        rows=config.comparison_grid_side,
        cols=config.comparison_grid_side,
    )
    save_image_grid_pair(
        comparison_samples,
        reference_samples,
        output_dir / "generated_vs_real_4x4.png",
        rows=config.comparison_grid_side,
        cols=config.comparison_grid_side,
    )

    result = {
        "class_name": config.class_name,
        "artist_name": config.artist_name,
        "num_samples": config.num_samples,
        "output_dir": str(output_dir.resolve()),
        "comparison_grid_path": str((output_dir / "generated_vs_real_4x4.png").resolve()),
        "comparison_classes": [class_names[index] for index in comparison_class_labels.cpu().tolist()],
    }
    print(result)
    return result


def main(config: DiffusionSamplingConfig | None = None) -> dict[str, Any]:
    return generate(config or DiffusionSamplingConfig())


if __name__ == "__main__":
    main()
