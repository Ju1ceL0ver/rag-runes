from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rune_diffusion.dataset import sample_reference_images
from rune_diffusion.gan import ConditionalGenerator
from rune_diffusion.utils import (
    save_image_grid,
    save_image_grid_pair,
    select_device,
    set_seed,
    tensor_to_pil,
    timestamp_string,
)


@dataclass(slots=True)
class GanSamplingConfig:
    checkpoint: Path | None = Path(r'C:\Users\Juice_Lover\rag-runes\artifacts\conditional_gan\20260319_155611\checkpoints\last.pt')
    class_name: str | None = '3'
    num_samples: int = 16
    seed: int = 123
    device: str = "auto"
    output_dir: Path = Path("artifacts") / "generated_runes_gan"
    comparison_grid_side: int = 4


def generate(config: GanSamplingConfig) -> dict[str, Any]:
    if config.checkpoint is None:
        raise ValueError("GanSamplingConfig.checkpoint must point to a trained checkpoint")

    set_seed(config.seed)
    device = select_device(config.device)

    checkpoint = torch.load(config.checkpoint, map_location=device)
    metadata = checkpoint["dataset_metadata"]
    config_payload = checkpoint["config"]
    class_names: list[str] = metadata["class_names"]
    if config.class_name and config.class_name not in class_names:
        raise ValueError(f"class_name must be one of: {', '.join(class_names)}")

    generator = ConditionalGenerator(
        num_classes=metadata["num_classes"],
        latent_dim=config_payload["latent_dim"],
        base_channels=config_payload["g_base_channels"],
    ).to(device)
    generator.load_state_dict(checkpoint.get("ema_generator_state") or checkpoint["generator_state"])
    generator.eval()

    output_tag = config.class_name or "random"
    output_dir = config.output_dir / f"class_{output_tag}_{timestamp_string()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.class_name:
        class_idx = class_names.index(config.class_name)
        class_labels = torch.full((config.num_samples,), class_idx, device=device, dtype=torch.long)
        noise = torch.randn(config.num_samples, config_payload["latent_dim"], device=device)
        samples = generator(noise, class_labels)
        save_image_grid(samples.cpu(), output_dir / "grid.png", rows=1, cols=config.num_samples)
        for index, sample in enumerate(samples):
            tensor_to_pil(sample.cpu()).save(output_dir / f"{config.class_name}_{index:02d}.png")

    comparison_count = config.comparison_grid_side * config.comparison_grid_side
    comparison_labels = torch.randint(
        low=0,
        high=metadata["num_classes"],
        size=(comparison_count,),
        device=device,
        dtype=torch.long,
    )
    comparison_noise = torch.randn(comparison_count, config_payload["latent_dim"], device=device)
    comparison_samples = generator(comparison_noise, comparison_labels).cpu()
    reference_samples = sample_reference_images(
        data_dir=Path(config_payload["data_dir"]),
        image_size=int(config_payload["image_size"]),
        class_indices=comparison_labels.cpu().tolist(),
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
        "num_samples": config.num_samples,
        "output_dir": str(output_dir.resolve()),
        "comparison_grid_path": str((output_dir / "generated_vs_real_4x4.png").resolve()),
        "comparison_classes": [class_names[index] for index in comparison_labels.cpu().tolist()],
    }
    print(result)
    return result


def main(config: GanSamplingConfig | None = None) -> dict[str, Any]:
    return generate(config or GanSamplingConfig())


if __name__ == "__main__":
    main()
