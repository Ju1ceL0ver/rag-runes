from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rune_diffusion.gan import ConditionalGenerator
from rune_diffusion.utils import save_image_grid, set_seed, tensor_to_pil, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rune samples from a trained class-conditional GAN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--class-name", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "generated_runes_gan")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint["dataset_metadata"]
    config = checkpoint["config"]
    class_names: list[str] = metadata["class_names"]
    if args.class_name not in class_names:
        raise ValueError(f"class-name must be one of: {', '.join(class_names)}")

    generator = ConditionalGenerator(
        num_classes=metadata["num_classes"],
        latent_dim=config["latent_dim"],
        base_channels=config["g_base_channels"],
    ).to(device)
    generator.load_state_dict(checkpoint.get("ema_generator_state") or checkpoint["generator_state"])
    generator.eval()

    class_idx = class_names.index(args.class_name)
    class_labels = torch.full((args.num_samples,), class_idx, device=device, dtype=torch.long)
    noise = torch.randn(args.num_samples, config["latent_dim"], device=device)
    samples = generator(noise, class_labels)

    output_dir = args.output_dir / f"class_{args.class_name}_{timestamp_string()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image_grid(samples.cpu(), output_dir / "grid.png", rows=1, cols=args.num_samples)
    for index, sample in enumerate(samples):
        tensor_to_pil(sample.cpu()).save(output_dir / f"{args.class_name}_{index:02d}.png")
    print({"output_dir": str(output_dir.resolve())})


if __name__ == "__main__":
    main()
