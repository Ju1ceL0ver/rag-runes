from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rune_diffusion.diffusion import GaussianDiffusion
from rune_diffusion.model import ConditionalUNet
from rune_diffusion.utils import save_image_grid, set_seed, tensor_to_pil, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rune samples from a trained class-conditioned diffusion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--class-name", type=str, required=True)
    parser.add_argument("--artist-name", type=str, default="random")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "generated_runes")
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
    artist_names: list[str] = metadata.get("artist_names", [])

    if args.class_name not in class_names:
        available = ", ".join(class_names)
        raise ValueError(f"class-name must be one of: {available}")

    state_dict = checkpoint.get("ema_state") or checkpoint["model_state"]
    expects_artist_condition = "artist_embedding.weight" in state_dict
    model = ConditionalUNet(
        num_classes=metadata["num_classes"],
        num_artists=metadata.get("num_artists", 0) if expects_artist_condition else 0,
        base_channels=config["base_channels"],
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=config["timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
    ).to(device)

    class_idx = class_names.index(args.class_name)
    class_labels = torch.full((args.num_samples,), class_idx, device=device, dtype=torch.long)
    artist_labels = None
    if artist_names:
        if args.artist_name == "random":
            artist_labels = torch.randint(0, len(artist_names), (args.num_samples,), device=device, dtype=torch.long)
        else:
            if args.artist_name not in artist_names:
                available = ", ".join(artist_names)
                raise ValueError(f"artist-name must be one of: random, {available}")
            artist_idx = artist_names.index(args.artist_name)
            artist_labels = torch.full((args.num_samples,), artist_idx, device=device, dtype=torch.long)

    samples = diffusion.ddim_sample(
        model=model,
        shape=(args.num_samples, 1, config["image_size"], config["image_size"]),
        class_labels=class_labels,
        artist_labels=artist_labels,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        device=device,
    )

    output_dir = args.output_dir / f"class_{args.class_name}_{timestamp_string()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image_grid(samples, output_dir / "grid.png", rows=1, cols=args.num_samples)

    for index, sample in enumerate(samples):
        tensor_to_pil(sample).save(output_dir / f"{args.class_name}_{index:02d}.png")

    print(
        {
            "class_name": args.class_name,
            "artist_name": args.artist_name,
            "num_samples": args.num_samples,
            "output_dir": str(output_dir.resolve()),
        }
    )


if __name__ == "__main__":
    main()
