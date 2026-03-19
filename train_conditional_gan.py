from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from rune_diffusion.dataset import build_datasets
from rune_diffusion.gan import ConditionalGenerator, ProjectionDiscriminator
from rune_diffusion.utils import count_parameters, save_image_grid, save_json, set_seed, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a class-conditional GAN for rune generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("runes"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "conditional_gan")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--g-base-channels", type=int, default=64)
    parser.add_argument("--d-base-channels", type=int, default=64)
    parser.add_argument("--g-lr", type=float, default=2e-4)
    parser.add_argument("--d-lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    run_dir = args.output_dir / timestamp_string()
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, _val_dataset, metadata = build_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        val_fraction=0.0,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    generator = ConditionalGenerator(
        num_classes=metadata["num_classes"],
        latent_dim=args.latent_dim,
        base_channels=args.g_base_channels,
    ).to(device)
    discriminator = ProjectionDiscriminator(
        num_classes=metadata["num_classes"],
        base_channels=args.d_base_channels,
    ).to(device)
    ema_generator = copy.deepcopy(generator).to(device)
    ema_generator.eval()
    for parameter in ema_generator.parameters():
        parameter.requires_grad_(False)

    g_optimizer = Adam(generator.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    d_optimizer = Adam(discriminator.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(4, args.latent_dim, device=device)
    fixed_labels = torch.arange(min(metadata["num_classes"], 4), device=device, dtype=torch.long)
    if fixed_labels.shape[0] < 4:
        fixed_labels = fixed_labels.repeat(4)[:4]

    config = {
        "data_dir": str(args.data_dir.resolve()),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "g_base_channels": args.g_base_channels,
        "d_base_channels": args.d_base_channels,
        "g_lr": args.g_lr,
        "d_lr": args.d_lr,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
        "device": str(device),
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
    }
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "dataset_metadata.json", metadata)

    history: list[dict[str, float | int]] = []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        seen_items = 0

        for real_images, class_ids, _artist_ids in train_loader:
            real_images = real_images.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            batch_size = real_images.shape[0]

            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise, class_ids)

            d_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images, class_ids)
            fake_logits = discriminator(fake_images.detach(), class_ids)
            d_loss = torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise, class_ids)
            g_loss = -discriminator(fake_images, class_ids).mean()
            g_loss.backward()
            g_optimizer.step()
            update_ema(ema_generator, generator, args.ema_decay)

            epoch_d_loss += d_loss.item() * batch_size
            epoch_g_loss += g_loss.item() * batch_size
            seen_items += batch_size

        mean_d_loss = epoch_d_loss / max(seen_items, 1)
        mean_g_loss = epoch_g_loss / max(seen_items, 1)
        history.append(
            {
                "epoch": epoch,
                "g_loss": mean_g_loss,
                "d_loss": mean_d_loss,
            }
        )
        save_json(run_dir / "history.json", {"history": history})

        with torch.no_grad():
            preview = ema_generator(fixed_noise, fixed_labels).cpu()
        save_image_grid(preview, samples_dir / f"epoch_{epoch:03d}.png", rows=1, cols=4)

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint = {
                "generator_state": generator.state_dict(),
                "ema_generator_state": ema_generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "g_optimizer_state": g_optimizer.state_dict(),
                "d_optimizer_state": d_optimizer.state_dict(),
                "history": history,
                "dataset_metadata": metadata,
                "config": config,
                "epoch": epoch,
            }
            torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch:03d}.pt")
            torch.save(checkpoint, checkpoints_dir / "last.pt")

        print(
            f"epoch={epoch:03d} "
            f"g_loss={mean_g_loss:.6f} "
            f"d_loss={mean_d_loss:.6f}"
        )

    summary = {
        "elapsed_minutes": (time.time() - train_start) / 60.0,
        "run_dir": str(run_dir.resolve()),
    }
    save_json(run_dir / "training_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

