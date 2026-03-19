from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import time

import torch
from torch import amp
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rune_diffusion.dataset import build_datasets
from rune_diffusion.diffusion import GaussianDiffusion
from rune_diffusion.model import ConditionalUNet
from rune_diffusion.utils import count_parameters, save_image_grid, save_json, set_seed, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a class-conditioned diffusion model for rune generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("runes"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "conditional_diffusion")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--cond-drop-prob", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--preview-every", type=int, default=1)
    parser.add_argument("--preview-classes", type=int, default=4)
    parser.add_argument("--preview-samples-per-class", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
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


def evaluate(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    for images, class_ids, artist_ids in dataloader:
        images = images.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)
        artist_ids = artist_ids.to(device, non_blocking=True)
        noise = torch.randn_like(images)
        timesteps = diffusion.sample_timesteps(images.shape[0], device)
        noisy_images = diffusion.q_sample(images, timesteps, noise)
        predicted_noise = model(noisy_images, timesteps, class_ids, artist_ids)
        loss = F.mse_loss(predicted_noise, noise, reduction="mean")
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def make_preview_conditions(
    num_classes: int,
    num_artists: int,
    classes_to_show: int,
    samples_per_class: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    class_ids = torch.arange(min(num_classes, classes_to_show), device=device, dtype=torch.long)
    class_ids = class_ids.repeat_interleave(samples_per_class)
    if num_artists <= 0:
        return class_ids, None

    artist_ids: list[int] = []
    for _class_idx in range(min(num_classes, classes_to_show)):
        for sample_idx in range(samples_per_class):
            artist_ids.append(sample_idx % num_artists)
    return class_ids, torch.tensor(artist_ids, device=device, dtype=torch.long)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)
    run_dir = args.output_dir / timestamp_string() if args.resume is None else args.resume.resolve().parent.parent
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, metadata = build_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = ConditionalUNet(
        num_classes=metadata["num_classes"],
        num_artists=metadata["num_artists"],
        base_channels=args.base_channels,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    diffusion = GaussianDiffusion(
        num_timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    best_checkpoint_path: Path | None = None
    last_checkpoint_path: Path | None = None

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["weight_decay"] = args.weight_decay
        history = checkpoint.get("history", [])
        if history:
            best_entry = min(history, key=lambda item: float(item["val_loss"]))
            best_val_loss = float(best_entry["val_loss"])
            best_epoch = int(best_entry["epoch"])
        for path in sorted(checkpoints_dir.glob("best_epoch_*.pt")):
            best_checkpoint_path = path
        for path in sorted(checkpoints_dir.glob("epoch_*.pt")):
            last_checkpoint_path = path
        start_epoch = int(checkpoint["epoch"]) + 1

    config = {
        "data_dir": str(args.data_dir.resolve()),
        "resume": str(args.resume.resolve()) if args.resume is not None else None,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "cond_drop_prob": args.cond_drop_prob,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "base_channels": args.base_channels,
        "timesteps": args.timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "guidance_scale": args.guidance_scale,
        "sample_steps": args.sample_steps,
        "device": str(device),
        "parameter_count": count_parameters(model),
    }
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "dataset_metadata.json", metadata)

    fixed_labels, fixed_artists = make_preview_conditions(
        metadata["num_classes"],
        metadata["num_artists"],
        args.preview_classes,
        args.preview_samples_per_class,
        device,
    )
    fixed_noise = torch.randn(
        fixed_labels.shape[0],
        1,
        args.image_size,
        args.image_size,
        device=device,
    )

    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen_items = 0

        for images, class_ids, artist_ids in train_loader:
            images = images.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            artist_ids = artist_ids.to(device, non_blocking=True)

            drop_mask = torch.rand(class_ids.shape[0], device=device) < args.cond_drop_prob
            class_input = class_ids.clone()
            class_input[drop_mask] = model.null_class_idx
            artist_input = artist_ids.clone()
            if model.null_artist_idx is not None:
                artist_input[drop_mask] = model.null_artist_idx

            noise = torch.randn_like(images)
            timesteps = diffusion.sample_timesteps(images.shape[0], device)
            noisy_images = diffusion.q_sample(images, timesteps, noise)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                predicted_noise = model(noisy_images, timesteps, class_input, artist_input)
                loss = F.mse_loss(predicted_noise, noise, reduction="mean")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, args.ema_decay)

            batch_size = images.shape[0]
            epoch_loss += loss.item() * batch_size
            seen_items += batch_size

        train_loss = epoch_loss / max(seen_items, 1)
        val_loss = evaluate(ema_model, diffusion, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        save_json(run_dir / "history.json", {"history": history})

        checkpoint = {
            "model_state": model.state_dict(),
            "ema_state": ema_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "dataset_metadata": metadata,
            "config": config,
            "epoch": epoch,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            next_best_path = checkpoints_dir / f"best_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, next_best_path)
            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                try:
                    best_checkpoint_path.unlink()
                except OSError:
                    pass
            best_checkpoint_path = next_best_path

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            last_checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, last_checkpoint_path)

        if epoch == 1 or epoch == args.epochs or epoch % args.preview_every == 0:
            preview = diffusion.ddim_sample(
                model=ema_model,
                shape=(fixed_labels.shape[0], 1, args.image_size, args.image_size),
                class_labels=fixed_labels,
                artist_labels=fixed_artists,
                initial_noise=fixed_noise,
                guidance_scale=args.guidance_scale,
                steps=args.sample_steps,
                device=device,
            )
            save_image_grid(
                preview,
                samples_dir / f"epoch_{epoch:03d}.png",
                rows=min(metadata["num_classes"], args.preview_classes),
                cols=args.preview_samples_per_class,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"best_val={best_val_loss:.6f}"
        )

    elapsed_minutes = (time.time() - train_start) / 60.0
    if best_checkpoint_path is not None:
        try:
            shutil.copyfile(best_checkpoint_path, checkpoints_dir / "best.pt")
        except OSError:
            pass
    if last_checkpoint_path is not None:
        try:
            shutil.copyfile(last_checkpoint_path, checkpoints_dir / "last.pt")
        except OSError:
            pass
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_minutes": elapsed_minutes,
        "run_dir": str(run_dir.resolve()),
        "best_checkpoint_path": str(best_checkpoint_path.resolve()) if best_checkpoint_path else None,
        "last_checkpoint_path": str(last_checkpoint_path.resolve()) if last_checkpoint_path else None,
    }
    save_json(run_dir / "training_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
