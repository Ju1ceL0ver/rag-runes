from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Any

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
from rune_diffusion.utils import (
    count_parameters,
    save_image_grid,
    save_json,
    select_device,
    set_seed,
    timestamp_string,
)


@dataclass(slots=True)
class DiffusionTrainingConfig:
    data_dir: Path = Path("runes")
    output_dir: Path = Path("artifacts") / "conditional_diffusion"
    resume: Path | None = None
    image_size: int = 64
    epochs: int = 40
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    cond_drop_prob: float = 0.1
    val_fraction: float = 0.1
    seed: int = 42
    base_channels: int = 64
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str = "auto"
    preview_every: int = 1
    preview_classes: int = 4
    preview_samples_per_class: int = 1
    sample_steps: int = 50
    guidance_scale: float = 4.0
    checkpoint_every: int = 5


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


def train(config: DiffusionTrainingConfig | None = None) -> dict[str, Any]:
    cfg = config or DiffusionTrainingConfig()
    set_seed(cfg.seed)

    device = select_device(cfg.device)
    run_dir = cfg.output_dir / timestamp_string() if cfg.resume is None else cfg.resume.resolve().parent.parent
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, metadata = build_datasets(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    model = ConditionalUNet(
        num_classes=metadata["num_classes"],
        num_artists=metadata["num_artists"],
        base_channels=cfg.base_channels,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    diffusion = GaussianDiffusion(
        num_timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    best_checkpoint_path: Path | None = None
    last_checkpoint_path: Path | None = None

    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr
            group["weight_decay"] = cfg.weight_decay
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

    config_payload = {
        "data_dir": str(cfg.data_dir.resolve()),
        "resume": str(cfg.resume.resolve()) if cfg.resume is not None else None,
        "image_size": cfg.image_size,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "grad_clip": cfg.grad_clip,
        "ema_decay": cfg.ema_decay,
        "cond_drop_prob": cfg.cond_drop_prob,
        "val_fraction": cfg.val_fraction,
        "seed": cfg.seed,
        "base_channels": cfg.base_channels,
        "timesteps": cfg.timesteps,
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,
        "guidance_scale": cfg.guidance_scale,
        "sample_steps": cfg.sample_steps,
        "device": str(device),
        "parameter_count": count_parameters(model),
    }
    save_json(run_dir / "config.json", config_payload)
    save_json(run_dir / "dataset_metadata.json", metadata)

    fixed_labels, fixed_artists = make_preview_conditions(
        metadata["num_classes"],
        metadata["num_artists"],
        cfg.preview_classes,
        cfg.preview_samples_per_class,
        device,
    )
    fixed_noise = torch.randn(
        fixed_labels.shape[0],
        1,
        cfg.image_size,
        cfg.image_size,
        device=device,
    )

    train_start = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen_items = 0

        for images, class_ids, artist_ids in train_loader:
            images = images.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            artist_ids = artist_ids.to(device, non_blocking=True)

            drop_mask = torch.rand(class_ids.shape[0], device=device) < cfg.cond_drop_prob
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
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, cfg.ema_decay)

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
            "config": config_payload,
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

        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            last_checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, last_checkpoint_path)

        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.preview_every == 0:
            preview = diffusion.ddim_sample(
                model=ema_model,
                shape=(fixed_labels.shape[0], 1, cfg.image_size, cfg.image_size),
                class_labels=fixed_labels,
                artist_labels=fixed_artists,
                initial_noise=fixed_noise,
                guidance_scale=cfg.guidance_scale,
                steps=cfg.sample_steps,
                device=device,
            )
            save_image_grid(
                preview,
                samples_dir / f"epoch_{epoch:03d}.png",
                rows=min(metadata["num_classes"], cfg.preview_classes),
                cols=cfg.preview_samples_per_class,
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
        "history_length": len(history),
    }
    save_json(run_dir / "training_summary.json", summary)
    print(summary)
    return summary


def main(config: DiffusionTrainingConfig | None = None) -> dict[str, Any]:
    return train(config)


if __name__ == "__main__":
    main()
