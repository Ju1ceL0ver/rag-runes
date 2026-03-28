from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch
from torch import amp
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rune_diffusion.dataset import build_datasets
from rune_diffusion.gan import ConditionalGenerator, ProjectionDiscriminator
from rune_diffusion.utils import (
    count_parameters,
    save_image_grid,
    save_json,
    select_device,
    set_seed,
    timestamp_string,
)


@dataclass(slots=True)
class GanTrainingConfig:
    data_dir: Path = Path("all_runes")
    output_dir: Path = Path("artifacts") / "conditional_gan"
    image_size: int = 64
    epochs: int = 200
    batch_size: int = 64
    num_workers: int = 4
    val_fraction: float = 0.1
    latent_dim: int = 128
    g_base_channels: int = 64
    d_base_channels: int = 64
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.9
    ema_decay: float = 0.999
    seed: int = 42
    device: str = "auto"
    checkpoint_every: int = 5
    preview_every: int = 1
    preview_classes: int = 8
    amp_dtype: str = "float16"
    matmul_precision: str = "high"
    allow_tf32: bool = True


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def resolve_amp_dtype(device: torch.device, amp_dtype: str) -> torch.dtype | None:
    if device.type != "cuda" or amp_dtype == "none":
        return None
    if amp_dtype == "float16":
        return torch.float16
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "auto":
        return torch.float16
    raise ValueError("amp_dtype must be one of: float16, bfloat16, auto, none")


def build_autocast_kwargs(device: torch.device, amp_dtype: torch.dtype | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "device_type": device.type,
        "enabled": amp_dtype is not None,
    }
    if amp_dtype is not None:
        kwargs["dtype"] = amp_dtype
    return kwargs


def configure_runtime(device: torch.device, config: GanTrainingConfig) -> dict[str, Any]:
    torch.set_float32_matmul_precision(config.matmul_precision)
    runtime_info = {
        "matmul_precision": config.matmul_precision,
        "allow_tf32": False,
        "cudnn_benchmark": False,
    }
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.benchmark = True
        runtime_info["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        runtime_info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    return runtime_info


@torch.no_grad()
def evaluate(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    latent_dim: int,
    amp_dtype: torch.dtype | None,
    seed: int,
) -> tuple[float, float]:
    generator.eval()
    discriminator.eval()
    autocast_kwargs = build_autocast_kwargs(device, amp_dtype)
    noise_generator = torch.Generator(device=device.type)
    noise_generator.manual_seed(seed)

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_items = 0

    for real_images, class_ids, _artist_ids in dataloader:
        real_images = real_images.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)
        batch_size = real_images.shape[0]
        noise = torch.randn(batch_size, latent_dim, device=device, generator=noise_generator)

        with amp.autocast(**autocast_kwargs):
            fake_images = generator(noise, class_ids)
            real_logits = discriminator(real_images, class_ids)
            fake_logits = discriminator(fake_images, class_ids)
            d_loss = torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()
            g_loss = -fake_logits.mean()

        total_d_loss += d_loss.item() * batch_size
        total_g_loss += g_loss.item() * batch_size
        total_items += batch_size

    return (
        total_g_loss / max(total_items, 1),
        total_d_loss / max(total_items, 1),
    )


def train(config: GanTrainingConfig | None = None) -> dict[str, Any]:
    cfg = config or GanTrainingConfig()
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    runtime_info = configure_runtime(device, cfg)
    amp_dtype = resolve_amp_dtype(device, cfg.amp_dtype)
    autocast_kwargs = build_autocast_kwargs(device, amp_dtype)
    scaler_enabled = amp_dtype == torch.float16

    run_dir = cfg.output_dir / timestamp_string()
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
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )

    generator = ConditionalGenerator(
        num_classes=metadata["num_classes"],
        latent_dim=cfg.latent_dim,
        base_channels=cfg.g_base_channels,
    ).to(device)
    discriminator = ProjectionDiscriminator(
        num_classes=metadata["num_classes"],
        base_channels=cfg.d_base_channels,
    ).to(device)
    ema_generator = copy.deepcopy(generator).to(device)
    ema_generator.eval()
    for parameter in ema_generator.parameters():
        parameter.requires_grad_(False)

    g_optimizer = Adam(generator.parameters(), lr=cfg.g_lr, betas=(cfg.beta1, cfg.beta2))
    d_optimizer = Adam(discriminator.parameters(), lr=cfg.d_lr, betas=(cfg.beta1, cfg.beta2))
    scaler = amp.GradScaler(device.type, enabled=scaler_enabled)

    fixed_labels = torch.arange(
        min(metadata["num_classes"], cfg.preview_classes),
        device=device,
        dtype=torch.long,
    )
    if fixed_labels.shape[0] == 0:
        raise RuntimeError("Dataset must contain at least one class")
    fixed_noise = torch.randn(fixed_labels.shape[0], cfg.latent_dim, device=device)

    config_payload = {
        "data_dir": str(cfg.data_dir.resolve()),
        "image_size": cfg.image_size,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "val_fraction": cfg.val_fraction,
        "latent_dim": cfg.latent_dim,
        "g_base_channels": cfg.g_base_channels,
        "d_base_channels": cfg.d_base_channels,
        "g_lr": cfg.g_lr,
        "d_lr": cfg.d_lr,
        "beta1": cfg.beta1,
        "beta2": cfg.beta2,
        "ema_decay": cfg.ema_decay,
        "seed": cfg.seed,
        "device": str(device),
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
        "scaler_enabled": scaler_enabled,
        "preview_classes": cfg.preview_classes,
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "runtime": runtime_info,
    }
    save_json(run_dir / "config.json", config_payload)
    save_json(run_dir / "dataset_metadata.json", metadata)

    history: list[dict[str, float | int]] = []
    best_metric_name = "val_g_loss" if val_loader is not None else "train_g_loss"
    best_metric = float("inf")
    best_epoch = 0
    best_checkpoint_path: Path | None = None
    last_checkpoint_path: Path | None = None
    train_start = time.time()
    show_progress = sys.stderr.isatty() or sys.stdout.isatty()

    epoch_iterator = tqdm(
        range(1, cfg.epochs + 1),
        desc="cgan epochs",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for epoch in epoch_iterator:
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        seen_items = 0

        batch_iterator = tqdm(
            train_loader,
            desc=f"epoch {epoch:03d}",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for real_images, class_ids, _artist_ids in batch_iterator:
            real_images = real_images.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            batch_size = real_images.shape[0]

            d_optimizer.zero_grad(set_to_none=True)
            with amp.autocast(**autocast_kwargs):
                noise = torch.randn(batch_size, cfg.latent_dim, device=device)
                fake_images = generator(noise, class_ids)
                real_logits = discriminator(real_images, class_ids)
                fake_logits = discriminator(fake_images.detach(), class_ids)
                d_loss = torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()
            if scaler_enabled:
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
            else:
                d_loss.backward()
                d_optimizer.step()

            g_optimizer.zero_grad(set_to_none=True)
            with amp.autocast(**autocast_kwargs):
                noise = torch.randn(batch_size, cfg.latent_dim, device=device)
                fake_images = generator(noise, class_ids)
                g_loss = -discriminator(fake_images, class_ids).mean()
            if scaler_enabled:
                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                g_loss.backward()
                g_optimizer.step()
            update_ema(ema_generator, generator, cfg.ema_decay)

            epoch_d_loss += d_loss.item() * batch_size
            epoch_g_loss += g_loss.item() * batch_size
            seen_items += batch_size
            if show_progress:
                batch_iterator.set_postfix(
                    g=f"{g_loss.item():.4f}",
                    d=f"{d_loss.item():.4f}",
                )

        mean_d_loss = epoch_d_loss / max(seen_items, 1)
        mean_g_loss = epoch_g_loss / max(seen_items, 1)
        val_g_loss = None
        val_d_loss = None
        if val_loader is not None:
            val_g_loss, val_d_loss = evaluate(
                ema_generator,
                discriminator,
                val_loader,
                device,
                cfg.latent_dim,
                amp_dtype,
                cfg.seed,
            )
        current_metric = mean_g_loss if val_g_loss is None else val_g_loss
        history.append(
            {
                "epoch": epoch,
                "train_g_loss": mean_g_loss,
                "train_d_loss": mean_d_loss,
                "val_g_loss": val_g_loss,
                "val_d_loss": val_d_loss,
            }
        )
        save_json(run_dir / "history.json", {"history": history})

        checkpoint = {
            "generator_state": generator.state_dict(),
            "ema_generator_state": ema_generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "g_optimizer_state": g_optimizer.state_dict(),
            "d_optimizer_state": d_optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler_enabled else None,
            "history": history,
            "dataset_metadata": metadata,
            "config": config_payload,
            "epoch": epoch,
            "best_metric_name": best_metric_name,
            "best_metric_value": current_metric,
        }

        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
            next_best_path = checkpoints_dir / f"best_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, next_best_path)
            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                try:
                    best_checkpoint_path.unlink()
                except OSError:
                    pass
            best_checkpoint_path = next_best_path
            try:
                shutil.copyfile(best_checkpoint_path, checkpoints_dir / "best.pt")
            except OSError:
                pass

        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            last_checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, last_checkpoint_path)
            try:
                shutil.copyfile(last_checkpoint_path, checkpoints_dir / "last.pt")
            except OSError:
                pass

        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.preview_every == 0:
            with torch.no_grad():
                preview = ema_generator(fixed_noise, fixed_labels).cpu()
            save_image_grid(
                preview,
                samples_dir / f"epoch_{epoch:03d}.png",
                rows=1,
                cols=fixed_labels.shape[0],
            )

        if show_progress:
            epoch_iterator.set_postfix(
                train_g=f"{mean_g_loss:.4f}",
                train_d=f"{mean_d_loss:.4f}",
                best=f"{best_metric:.4f}",
            )
        print(
            f"epoch={epoch:03d} "
            f"train_g_loss={mean_g_loss:.6f} "
            f"train_d_loss={mean_d_loss:.6f} "
            f"{best_metric_name}={current_metric:.6f} "
            f"best={best_metric:.6f}",
            flush=True,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric,
        "elapsed_minutes": (time.time() - train_start) / 60.0,
        "run_dir": str(run_dir.resolve()),
        "best_checkpoint_path": str(best_checkpoint_path.resolve()) if best_checkpoint_path else None,
        "last_checkpoint_path": str(last_checkpoint_path.resolve()) if last_checkpoint_path else None,
        "history_length": len(history),
        "last_train_g_loss": float(history[-1]["train_g_loss"]) if history else None,
        "last_train_d_loss": float(history[-1]["train_d_loss"]) if history else None,
        "last_val_g_loss": float(history[-1]["val_g_loss"]) if history and history[-1]["val_g_loss"] is not None else None,
        "last_val_d_loss": float(history[-1]["val_d_loss"]) if history and history[-1]["val_d_loss"] is not None else None,
    }
    save_json(run_dir / "training_summary.json", summary)
    print(summary, flush=True)
    return summary


def main(config: GanTrainingConfig | None = None) -> dict[str, Any]:
    return train(config)


if __name__ == "__main__":
    main()
