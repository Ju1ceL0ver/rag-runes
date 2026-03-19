from __future__ import annotations

import torch


def extract(buffer: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    values = buffer.gather(0, timesteps)
    return values.view(timesteps.shape[0], *([1] * (len(x_shape) - 1)))


class GaussianDiffusion:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]],
            dim=0,
        )
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device: torch.device) -> "GaussianDiffusion":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int, int],
        class_labels: torch.Tensor,
        artist_labels: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        guidance_scale: float = 4.0,
        steps: int = 50,
        eta: float = 0.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = class_labels.device

        model.eval()
        sample = torch.randn(shape, device=device) if initial_noise is None else initial_noise.clone().to(device)
        null_labels = torch.full_like(class_labels, getattr(model, "null_class_idx"))
        null_artist_labels = None
        if artist_labels is not None and getattr(model, "null_artist_idx", None) is not None:
            null_artist_labels = torch.full_like(artist_labels, getattr(model, "null_artist_idx"))
        step_indices = torch.linspace(
            self.num_timesteps - 1,
            0,
            steps,
            device=device,
            dtype=torch.float32,
        ).round().long()

        for index, step in enumerate(step_indices):
            timestep = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            eps_cond = model(sample, timestep, class_labels, artist_labels)
            if guidance_scale != 1.0:
                eps_uncond = model(sample, timestep, null_labels, null_artist_labels)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            alpha_bar = self.alphas_cumprod[step]
            if index + 1 < len(step_indices):
                prev_step = step_indices[index + 1]
                alpha_bar_prev = self.alphas_cumprod[prev_step]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            sqrt_alpha_bar = alpha_bar.sqrt()
            sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
            x0 = (sample - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar
            x0 = x0.clamp(-1.0, 1.0)

            sigma = eta * (
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                * (1.0 - alpha_bar / alpha_bar_prev)
            ).clamp(min=0.0).sqrt()
            direction = (1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0).sqrt() * eps
            noise = torch.randn_like(sample) if index + 1 < len(step_indices) else torch.zeros_like(sample)
            sample = alpha_bar_prev.sqrt() * x0 + direction + sigma * noise

        return sample.clamp(-1.0, 1.0)
