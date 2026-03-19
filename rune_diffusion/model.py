from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(batch_size, self.num_heads, self.head_dim, height * width).permute(0, 1, 3, 2)
        k = k.view(batch_size, self.num_heads, self.head_dim, height * width)
        v = v.view(batch_size, self.num_heads, self.head_dim, height * width).permute(0, 1, 3, 2)

        attention = torch.matmul(q, k) * self.scale
        attention = attention.softmax(dim=-1)
        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, channels, height, width)
        return x + self.proj(out)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_artists: int = 0,
        image_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.null_class_idx = num_classes
        self.num_artists = num_artists
        self.null_artist_idx = num_artists if num_artists > 0 else None

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_embedding = nn.Embedding(num_classes + 1, time_emb_dim)
        self.artist_embedding = (
            nn.Embedding(num_artists + 1, time_emb_dim) if num_artists > 0 else None
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.input_conv = nn.Conv2d(image_channels, c1, kernel_size=3, padding=1)

        self.down1_block1 = ResidualBlock(c1, c1, time_emb_dim)
        self.down1_block2 = ResidualBlock(c1, c1, time_emb_dim)
        self.downsample1 = nn.Conv2d(c1, c1, kernel_size=4, stride=2, padding=1)

        self.down2_block1 = ResidualBlock(c1, c2, time_emb_dim)
        self.down2_block2 = ResidualBlock(c2, c2, time_emb_dim)
        self.downsample2 = nn.Conv2d(c2, c2, kernel_size=4, stride=2, padding=1)

        self.down3_block1 = ResidualBlock(c2, c3, time_emb_dim)
        self.down3_attention = SelfAttention2d(c3, num_heads=num_heads)
        self.down3_block2 = ResidualBlock(c3, c3, time_emb_dim)
        self.downsample3 = nn.Conv2d(c3, c3, kernel_size=4, stride=2, padding=1)

        self.mid_block1 = ResidualBlock(c3, c3, time_emb_dim)
        self.mid_attention = SelfAttention2d(c3, num_heads=num_heads)
        self.mid_block2 = ResidualBlock(c3, c3, time_emb_dim)

        self.upsample3 = nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1)
        self.up3_block1 = ResidualBlock(c3 + c3, c3, time_emb_dim)
        self.up3_attention = SelfAttention2d(c3, num_heads=num_heads)
        self.up3_block2 = ResidualBlock(c3, c2, time_emb_dim)

        self.upsample2 = nn.ConvTranspose2d(c2, c2, kernel_size=4, stride=2, padding=1)
        self.up2_block1 = ResidualBlock(c2 + c2, c2, time_emb_dim)
        self.up2_block2 = ResidualBlock(c2, c1, time_emb_dim)

        self.upsample1 = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        self.up1_block1 = ResidualBlock(c1 + c1, c1, time_emb_dim)
        self.up1_block2 = ResidualBlock(c1, c1, time_emb_dim)

        self.output_norm = nn.GroupNorm(_group_count(c1), c1)
        self.output_conv = nn.Conv2d(c1, image_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        artist_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if class_labels is None:
            class_labels = torch.full(
                (x.shape[0],),
                self.null_class_idx,
                device=x.device,
                dtype=torch.long,
            )

        emb = self.time_embedding(timesteps) + self.class_embedding(class_labels)
        if self.artist_embedding is not None:
            if artist_labels is None:
                artist_labels = torch.full(
                    (x.shape[0],),
                    self.null_artist_idx,
                    device=x.device,
                    dtype=torch.long,
                )
            emb = emb + self.artist_embedding(artist_labels)

        x = self.input_conv(x)

        skip1 = self.down1_block2(self.down1_block1(x, emb), emb)
        x = self.downsample1(skip1)

        skip2 = self.down2_block2(self.down2_block1(x, emb), emb)
        x = self.downsample2(skip2)

        x = self.down3_block1(x, emb)
        x = self.down3_attention(x)
        skip3 = self.down3_block2(x, emb)
        x = self.downsample3(skip3)

        x = self.mid_block1(x, emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, emb)

        x = self.upsample3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up3_block1(x, emb)
        x = self.up3_attention(x)
        x = self.up3_block2(x, emb)

        x = self.upsample2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2_block1(x, emb)
        x = self.up2_block2(x, emb)

        x = self.upsample1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1_block1(x, emb)
        x = self.up1_block2(x, emb)

        return self.output_conv(F.silu(self.output_norm(x)))
