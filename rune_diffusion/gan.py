from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embedding = nn.Embedding(num_classes, num_features * 2)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.embedding(class_labels).chunk(2, dim=1)
        gamma = 1.0 + gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return self.bn(x) * gamma + beta


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(F.relu(self.cbn1(x, class_labels)), scale_factor=2.0, mode="nearest")
        h = self.conv1(h)
        h = self.conv2(F.relu(self.cbn2(h, class_labels)))

        skip = F.interpolate(x, scale_factor=2.0, mode="nearest")
        skip = self.skip(skip)
        return h + skip


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 128,
        base_channels: int = 64,
        image_channels: int = 1,
    ) -> None:
        super().__init__()
        c4 = base_channels * 8
        c3 = base_channels * 4
        c2 = base_channels * 2
        c1 = base_channels
        c0 = max(base_channels // 2, 32)

        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 4 * 4 * c4)
        self.block1 = GeneratorBlock(c4, c3, num_classes)
        self.block2 = GeneratorBlock(c3, c2, num_classes)
        self.block3 = GeneratorBlock(c2, c1, num_classes)
        self.block4 = GeneratorBlock(c1, c0, num_classes)
        self.bn = nn.BatchNorm2d(c0)
        self.output = nn.Conv2d(c0, image_channels, kernel_size=3, padding=1)

    def forward(self, noise: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        x = self.fc(noise).view(noise.shape[0], -1, 4, 4)
        x = self.block1(x, class_labels)
        x = self.block2(x, class_labels)
        x = self.block3(x, class_labels)
        x = self.block4(x, class_labels)
        x = self.output(F.relu(self.bn(x)))
        return torch.tanh(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True) -> None:
        super().__init__()
        self.downsample = downsample
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.skip = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            if in_channels != out_channels or downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(x, negative_slope=0.2))
        h = self.conv2(F.leaky_relu(h, negative_slope=0.2))
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2)

        skip = self.skip(x)
        if self.downsample:
            skip = F.avg_pool2d(skip, kernel_size=2)
        return h + skip


class ProjectionDiscriminator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 64,
        image_channels: int = 1,
    ) -> None:
        super().__init__()
        c0 = max(base_channels // 2, 32)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.block0 = DiscriminatorBlock(image_channels, c0)
        self.block1 = DiscriminatorBlock(c0, c1)
        self.block2 = DiscriminatorBlock(c1, c2)
        self.block3 = DiscriminatorBlock(c2, c3)
        self.block4 = DiscriminatorBlock(c3, c4, downsample=False)
        self.linear = spectral_norm(nn.Linear(c4, 1))
        self.embedding = spectral_norm(nn.Embedding(num_classes, c4))

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        pooled = x.sum(dim=(2, 3))
        logits = self.linear(pooled).squeeze(1)
        projection = (self.embedding(class_labels) * pooled).sum(dim=1)
        return logits + projection
