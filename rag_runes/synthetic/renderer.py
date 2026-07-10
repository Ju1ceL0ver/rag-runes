from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


STONE_PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "granite": ((116, 111, 104), (145, 139, 132), (94, 90, 84)),
    "slate": ((89, 95, 103), (111, 118, 126), (72, 78, 86)),
    "sandstone": ((164, 145, 116), (188, 170, 138), (137, 120, 96)),
    "basalt": ((72, 72, 75), (99, 101, 106), (54, 54, 58)),
    "limestone": ((186, 181, 168), (212, 208, 196), (154, 149, 138)),
    "marble": ((198, 198, 202), (228, 228, 232), (162, 162, 168)),
    "gneiss": ((118, 113, 120), (148, 142, 150), (88, 84, 92)),
    "quartzite": ((170, 168, 160), (201, 198, 190), (136, 133, 126)),
    "obsidian": ((44, 47, 53), (70, 74, 82), (26, 28, 33)),
    "travertine": ((188, 172, 146), (214, 198, 172), (154, 140, 116)),
    "serpentine": ((92, 108, 94), (124, 142, 126), (70, 83, 71)),
    "dolomite": ((176, 174, 170), (205, 203, 198), (142, 140, 136)),
}

RUNE_TEXTURES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "ash": ((176, 176, 176), (134, 134, 134), (218, 218, 218)),
    "graphite": ((118, 120, 124), (76, 78, 82), (162, 164, 168)),
    "chalk": ((204, 201, 194), (160, 156, 150), (236, 234, 228)),
    "ink": ((72, 74, 78), (38, 40, 44), (118, 121, 128)),
    "smoke": ((150, 152, 156), (106, 108, 112), (194, 196, 200)),
}


def _resolve_palette_name(palette_name: str, rng: np.random.Generator) -> str:
    normalized = palette_name.strip().lower()
    if normalized == "auto":
        names = tuple(STONE_PALETTES.keys())
        return names[int(rng.integers(0, len(names)))]
    if normalized in STONE_PALETTES:
        return normalized
    return "granite"


def _resolve_rune_texture(
    rng: np.random.Generator,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(RUNE_TEXTURES.keys())
    name = names[int(rng.integers(0, len(names)))]
    base_rgb, dark_rgb, light_rgb = RUNE_TEXTURES[name]
    return (
        name,
        np.asarray(base_rgb, dtype=np.float32) / 255.0,
        np.asarray(dark_rgb, dtype=np.float32) / 255.0,
        np.asarray(light_rgb, dtype=np.float32) / 255.0,
    )


def _augment_glyph_mask(glyph: Image.Image, rng: np.random.Generator) -> Image.Image:
    width, height = glyph.size
    stretch_x = float(rng.uniform(0.88, 1.16))
    stretch_y = float(rng.uniform(0.88, 1.16))
    stretched = glyph.resize(
        (max(8, int(round(width * stretch_x))), max(8, int(round(height * stretch_y)))),
        resample=Image.Resampling.LANCZOS,
    )

    pad = max(6, int(round(min(stretched.size) * 0.12)))
    padded = ImageOps.expand(stretched, border=pad, fill=0)
    pw, ph = padded.size

    shear_x = float(rng.uniform(-0.12, 0.12))
    shear_y = float(rng.uniform(-0.08, 0.08))
    affine = padded.transform(
        (pw, ph),
        Image.Transform.AFFINE,
        (1.0, shear_x, -shear_x * pw * 0.5, shear_y, 1.0, -shear_y * ph * 0.5),
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )

    perspective = float(rng.uniform(0.04, 0.16))
    dx = max(2.0, pw * perspective)
    dy = max(2.0, ph * perspective)
    quad = (
        float(rng.uniform(0.0, dx)), float(rng.uniform(0.0, dy)),
        float(pw - rng.uniform(0.0, dx)), float(rng.uniform(0.0, dy)),
        float(pw - rng.uniform(0.0, dx)), float(ph - rng.uniform(0.0, dy)),
        float(rng.uniform(0.0, dx)), float(ph - rng.uniform(0.0, dy)),
    )
    warped = affine.transform(
        (pw, ph),
        Image.Transform.QUAD,
        quad,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )

    warped = warped.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.45))))
    bbox = warped.getbbox()
    if bbox is None:
        return glyph
    cropped = warped.crop(bbox)
    return cropped if cropped.size[0] > 0 and cropped.size[1] > 0 else glyph


@dataclass(slots=True)
class RenderedRune:
    composite: Image.Image
    mask: Image.Image
    cutout: Image.Image
    overlay: Image.Image
    bbox: tuple[int, int, int, int]
    metadata: dict[str, float | int | str]


def _resolve_canvas_size(canvas_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(canvas_size, tuple):
        width, height = int(canvas_size[0]), int(canvas_size[1])
    else:
        width = int(canvas_size)
        height = int(canvas_size)
    if width <= 0 or height <= 0:
        raise ValueError("canvas_size must be positive")
    return width, height


def _fractal_noise(
    width: int,
    height: int,
    rng: np.random.Generator,
    octaves: int = 5,
) -> np.ndarray:
    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    amplitude_sum = 0.0
    for octave in range(octaves):
        divisor_low = 3 + octave * 3
        divisor_high = 7 + octave * 9
        grid_w = max(3, int(rng.integers(max(3, width // divisor_high), max(4, width // divisor_low + 1))))
        grid_h = max(3, int(rng.integers(max(3, height // divisor_high), max(4, height // divisor_low + 1))))
        grid = rng.random((grid_h, grid_w), dtype=np.float32)
        image = Image.fromarray((grid * 255).astype(np.uint8), mode="L").resize(
            (width, height),
            resample=Image.Resampling.BICUBIC,
        )
        noise += amplitude * (np.asarray(image, dtype=np.float32) / 255.0)
        amplitude_sum += amplitude
        amplitude *= 0.55
    noise /= max(amplitude_sum, 1e-6)
    noise -= noise.min()
    noise /= max(float(noise.max()), 1e-6)
    return noise


def _build_rune_texture_rgb(
    width: int,
    height: int,
    rng: np.random.Generator,
    base_rgb: np.ndarray,
    dark_rgb: np.ndarray,
    light_rgb: np.ndarray,
) -> np.ndarray:
    broad_noise = _fractal_noise(width, height, rng, octaves=4)
    grain_noise = _fractal_noise(width, height, rng, octaves=6)
    vein_noise = _fractal_noise(width, height, rng, octaves=5)
    speck_noise = _fractal_noise(width, height, rng, octaves=7)

    texture = np.broadcast_to(base_rgb.reshape(1, 1, 3), (height, width, 3)).astype(np.float32).copy()

    light_mask = np.clip((broad_noise - 0.42) * 1.6, 0.0, 1.0)
    dark_mask = np.clip((0.58 - broad_noise) * 1.5, 0.0, 1.0)
    texture += (light_rgb.reshape(1, 1, 3) - texture) * light_mask[:, :, None] * 0.52
    texture += (dark_rgb.reshape(1, 1, 3) - texture) * dark_mask[:, :, None] * 0.48

    grain_centered = grain_noise - 0.5
    texture += (light_rgb.reshape(1, 1, 3) - texture) * np.clip(grain_centered, 0.0, 1.0)[:, :, None] * 0.24
    texture += (dark_rgb.reshape(1, 1, 3) - texture) * np.clip(-grain_centered, 0.0, 1.0)[:, :, None] * 0.24

    vein_mask = np.clip((vein_noise - 0.70) * 2.8, 0.0, 1.0)
    texture += (dark_rgb.reshape(1, 1, 3) - texture) * vein_mask[:, :, None] * 0.34

    speck_mask = np.clip((speck_noise - 0.86) * 4.5, 0.0, 1.0)
    texture += (light_rgb.reshape(1, 1, 3) - texture) * speck_mask[:, :, None] * 0.16

    return np.clip(texture, 0.0, 1.0)


def _procedural_stone(
    width: int,
    height: int,
    rng: np.random.Generator,
    palette_name: str,
) -> np.ndarray:
    palette = STONE_PALETTES.get(palette_name, STONE_PALETTES["granite"])
    base_noise = _fractal_noise(width, height, rng, octaves=5)
    grain_noise = _fractal_noise(width, height, rng, octaves=6)
    crack_noise = _fractal_noise(width, height, rng, octaves=4)
    broad_noise = _fractal_noise(width, height, rng, octaves=3)

    palette_arr = np.asarray(palette, dtype=np.float32) / 255.0
    base = palette_arr[rng.integers(0, len(palette_arr))]
    accent = palette_arr[rng.integers(0, len(palette_arr))]

    brightness = 0.70 + (broad_noise - 0.5) * 0.34 + (base_noise - 0.5) * 0.16
    brightness = brightness.clip(0.32, 1.02)

    stone = base.reshape(1, 1, 3) * brightness[:, :, None]
    vein_strength = np.clip((crack_noise - 0.68) * 2.6, 0.0, 1.0)
    stone = stone * (1.0 - vein_strength[:, :, None] * 0.16)
    stone = stone + accent.reshape(1, 1, 3) * vein_strength[:, :, None] * 0.08

    grain = (grain_noise - 0.5)[:, :, None] * 0.14
    stone = np.clip(stone + grain, 0.0, 1.0)
    speckles = rng.random((height, width, 1), dtype=np.float32)
    speckles = (speckles > 0.994).astype(np.float32) * rng.uniform(0.06, 0.16)
    stone = np.clip(stone + speckles, 0.0, 1.0)
    return stone


def _load_stone_texture(
    texture_dir: Path,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if not texture_dir.exists():
        return None
    candidates = sorted(
        path
        for path in texture_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not candidates:
        return None

    texture_path = candidates[int(rng.integers(0, len(candidates)))]
    image = Image.open(texture_path).convert("RGB")
    if image.width < width or image.height < height:
        image = ImageOps.fit(
            image,
            (max(width, image.width), max(height, image.height)),
            method=Image.Resampling.LANCZOS,
        )
    left = int(rng.integers(0, max(1, image.width - width + 1)))
    top = int(rng.integers(0, max(1, image.height - height + 1)))
    crop = image.crop((left, top, left + width, top + height))
    return np.asarray(crop, dtype=np.float32) / 255.0


def _soft_shift(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.roll(values, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    return shifted


def _adjust_mask_thickness(mask_image: Image.Image, thickness: float) -> Image.Image:
    if abs(thickness - 1.0) < 1e-3:
        return mask_image
    grayscale = mask_image.convert("L")
    if thickness > 1.0:
        radius = max(1, int(round((thickness - 1.0) * 3.0)))
        for _ in range(radius):
            grayscale = grayscale.filter(ImageFilter.MaxFilter(size=3))
        return grayscale
    radius = max(1, int(round((1.0 - thickness) * 2.0)))
    for _ in range(radius):
        grayscale = grayscale.filter(ImageFilter.MinFilter(size=3))
    return grayscale


def _apply_line_defects(
    mask_image: Image.Image,
    rng: np.random.Generator,
    defect_strength: float,
) -> Image.Image:
    if defect_strength <= 0:
        return mask_image

    width, height = mask_image.size
    mask_arr = np.asarray(mask_image, dtype=np.float32) / 255.0
    active_points = np.argwhere(mask_arr > 0.20)
    if active_points.size == 0:
        return mask_image

    damage = Image.new("L", (width, height), 0)
    blob_draw = ImageDraw.Draw(damage)
    cuts = max(1, int(round(2.0 + defect_strength * 10.0)))

    for _ in range(cuts):
        cy, cx = active_points[int(rng.integers(0, len(active_points)))]
        rx = int(rng.integers(max(4, int(width * 0.015)), max(7, int(width * 0.05))))
        ry = int(rng.integers(max(3, int(height * 0.012)), max(6, int(height * 0.045))))
        angle = float(rng.uniform(0.0, 180.0))

        blob = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(blob)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
        blob = blob.rotate(angle, resample=Image.Resampling.BICUBIC, center=(cx, cy))
        damage = ImageChops.lighter(damage, blob)

    if defect_strength > 0.35:
        for _ in range(max(1, int(defect_strength * 3.0))):
            y0, x0 = active_points[int(rng.integers(0, len(active_points)))]
            x1 = int(np.clip(x0 + rng.integers(-width // 10, width // 10 + 1), 0, width - 1))
            y1 = int(np.clip(y0 + rng.integers(-height // 10, height // 10 + 1), 0, height - 1))
            line_w = int(rng.integers(2, max(3, int(width * 0.025))))
            blob_draw.line((x0, y0, x1, y1), fill=255, width=line_w)

    damage = damage.filter(ImageFilter.GaussianBlur(radius=1.2 + defect_strength * 1.8))
    damage_arr = np.asarray(damage, dtype=np.float32) / 255.0

    noise = _fractal_noise(width, height, rng, octaves=6)
    erosion = np.clip((noise - (0.63 - defect_strength * 0.24)) * 2.2, 0.0, 1.0)
    direct_breaks = (damage_arr > (0.20 + (1.0 - defect_strength) * 0.12)).astype(np.float32)
    degraded = mask_arr * (1.0 - damage_arr * (0.72 + defect_strength * 0.55))
    degraded = degraded * (1.0 - erosion * (0.14 + defect_strength * 0.28))
    degraded = np.where(direct_breaks > 0, degraded * (0.04 + noise * 0.08), degraded)
    degraded = np.clip(degraded, 0.0, 1.0)
    return Image.fromarray((degraded * 255).astype(np.uint8), mode="L")


def _stone_color_profile(stone: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = stone.reshape(-1, 3)
    mid = np.quantile(flat, 0.50, axis=0).astype(np.float32)
    dark = np.quantile(flat, 0.18, axis=0).astype(np.float32)
    light = np.quantile(flat, 0.84, axis=0).astype(np.float32)
    groove = np.clip(mid * 0.62 + dark * 0.38, 0.0, 1.0)
    shadow = np.clip(mid * 0.48 + dark * 0.52, 0.0, 1.0)
    highlight = np.clip(mid * 0.70 + light * 0.30, 0.0, 1.0)
    return mid, dark, groove, shadow, highlight


class StoneRuneRenderer:
    def __init__(self, texture_dir: Path | None = None) -> None:
        self.texture_dir = texture_dir

    def _background(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
        palette_name: str,
    ) -> np.ndarray:
        resolved_palette = _resolve_palette_name(palette_name, rng)
        if self.texture_dir is not None:
            texture = _load_stone_texture(self.texture_dir, width, height, rng)
            if texture is not None:
                return texture
        return _procedural_stone(width, height, rng, palette_name=resolved_palette)

    def _place_mask(
        self,
        mask_image: Image.Image,
        canvas_size: int | tuple[int, int],
        rng: np.random.Generator,
        defect_strength: float,
        thickness: float,
        centered: bool = False,
    ) -> Image.Image:
        canvas_width, canvas_height = _resolve_canvas_size(canvas_size)
        bbox = mask_image.getbbox()
        if bbox is None:
            raise RuntimeError("Generated rune mask is empty.")
        glyph = mask_image.crop(bbox)

        angle = float(rng.uniform(-13.0, 13.0))
        scale = float(rng.uniform(0.58, 0.80))
        short_side = min(canvas_width, canvas_height)
        target = max(10, min(short_side - 2, int(short_side * scale)))
        aspect = glyph.width / max(glyph.height, 1)
        if glyph.width >= glyph.height:
            target_w = target
            target_h = max(8, min(canvas_height - 2, int(target / max(aspect, 1e-6))))
        else:
            target_h = target
            target_w = max(8, min(canvas_width - 2, int(target * aspect)))

        glyph = glyph.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        glyph = glyph.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=0,
        )
        glyph = _augment_glyph_mask(glyph, rng=rng)
        glyph = glyph.filter(ImageFilter.GaussianBlur(radius=0.4))
        glyph = _adjust_mask_thickness(glyph, thickness=thickness)

        canvas = Image.new("L", (canvas_width, canvas_height), 0)
        margin = max(12, min(canvas_width, canvas_height) // 12)
        if centered:
            jitter_x = max(1, min(canvas_width, canvas_height) // 30)
            jitter_y = max(1, min(canvas_width, canvas_height) // 30)
            center_left = max(margin, (canvas_width - glyph.width) // 2)
            center_top = max(margin, (canvas_height - glyph.height) // 2)
            left = int(np.clip(center_left + rng.integers(-jitter_x, jitter_x + 1), margin, max(margin, canvas_width - glyph.width - margin)))
            top = int(np.clip(center_top + rng.integers(-jitter_y, jitter_y + 1), margin, max(margin, canvas_height - glyph.height - margin)))
        else:
            left = int(rng.integers(margin, max(margin + 1, canvas_width - glyph.width - margin + 1)))
            top = int(rng.integers(margin, max(margin + 1, canvas_height - glyph.height - margin + 1)))
        canvas.paste(glyph, (left, top))

        # Introduce local edge erosion for carved effect.
        erosion_noise = _fractal_noise(canvas_width, canvas_height, rng, octaves=5)
        canvas_arr = np.asarray(canvas, dtype=np.float32) / 255.0
        canvas_arr *= (0.80 + erosion_noise * 0.35)
        canvas = Image.fromarray((canvas_arr.clip(0, 1) * 255).astype(np.uint8), mode="L")
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.9))
        return _apply_line_defects(canvas, rng=rng, defect_strength=defect_strength)

    def render(
        self,
        mask_image: Image.Image,
        seed: int,
        canvas_size: int | tuple[int, int] = 256,
        palette_name: str = "granite",
        style: str = "engraved",
        background_mode: str = "transparent",
        defect_strength: float = 0.0,
        thickness: float = 1.0,
        background_image: Image.Image | None = None,
        centered: bool = False,
    ) -> RenderedRune:
        rng = np.random.default_rng(seed)
        normalized_style = style.strip().lower()
        normalized_bg = background_mode.strip().lower()
        if normalized_style not in {"engraved", "scratch"}:
            raise ValueError(f"Unsupported style '{style}'.")
        if normalized_bg not in {"stone", "transparent"}:
            raise ValueError(f"Unsupported background_mode '{background_mode}'.")
        canvas_width, canvas_height = (
            background_image.size
            if background_image is not None
            else _resolve_canvas_size(canvas_size)
        )

        mask_canvas = self._place_mask(
            mask_image,
            canvas_size=(canvas_width, canvas_height),
            rng=rng,
            defect_strength=defect_strength,
            thickness=thickness,
            centered=centered,
        )
        mask_soft = np.asarray(mask_canvas, dtype=np.float32) / 255.0
        if float(mask_soft.max()) <= 0.01:
            raise RuntimeError("Rendered rune mask is empty after placement.")

        if background_image is not None:
            stone = np.asarray(background_image.convert("RGB"), dtype=np.float32) / 255.0
            if stone.shape[1] != canvas_width or stone.shape[0] != canvas_height:
                fitted = ImageOps.fit(
                    background_image.convert("RGB"),
                    (canvas_width, canvas_height),
                    method=Image.Resampling.LANCZOS,
                )
                stone = np.asarray(fitted, dtype=np.float32) / 255.0
            resolved_palette = "image-adaptive"
        else:
            resolved_palette = _resolve_palette_name(palette_name, rng)
            stone = self._background(canvas_width, canvas_height, rng, palette_name=resolved_palette)
        local_noise = _fractal_noise(canvas_width, canvas_height, rng, octaves=6)
        detail_noise = _fractal_noise(canvas_width, canvas_height, rng, octaves=7)
        groove_strength = mask_soft ** 1.35
        if normalized_style == "scratch":
            if background_image is not None:
                scratch_core = np.clip(
                    (mask_soft ** 1.65) * (0.92 + detail_noise * 0.22),
                    0.0,
                    1.0,
                )
                scratch_mid = np.clip(
                    (mask_soft ** 1.18) * (0.96 + detail_noise * 0.18),
                    0.0,
                    1.0,
                )
            else:
                scratch_core = np.clip(
                    (mask_soft ** 2.35) * (0.76 + detail_noise * 0.30),
                    0.0,
                    1.0,
                )
                scratch_mid = np.clip(
                    (mask_soft ** 1.55) * (0.78 + detail_noise * 0.24),
                    0.0,
                    1.0,
                )
            scratch_edge = np.clip(scratch_mid - scratch_core * 0.70, 0.0, 1.0)
            groove_depth = scratch_core * (0.18 + local_noise * 0.11)
        else:
            groove_depth = groove_strength * (0.58 + local_noise * 0.42)
        ambient = np.asarray(
            mask_canvas.filter(ImageFilter.GaussianBlur(radius=6.0)),
            dtype=np.float32,
        ) / 255.0
        texture_gray = (
            stone[:, :, 0] * 0.299
            + stone[:, :, 1] * 0.587
            + stone[:, :, 2] * 0.114
        )
        texture_relief = np.clip(
            np.abs(texture_gray - _soft_shift(texture_gray, 1, 1)) * 3.0,
            0.0,
            1.0,
        )
        _local_mid_rgb, local_dark_rgb, groove_tint, shadow_tint, highlight_tint = _stone_color_profile(stone)
        rune_texture_name, rune_base_rgb, rune_dark_rgb, rune_light_rgb = _resolve_rune_texture(rng)
        rune_texture_rgb = _build_rune_texture_rgb(
            width=canvas_width,
            height=canvas_height,
            rng=rng,
            base_rgb=rune_base_rgb,
            dark_rgb=rune_dark_rgb,
            light_rgb=rune_light_rgb,
        )
        rune_texture_luma = (
            rune_texture_rgb[:, :, 0] * 0.299
            + rune_texture_rgb[:, :, 1] * 0.587
            + rune_texture_rgb[:, :, 2] * 0.114
        )
        rune_texture_luma = np.clip((rune_texture_luma - 0.5) * 1.8 + 0.5, 0.0, 1.0)

        offset = 1 if normalized_style == "scratch" else 2
        shadow = np.clip(groove_depth - _soft_shift(groove_depth, offset, offset), 0.0, 1.0)
        highlight = np.clip(groove_depth - _soft_shift(groove_depth, -offset, -offset), 0.0, 1.0)
        chipped = np.clip(mask_soft - ambient * 0.82, 0.0, 1.0) * (0.6 + local_noise * 0.4)
        edge_wear = chipped * np.clip(defect_strength * 1.25, 0.0, 1.0)

        cavity = stone * 0.42 + groove_tint.reshape(1, 1, 3) * 0.58
        cavity *= (0.78 + local_noise[:, :, None] * 0.22)

        if normalized_style == "scratch":
            striations = np.clip((detail_noise - 0.52) * 1.7, 0.0, 1.0) * scratch_mid
            if background_image is not None:
                visibility_gain = 1.42 + max(0.0, thickness - 1.0) * 0.34
                textured_carve = rune_texture_rgb.copy()
                textured_carve *= (
                    0.92
                    + (rune_texture_luma[:, :, None] - 0.5) * 0.22
                    + (detail_noise[:, :, None] - 0.5) * 0.10
                )
                textured_carve = textured_carve * 0.88 + stone * 0.12
                textured_carve = np.minimum(textured_carve, stone * 0.74 + 0.08)
                textured_carve += (
                    groove_tint.reshape(1, 1, 3) - textured_carve
                ) * scratch_core[:, :, None] * 0.30
                textured_carve += (
                    shadow_tint.reshape(1, 1, 3) - textured_carve
                ) * shadow[:, :, None] * 0.40
                textured_carve += (
                    highlight_tint.reshape(1, 1, 3) - textured_carve
                ) * highlight[:, :, None] * 0.12
                textured_carve += (
                    shadow_tint.reshape(1, 1, 3) - textured_carve
                ) * striations[:, :, None] * 0.24
                textured_carve += (
                    shadow_tint.reshape(1, 1, 3) - textured_carve
                ) * edge_wear[:, :, None] * 0.14

                blend_alpha = np.clip(
                    scratch_mid * (0.54 + max(0.0, thickness - 1.0) * 0.12)
                    + scratch_core * (0.30 + max(0.0, thickness - 1.0) * 0.10),
                    0.0,
                    0.92,
                )
                scratch_rgb = stone * (1.0 - blend_alpha[:, :, None]) + textured_carve * blend_alpha[:, :, None]
                scratch_rgb += (
                    highlight_tint.reshape(1, 1, 3) - scratch_rgb
                ) * scratch_edge[:, :, None] * 0.16 * visibility_gain
                scratch_rgb += (
                    shadow_tint.reshape(1, 1, 3) - scratch_rgb
                ) * scratch_core[:, :, None] * 0.22 * visibility_gain
            else:
                visibility_gain = 1.0 + max(0.0, thickness - 1.0) * 0.9
                scratch_base = stone * (1.0 - scratch_core[:, :, None] * (0.06 * visibility_gain))
                scratch_rgb = scratch_base + highlight[:, :, None] * ((0.28 + texture_relief[:, :, None] * 0.08) * visibility_gain)
                scratch_rgb -= shadow[:, :, None] * ((0.18 + texture_relief[:, :, None] * 0.10) * visibility_gain)
                scratch_rgb -= scratch_core[:, :, None] * ((0.08 + texture_relief[:, :, None] * 0.06) * visibility_gain)
                scratch_rgb += scratch_edge[:, :, None] * (0.05 * visibility_gain)
                scratch_rgb += striations[:, :, None] * (0.05 * visibility_gain)
                scratch_rgb += edge_wear[:, :, None] * ((0.03 + texture_relief[:, :, None] * 0.03) * visibility_gain)
            composite = np.clip(scratch_rgb, 0.0, 1.0)
            local_base_luma = float(texture_gray.mean())
            contrast_boost = float(texture_gray.std()) * 0.9
            scratch_luma = np.clip(
                local_base_luma
                + highlight * (0.12 + contrast_boost * 0.38 + max(0.0, thickness - 1.0) * 0.04)
                - shadow * (0.12 + contrast_boost * 0.34 + max(0.0, thickness - 1.0) * 0.04)
                - scratch_core * (0.11 + contrast_boost * 0.24 + max(0.0, thickness - 1.0) * 0.05)
                + scratch_edge * (0.05 + contrast_boost * 0.12 + max(0.0, thickness - 1.0) * 0.02)
                + striations * 0.03
                + edge_wear * 0.03
                + (detail_noise - 0.5) * 0.04,
                0.08,
                0.96,
            )
            if normalized_bg == "transparent":
                cutout_rgb = rune_texture_rgb.copy()
                cutout_rgb += (
                    rune_light_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * highlight[:, :, None] * (0.46 + max(0.0, thickness - 1.0) * 0.05)
                cutout_rgb += (
                    rune_dark_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * shadow[:, :, None] * (0.40 + max(0.0, thickness - 1.0) * 0.05)
                cutout_rgb += (
                    rune_dark_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * scratch_core[:, :, None] * 0.22
                cutout_rgb += (
                    rune_light_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * scratch_edge[:, :, None] * 0.16
                cutout_rgb += (
                    rune_dark_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * striations[:, :, None] * 0.14
                cutout_rgb += (
                    rune_dark_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * edge_wear[:, :, None] * 0.12
                cutout_rgb *= 0.90 + detail_noise[:, :, None] * 0.18
            else:
                cutout_rgb = np.repeat(scratch_luma[:, :, None], 3, axis=2)
        else:
            composite = stone * (1.0 - groove_depth[:, :, None] * 0.46)
            composite = composite * (1.0 - ambient[:, :, None] * 0.08)
            composite = composite * (1.0 - shadow[:, :, None] * 0.40)
            composite += cavity * groove_depth[:, :, None] * 0.94
            composite += highlight[:, :, None] * 0.22
            composite -= shadow[:, :, None] * 0.10
            composite += edge_wear[:, :, None] * 0.035
            composite = np.clip(composite, 0.0, 1.0)
            if normalized_bg == "transparent":
                cutout_rgb = rune_texture_rgb.copy()
                cutout_rgb += (
                    rune_light_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * highlight[:, :, None] * 0.26
                cutout_rgb += (
                    rune_dark_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * shadow[:, :, None] * 0.22
                cutout_rgb += (
                    rune_dark_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * groove_depth[:, :, None] * 0.18
                cutout_rgb += (
                    rune_dark_rgb.reshape(1, 1, 3) - cutout_rgb
                ) * edge_wear[:, :, None] * 0.08
                cutout_rgb *= 0.92 + local_noise[:, :, None] * 0.16
            else:
                cutout_rgb = cavity + highlight[:, :, None] * 0.18 - shadow[:, :, None] * 0.13
                cutout_rgb += edge_wear[:, :, None] * 0.05
        cutout_rgb = np.clip(cutout_rgb, 0.0, 1.0)
        if normalized_style == "scratch":
            alpha = np.clip((mask_soft ** 1.15) * 215.0, 0, 255).astype(np.uint8)
        else:
            alpha = np.clip(mask_soft * 255.0, 0, 255).astype(np.uint8)
        if normalized_style == "scratch":
            scratch_alpha = (
                scratch_core * (118.0 + max(0.0, thickness - 1.0) * 24.0)
                + (highlight + shadow) * (38.0 + max(0.0, thickness - 1.0) * 10.0)
                + scratch_edge * (16.0 + max(0.0, thickness - 1.0) * 6.0)
                + edge_wear * 10.0
            ) * (0.76 + detail_noise * 0.12 + texture_relief * 0.14)
            alpha = np.clip(scratch_alpha, 0, 196).astype(np.uint8)
        cutout_rgba = np.dstack(
            [
                (cutout_rgb[:, :, 0] * 255).astype(np.uint8),
                (cutout_rgb[:, :, 1] * 255).astype(np.uint8),
                (cutout_rgb[:, :, 2] * 255).astype(np.uint8),
                alpha,
            ]
        )
        overlay_rgba = cutout_rgba.copy()
        if normalized_bg == "transparent":
            composite_image = Image.fromarray(overlay_rgba, mode="RGBA")
        else:
            composite_image = Image.fromarray((composite * 255).astype(np.uint8), mode="RGB")

        size_factor = max(0.0, (96.0 - float(min(canvas_width, canvas_height))) / 96.0)
        if normalized_style == "scratch":
            mask_threshold = 0.17 - size_factor * 0.07 - max(0.0, thickness - 1.0) * 0.02
        else:
            mask_threshold = 0.22 - size_factor * 0.05 - max(0.0, thickness - 1.0) * 0.015
        mask_threshold = float(np.clip(mask_threshold, 0.07, 0.24))
        mask_binary = (mask_soft > mask_threshold).astype(np.uint8) * 255
        ys, xs = np.where(mask_binary > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("Binary rune mask is empty.")
        bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        return RenderedRune(
            composite=composite_image,
            mask=Image.fromarray(mask_binary.astype(np.uint8), mode="L"),
            cutout=Image.fromarray(cutout_rgba, mode="RGBA"),
            overlay=Image.fromarray(overlay_rgba, mode="RGBA"),
            bbox=bbox,
            metadata={
                "bbox_left": bbox[0],
                "bbox_top": bbox[1],
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "coverage": float(mask_binary.mean()) / 255.0,
                "palette": resolved_palette,
                "style": normalized_style,
                "background_mode": normalized_bg,
                "defect_strength": defect_strength,
                "thickness": thickness,
                "background_source": "image" if background_image is not None else "synthetic",
                "rune_texture": rune_texture_name,
            },
        )
