from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from rag_runes.synthetic.generator import RuneMaskGenerator
from rag_runes.synthetic.renderer import StoneRuneRenderer


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = float((right - left) * (bottom - top))
    area_a = float(max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = float(max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    return intersection / max(area_a + area_b - intersection, 1.0)


def _bbox_intersects(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    padding: int = 0,
) -> bool:
    return not (
        box_a[2] + padding <= box_b[0]
        or box_b[2] + padding <= box_a[0]
        or box_a[3] + padding <= box_b[1]
        or box_b[3] + padding <= box_a[1]
    )


def _stone_likelihood_map(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    saturation = np.zeros_like(max_rgb)
    np.divide(max_rgb - min_rgb, max_rgb, out=saturation, where=max_rgb > 1e-6)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    luma = red * 0.299 + green * 0.587 + blue * 0.114
    green_bias = np.clip(green - np.maximum(red, blue), 0.0, 1.0)
    neutrality = 1.0 - np.std(rgb, axis=2) * 2.4
    neutrality = np.clip(neutrality, 0.0, 1.0)

    grad_x = np.abs(luma - np.roll(luma, 1, axis=1))
    grad_y = np.abs(luma - np.roll(luma, 1, axis=0))
    texture = np.clip((grad_x + grad_y) * 4.0, 0.0, 1.0)
    texture_pref = 1.0 - np.abs(texture - 0.22) / 0.22
    texture_pref = np.clip(texture_pref, 0.0, 1.0)

    light_pref = 1.0 - np.abs(luma - 0.52) / 0.52
    light_pref = np.clip(light_pref, 0.0, 1.0)

    score = (
        neutrality * 0.38
        + (1.0 - saturation) * 0.26
        + (1.0 - green_bias) * 0.22
        + texture_pref * 0.08
        + light_pref * 0.06
    )
    return np.clip(score, 0.0, 1.0)


def _dominant_stone_mask(score_map: np.ndarray, threshold: float, max_components: int = 3) -> np.ndarray:
    binary = score_map >= threshold
    height, width = binary.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[tuple[float, list[tuple[int, int]]]] = []

    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            score_sum = 0.0
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                score_sum += float(score_map[cy, cx])
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            component_score = score_sum * len(pixels)
            components.append((component_score, pixels))

    if not components:
        return np.ones_like(score_map, dtype=np.float32)

    components.sort(key=lambda item: item[0], reverse=True)
    selected = components[:max_components]
    mask = np.zeros_like(score_map, dtype=np.float32)
    for _, pixels in selected:
        for py, px in pixels:
            mask[py, px] = 1.0
    return mask


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask >= 0.5)
    height, width = mask.shape
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, width, height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


@dataclass(slots=True)
class SceneInstance:
    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    seed: int
    score: float
    occupancy: float
    renderer: dict[str, float | int | str]

    def to_dict(self, image_width: int, image_height: int) -> dict:
        left, top, right, bottom = self.bbox
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            },
            "yolo_bbox": {
                "cx": ((left + right) / 2.0) / image_width,
                "cy": ((top + bottom) / 2.0) / image_height,
                "width": (right - left) / image_width,
                "height": (bottom - top) / image_height,
            },
            "seed": self.seed,
            "score": self.score,
            "occupancy": self.occupancy,
            "renderer": self.renderer,
        }

    def to_yolo_line(self, image_width: int, image_height: int) -> str:
        payload = self.to_dict(image_width, image_height)["yolo_bbox"]
        return (
            f"{self.class_id} "
            f"{payload['cx']:.6f} "
            f"{payload['cy']:.6f} "
            f"{payload['width']:.6f} "
            f"{payload['height']:.6f}"
        )


class PhotoRuneSceneBuilder:
    def __init__(
        self,
        generator: RuneMaskGenerator,
        renderer: StoneRuneRenderer,
    ) -> None:
        self.generator = generator
        self.renderer = renderer

    def _load_background_candidates(
        self,
        background_path: Path | None = None,
        backgrounds_dir: Path | None = None,
    ) -> list[Path]:
        candidates: list[Path] = []
        if background_path is not None:
            if not background_path.exists() or not background_path.is_file():
                raise ValueError(f"Background image not found: {background_path}")
            candidates.append(background_path)
        if backgrounds_dir is not None:
            if not backgrounds_dir.exists() or not backgrounds_dir.is_dir():
                raise ValueError(f"Background directory not found: {backgrounds_dir}")
            for path in sorted(backgrounds_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                    candidates.append(path)
        unique_candidates = list(dict.fromkeys(candidates))
        if not unique_candidates:
            raise ValueError("No background images found.")
        return unique_candidates

    def _prepare_background(
        self,
        background_path: Path,
        min_scene_side: int = 512,
    ) -> Image.Image:
        background = Image.open(background_path).convert("RGB")
        width, height = background.size
        short_side = min(width, height)
        if short_side >= min_scene_side:
            return background
        scale = min_scene_side / max(short_side, 1)
        resized = background.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            resample=Image.Resampling.LANCZOS,
        )
        return resized

    def _sample_patch_size(
        self,
        glyph_bbox: tuple[int, int, int, int],
        scene_width: int,
        scene_height: int,
        rng: np.random.Generator,
        scale_min: float,
        scale_max: float,
    ) -> tuple[int, int]:
        glyph_width = max(1, glyph_bbox[2] - glyph_bbox[0])
        glyph_height = max(1, glyph_bbox[3] - glyph_bbox[1])
        short_side = min(scene_width, scene_height)
        target_short = int(short_side * float(rng.uniform(scale_min, scale_max)))
        target_short = max(14, min(target_short, int(short_side * 0.72)))

        aspect = glyph_width / max(glyph_height, 1)
        if aspect >= 1.0:
            patch_width = int(target_short * aspect * float(rng.uniform(1.30, 1.55)))
            patch_height = int(target_short * float(rng.uniform(1.25, 1.45)))
        else:
            patch_width = int(target_short * float(rng.uniform(1.25, 1.45)))
            patch_height = int(target_short / max(aspect, 1e-6) * float(rng.uniform(1.30, 1.55)))

        patch_width = max(18, min(patch_width, scene_width - 4))
        patch_height = max(18, min(patch_height, scene_height - 4))
        return patch_width, patch_height

    def _choose_patch_location(
        self,
        score_map: np.ndarray,
        patch_width: int,
        patch_height: int,
        rng: np.random.Generator,
        occupied_boxes: list[tuple[int, int, int, int]],
        max_iou: float,
        score_threshold: float,
        center_threshold: float,
        roi_mask: np.ndarray,
        placement_bounds: tuple[int, int, int, int] | None = None,
        min_gap: int = 0,
        attempts: int = 80,
    ) -> tuple[int, int, int, int] | None:
        image_height, image_width = score_map.shape
        if placement_bounds is None:
            bound_left, bound_top, bound_right, bound_bottom = (0, 0, image_width, image_height)
        else:
            bound_left, bound_top, bound_right, bound_bottom = placement_bounds
        if bound_right - bound_left < patch_width or bound_bottom - bound_top < patch_height:
            return None
        placement_center_x = (bound_left + bound_right) / 2.0
        placement_center_y = (bound_top + bound_bottom) / 2.0
        placement_half_w = max((bound_right - bound_left) / 2.0, 1.0)
        placement_half_h = max((bound_bottom - bound_top) / 2.0, 1.0)
        best: tuple[float, tuple[int, int, int, int]] | None = None
        for _ in range(attempts):
            left = int(rng.integers(bound_left, max(bound_left + 1, bound_right - patch_width + 1)))
            top = int(rng.integers(bound_top, max(bound_top + 1, bound_bottom - patch_height + 1)))
            box = (left, top, left + patch_width, top + patch_height)
            if any(
                _bbox_intersects(box, existing, padding=min_gap) or _bbox_iou(box, existing) > max_iou
                for existing in occupied_boxes
            ):
                continue
            center_x = left + patch_width // 2
            center_y = top + patch_height // 2
            if roi_mask[center_y, center_x] < 0.5:
                continue
            patch_score = score_map[top : top + patch_height, left : left + patch_width]
            patch_roi = roi_mask[top : top + patch_height, left : left + patch_width]
            inner_top = max(0, patch_height // 5)
            inner_bottom = max(inner_top + 1, patch_height - patch_height // 5)
            inner_left = max(0, patch_width // 5)
            inner_right = max(inner_left + 1, patch_width - patch_width // 5)
            center_score = float(patch_score[inner_top:inner_bottom, inner_left:inner_right].mean())
            local_score = float(patch_score.mean())
            stone_fraction = float((patch_score >= score_threshold).mean())
            roi_fraction = float((patch_roi >= 0.5).mean())
            blended_score = local_score * 0.40 + center_score * 0.60
            if (
                local_score < score_threshold
                or center_score < center_threshold
                or stone_fraction < 0.66
                or roi_fraction < 0.80
            ):
                continue
            center_bias_x = abs(center_x - placement_center_x) / placement_half_w
            center_bias_y = abs(center_y - placement_center_y) / placement_half_h
            center_bias = np.clip(1.0 - (center_bias_x * 0.65 + center_bias_y * 0.35), 0.0, 1.0)
            weighted_score = blended_score * 0.82 + float(center_bias) * 0.18
            if best is None or weighted_score > best[0]:
                best = (weighted_score, box)
        return None if best is None else best[1]

    def render_scene(
        self,
        output_dir: Path,
        background_path: Path,
        instance_count: int | None,
        seed: int = 42,
        instance_count_min: int = 25,
        instance_count_max: int = 45,
        center_coverage: float = 0.70,
        class_ids: list[int] | None = None,
        style: str = "scratch",
        defect_strength: float = 0.0,
        thickness: float = 1.6,
        scale_min: float = 0.08,
        scale_max: float = 0.16,
        max_iou: float = 0.14,
        export_name: str | None = None,
        split: str | None = None,
    ) -> dict:
        if instance_count is None:
            if instance_count_min <= 0 or instance_count_max < instance_count_min:
                raise ValueError("Invalid instance_count_min/instance_count_max values.")
            target_instance_count = int(np.random.default_rng(seed).integers(instance_count_min, instance_count_max + 1))
        else:
            if instance_count <= 0:
                raise ValueError("instance_count must be > 0")
            target_instance_count = int(instance_count)
        if not (0.30 <= center_coverage <= 1.0):
            raise ValueError("center_coverage must be between 0.30 and 1.0")
        if not (0.02 <= scale_min <= scale_max <= 0.95):
            raise ValueError("scale_min/scale_max are out of range")

        background = self._prepare_background(background_path)
        scene = background.copy()
        scene_width, scene_height = scene.size
        score_map = _stone_likelihood_map(background)
        density_ratio = min(1.0, max(0.0, (target_instance_count - 10) / 24.0))
        adaptive_scale_min = max(0.035, scale_min * (1.0 - density_ratio * 0.45))
        adaptive_scale_max = max(adaptive_scale_min + 0.01, scale_max * (1.0 - density_ratio * 0.42))
        adaptive_max_iou = min(0.28, max_iou + density_ratio * 0.08)
        score_threshold = max(0.42, float(np.quantile(score_map, 0.66)) - density_ratio * 0.10)
        center_threshold = max(score_threshold + 0.02, float(np.quantile(score_map, 0.78)) - density_ratio * 0.10)
        roi_mask = _dominant_stone_mask(score_map, threshold=center_threshold, max_components=1)
        roi_bbox = _mask_bbox(roi_mask)
        shrink_ratio_x = max(0.06, 0.16 - density_ratio * 0.07)
        shrink_ratio_y = max(0.08, 0.20 - density_ratio * 0.09)
        pad_x = int((roi_bbox[2] - roi_bbox[0]) * shrink_ratio_x)
        pad_y = int((roi_bbox[3] - roi_bbox[1]) * shrink_ratio_y)
        placement_bounds = (
            roi_bbox[0] + pad_x,
            roi_bbox[1] + pad_y,
            roi_bbox[2] - pad_x,
            roi_bbox[3] - pad_y,
        )
        side_margin_x = (1.0 - center_coverage) * 0.5
        side_margin_y = (1.0 - center_coverage) * 0.5
        image_center_bounds = (
            int(scene_width * side_margin_x),
            int(scene_height * side_margin_y),
            int(scene_width * (1.0 - side_margin_x)),
            int(scene_height * (1.0 - side_margin_y)),
        )
        centered_bounds = (
            max(placement_bounds[0], image_center_bounds[0]),
            max(placement_bounds[1], image_center_bounds[1]),
            min(placement_bounds[2], image_center_bounds[2]),
            min(placement_bounds[3], image_center_bounds[3]),
        )
        if centered_bounds[2] - centered_bounds[0] >= max(48, scene_width // 5) and centered_bounds[3] - centered_bounds[1] >= max(48, scene_height // 5):
            placement_bounds = centered_bounds
        if (
            placement_bounds[2] - placement_bounds[0] < max(48, scene_width // 5)
            or placement_bounds[3] - placement_bounds[1] < max(48, scene_height // 5)
        ):
            placement_bounds = roi_bbox
        min_gap = max(2, min(scene_width, scene_height) // 140)
        placement_width = max(32, placement_bounds[2] - placement_bounds[0])
        placement_height = max(32, placement_bounds[3] - placement_bounds[1])
        occupied_patch_boxes: list[tuple[int, int, int, int]] = []
        instances: list[SceneInstance] = []
        mask_scene = Image.new("L", scene.size, 0)
        rng = np.random.default_rng(seed)
        class_pool = (
            list(class_ids)
            if class_ids is not None and len(class_ids) > 0
            else list(range(self.generator.num_classes))
        )
        if not class_pool:
            raise ValueError("No classes available for scene generation.")

        max_candidates = target_instance_count * (6 if target_instance_count >= 25 else 3)
        for index in range(max_candidates):
            if len(instances) >= target_instance_count:
                break
            placed = False
            for attempt in range(140 if target_instance_count >= 25 else 80):
                instance_seed = int(seed + index * 1009 + attempt * 97 + rng.integers(0, 97))
                class_id = int(class_pool[int(rng.integers(0, len(class_pool)))])
                sample = self.generator.generate(class_id=class_id, seed=instance_seed)
                glyph_bbox = sample.mask_image.getbbox()
                if glyph_bbox is None:
                    continue
                patch_width, patch_height = self._sample_patch_size(
                    glyph_bbox=glyph_bbox,
                    scene_width=placement_width,
                    scene_height=placement_height,
                    rng=rng,
                    scale_min=adaptive_scale_min,
                    scale_max=adaptive_scale_max,
                )
                patch_box = self._choose_patch_location(
                    score_map=score_map,
                    patch_width=patch_width,
                    patch_height=patch_height,
                    rng=rng,
                    occupied_boxes=occupied_patch_boxes,
                    max_iou=adaptive_max_iou,
                    score_threshold=score_threshold,
                    center_threshold=center_threshold,
                    roi_mask=roi_mask,
                    placement_bounds=placement_bounds,
                    min_gap=min_gap,
                    attempts=140 if target_instance_count >= 25 else 90,
                )
                if patch_box is None:
                    continue

                patch = background.crop(patch_box)
                try:
                    rendered = self.renderer.render(
                        sample.mask_image,
                        seed=instance_seed,
                        canvas_size=(patch_width, patch_height),
                        style=style,
                        background_mode="stone",
                        defect_strength=defect_strength,
                        thickness=thickness,
                        background_image=patch,
                        centered=True,
                    )
                except RuntimeError:
                    continue
                local_bbox = rendered.bbox
                global_bbox = (
                    patch_box[0] + local_bbox[0],
                    patch_box[1] + local_bbox[1],
                    patch_box[0] + local_bbox[2],
                    patch_box[1] + local_bbox[3],
                )
                if any(
                    _bbox_intersects(global_bbox, item.bbox, padding=min_gap)
                    or _bbox_iou(global_bbox, item.bbox) > adaptive_max_iou
                    for item in instances
                ):
                    continue

                scene.paste(rendered.composite, patch_box[:2])
                existing_mask = mask_scene.crop(patch_box)
                merged_mask = ImageChops.lighter(existing_mask, rendered.mask)
                mask_scene.paste(merged_mask, patch_box[:2])
                occupied_patch_boxes.append(patch_box)
                instances.append(
                    SceneInstance(
                        class_id=sample.class_id,
                        class_name=sample.class_name,
                        bbox=global_bbox,
                        seed=sample.seed,
                        score=sample.score,
                        occupancy=sample.occupancy,
                        renderer=rendered.metadata,
                    )
                )
                placed = True
                break
            if not placed:
                continue

        if not instances:
            raise RuntimeError("Failed to place any runes on the background image.")

        split_prefix = f"{split}/" if split else ""
        images_dir = output_dir / "images" / split if split else output_dir / "images"
        labels_dir = output_dir / "labels" / split if split else output_dir / "labels"
        masks_dir = output_dir / "masks" / split if split else output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        scene_name = _safe_name(export_name or f"{background_path.stem}_{seed}")
        image_path = images_dir / f"{scene_name}.png"
        label_path = labels_dir / f"{scene_name}.txt"
        mask_path = masks_dir / f"{scene_name}.png"
        scene.save(image_path)
        mask_scene.save(mask_path)
        label_path.write_text(
            "\n".join(item.to_yolo_line(scene_width, scene_height) for item in instances) + "\n",
            encoding="utf-8",
        )

        return {
            "status": "ok",
            "background_path": str(background_path),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "mask_path": str(mask_path),
            "width": scene_width,
            "height": scene_height,
            "split": split or "",
            "instances": [item.to_dict(scene_width, scene_height) for item in instances],
            "stone_score_mean": float(score_map.mean()),
            "stone_score_max": float(score_map.max()),
            "thickness": thickness,
            "instance_count": len(instances),
            "scene_name": scene_name,
            "scene_relpath": f"{split_prefix}{scene_name}.png" if split else f"{scene_name}.png",
        }

    def build_detection_dataset(
        self,
        output_dir: Path,
        backgrounds_dir: Path | None = None,
        background_path: Path | None = None,
        images_per_background: int = 4,
        instances_min: int = 25,
        instances_max: int = 45,
        center_coverage: float = 0.70,
        class_ids: list[int] | None = None,
        base_seed: int = 42,
        style: str = "scratch",
        defect_strength: float = 0.0,
        thickness: float = 1.6,
        scale_min: float = 0.08,
        scale_max: float = 0.16,
        max_iou: float = 0.14,
    ) -> dict:
        if images_per_background <= 0:
            raise ValueError("images_per_background must be > 0")
        if instances_min <= 0 or instances_max < instances_min:
            raise ValueError("Invalid instances_min/instances_max values.")

        output_dir.mkdir(parents=True, exist_ok=True)
        candidates = self._load_background_candidates(
            background_path=background_path,
            backgrounds_dir=backgrounds_dir,
        )

        total_scenes = len(candidates) * images_per_background
        manifest_path = output_dir / "manifest.jsonl"
        records: list[dict] = []
        rng = np.random.default_rng(base_seed)
        counter = 0

        for background_index, candidate in enumerate(candidates):
            for repeat_index in range(images_per_background):
                split = "val" if (counter % 10 == 0) else "train"
                instance_count = int(rng.integers(instances_min, instances_max + 1))
                scene_seed = int(base_seed + background_index * 10007 + repeat_index * 389 + rng.integers(0, 997))
                record = self.render_scene(
                    output_dir=output_dir,
                    background_path=candidate,
                    instance_count=instance_count,
                    seed=scene_seed,
                    center_coverage=center_coverage,
                    class_ids=class_ids,
                    style=style,
                    defect_strength=defect_strength,
                    thickness=thickness,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    max_iou=max_iou,
                    export_name=f"{candidate.stem}_{repeat_index:04d}",
                    split=split,
                )
                records.append(record)
                counter += 1

        with manifest_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        dataset_yaml_path = output_dir / "dataset.yaml"
        class_names = list(self.generator.class_names)
        dataset_yaml_path.write_text(
            "\n".join(
                [
                    f"path: {output_dir}",
                    "train: images/train",
                    "val: images/val",
                    f"nc: {len(class_names)}",
                    "names:",
                    *[f"  {idx}: {name}" for idx, name in enumerate(class_names)],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = {
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "dataset_yaml_path": str(dataset_yaml_path),
            "backgrounds": len(candidates),
            "images_per_background": images_per_background,
            "scenes": total_scenes,
            "train_scenes": sum(1 for record in records if record["split"] == "train"),
            "val_scenes": sum(1 for record in records if record["split"] == "val"),
            "instances_min": instances_min,
            "instances_max": instances_max,
            "center_coverage": center_coverage,
            "style": style,
            "defect_strength": defect_strength,
            "thickness": thickness,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "max_iou": max_iou,
            "selected_class_ids": list(class_ids) if class_ids is not None else None,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
