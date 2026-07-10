from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from rag_runes.synthetic.dataset import SyntheticRuneDatasetBuilder
from rag_runes.synthetic.generator import RuneMaskGenerator
from rag_runes.synthetic.renderer import StoneRuneRenderer
from rag_runes.synthetic.scene import PhotoRuneSceneBuilder


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


class RuneSynthesisService:
    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        device: str = "cpu",
        stone_texture_dir: Path | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self.generator = RuneMaskGenerator(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.renderer = StoneRuneRenderer(texture_dir=stone_texture_dir)
        self.builder = SyntheticRuneDatasetBuilder(
            generator=self.generator,
            renderer=self.renderer,
        )
        self.scene_builder = PhotoRuneSceneBuilder(
            generator=self.generator,
            renderer=self.renderer,
        )

    def render_single(
        self,
        class_id: int | None = None,
        class_name: str | None = None,
        seed: int = 42,
        canvas_size: int = 256,
        palette_name: str = "auto",
        style: str = "engraved",
        background_mode: str = "transparent",
        defect_strength: float = 0.0,
        thickness: float = 1.0,
        export_name: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            sample = self.generator.generate(
                class_id=class_id,
                class_name=class_name,
                seed=seed,
            )
            rendered = self.renderer.render(
                sample.mask_image,
                seed=seed,
                canvas_size=canvas_size,
                palette_name=palette_name,
                style=style,
                background_mode=background_mode,
                defect_strength=defect_strength,
                thickness=thickness,
            )

            tag = export_name or f"{sample.class_name}_{seed}"
            safe_name = _safe_name(tag)
            run_dir = self.output_dir / "single"
            run_dir.mkdir(parents=True, exist_ok=True)
            image_path = run_dir / f"{safe_name}.png"
            mask_path = run_dir / f"{safe_name}_mask.png"
            cutout_path = run_dir / f"{safe_name}_cutout.png"
            overlay_path = run_dir / f"{safe_name}_overlay.png"
            raw_path = run_dir / f"{safe_name}_raw.png"

            rendered.composite.save(image_path)
            rendered.mask.save(mask_path)
            rendered.cutout.save(cutout_path)
            rendered.overlay.save(overlay_path)
            sample.raw_image.save(raw_path)

            return {
                "status": "ok",
                "class_id": sample.class_id,
                "class_name": sample.class_name,
                "seed": sample.seed,
                "score": sample.score,
                "occupancy": sample.occupancy,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "cutout_path": str(cutout_path),
                "overlay_path": str(overlay_path),
                "raw_path": str(raw_path),
                "bbox": {
                    "left": rendered.bbox[0],
                    "top": rendered.bbox[1],
                    "right": rendered.bbox[2],
                    "bottom": rendered.bbox[3],
                },
                "renderer": rendered.metadata,
            }

    def build_dataset(
        self,
        variants_per_class: int,
        class_ids: list[int] | None = None,
        canvas_size: int = 256,
        base_seed: int = 42,
        palette_name: str = "auto",
        style: str = "engraved",
        background_mode: str = "transparent",
        defect_strength: float = 0.0,
        thickness: float = 1.0,
        output_name: str | None = None,
    ) -> dict[str, Any]:
        tag = output_name or f"dataset_{base_seed}_{variants_per_class}"
        safe_tag = _safe_name(tag)
        digest = hashlib.blake2b(
            f"{safe_tag}:{base_seed}:{variants_per_class}".encode("utf-8"),
            digest_size=4,
        ).hexdigest()
        output_dir = self.output_dir / "datasets" / f"{safe_tag}_{digest}"

        with self._lock:
            summary = self.builder.build(
                output_dir=output_dir,
                variants_per_class=variants_per_class,
                class_ids=class_ids,
                canvas_size=canvas_size,
                base_seed=base_seed,
                palette_name=palette_name,
                style=style,
                background_mode=background_mode,
                defect_strength=defect_strength,
                thickness=thickness,
            )
        summary["status"] = "ok"
        summary["checkpoint_path"] = str(self.checkpoint_path)
        return summary

    def render_scene(
        self,
        background_path: Path,
        instance_count: int | None = None,
        instance_count_min: int = 25,
        instance_count_max: int = 45,
        center_coverage: float = 0.70,
        seed: int = 42,
        class_ids: list[int] | None = None,
        style: str = "scratch",
        defect_strength: float = 0.0,
        thickness: float = 1.6,
        scale_min: float = 0.08,
        scale_max: float = 0.16,
        max_iou: float = 0.14,
        export_name: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            result = self.scene_builder.render_scene(
                output_dir=self.output_dir / "scenes",
                background_path=background_path,
                instance_count=instance_count,
                instance_count_min=instance_count_min,
                instance_count_max=instance_count_max,
                center_coverage=center_coverage,
                seed=seed,
                class_ids=class_ids,
                style=style,
                defect_strength=defect_strength,
                thickness=thickness,
                scale_min=scale_min,
                scale_max=scale_max,
                max_iou=max_iou,
                export_name=export_name,
            )
        result["checkpoint_path"] = str(self.checkpoint_path)
        return result

    def build_detection_dataset(
        self,
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
        output_name: str | None = None,
    ) -> dict[str, Any]:
        tag = output_name or f"detection_{base_seed}_{images_per_background}"
        safe_tag = _safe_name(tag)
        digest = hashlib.blake2b(
            f"{safe_tag}:{base_seed}:{images_per_background}:{instances_min}:{instances_max}".encode("utf-8"),
            digest_size=4,
        ).hexdigest()
        output_dir = self.output_dir / "detection" / f"{safe_tag}_{digest}"
        with self._lock:
            summary = self.scene_builder.build_detection_dataset(
                output_dir=output_dir,
                backgrounds_dir=backgrounds_dir,
                background_path=background_path,
                images_per_background=images_per_background,
                instances_min=instances_min,
                instances_max=instances_max,
                center_coverage=center_coverage,
                class_ids=class_ids,
                base_seed=base_seed,
                style=style,
                defect_strength=defect_strength,
                thickness=thickness,
                scale_min=scale_min,
                scale_max=scale_max,
                max_iou=max_iou,
            )
        summary["status"] = "ok"
        summary["checkpoint_path"] = str(self.checkpoint_path)
        return summary

    def health(self) -> dict[str, Any]:
        return {
            "ready": True,
            "checkpoint_path": str(self.checkpoint_path),
            "num_classes": self.generator.num_classes,
            "class_names": self.generator.class_names,
            "output_dir": str(self.output_dir),
        }
