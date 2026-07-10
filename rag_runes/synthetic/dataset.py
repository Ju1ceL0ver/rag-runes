from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_runes.synthetic.generator import RuneMaskGenerator
from rag_runes.synthetic.renderer import RenderedRune, StoneRuneRenderer


@dataclass(slots=True)
class SyntheticSampleRecord:
    image_path: str
    mask_path: str
    cutout_path: str
    overlay_path: str
    class_id: int
    class_name: str
    seed: int
    score: float
    occupancy: float
    bbox: dict[str, int]
    renderer: dict[str, float | int | str]

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "cutout_path": self.cutout_path,
            "overlay_path": self.overlay_path,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "seed": self.seed,
            "score": self.score,
            "occupancy": self.occupancy,
            "bbox": self.bbox,
            "renderer": self.renderer,
        }


class SyntheticRuneDatasetBuilder:
    def __init__(
        self,
        generator: RuneMaskGenerator,
        renderer: StoneRuneRenderer,
    ) -> None:
        self.generator = generator
        self.renderer = renderer

    def _save_artifact(
        self,
        output_dir: Path,
        class_name: str,
        sample_name: str,
        rendered: RenderedRune,
    ) -> tuple[Path, Path, Path, Path]:
        images_dir = output_dir / "images" / class_name
        masks_dir = output_dir / "masks" / class_name
        cutouts_dir = output_dir / "cutouts" / class_name
        overlays_dir = output_dir / "overlays" / class_name
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        cutouts_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        image_path = images_dir / f"{sample_name}.png"
        mask_path = masks_dir / f"{sample_name}.png"
        cutout_path = cutouts_dir / f"{sample_name}.png"
        overlay_path = overlays_dir / f"{sample_name}.png"
        rendered.composite.save(image_path)
        rendered.mask.save(mask_path)
        rendered.cutout.save(cutout_path)
        rendered.overlay.save(overlay_path)
        return image_path, mask_path, cutout_path, overlay_path

    def build(
        self,
        output_dir: Path,
        variants_per_class: int,
        class_ids: list[int] | None = None,
        canvas_size: int = 256,
        base_seed: int = 42,
        palette_name: str = "auto",
        style: str = "engraved",
        background_mode: str = "transparent",
        defect_strength: float = 0.28,
        thickness: float = 1.0,
    ) -> dict:
        if variants_per_class <= 0:
            raise ValueError("variants_per_class must be > 0")

        output_dir.mkdir(parents=True, exist_ok=True)
        selected_classes = (
            list(class_ids)
            if class_ids is not None
            else list(range(self.generator.num_classes))
        )
        manifest_path = output_dir / "manifest.jsonl"
        records: list[SyntheticSampleRecord] = []
        counter = 0

        for class_id in selected_classes:
            class_name = self.generator.class_names[class_id]
            for variant_idx in range(variants_per_class):
                sample_seed = base_seed + counter
                counter += 1
                sample = self.generator.generate(
                    class_id=class_id,
                    seed=sample_seed,
                )
                rendered = self.renderer.render(
                    sample.mask_image,
                    seed=sample_seed,
                    canvas_size=canvas_size,
                    palette_name=palette_name,
                    style=style,
                    background_mode=background_mode,
                    defect_strength=defect_strength,
                    thickness=thickness,
                )
                sample_name = f"{class_name}_{variant_idx:05d}"
                image_path, mask_path, cutout_path, overlay_path = self._save_artifact(
                    output_dir=output_dir,
                    class_name=class_name,
                    sample_name=sample_name,
                    rendered=rendered,
                )

                record = SyntheticSampleRecord(
                    image_path=str(image_path.relative_to(output_dir)),
                    mask_path=str(mask_path.relative_to(output_dir)),
                    cutout_path=str(cutout_path.relative_to(output_dir)),
                    overlay_path=str(overlay_path.relative_to(output_dir)),
                    class_id=sample.class_id,
                    class_name=sample.class_name,
                    seed=sample.seed,
                    score=sample.score,
                    occupancy=sample.occupancy,
                    bbox={
                        "left": rendered.bbox[0],
                        "top": rendered.bbox[1],
                        "right": rendered.bbox[2],
                        "bottom": rendered.bbox[3],
                    },
                    renderer=rendered.metadata,
                )
                records.append(record)

        with manifest_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        summary = {
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "classes": len(selected_classes),
            "variants_per_class": variants_per_class,
            "samples": len(records),
            "canvas_size": canvas_size,
            "palette_name": palette_name,
            "style": style,
            "background_mode": background_mode,
            "defect_strength": defect_strength,
            "thickness": thickness,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
