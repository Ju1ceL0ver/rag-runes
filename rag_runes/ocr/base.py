from __future__ import annotations

from typing import Protocol


class OCRClient(Protocol):
    def extract_text(self, image_path: str) -> str:
        ...
