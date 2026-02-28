from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import requests

from rag_runes.ocr.base import OCRClient


class GLMOCRClient:
    """OpenAI-compatible client for GLM-OCR style vision endpoints."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_s: int = 90,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    @staticmethod
    def _data_url(image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        raw = Path(image_path).read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def extract_text(self, image_path: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract text and symbolic labels from this image. "
                                "Return plain text only."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self._data_url(image_path)},
                        },
                    ],
                }
            ],
        }

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"GLM-OCR request failed for '{image_path}': {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"GLM-OCR response has no choices for '{image_path}'")
        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        text_parts.append(str(text))
            return " ".join(text_parts).strip()
        return str(content).strip()


def build_ocr_client(
    endpoint: str | None,
    model: str,
    api_key: str | None = None,
) -> OCRClient:
    if not endpoint:
        raise ValueError("OCR endpoint is not configured. Set GLM_OCR endpoint.")
    if not model:
        raise ValueError("OCR model is not configured. Set GLM_OCR model name.")
    return GLMOCRClient(endpoint=endpoint, model=model, api_key=api_key)
