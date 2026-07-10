from __future__ import annotations

import base64
import logging
import mimetypes
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests

from rag_runes.ocr.base import OCRClient
from rag_runes.text_utils import normalize_whitespace


CODE_FENCE_LINE_RE = re.compile(r"^```(?:\w+)?\s*$|^```\s*$")
EMPTY_OCR_RE = re.compile(
    r"^(?:no text|no visible text|empty|none|null|нет текста|текста нет)$",
    re.IGNORECASE,
)
LOGGER = logging.getLogger(__name__)


class NoopOCRClient:
    def extract_text(self, image_path: str) -> str:
        return ""


class TesseractOCRClient:
    """Local OCR client backed by the tesseract CLI."""

    def __init__(
        self,
        languages: str = "rus+eng",
        psm: int = 6,
        timeout_s: int = 90,
        executable: str = "tesseract",
    ) -> None:
        self.languages = languages
        self.psm = psm
        self.timeout_s = timeout_s
        self.executable = executable

    @staticmethod
    def _clean_response_text(value: str) -> str:
        lines = [
            normalize_whitespace(line)
            for line in value.splitlines()
            if normalize_whitespace(line)
        ]
        return "\n".join(lines).strip()

    def extract_text(self, image_path: str) -> str:
        if shutil.which(self.executable) is None:
            raise RuntimeError(f"tesseract executable not found: {self.executable}")
        cmd = [
            self.executable,
            image_path,
            "stdout",
            "-l",
            self.languages,
            "--psm",
            str(self.psm),
            "--oem",
            "1",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Tesseract OCR timed out for '{image_path}' after {self.timeout_s}s"
            ) from exc
        if result.returncode != 0:
            stderr = result.stderr.strip() or "no stderr"
            raise RuntimeError(f"Tesseract OCR failed for '{image_path}': {stderr}")
        return self._clean_response_text(result.stdout)


class GLMOCRClient:
    """OpenAI-compatible client for GLM-OCR style vision endpoints."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_s: int = 90,
        max_retries: int = 3,
        retry_delay_s: float = 30.0,
        min_interval_s: float = 0.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self.retry_delay_s = max(0.0, retry_delay_s)
        self.min_interval_s = max(0.0, min_interval_s)
        self._last_request_at = 0.0

    @staticmethod
    def _data_url(image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        raw = Path(image_path).read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _clean_response_text(value: str) -> str:
        lines = [
            line.strip()
            for line in value.splitlines()
            if line.strip() and not CODE_FENCE_LINE_RE.match(line.strip())
        ]
        cleaned = "\n".join(lines).strip()
        if EMPTY_OCR_RE.match(cleaned):
            return ""
        return cleaned

    def _wait_for_rate_limit(self) -> None:
        if self.min_interval_s <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait_s = self.min_interval_s - elapsed
        if wait_s > 0:
            time.sleep(wait_s)

    def _retry_delay(self, response: requests.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        return self.retry_delay_s * max(1, attempt)

    def extract_text(self, image_path: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict OCR engine. Transcribe visible text exactly. "
                        "Do not summarize, translate, normalize spelling, correct typos, "
                        "expand abbreviations, or add explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe every visible text fragment from this image as accurately "
                                "as possible. Preserve original language, order, punctuation, digits, "
                                "diacritics, line breaks, and symbolic/runic characters. If a character "
                                "is unreadable, write [?] instead of guessing. Return only the OCR text."
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

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            response: requests.Response | None = None
            try:
                self._wait_for_rate_limit()
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                self._last_request_at = time.monotonic()
                if response.status_code == 429:
                    if attempt <= self.max_retries:
                        wait_s = self._retry_delay(response=response, attempt=attempt)
                        LOGGER.warning(
                            "OCR rate limited for '%s' on attempt %s/%s; sleeping %.1fs",
                            image_path,
                            attempt,
                            self.max_retries + 1,
                            wait_s,
                        )
                        time.sleep(wait_s)
                        continue
                    LOGGER.warning(
                        "OCR rate limited for '%s' on final attempt %s/%s",
                        image_path,
                        attempt,
                        self.max_retries + 1,
                    )
                response.raise_for_status()
                body = response.json()
                break
            except requests.Timeout as exc:
                last_error = exc
                wait_s = self._retry_delay(response=response, attempt=attempt)
                LOGGER.warning(
                    "OCR timeout for '%s' on attempt %s/%s; sleeping %.1fs",
                    image_path,
                    attempt,
                    self.max_retries + 1,
                    wait_s,
                )
                if attempt <= self.max_retries:
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"GLM-OCR request failed for '{image_path}': {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                raise RuntimeError(f"GLM-OCR request failed for '{image_path}': {exc}") from exc
        else:
            raise RuntimeError(f"GLM-OCR request failed for '{image_path}': {last_error}")

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"GLM-OCR response has no choices for '{image_path}'")
        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return self._clean_response_text(content)
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        text_parts.append(str(text))
            return self._clean_response_text(" ".join(text_parts))
        return self._clean_response_text(str(content))


def build_ocr_client(
    endpoint: str | None,
    model: str,
    api_key: str | None = None,
    timeout_s: int = 90,
    max_retries: int = 3,
    retry_delay_s: float = 30.0,
    min_interval_s: float = 0.0,
) -> OCRClient:
    selected_model = (model or "").strip().lower()
    if selected_model in {"none", "off", "disabled", "skip"}:
        return NoopOCRClient()
    if selected_model == "tesseract" or selected_model.startswith("tesseract:"):
        parts = (model or "tesseract").split(":")
        languages = parts[1] if len(parts) >= 2 and parts[1] else "rus+eng"
        psm = int(parts[2]) if len(parts) >= 3 and parts[2] else 6
        return TesseractOCRClient(languages=languages, psm=psm, timeout_s=timeout_s)
    if not endpoint:
        raise ValueError("OCR endpoint is not configured. Set GLM_OCR endpoint.")
    if not model:
        raise ValueError("OCR model is not configured. Set GLM_OCR model name.")
    return GLMOCRClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_delay_s=retry_delay_s,
        min_interval_s=min_interval_s,
    )
