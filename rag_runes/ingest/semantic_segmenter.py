from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from rag_runes.text_utils import normalize_whitespace, split_text

JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s+(.*\S)\s*$")
PLACEHOLDER_RE = re.compile(
    r"^(chunk|segment)\s*(text)?\s*\d*$|^placeholder$|^example$",
    re.IGNORECASE,
)


class SemanticSegmenter:
    def split(
        self,
        text: str,
        section_title: str = "",
        max_chars: int = 1200,
        overlap: int = 180,
    ) -> list[str]:
        raise NotImplementedError


@dataclass(slots=True)
class LLMSemanticSegmenter(SemanticSegmenter):
    endpoint: str
    model: str
    api_key: str | None = None
    timeout_s: int = 180
    max_input_chars: int = 16000

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_message(self, body: dict[str, Any]) -> str:
        choices = body.get("choices") or []
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts).strip()
        return str(content).strip()

    def _normalize_list(self, values: list[Any]) -> list[str]:
        output: list[str] = []
        for value in values:
            text_value: str | None = None
            if isinstance(value, str):
                text_value = value
            elif isinstance(value, dict):
                for key in ("text", "content", "chunk", "value"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        text_value = candidate
                        break
            if text_value is None:
                continue
            cleaned = normalize_whitespace(text_value)
            if cleaned:
                output.append(cleaned)
        return output

    def _parse_json_payload(self, payload: Any) -> list[str]:
        if isinstance(payload, list):
            return self._normalize_list(payload)
        if isinstance(payload, dict):
            for key in ("chunks", "segments", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    parsed = self._normalize_list(value)
                    if parsed:
                        return parsed
        return []

    def _json_candidates(self, raw_text: str) -> list[str]:
        candidates: list[str] = [raw_text]

        for block in JSON_CODE_BLOCK_RE.findall(raw_text):
            if block.strip():
                candidates.append(block.strip())

        array_match = JSON_ARRAY_RE.search(raw_text)
        if array_match:
            candidates.append(array_match.group(0))

        object_match = JSON_OBJECT_RE.search(raw_text)
        if object_match:
            candidates.append(object_match.group(0))

        return candidates

    def _parse_segments(self, raw_text: str) -> list[str]:
        text = raw_text.strip()
        if not text:
            return []

        for candidate in self._json_candidates(text):
            try:
                payload = json.loads(candidate)
            except Exception:  # noqa: BLE001
                continue
            parsed = self._parse_json_payload(payload)
            if parsed:
                return parsed

        lines = [normalize_whitespace(line) for line in text.splitlines()]
        line_items: list[str] = []
        for line in lines:
            if not line:
                continue
            match = LIST_ITEM_RE.match(line)
            if match:
                line_items.append(normalize_whitespace(match.group(1)))
        if line_items:
            return [item for item in line_items if item]
        return []

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("Segmenter endpoint returned non-JSON response object.")
        return body

    def _segment_prompt(self, section_title: str, source_text: str, max_chars: int) -> str:
        return (
            "You split scientific text into semantic chunks for retrieval.\n"
            "Return STRICT JSON only.\n"
            "Output format:\n"
            '{"chunks":[]}\n'
            "Rules:\n"
            f"- each chunk length <= {max_chars} characters\n"
            "- split by topic/meaning shifts\n"
            "- each chunk must be copied from source text (no summary)\n"
            "- keep [IMAGE ...] and [OCR] lines with relevant text\n"
            "- do not output markdown, comments, or explanations\n"
            "- forbidden output: placeholders like 'chunk text 1', 'segment 2', 'example'\n\n"
            f"Section title: {section_title}\n\n"
            f"Text:\n{source_text}"
        )

    def _repair_prompt(self, section_title: str, source_text: str, max_chars: int) -> str:
        clipped = source_text[:12000]
        return (
            "Retry segmentation with strict constraints.\n"
            "Return exactly one JSON object with key 'chunks'.\n"
            "Format:\n"
            '{"chunks":[]}\n'
            "Do not use placeholders. Chunks must be direct excerpts from source text.\n"
            f"Each chunk <= {max_chars} chars.\n"
            "No markdown, no prose.\n\n"
            f"Section title: {section_title}\n\n"
            f"Source text:\n{clipped}"
        )

    def _bounded(self, chunks: list[str], max_chars: int, overlap: int) -> list[str]:
        bounded: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                bounded.append(chunk)
            else:
                bounded.extend(split_text(chunk, max_chars=max_chars, overlap=overlap))
        return [normalize_whitespace(chunk) for chunk in bounded if chunk.strip()]

    def _looks_placeholder(self, chunks: list[str]) -> bool:
        if not chunks:
            return True
        invalid_count = 0
        for chunk in chunks:
            value = normalize_whitespace(chunk).strip().strip('"').strip("'")
            if len(value) < 5:
                invalid_count += 1
                continue
            if PLACEHOLDER_RE.match(value):
                invalid_count += 1
                continue
            if "chunk text" in value.lower():
                invalid_count += 1
        return invalid_count == len(chunks)

    def split(
        self,
        text: str,
        section_title: str = "",
        max_chars: int = 1200,
        overlap: int = 180,
    ) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) < int(max_chars * 1.2):
            return [normalize_whitespace(cleaned)]

        clipped = cleaned[: self.max_input_chars]
        base_payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": self._segment_prompt(section_title, clipped, max_chars)}],
        }

        responses: list[str] = []
        payloads = [
            {**base_payload, "response_format": {"type": "json_object"}},
            base_payload,
        ]

        for payload in payloads:
            try:
                body = self._request(payload)
            except Exception as exc:  # noqa: BLE001
                responses.append(f"[request_error] {exc}")
                continue
            raw = self._extract_message(body)
            if raw:
                responses.append(raw)
                parsed = self._parse_segments(raw)
                if parsed:
                    bounded = self._bounded(parsed, max_chars=max_chars, overlap=overlap)
                    if bounded and not self._looks_placeholder(bounded):
                        return bounded

        if responses:
            repair_payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": self._repair_prompt(
                            section_title=section_title,
                            source_text=clipped,
                            max_chars=max_chars,
                        ),
                    }
                ],
            }
            try:
                repair_body = self._request(repair_payload)
                repair_raw = self._extract_message(repair_body)
                parsed = self._parse_segments(repair_raw)
                if parsed:
                    bounded = self._bounded(parsed, max_chars=max_chars, overlap=overlap)
                    if bounded and not self._looks_placeholder(bounded):
                        return bounded
            except Exception:  # noqa: BLE001
                pass

        sample = responses[-1][:300] if responses else "<empty>"
        raise RuntimeError(
            "LLM segmenter returned invalid format after retries. "
            f"Expected JSON with chunks. Last sample: {sample}"
        )


def build_segmenter(
    mode: str,
    endpoint: str | None,
    model: str,
    api_key: str | None = None,
) -> SemanticSegmenter:
    selected = (mode or "").strip().lower()
    if selected != "llm":
        raise ValueError(
            f"Segmenter mode '{mode}' is disabled in strict mode. Use mode='llm'."
        )
    if not endpoint:
        raise ValueError(
            "Segmenter endpoint is not configured. Set RAG_SEGMENTER endpoint."
        )
    if not model:
        raise ValueError("Segmenter model is not configured.")
    return LLMSemanticSegmenter(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
    )
