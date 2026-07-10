from __future__ import annotations

import json
import re
import signal
import threading
import time
from dataclasses import dataclass, field
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
MARKER_LINE_RE = re.compile(r"^\[(?:OCR|IMAGE\b)", re.IGNORECASE)
LIST_START_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[\.\)]|[a-zа-я][\.\)]|[-*])\s+",
    re.IGNORECASE,
)
TOPIC_START_RE = re.compile(
    r"^(?:"
    r"fig(?:ure)?\.?|table|scheme|plate|appendix|summary|contents|bibliography|"
    r"рис\.?|табл\.?|схема|приложение|выводы|содержание|литература|"
    r"written in|ipa phonetic transcription|translation from|"
    r"перевод|транскрипция|надпись|памятник"
    r")\b",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"""[.!?…:;]['")\]]*$""")


class SemanticSegmenter:
    def split(
        self,
        text: str,
        section_title: str = "",
        max_chars: int = 1200,
        overlap: int = 180,
    ) -> list[str]:
        raise NotImplementedError


class RuleBasedSemanticSegmenter(SemanticSegmenter):
    """Source-preserving splitter used to keep ingestion stable and auditable."""

    def _starts_new_block(self, line: str) -> bool:
        return bool(
            MARKER_LINE_RE.match(line)
            or TOPIC_START_RE.match(line)
            or LIST_START_RE.match(line)
        )

    def _ends_sentence(self, line: str) -> bool:
        return bool(SENTENCE_END_RE.search(line))

    def _semantic_units(self, text: str, max_chars: int) -> list[str]:
        lines = [normalize_whitespace(line) for line in text.splitlines()]
        if len(lines) <= 1:
            return split_text(text, max_chars=max_chars, overlap=0) or [
                normalize_whitespace(text)
            ]

        units: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if not current:
                return
            unit = normalize_whitespace(" ".join(current))
            if unit:
                units.append(unit)
            current.clear()

        for line in lines:
            if not line:
                flush()
                continue

            current_text = normalize_whitespace(" ".join(current))
            current_len = len(current_text)
            starts_new_block = self._starts_new_block(line)

            if current and starts_new_block:
                flush()
            elif (
                current
                and current_len >= max(240, int(max_chars * 0.45))
                and self._ends_sentence(current[-1])
                and line[:1].isupper()
            ):
                flush()
            elif current and current_len + len(line) + 1 > max_chars:
                flush()

            if current and current[-1].endswith("-") and line[:1].islower():
                current[-1] = current[-1][:-1] + line
            else:
                current.append(line)

            if (
                len(normalize_whitespace(" ".join(current))) >= max_chars
                and self._ends_sentence(line)
            ):
                flush()

        flush()
        return units

    def _split_oversized(self, unit: str, max_chars: int, overlap: int) -> list[str]:
        if len(unit) <= max_chars:
            return [unit]
        chunks = split_text(unit, max_chars=max_chars, overlap=overlap)
        output: list[str] = []
        for chunk in chunks or [unit]:
            if len(chunk) <= max_chars:
                output.append(chunk)
                continue
            output.extend(
                normalize_whitespace(chunk[index : index + max_chars])
                for index in range(0, len(chunk), max_chars)
            )
        return [chunk for chunk in output if chunk]

    def _tail_units(self, units: list[str], overlap: int) -> list[str]:
        if overlap <= 0:
            return []
        tail: list[str] = []
        tail_len = 0
        for unit in reversed(units):
            add_len = len(unit) + (1 if tail else 0)
            if tail_len + add_len > overlap:
                break
            tail.insert(0, unit)
            tail_len += add_len
        return tail

    def _pack_units(self, units: list[str], max_chars: int, overlap: int) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []

        def current_text() -> str:
            return normalize_whitespace(" ".join(current))

        def flush(with_overlap: bool) -> list[str]:
            text = current_text()
            if text:
                chunks.append(text)
            carry = self._tail_units(current, overlap) if with_overlap else []
            current.clear()
            return carry

        for unit in units:
            for piece in self._split_oversized(
                unit,
                max_chars=max_chars,
                overlap=overlap,
            ):
                if not current:
                    current.append(piece)
                    continue

                candidate_len = len(current_text()) + 1 + len(piece)
                topic_boundary = self._starts_new_block(piece) and len(
                    current_text()
                ) >= max(220, int(max_chars * 0.35))
                if candidate_len > max_chars or topic_boundary:
                    carry = flush(with_overlap=not topic_boundary)
                    carry_len = len(normalize_whitespace(" ".join(carry)))
                    if carry and carry_len + 1 + len(piece) > max_chars:
                        carry = []
                    current.extend(carry)
                current.append(piece)

        if current:
            flush(with_overlap=False)
        return [chunk for chunk in chunks if chunk]

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
        units = self._semantic_units(cleaned, max_chars=max_chars)
        return self._pack_units(units, max_chars=max_chars, overlap=overlap)

    def source_windows(self, text: str, max_chars: int) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        units = self._semantic_units(cleaned, max_chars=max_chars)
        return self._pack_units(units, max_chars=max_chars, overlap=0)


@dataclass(slots=True)
class LLMSemanticSegmenter(SemanticSegmenter):
    endpoint: str
    model: str
    api_key: str | None = None
    timeout_s: int = 180
    max_input_chars: int = 8000
    request_retries: int = 5
    retry_backoff_s: float = 4.0
    hard_timeout_s: int | None = None
    local_segmenter: RuleBasedSemanticSegmenter = field(
        default_factory=RuleBasedSemanticSegmenter
    )

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

    def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    def _post_with_hard_timeout(self, payload: dict[str, Any]) -> dict[str, Any]:
        if (
            self.hard_timeout_s is None
            or self.hard_timeout_s <= 0
            or threading.current_thread() is not threading.main_thread()
        ):
            return self._post_once(payload)

        def raise_timeout(signum, frame):  # noqa: ANN001
            raise TimeoutError(
                f"Segmenter request exceeded {self.hard_timeout_s}s hard timeout."
            )

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(self.hard_timeout_s))
        try:
            return self._post_once(payload)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.request_retries) + 1):
            try:
                body = self._post_with_hard_timeout(payload)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= max(1, self.request_retries):
                    raise
                sleep_s = self.retry_backoff_s * attempt
                time.sleep(sleep_s)
        else:
            raise RuntimeError(f"Segmenter request failed: {last_error}")

        return body

    def _segment_prompt(self, section_title: str, source_text: str, max_chars: int) -> str:
        return (
            "You split scientific text into semantic chunks for retrieval.\n"
            "Return STRICT JSON only.\n"
            "Output format:\n"
            '{"chunks":[]}\n'
            "Rules:\n"
            f"- each chunk length <= {max_chars} characters\n"
            "- cover the complete source text in original order; do not drop tail text\n"
            "- split only at topic, paragraph, sentence, caption, list, table, "
            "or OCR/image boundaries\n"
            "- each chunk must be copied from source text exactly; no summary, "
            "rewrite, translation, or interpretation\n"
            "- keep [IMAGE ...] and [OCR] lines with the text they explain\n"
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
            "Do not use placeholders. Chunks must be direct contiguous excerpts from source text.\n"
            "Cover the whole source in order. Do not summarize or omit the end.\n"
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

    def _source_split_is_valid(
        self,
        source_text: str,
        chunks: list[str],
        overlap: int,
    ) -> bool:
        if self._looks_placeholder(chunks):
            return False

        source = normalize_whitespace(source_text)
        if not source:
            return not chunks

        intervals: list[tuple[int, int]] = []
        cursor = 0
        for chunk in chunks:
            candidate = normalize_whitespace(chunk)
            if not candidate:
                return False
            search_from = max(0, cursor - max(overlap, 0) - 32)
            start = source.find(candidate, search_from)
            if start < 0:
                start = source.find(candidate)
            if start < 0:
                return False
            end = start + len(candidate)
            intervals.append((start, end))
            cursor = max(cursor, end)

        intervals.sort()
        covered = 0
        current_start = -1
        current_end = -1
        for start, end in intervals:
            if current_start < 0:
                current_start = start
                current_end = end
                continue
            if start <= current_end:
                current_end = max(current_end, end)
                continue
            covered += current_end - current_start
            current_start = start
            current_end = end
        if current_start >= 0:
            covered += current_end - current_start

        coverage = covered / max(len(source), 1)
        return coverage >= 0.9

    def _llm_split_window(
        self,
        text: str,
        section_title: str = "",
        max_chars: int = 1200,
        overlap: int = 180,
    ) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= max_chars:
            return [normalize_whitespace(cleaned)]
        if len(cleaned) < int(max_chars * 1.2):
            return self.local_segmenter.split(
                cleaned,
                section_title=section_title,
                max_chars=max_chars,
                overlap=overlap,
            )

        clipped = cleaned[: self.max_input_chars]
        base_payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": self._segment_prompt(
                        section_title,
                        clipped,
                        max_chars,
                    ),
                }
            ],
        }

        responses: list[str] = []
        request_errors: list[str] = []
        payloads = [
            {**base_payload, "response_format": {"type": "json_object"}},
            base_payload,
        ]

        for payload in payloads:
            try:
                body = self._request(payload)
            except Exception as exc:  # noqa: BLE001
                request_errors.append(str(exc))
                continue
            raw = self._extract_message(body)
            if raw:
                responses.append(raw)
                parsed = self._parse_segments(raw)
                if parsed:
                    bounded = self._bounded(
                        parsed,
                        max_chars=max_chars,
                        overlap=overlap,
                    )
                    if self._source_split_is_valid(
                        source_text=clipped,
                        chunks=bounded,
                        overlap=overlap,
                    ):
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
                    bounded = self._bounded(
                        parsed,
                        max_chars=max_chars,
                        overlap=overlap,
                    )
                    if self._source_split_is_valid(
                        source_text=clipped,
                        chunks=bounded,
                        overlap=overlap,
                    ):
                        return bounded
            except Exception:  # noqa: BLE001
                pass

        if request_errors and not responses:
            return []
        return []

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

        chunks: list[str] = []
        windows = self.local_segmenter.source_windows(
            cleaned,
            max_chars=self.max_input_chars,
        )
        for window in windows:
            llm_chunks = self._llm_split_window(
                text=window,
                section_title=section_title,
                max_chars=max_chars,
                overlap=overlap,
            )
            if llm_chunks:
                chunks.extend(llm_chunks)
                continue
            chunks.extend(
                self.local_segmenter.split(
                    text=window,
                    section_title=section_title,
                    max_chars=max_chars,
                    overlap=overlap,
                )
            )

        return [normalize_whitespace(chunk) for chunk in chunks if chunk.strip()]


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
