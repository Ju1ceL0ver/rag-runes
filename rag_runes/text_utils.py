from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"[\w-]+", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
RUNE_QUERY_RE = re.compile(
    r"\b(руна|руны|рун|rune|runes|futhark|орхон|старотюрк|runic)\b",
    flags=re.IGNORECASE,
)


def normalize_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "book"


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(value)]


def split_sentences(value: str) -> list[str]:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]


def _split_long_sentence(value: str, max_chars: int) -> list[str]:
    if len(value) <= max_chars:
        return [value]

    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in value.split():
        if len(word) > max_chars:
            if current:
                parts.append(" ".join(current))
                current = []
                current_len = 0
            parts.extend(
                word[index : index + max_chars]
                for index in range(0, len(word), max_chars)
            )
            continue

        add_len = len(word) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            parts.append(" ".join(current))
            current = []
            current_len = 0
            add_len = len(word)

        current.append(word)
        current_len += add_len

    if current:
        parts.append(" ".join(current))
    return parts


def split_text(value: str, max_chars: int, overlap: int) -> list[str]:
    sentences = [
        part
        for sentence in split_sentences(value)
        for part in _split_long_sentence(sentence, max_chars=max_chars)
    ]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if overlap > 0:
                carry: list[str] = []
                carry_len = 0
                for old_sentence in reversed(current):
                    sentence_len = len(old_sentence) + (1 if carry else 0)
                    if carry_len + sentence_len > overlap:
                        break
                    carry.insert(0, old_sentence)
                    carry_len += sentence_len
                current = carry
                current_len = carry_len
            else:
                current = []
                current_len = 0

            add_len = len(sentence) + (1 if current else 0)

        current.append(sentence)
        current_len += add_len

    tail = " ".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks


def is_runic_char(ch: str) -> bool:
    code = ord(ch)
    return (0x16A0 <= code <= 0x16FF) or (0x10C00 <= code <= 0x10C4F)


def rune_density(value: str) -> float:
    if not value:
        return 0.0
    chars = [ch for ch in value if not ch.isspace()]
    if not chars:
        return 0.0
    runic_chars = sum(1 for ch in chars if is_runic_char(ch))
    return runic_chars / len(chars)


def contains_rune_query(value: str) -> bool:
    return bool(RUNE_QUERY_RE.search(value or ""))
