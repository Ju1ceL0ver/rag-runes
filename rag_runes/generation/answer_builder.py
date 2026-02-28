from __future__ import annotations

from dataclasses import asdict, dataclass

import requests

from rag_runes.retrieval.tree_search import RetrievalHit


@dataclass(slots=True)
class ReadingPage:
    page_from: int
    page_to: int
    level: str
    path: str


@dataclass(slots=True)
class AnswerBundle:
    answer: str
    reading_pages: list[ReadingPage]
    context_hits: list[RetrievalHit]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "reading_pages": [asdict(item) for item in self.reading_pages],
            "context_hits": [asdict(hit) for hit in self.context_hits],
        }


class AnswerBuilder:
    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "glm-4.5",
        api_key: str | None = None,
        timeout_s: int = 90,
        per_hit_chars: int = 900,
    ) -> None:
        if not endpoint:
            raise ValueError(
                "Answer endpoint is not configured. Set RAG_ANSWER endpoint."
            )
        if not model:
            raise ValueError("Answer model is not configured.")
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.per_hit_chars = per_hit_chars

    def _snippet(self, text: str, max_chars: int) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3] + "..."

    def _context_text(self, hits: list[RetrievalHit], max_items: int = 8) -> str:
        blocks: list[str] = []
        for idx, hit in enumerate(hits[:max_items], start=1):
            snippet = self._snippet(hit.text, self.per_hit_chars)
            blocks.append(
                (
                    f"[CTX {idx}] level={hit.level} pages={hit.page_from}-{hit.page_to}\n"
                    f"path: {hit.path}\n"
                    f"text: {snippet}"
                )
            )
        return "\n\n".join(blocks)

    def _extract_message(self, body: dict) -> str:
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

    def _llm_answer(self, query: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            raise RuntimeError(
                "No context hits found for answer generation. "
                "Upload more documents or adjust retrieval settings."
            )
        context = self._context_text(hits=hits)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        prompt = (
            "Ты ассистент по научному корпусу книг о рунах.\n"
            "Ответь по контексту ниже.\n"
            "Требования:\n"
            "- Пиши по-русски.\n"
            "- Дай точный ответ в 1-3 абзацах.\n"
            "- Каждое утверждение ссылай на страницы в формате [p.X-Y].\n"
            "- Если данных недостаточно, явно это скажи.\n\n"
            f"Вопрос: {query}\n\n"
            f"Контекст:\n{context}"
        )
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            text = self._extract_message(response.json()).strip()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM answer generation failed: {exc}") from exc
        if not text:
            raise RuntimeError("LLM answer generation returned empty content.")
        return text

    def _reading_pages(self, hits: list[RetrievalHit], max_items: int = 12) -> list[ReadingPage]:
        unique: dict[tuple[int, int, str], ReadingPage] = {}
        for hit in hits:
            key = (hit.page_from, hit.page_to, hit.path)
            if key not in unique:
                unique[key] = ReadingPage(
                    page_from=hit.page_from,
                    page_to=hit.page_to,
                    level=hit.level,
                    path=hit.path,
                )
            if len(unique) >= max_items:
                break
        return list(unique.values())

    def build(self, query: str, hits: list[RetrievalHit]) -> AnswerBundle:
        answer = self._llm_answer(query=query, hits=hits)
        return AnswerBundle(
            answer=answer,
            reading_pages=self._reading_pages(hits=hits),
            context_hits=hits,
        )
