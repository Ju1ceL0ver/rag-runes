# rag-runes (FastAPI)

API-сервис для мультимодального Tree-RAG по книгам:
- текст из PDF,
- изображения и OCR по изображениям (GLM-OCR),
- дерево `book -> chapter -> section -> (chunk|image)`,
- гибридный поиск `dense + BM25 + rune boost`,
- reranker и финальный ответ с цитатами страниц.

Важно: система работает в strict-режиме без fallback.
Если модель/endpoint не настроены, API вернет явную ошибку.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python server.py
```

Сервер поднимется на `http://0.0.0.0:8000`.

Конфиг загружается из:
- [config/settings.toml](/Users/aleksejzagorskij/rag-runes/config/settings.toml) (параметры/модели/endpoint'ы),
- [/.env](/Users/aleksejzagorskij/rag-runes/.env) (ключи и env-overrides).

Безопасность для GitHub:
- [/.env](/Users/aleksejzagorskij/rag-runes/.env) добавлен в `.gitignore` и не пушится.
- [/.env.example](/Users/aleksejzagorskij/rag-runes/.env.example) можно пушить как шаблон без секретов.

Для вашего `ollama pull glm-ocr:q8_0` уже выставлены дефолты:
- `ocr_model = "glm-ocr:q8_0"`
- `segmenter_model = "minimax-m2.5:cloud"`
- `answer_model = "minimax-m2.5:cloud"`
- endpoint'ы: `http://127.0.0.1:11434/v1/chat/completions`

Почему так: `glm-ocr:q8_0` используется только для локального OCR изображений, а разбиение текста/ответ делается облачной LLM через Ollama (`minimax-m2.5:cloud`).

## Endpoint'ы

### 1) Health

`GET /health`

```bash
curl http://localhost:8000/health
```

Если чего-то не хватает, в ответе будет:
- `status: "error"`
- список `errors` с точной причиной (например, нет OCR endpoint или не загрузилась модель).

### 2) Upload + Ingest

`POST /books/upload-ingest`

Добавляет PDF в `books/`, прогоняет ingestion и обновляет индекс.

```bash
curl -X POST \
  -F "file=@/absolute/path/to/book.pdf" \
  http://localhost:8000/books/upload-ingest
```

### 3) Query

`POST /query`

Возвращает:
- `answer` (итоговый ответ с ссылками на страницы),
- `reading_pages` (отдельный блок страниц для самостоятельного чтения),
- `context_hits` (сырой контекст: chunk/image).

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Какие рунические надписи встречаются в тюркских памятниках?",
    "top_k": 8,
    "branch_factor": 4,
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
  }'
```

## Архитектура

- Runtime boot: [server.py](/Users/aleksejzagorskij/rag-runes/server.py)
- API: [app.py](/Users/aleksejzagorskij/rag-runes/rag_runes/api/app.py)
- Config loader: [config.py](/Users/aleksejzagorskij/rag-runes/rag_runes/api/config.py)
- Pipeline service: [service.py](/Users/aleksejzagorskij/rag-runes/rag_runes/api/service.py)
- Ingestion:
  - [pdf_ingestor.py](/Users/aleksejzagorskij/rag-runes/rag_runes/ingest/pdf_ingestor.py)
  - [semantic_segmenter.py](/Users/aleksejzagorskij/rag-runes/rag_runes/ingest/semantic_segmenter.py)
  - [tree_builder.py](/Users/aleksejzagorskij/rag-runes/rag_runes/ingest/tree_builder.py)
- OCR: [glm_ocr.py](/Users/aleksejzagorskij/rag-runes/rag_runes/ocr/glm_ocr.py)
- Index: [hybrid_index.py](/Users/aleksejzagorskij/rag-runes/rag_runes/index/hybrid_index.py)
- Retrieval:
  - [tree_search.py](/Users/aleksejzagorskij/rag-runes/rag_runes/retrieval/tree_search.py)
  - [reranker.py](/Users/aleksejzagorskij/rag-runes/rag_runes/retrieval/reranker.py)
- Answer generation: [answer_builder.py](/Users/aleksejzagorskij/rag-runes/rag_runes/generation/answer_builder.py)
