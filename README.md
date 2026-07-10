# rag-runes (FastAPI)

API-сервис для мультимодального Tree-RAG по книгам:
- текст из PDF,
- изображения и OCR по изображениям (GLM-OCR),
- дерево `book -> chapter -> section -> (chunk|image)`,
- гибридный поиск `dense + BM25 + rune boost`,
- reranker и финальный ответ с цитатами страниц,
- synthetic pipeline для рун из `epoch_200.pt` с прозрачными overlay PNG, глубокой гравировкой, режимом поверхностных царапин и photo-scene генерацией для детекции на реальных фото камней.

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

Synthetic pipeline берет параметры из секции `[synthetic]`:
- `checkpoint_path = "epoch_200.pt"`
- `device = "cpu"`
- `stone_texture_dir = ""`

Если `stone_texture_dir` пустой, каменная фактура генерируется процедурно.
Если указать директорию с фото камня, renderer будет брать реальные текстуры оттуда.

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

### 4) Synthetic Classes

`GET /synthetic-runes/classes`

Возвращает классы из `epoch_200.pt`.

```bash
curl http://localhost:8000/synthetic-runes/classes
```

### 5) Synthetic Render

`POST /synthetic-runes/render`

Генерирует одну руну по классу:
- `raw` из генератора,
- `mask`,
- `cutout` с прозрачным фоном,
- `overlay` с прозрачным фоном для наложения,
- финальный PNG в одном из режимов:
  - `style = "engraved"` для глубокой вырезанной руны,
  - `style = "scratch"` для поверхностной царапины,
  - `background_mode = "transparent"` для наложения без фона по умолчанию,
  - `background_mode = "stone"` для готовой каменной сцены,
  - `defect_strength` для случайных локальных повреждений штриха,
  - `thickness` для толщины штриха руны.

```bash
curl -X POST http://localhost:8000/synthetic-runes/render \
  -H "Content-Type: application/json" \
  -d '{
    "class_id": 0,
    "seed": 321,
    "canvas_size": 256,
    "palette_name": "granite",
    "style": "scratch",
    "background_mode": "transparent",
    "defect_strength": 0.35,
    "thickness": 1.2,
    "export_name": "sample_rune"
  }'
```

### 6) Synthetic Dataset

`POST /synthetic-runes/dataset`

Строит synthetic dataset:
- `images/<class>/...png`
- `masks/<class>/...png`
- `cutouts/<class>/...png`
- `overlays/<class>/...png`
- `manifest.jsonl`

```bash
curl -X POST http://localhost:8000/synthetic-runes/dataset \
  -H "Content-Type: application/json" \
  -d '{
    "class_ids": [0, 1, 2],
    "variants_per_class": 20,
    "canvas_size": 256,
    "base_seed": 1000,
    "palette_name": "granite",
    "style": "scratch",
    "background_mode": "transparent",
    "defect_strength": 0.4,
    "thickness": 1.2,
    "output_name": "stone_runes_v1"
  }'
```

### 7) Synthetic Scene

`POST /synthetic-runes/scene`

Строит одну сцену на реальном фото камня:
- автоматически ищет каменные зоны в кадре,
- размещает несколько рун без сильного overlap,
- рендерит руну прямо в локальный фотопатч камня,
- сохраняет изображение, mask и YOLO label,
- автоматически увеличивает слишком маленькие фоны,
- по умолчанию ориентирован на плотную сцену около `30` рун на изображение.

```bash
curl -X POST http://localhost:8000/synthetic-runes/scene \
  -H "Content-Type: application/json" \
  -d '{
    "background_path": "rock_example.png",
    "instance_count": 30,
    "seed": 2026,
    "style": "scratch",
    "defect_strength": 0.42,
    "thickness": 1.8,
    "scale_min": 0.08,
    "scale_max": 0.16,
    "max_iou": 0.14,
    "export_name": "rock_example_multi"
  }'
```

### 8) Detection Dataset On Rock Photos

`POST /synthetic-runes/detection-dataset`

Строит YOLO-совместимый detection dataset по одному фото или по папке с фото камней:
- `images/train`, `images/val`
- `labels/train`, `labels/val`
- `masks/train`, `masks/val`
- `manifest.jsonl`
- `dataset.yaml`

Поддерживает:
- `background_path` для одного изображения,
- `backgrounds_dir` для директории с фонами,
- `thickness` для управления толщиной и заметностью штриха,
- defaults под плотные сцены около `30` рун на кадр.

```bash
curl -X POST http://localhost:8000/synthetic-runes/detection-dataset \
  -H "Content-Type: application/json" \
  -d '{
    "background_path": "rock_example.png",
    "images_per_background": 3,
    "instances_min": 26,
    "instances_max": 34,
    "base_seed": 3030,
    "style": "scratch",
    "defect_strength": 0.45,
    "thickness": 1.8,
    "scale_min": 0.08,
    "scale_max": 0.16,
    "max_iou": 0.14,
    "output_name": "rock_single_bg_detection"
  }'
```

## Архитектура

- Runtime boot: [server.py](/Users/aleksejzagorskij/rag-runes/server.py)
- API: [app.py](/Users/aleksejzagorskij/rag-runes/rag_runes/api/app.py)
- Config loader: [config.py](/Users/aleksejzagorskij/rag-runes/rag_runes/api/config.py)
- Pipeline service: [service.py](/Users/aleksejzagorskij/rag-runes/rag_runes/api/service.py)
- Synthetic:
  - [generator.py](/Users/aleksejzagorskij/rag-runes/rag_runes/synthetic/generator.py)
  - [renderer.py](/Users/aleksejzagorskij/rag-runes/rag_runes/synthetic/renderer.py)
  - [dataset.py](/Users/aleksejzagorskij/rag-runes/rag_runes/synthetic/dataset.py)
  - [scene.py](/Users/aleksejzagorskij/rag-runes/rag_runes/synthetic/scene.py)
  - [service.py](/Users/aleksejzagorskij/rag-runes/rag_runes/synthetic/service.py)
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
