# CodeLens — умный поиск по кодовой базе

> Прототип RAG-системы: семантический и гибридный поиск по коду на Streamlit.  
> Языки: Python · JavaScript/TypeScript · Java

---

## Архитектура

```
.py / .js / .ts / .java
        │
        ▼
[index.py] — парсинг
├── Python  → модуль ast (встроенный, точный)
├── JS/TS   → регулярные выражения
└── Java    → регулярные выражения
        │
        ▼
sentence-transformers
paraphrase-multilingual-MiniLM-L12-v2  (dim=384, RU+EN)
        │
        ▼
ChromaDB  (persistent, hnsw:space=cosine)
        │
        ▼
[app.py / Streamlit]
├── 🧠 Семантический поиск         — вектор → cosine top-K
├── 🔀 Гибридный поиск (бонус)     — вектор + keyword overlap, настраиваемый вес
├── 🤖 LLM-ответ (бонус)           — Ollama / OpenAI / Anthropic
├── 📊 Precision@5 (бонус)         — авто-оценка на eval_questions.json
└── 📦 Расширение базы (бонус)     — open-source репозитории
```

---

## Стратегия чанкования

**Один чанк = одна функция или один класс. Методы класса — отдельный чанк с именем `ClassName.method_name`.**

Функция — минимальная семантическая единица кода: имеет имя, сигнатуру и единственную ответственность (~10–80 строк). Методы индексируются отдельно, чтобы при поиске возвращался конкретный метод, а не всё тело класса. Граница чанка определяется синтаксисом через `ast` (Python) — не произвольна. Это обеспечивает совместимость `chunk_id` с форматом `eval_questions.json`.

---

## Технический стек

| Слой | Инструмент | Примечание |
|---|---|---|
| Язык | Python 3.12 | Требование ТЗ |
| Парсинг Python | `ast` (встроенный) | Надёжно, без зависимостей |
| Парсинг JS/Java | Регулярные выражения | Без внешних зависимостей |
| Эмбеддинги | `paraphrase-multilingual-MiniLM-L12-v2` | RU+EN, dim=384, Apache 2.0 |
| Векторная БД | ChromaDB (persistent, cosine) | Без отдельного сервера |
| UI | Streamlit | Python-only, `st.code()` встроена |
| Гибридный поиск | keyword overlap + vector | Лучше на точных именах функций |
| LLM (опц.) | Ollama / OpenAI / Anthropic | Чат-режим по коду |

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Индексирование

```bash
# Только основная кодовая база (минимум для чемпионата):
python index.py ./codebase_python/gymhero --reset

# С расширением базы open-source репозиториями (бонус):
python index.py ./codebase_python/gymhero --fetch-repos --reset

# Python + JS вместе:
python index.py ./backend ./frontend --lang python js --reset

# Java:
python index.py ./my_java_project --lang java
```



### 3. Запуск веб-интерфейса

```bash
python -m streamlit run app.py

```

Откроется браузер на `http://localhost:8501`.

### Docker (бонус)

```bash
docker compose up
# http://localhost:8501
```

---

## 5 примеров запросов с ответами

| Запрос | Найденный фрагмент |
|---|---|
| `как создаётся токен доступа?` | `security.py: create_access_token` |
| `how does JWT verification work?` | `dependencies.py: get_token, get_current_user` |
| `где проверяется суперпользователь?` | `crud/user.py: UserCRUDRepository.is_super_user` |
| `строка подключения к PostgreSQL` | `database/session.py: build_sqlalchemy_database_url_from_settings` |
| `how does session management work?` | `database/session.py: get_db, get_local_session` |

---

## Расширение тестовой базы (бонус)

Участники вправе использовать открытые Python-репозитории для расширения тестовой базы.  
Автоматически загружаются три репозитория:

| Репозиторий | Описание |
|---|---|
| `encode/httpx` | Async HTTP client — паттерны запросов, middleware, auth |
| `Textualize/rich` | Terminal formatting — хорошо структурированные классы |
| `encode/starlette` | ASGI framework — основа FastAPI, routing, requests |

Запуск: `python index.py ./codebase_python/gymhero --fetch-repos --reset`  
Или через вкладку **📦 Расширение базы** в интерфейсе.

---

## Поддержка второго языка (бонус)

`index.py` поддерживает JavaScript, TypeScript и Java через регулярные выражения:

```bash
python index.py ./my_js_project --lang js
python index.py ./my_java_project --lang java
python index.py ./backend ./frontend --lang python js
```

---

## Оценка Precision@5

На странице **«📊 Метрики»** → кнопка «▶ Запустить оценку».

Или через CLI:
```bash
# Скачать results.json через UI (кнопка «Скачать»)
python score.py --predictions results.json --questions eval_questions.json
```

Целевое значение по ТЗ: **Precision@5 ≥ 60%**.

---

## Гибридный поиск (бонус)

На странице «🔍 Поиск» → «⚙️ Настройки» → режим «🔀 Гибридный».

`score = w × vector_score + (1-w) × text_overlap_score`

Слайдер `w` (по умолчанию 0.7). Гибридный режим лучше находит фрагменты по точным именам функций.

---

## LLM-режим (бонус)

На странице «🔍 Поиск» → «🤖 LLM-режим».

- **Ollama** API-ключ →  `open api kye`
- **Qwen** API-ключ →  `HuggingFace`

---

## Структура проекта

```
codelens/
├── index.py              # Индексирование (Python/JS/Java + open-source repos)
├── app.py                # Streamlit веб-интерфейс
├── requirements.txt      # Зависимости
├── README.md             # Документация
├── Dockerfile            # Docker-образ
├── docker-compose.yml    # Docker Compose (бонус)
├── score.py              # Скрипт оценки Precision@5
├── eval_questions.json   # 15 тестовых вопросов
├── sample_queries.txt    # 20 примеров запросов
├── chroma_db/            # Векторная БД (создаётся при индексировании)
├── codebase_python/      # Основная кодовая база (датасет чемпионата)
└── extra_repos/          # Open-source репозитории (создаётся при --fetch-repos)
    ├── httpx/
    ├── rich/
    └── starlette/
```
---
### Работу выполнили:
- Пирматова Мария Дмитриевна
- Овчинникова Анна Вадимовна
- Столбова Вероника Евгеньевна
