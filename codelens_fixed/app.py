"""
CodeLens — веб-интерфейс на Streamlit.

Запуск:
    streamlit run app.py

Предварительно проиндексируйте кодовую базу:
    python index.py ./codebase_python/gymhero --reset

Для расширения базы open-source репозиториями:
    python index.py ./codebase_python/gymhero --fetch-repos --reset
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Загружаем токены из .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# ─── Константы ────────────────────────────────────────────────────────────────
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "codelens"
DEFAULT_DB_PATH = "./chroma_db"
EVAL_PATH       = "./eval_questions.json"
TOP_K = 5

# Соответствие расширений языков для подсветки синтаксиса в st.code()
LANG_HIGHLIGHT = {
    "python":      "python",
    "js":          "javascript",
    "javascript":  "javascript",
    "typescript":  "typescript",
    "java":        "java",
    "async_function": "python",
    "function":    "python",
    "class":       "python",
    "method":      "python",
}


# ─── Кеш ресурсов ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Загружаю модель эмбеддингов...")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="⏳ Подключаюсь к ChromaDB...", hash_funcs={str: lambda s: s})
def load_collection(db_path: str):
    try:
        client = chromadb.PersistentClient(path=db_path)
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _make_hit(doc: str, meta: dict, similarity: float) -> dict:
    return {
        "chunk_id":   f"{meta['file_path']}:{meta['name']}:{meta['start_line']}",
        "file_path":  meta["file_path"],
        "type":       meta["type"],
        "name":       meta["name"],
        "start_line": meta["start_line"],
        "end_line":   meta["end_line"],
        "docstring":  meta.get("docstring", ""),
        "language":   meta.get("language", "python"),
        "source_code": doc,
        "similarity":  similarity,
    }


def _highlight_lang(hit: dict) -> str:
    """Возвращает язык для st.code() на основе метаданных чанка."""
    lang = hit.get("language", "python")
    return LANG_HIGHLIGHT.get(lang, "python")


def _db_stats(collection) -> dict:
    """Считает статистику по языкам и типам чанков в БД."""
    if collection is None:
        return {}
    try:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        langs, types = {}, {}
        for m in all_meta:
            l = m.get("language", "python")
            t = m.get("type", "?")
            langs[l] = langs.get(l, 0) + 1
            types[t] = types.get(t, 0) + 1
        return {"langs": langs, "types": types, "total": len(all_meta)}
    except Exception:
        return {}


# ─── Поиск ────────────────────────────────────────────────────────────────────

def vector_search(
    query: str,
    collection,
    model,
    top_k: int,
    lang_filter: str = "all",
) -> tuple[list[dict], float]:
    """Семантический поиск по эмбеддингам (cosine similarity)."""
    t0 = time.perf_counter()
    vec = model.encode([query]).tolist()

    where = None
    if lang_filter != "all":
        where = {"language": {"$eq": lang_filter}}

    try:
        results = collection.query(
            query_embeddings=vec,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
    except Exception:
        # ChromaDB может не поддерживать where на малой базе — fallback без фильтра
        results = collection.query(
            query_embeddings=vec,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    latency = time.perf_counter() - t0
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = max(0.0, 1.0 - dist / 2.0)
        hits.append(_make_hit(doc, meta, similarity))
    return hits, latency


def fulltext_search(
    query: str,
    collection,
    top_k: int,
    lang_filter: str = "all",
) -> list[dict]:
    """
    Полнотекстовый keyword-поиск.
    Используется как второй канал в гибридном режиме.
    """
    all_data = collection.get(include=["documents", "metadatas"])
    keywords = set(re.findall(r"\w+", query.lower()))

    scored = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        if lang_filter != "all" and meta.get("language", "python") != lang_filter:
            continue
        doc_words = set(re.findall(r"\w+", doc.lower()))
        overlap = len(keywords & doc_words)
        if overlap > 0:
            score = overlap / max(len(keywords), 1)
            scored.append((score, doc, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [_make_hit(d, m, s) for s, d, m in scored[:top_k]]


def hybrid_search(
    query: str,
    collection,
    model,
    top_k: int,
    vector_weight: float = 0.7,
    lang_filter: str = "all",
) -> tuple[list[dict], float]:
    """
    Гибридный поиск: векторный + полнотекстовый.
    score = vector_weight × vec_score + (1 − vector_weight) × text_score
    Настраиваемый вес позволяет балансировать между семантикой и точными совпадениями.
    """
    t0 = time.perf_counter()
    vec_hits, _ = vector_search(query, collection, model, top_k * 2, lang_filter)
    txt_hits = fulltext_search(query, collection, top_k * 2, lang_filter)
    latency = time.perf_counter() - t0

    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for h in vec_hits:
        cid = h["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + vector_weight * h["similarity"]
        chunks[cid] = h

    for h in txt_hits:
        cid = h["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + (1 - vector_weight) * h["similarity"]
        if cid not in chunks:
            chunks[cid] = h

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for cid, score in ranked:
        hit = dict(chunks[cid])
        hit["similarity"] = min(score, 1.0)
        result.append(hit)

    return result, latency


def do_search(
    query: str,
    collection,
    model,
    top_k: int = TOP_K,
    mode: str = "vector",
    vector_weight: float = 0.7,
    lang_filter: str = "all",
) -> tuple[list[dict], float]:
    """Единая точка входа для поиска."""
    if mode == "hybrid":
        return hybrid_search(query, collection, model, top_k, vector_weight, lang_filter)
    return vector_search(query, collection, model, top_k, lang_filter)


# ─── Precision@5 ──────────────────────────────────────────────────────────────

def chunks_match(pred: str, ref: str, tol: int = 2) -> bool:
    """Сравнивает chunk_id с допуском tol строк по номеру строки."""
    def parse(s):
        parts = s.rsplit(":", 2)
        if len(parts) != 3:
            return None
        path, name, line = parts
        try:
            return path, name, int(line)
        except ValueError:
            return None

    p, r = parse(pred), parse(ref)
    if p is None or r is None:
        return False
    return p[0] == r[0] and p[1] == r[1] and abs(p[2] - r[2]) <= tol


def compute_precision_at_5(top5_ids: list[str], correct_ids: list[str]) -> float:
    matched, used = 0, set()
    for pred in top5_ids:
        for i, ref in enumerate(correct_ids):
            if i not in used and chunks_match(pred, ref):
                matched += 1
                used.add(i)
                break
    return matched / min(5, len(correct_ids)) if correct_ids else 0.0



# ─── HuggingFace: ротация токенов ─────────────────────────────────────────────

def _load_hf_tokens() -> list[str]:
    """Считывает до 5 HF-токенов из переменных окружения HF_TOKEN_1…HF_TOKEN_5."""
    tokens = []
    for i in range(1, 6):
        t = os.environ.get(f"HF_TOKEN_{i}", "").strip()
        if t:
            tokens.append(t)
    return tokens


def _hf_generate(prompt: str, tokens: list[str]) -> str:
    """
    Отправляет запрос в HuggingFace Inference API через InferenceClient.
    При ошибке 429 / лимите автоматически переключается на следующий токен.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return "❌ Установите библиотеку: pip install huggingface_hub"

    if not tokens:
        return "❌ HF-токены не найдены. Проверьте файл .env (HF_TOKEN_1 … HF_TOKEN_5)."

    HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    last_error = ""

    for idx, token in enumerate(tokens, 1):
        try:
            client = InferenceClient(model=HF_MODEL, token=token.strip())
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = str(e)
            error_lower = last_error.lower()
            if "429" in last_error or "503" in last_error or "rate limit" in error_lower:
                continue
            return f"❌ HuggingFace ошибка (токен {idx}): {last_error[:200]}"

    return (
        f"❌ Все {len(tokens)} HF-токена исчерпали лимит или недоступны.\n"
        f"Последняя ошибка: {last_error}"
    )

# ─── LLM-ответ (бонус) ────────────────────────────────────────────────────────

def llm_answer(query: str, hits: list[dict], provider: str, api_key: str = "") -> str:
    """
    Чат-режим: находит фрагменты кода → отправляет в LLM → возвращает ответ.
    Поддерживает Ollama (локально), OpenAI, Anthropic.
    """
    context_parts = []
    for i, hit in enumerate(hits[:3], 1):
        lang = hit.get("language", "python")
        context_parts.append(
            f"Фрагмент {i} ({hit['file_path']}, {hit['name']}, lang={lang}):\n"
            f"```{lang}\n{hit['source_code'][:1500]}\n```"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        f"Ты ассистент по кодовой базе. Пользователь спрашивает:\n«{query}»\n\n"
        f"Вот релевантные фрагменты кода:\n\n{context}\n\n"
        f"Дай чёткий ответ, опираясь на код. Указывай имена функций и файлы."
    )

    try:
        if provider == "ollama":
            import ollama
            resp = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp["message"]["content"]

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            return resp.choices[0].message.content

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        elif provider == "huggingface":
            tokens = _load_hf_tokens()
            return _hf_generate(prompt, tokens)

        else:
            return "❌ Неизвестный провайдер LLM."

    except ImportError as e:
        return f"❌ Библиотека не установлена: {e}\n\nУстановите: pip install {e.name}"
    except Exception as e:
        return f"❌ Ошибка LLM: {e}"


# ─── Страница: Поиск ──────────────────────────────────────────────────────────

def page_search(collection, model):
    st.title("🔍 CodeLens — умный поиск по коду")
    st.caption(
        "Введите вопрос на **русском или английском** — "
        "система найдёт релевантные фрагменты, даже если слова не совпадают с именами функций."
    )

    if collection is None:
        st.error(
            "⚠️ Индекс не найден. Сначала проиндексируйте кодовую базу:\n\n"
            "```bash\npython index.py ./codebase_python/gymhero --reset\n```"
        )
        return

    stats = _db_stats(collection)
    total = stats.get("total", collection.count())
    langs = stats.get("langs", {})
    lang_str = " · ".join(f"{l}: {n}" for l, n in sorted(langs.items()))
    st.info(f"📦 Документов в индексе: **{total}** ({lang_str})", icon="ℹ️")

    # ── Настройки ──
    with st.expander("⚙️ Настройки поиска", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            search_mode = st.radio(
                "Режим",
                ["vector", "hybrid"],
                format_func=lambda x: (
                    "🧠 Семантический" if x == "vector" else "🔀 Гибридный (семантика + текст)"
                ),
                horizontal=True,
                help="Гибридный режим лучше находит фрагменты по точным именам функций",
            )
            vector_weight = 0.7
            if search_mode == "hybrid":
                vector_weight = st.slider(
                    "Вес векторного поиска", 0.0, 1.0, 0.7, 0.05,
                    help="0 = только текст, 1 = только вектор",
                )
        with c2:
            top_k = st.number_input("Количество результатов (K)", 1, 10, TOP_K)
            available_langs = ["all"] + sorted(langs.keys())
            lang_filter = st.selectbox(
                "Фильтр по языку",
                available_langs,
                format_func=lambda x: (
                    "🌐 Все языки" if x == "all"
                    else {"python": "🐍 Python", "js": "⚡ JavaScript/TS", "java": "☕ Java"}.get(x, x)
                ),
                help="Искать только в коде на выбранном языке",
            )

    # ── LLM-настройки ──
    with st.expander("🤖 LLM-режим — генерация связного ответа по коду (бонус)", expanded=False):
        llm_enabled = st.toggle("Включить LLM-ответ", value=False)
        llm_provider = st.selectbox(
            "Провайдер LLM",
            ["huggingface", "ollama", "openai", "anthropic"],
            help=(
                "huggingface — бесплатно (токены из .env, авто-ротация), "
                "ollama — локально, openai / anthropic — облачные (нужен API-ключ)"
            ),
        )
        llm_api_key = ""
        if llm_provider in ("openai", "anthropic"):
            llm_api_key = st.text_input("API Key", type="password")
        if llm_provider == "huggingface":
            hf_tokens = _load_hf_tokens()
            if hf_tokens:
                st.success(f"✅ Найдено HF-токенов: {len(hf_tokens)} (авто-ротация включена)")
            else:
                st.warning("⚠️ HF-токены не найдены. Добавьте HF_TOKEN_1…HF_TOKEN_5 в файл .env")

    # ── Поле запроса ──
    query = st.text_input(
        "Запрос на русском или английском",
        placeholder="как создаётся токен доступа? / how does session management work?",
        key="search_input",
    )

    col_btn, col_sample = st.columns([2, 3])
    with col_btn:
        search_clicked = st.button("🔎 Найти", type="primary", use_container_width=True)
    with col_sample:
        sample_queries = [
            "как создаётся токен доступа?",
            "where is JWT verified?",
            "где проверяется суперпользователь?",
            "строка подключения к PostgreSQL",
            "how does session management work?",
        ]
        chosen = st.selectbox("💡 Примеры запросов", [""] + sample_queries,
                              label_visibility="collapsed")
        if chosen:
            query = chosen

    if not (search_clicked or chosen):
        return
    if not query or not query.strip():
        st.warning("⚠️ Введите запрос.")
        return

    # ── Поиск ──
    with st.spinner("🔄 Ищу..."):
        hits, latency = do_search(
            query.strip(), collection, model,
            top_k=int(top_k),
            mode=search_mode,
            vector_weight=vector_weight,
            lang_filter=lang_filter,
        )

    mode_label = "🔀 Гибридный" if search_mode == "hybrid" else "🧠 Семантический"
    lat_ok = latency <= 3.0
    st.success(
        f"{'✅' if lat_ok else '⚠️'} Время: **{latency:.2f}с** | "
        f"Режим: {mode_label} | Найдено: **{len(hits)}**"
    )
    if not lat_ok:
        st.warning("⚠️ Время ответа превысило целевые 3 сек.")

    # ── LLM-ответ ──
    if llm_enabled and hits:
        with st.spinner("🤖 Генерирую ответ через LLM..."):
            answer = llm_answer(query.strip(), hits, llm_provider, llm_api_key)
        with st.container(border=True):
            st.markdown("### 🤖 LLM-объяснение")
            st.markdown(answer)
        st.divider()

    if not hits:
        st.warning("Ничего не найдено. Попробуйте другой запрос или смените режим поиска.")
        return

    # ── Карточки результатов ──
    for i, hit in enumerate(hits, 1):
        pct = hit["similarity"] * 100
        badge = "🟢" if pct >= 65 else "🟡" if pct >= 45 else "🔴"
        lang_icon = {"python": "🐍", "js": "⚡", "java": "☕"}.get(hit.get("language", "python"), "📄")
        hl_lang = _highlight_lang(hit)

        with st.container(border=True):
            h1, h2 = st.columns([5, 1])
            with h1:
                st.markdown(
                    f"**{i}. {lang_icon} `{hit['name']}`** &nbsp;·&nbsp; "
                    f"`{hit['file_path']}` &nbsp;·&nbsp; "
                    f"строки {hit['start_line']}–{hit['end_line']} &nbsp;·&nbsp; "
                    f"тип: `{hit['type']}`"
                )
                if hit["docstring"]:
                    st.caption(f"📝 {hit['docstring'][:200]}")
            with h2:
                st.metric("Релевантность", f"{badge} {pct:.1f}%")

            # Подсветка синтаксиса — обязательное требование ТЗ
            # Язык определяется из метаданных (python / javascript / java)
            st.code(hit["source_code"], language=hl_lang)

            with st.expander("chunk_id"):
                st.code(hit["chunk_id"])


# ─── Страница: Метрики Precision@5 (бонус) ────────────────────────────────────

def page_metrics(collection, model):
    st.title("📊 Метрики качества — Precision@5")
    st.caption("Автоматическая оценка на тестовом наборе `eval_questions.json`.")

    if collection is None:
        st.error("Коллекция не найдена. Запустите индексирование.")
        return

    eval_path = Path(EVAL_PATH)
    if not eval_path.exists():
        st.warning(f"Файл `{EVAL_PATH}` не найден рядом с app.py.")
        return

    with open(eval_path, encoding="utf-8") as f:
        questions = json.load(f)

    st.info(f"Вопросов в тестовом наборе: **{len(questions)}** | "
            f"Документов в индексе: **{collection.count()}**")

    c1, c2 = st.columns(2)
    with c1:
        eval_mode = st.radio(
            "Режим поиска",
            ["vector", "hybrid"],
            format_func=lambda x: "🧠 Семантический" if x == "vector" else "🔀 Гибридный",
            horizontal=True,
        )
    with c2:
        eval_weight = 0.7
        if eval_mode == "hybrid":
            eval_weight = st.slider("Вес векторного", 0.0, 1.0, 0.7, 0.05)

    if not st.button("▶ Запустить оценку", type="primary"):
        return

    results, latencies = [], []
    bar = st.progress(0, text="Оцениваю...")

    for idx, q in enumerate(questions):
        hits, lat = do_search(
            q["query"], collection, model, top_k=5,
            mode=eval_mode, vector_weight=eval_weight,
        )
        latencies.append(lat)
        top5_ids = [h["chunk_id"] for h in hits]
        score = compute_precision_at_5(top5_ids, q["correct_chunk_ids"])
        results.append({
            "question_id": q["question_id"],
            "query":        q["query"],
            "language":     q.get("language", "?"),
            "difficulty":   q.get("difficulty", "?"),
            "score":        score,
            "top5":         top5_ids,
            "correct":      q["correct_chunk_ids"],
        })
        bar.progress(
            (idx + 1) / len(questions),
            text=f"Вопрос {idx+1}/{len(questions)}: {q['query'][:55]}…",
        )

    bar.empty()

    mean_p5  = sum(r["score"] for r in results) / len(results)
    mean_lat = sum(latencies) / len(latencies)
    max_lat  = max(latencies)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Mean Precision@5",
        f"{mean_p5:.3f}",
        delta="✅ цель ≥60%" if mean_p5 >= 0.6 else "❌ ниже цели",
        delta_color="normal" if mean_p5 >= 0.6 else "inverse",
    )
    m2.metric(
        "Ср. Latency",
        f"{mean_lat:.2f}с",
        delta="✅ ≤3с" if mean_lat <= 3 else "⚠️ >3с",
        delta_color="normal" if mean_lat <= 3 else "inverse",
    )
    m3.metric("Макс. Latency", f"{max_lat:.2f}с")
    m4.metric("Вопросов", len(results))

    if mean_p5 >= 0.6:
        st.success(f"✅ Precision@5 = {mean_p5:.1%} — цель ≥60% достигнута!")
    else:
        st.error(f"❌ Precision@5 = {mean_p5:.1%} — ниже целевых 60%.")

    # ── Breakdown ──
    st.divider()
    bc1, bc2 = st.columns(2)
    with bc1:
        st.subheader("По сложности")
        for diff in ["easy", "medium", "hard"]:
            grp = [r["score"] for r in results if r["difficulty"] == diff]
            if grp:
                avg = sum(grp) / len(grp)
                ok = "✅" if avg >= 0.6 else "⚠️"
                st.write(f"{ok} **{diff}**: {avg:.3f} ({len(grp)} вопр.)")
    with bc2:
        st.subheader("По языку запроса")
        for lang in ["ru", "en"]:
            grp = [r["score"] for r in results if r["language"] == lang]
            if grp:
                avg = sum(grp) / len(grp)
                ok = "✅" if avg >= 0.6 else "⚠️"
                st.write(f"{ok} **{lang}**: {avg:.3f} ({len(grp)} вопр.)")

    # ── Детали ──
    st.divider()
    st.subheader("Детальные результаты")
    for r in sorted(results, key=lambda x: x["question_id"]):
        icon = "✅" if r["score"] >= 0.6 else "⚠️" if r["score"] > 0 else "❌"
        with st.expander(
            f"{icon} [{r['question_id']}] [{r['difficulty']}, {r['language']}] "
            f"P@5={r['score']:.2f} — {r['query']}"
        ):
            st.markdown("**Топ-5 найденных:**")
            for cid in r["top5"]:
                matched = any(chunks_match(cid, ref) for ref in r["correct"])
                st.markdown(f"{'✅' if matched else '·'} `{cid}`")
            st.markdown("**Эталонные chunk_id:**")
            for ref in r["correct"]:
                st.markdown(f"→ `{ref}`")

    # ── Экспорт ──
    predictions = [
        {"question_id": r["question_id"], "top_5_chunks": r["top5"]}
        for r in results
    ]
    st.download_button(
        "💾 Скачать results.json (для score.py)",
        data=json.dumps(predictions, ensure_ascii=False, indent=2),
        file_name="results.json",
        mime="application/json",
    )


# ─── Страница: Расширение базы (бонус) ───────────────────────────────────────

def page_extend(collection):
    st.title("📦 Расширение тестовой базы")
    st.caption(
        "Участники вправе дополнительно использовать любые открытые Python-репозитории "
        "для расширения тестовой базы — это учитывается при оценке."
    )

    st.markdown("""
Система автоматически загружает три открытых Python-репозитория из GitHub
и индексирует их вместе с основной кодовой базой:

| Репозиторий | Описание | Почему полезен |
|---|---|---|
| **encode/httpx** | Async HTTP client | Паттерны запросов, middleware, auth |
| **Textualize/rich** | Terminal formatting | Хорошо структурированные классы/методы |
| **encode/starlette** | ASGI framework | Основа FastAPI — routing, requests, middleware |
    """)

    extra_dir = Path("./extra_repos")
    repos_present = {
        "httpx":      (extra_dir / "httpx").exists(),
        "rich":       (extra_dir / "rich").exists(),
        "starlette":  (extra_dir / "starlette").exists(),
    }

    st.subheader("Статус репозиториев")
    for name, present in repos_present.items():
        icon = "✅ загружен" if present else "⬜ не загружен"
        py_count = len(list((extra_dir / name).rglob("*.py"))) if present else 0
        col_name, col_status, col_files = st.columns([2, 2, 2])
        col_name.write(f"**{name}**")
        col_status.write(icon)
        col_files.write(f"{py_count} .py файлов" if present else "—")

    st.divider()

    if collection:
        st.info(f"Текущий размер индекса: **{collection.count()}** чанков")

    st.subheader("Индексирование с расширением базы")
    st.code(
        "# Загрузить репозитории и переиндексировать:\n"
        "python index.py ./codebase_python/gymhero --fetch-repos --reset\n\n"
        "# Если репозитории уже загружены (./extra_repos/):\n"
        "python index.py ./codebase_python/gymhero ./extra_repos/httpx "
        "./extra_repos/rich ./extra_repos/starlette --reset",
        language="bash",
    )

    # Кнопка для запуска прямо из UI
    st.subheader("Запуск из интерфейса")
    st.warning(
        "⚠️ Эта операция занимает 5–15 минут при первом запуске "
        "(загрузка + индексирование). Прогресс будет виден в терминале."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        run_fetch = st.button(
            "⬇️ Скачать репозитории + индексировать",
            type="primary",
            use_container_width=True,
        )
    with col_b:
        run_index_only = st.button(
            "♻️ Переиндексировать уже загруженные",
            use_container_width=True,
            disabled=not any(repos_present.values()),
        )

    if run_fetch or run_index_only:
        if run_fetch:
            cmd = [
                sys.executable, "index.py",
                "./codebase_python/gymhero",
                "--fetch-repos", "--reset",
            ]
        else:
            dirs = ["./codebase_python/gymhero"] + [
                f"./extra_repos/{n}" for n, p in repos_present.items() if p
            ]
            cmd = [sys.executable, "index.py"] + dirs + ["--reset"]

        with st.spinner("⏳ Индексирование... (см. терминал для деталей)"):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=900,
                )
                if result.returncode == 0:
                    st.success("✅ Индексирование завершено! Перезапустите app для обновления кеша.")
                    st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
                else:
                    st.error("❌ Ошибка при индексировании:")
                    st.code(result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                st.error("⏱ Превышено время ожидания (15 мин). Запустите вручную в терминале.")
            except Exception as e:
                st.error(f"❌ {e}")

    # Поддержка второго языка
    st.divider()
    st.subheader("🌐 Поддержка второго языка (JavaScript / Java)")
    st.markdown("""
`index.py` поддерживает парсинг JavaScript, TypeScript и Java через регулярные выражения.
Для индексирования JS/Java проекта используйте флаг `--lang`:
    """)
    st.code(
        "# JavaScript/TypeScript:\n"
        "python index.py ./my_js_project --lang js\n\n"
        "# Java:\n"
        "python index.py ./my_java_project --lang java\n\n"
        "# Python + JS вместе:\n"
        "python index.py ./backend ./frontend --lang python js",
        language="bash",
    )

    st.caption(
        "Парсер JS/Java использует регулярные выражения (без внешних зависимостей). "
        "Для максимальной точности на JS/Java можно установить `tree-sitter` — "
        "библиотеку синтаксического разбора, но для прототипа регулярок достаточно."
    )


# ─── Страница: О проекте ──────────────────────────────────────────────────────

def page_about():
    st.title("ℹ️ О проекте CodeLens")
    st.markdown("""
## Архитектура

```
.py / .js / .ts / .java файлы
          │
          ▼
  [index.py] — парсинг
  ├── Python  → ast (точный, встроенный)
  ├── JS/TS   → регулярные выражения
  └── Java    → регулярные выражения
          │
          ▼
  sentence-transformers
  paraphrase-multilingual-MiniLM-L12-v2
  dim=384, RU+EN, Apache 2.0
          │
          ▼
  ChromaDB (persistent, hnsw:space=cosine)
          │
          ▼
  [app.py / Streamlit]
  ├── 🧠 Семантический поиск    — вектор → cosine top-K
  ├── 🔀 Гибридный поиск        — вектор + keyword overlap (бонус)
  ├── 🤖 LLM-ответ              — Ollama / OpenAI / Anthropic  (бонус)
  ├── 📊 Precision@5             — авто-оценка на eval_questions.json (бонус)
  └── 📦 Расширение базы        — open-source репозитории (бонус)
```

## Стратегия чанкования

**Один чанк = одна функция или один класс. Методы — отдельный чанк (ClassName.method_name).**

Функция — минимальная семантическая единица кода: имеет имя, сигнатуру и единственную
ответственность (~10–80 строк). Методы индексируются отдельно, чтобы при поиске
возвращался конкретный метод, а не всё тело класса. Граница определяется синтаксисом
через `ast` — не произвольна. Это обеспечивает совместимость chunk_id с eval_questions.json.

## Технический стек

| Слой | Инструмент | Примечание |
|---|---|---|
| Язык | Python 3.12 | Требование ТЗ |
| Парсинг Python | `ast` (встроенный) | Надёжно, без зависимостей |
| Парсинг JS/Java | Регулярные выражения | Без внешних зависимостей |
| Эмбеддинги | `paraphrase-multilingual-MiniLM-L12-v2` | RU+EN, dim=384 |
| Векторная БД | ChromaDB (persistent, cosine) | Без отдельного сервера |
| UI | Streamlit | Python-only, `st.code()` встроена |
| Гибридный поиск | keyword overlap + vector | Лучше на точных именах |
| LLM | Ollama / OpenAI / Anthropic | Опционально |

## Запуск

```bash
pip install -r requirements.txt

# Основная база:
python index.py ./codebase_python/gymhero --reset

# С расширением open-source репозиториями:
python index.py ./codebase_python/gymhero --fetch-repos --reset

# Веб-интерфейс:
streamlit run app.py
```

## 5 примеров запросов

| Запрос | Что найдёт |
|---|---|
| `как создаётся токен доступа?` | `create_access_token` → security.py |
| `how does JWT verification work?` | `get_token`, `get_current_user` → dependencies.py |
| `где проверяется суперпользователь?` | `UserCRUDRepository.is_super_user`, `get_current_superuser` |
| `строка подключения к PostgreSQL` | `build_sqlalchemy_database_url_from_settings` → session.py |
| `how does session management work?` | `get_db`, `get_local_session` → session.py |

## Обоснование выбора модели и БД

**paraphrase-multilingual-MiniLM-L12-v2** — единственная рекомендованная в ТЗ модель,
поддерживает русский и английский (dim=384), работает быстро (latency < 1с на CPU),
лицензия Apache 2.0 — без ограничений для коммерческого использования.

**ChromaDB** выбран вместо FAISS: имеет встроенное хранение метаданных и фильтрацию,
не требует отдельного сервера, persist-режим сохраняет индекс между запусками,
API проще для прототипа. FAISS быстрее на очень больших базах, но требует внешнего
хранения метаданных.
    """)


# ─── Точка входа ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="CodeLens",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("🔍 CodeLens")
        st.caption("Умный поиск по кодовой базе")
        st.divider()

        db_path = st.text_input(
            "Путь к ChromaDB",
            value=DEFAULT_DB_PATH,
            help="Создаётся после python index.py",
        )

        page = st.radio(
            "Навигация",
            [
                "🔍 Поиск",
                "📊 Метрики (Precision@5)",
                "📦 Расширение базы",
                "ℹ️ О проекте",
            ],
        )

        st.divider()
        # Мини-статистика в сайдбаре
        col = load_collection(db_path)
        if col:
            stats = _db_stats(col)
            st.caption(f"📦 Чанков: **{stats.get('total', 0)}**")
            for lang, cnt in sorted(stats.get("langs", {}).items()):
                icon = {"python": "🐍", "js": "⚡", "java": "☕"}.get(lang, "📄")
                st.caption(f"{icon} {lang}: {cnt}")
        else:
            st.caption("⚠️ Индекс не найден")

        st.caption("Модель: `MiniLM-L12-v2`")
        st.caption("Языки запросов: 🇷🇺 + 🇬🇧")

    model = load_model()
    collection = load_collection(db_path)

    if page == "🔍 Поиск":
        page_search(collection, model)
    elif page == "📊 Метрики (Precision@5)":
        page_metrics(collection, model)
    elif page == "📦 Расширение базы":
        page_extend(collection)
    else:
        page_about()


if __name__ == "__main__":
    main()
