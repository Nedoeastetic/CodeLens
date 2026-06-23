"""
CodeLens — индексирующий скрипт.

Парсит Python/JavaScript/Java кодовую базу, разбивает на чанки
(функции/классы/методы), строит эмбеддинги и сохраняет в ChromaDB.

Поддерживаемые языки:
    Python (.py)     — через встроенный модуль ast (точный парсинг)
    JavaScript (.js, .ts, .jsx, .tsx) — через регулярки
    Java (.java)     — через регулярки

Стратегия чанкования:
    Один чанк = одна функция или один класс (верхнего уровня).
    Методы класса также индексируются отдельно — с именем ClassName.method_name.
    Это обеспечивает точечный поиск конкретного метода без возврата всего
    тела класса и совместимо с форматом eval_questions.json.
    Функция — минимальная семантическая единица кода: имеет имя,
    сигнатуру и единственную ответственность (~10–80 строк).

Расширение тестовой базы:
    Используйте флаг --fetch-repos для автоматической загрузки
    открытых Python-репозиториев с GitHub (FastAPI, httpx, rich и др.)

Запуск:
    # Только основная кодовая база (чемпионат):
    python index.py ./codebase_python/gymhero

    # Сбросить и переиндексировать:
    python index.py ./codebase_python/gymhero --reset

    # С расширением базы из open-source репозиториев:
    python index.py ./codebase_python/gymhero --fetch-repos

    # Несколько папок сразу (Python + JS):
    python index.py ./codebase_python/gymhero ./extra_repos --lang python js

    # Только JS-файлы:
    python index.py ./my_js_project --lang js

ВАЖНО для датасета чемпионата:
    Передавайте ./codebase_python/gymhero (а не ./codebase_python).
    Это обеспечивает пути вида gymhero/security.py:... совпадающие
    с eval_questions.json.
"""

import argparse
import ast
import re
import subprocess
import sys
import urllib.request
import zipfile
import io
from pathlib import Path
from typing import Iterator

import chromadb
from sentence_transformers import SentenceTransformer

# ─── Константы ────────────────────────────────────────────────────────────────
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "codelens"
DEFAULT_DB_PATH = "./chroma_db"
BATCH_SIZE = 64
MAX_CHUNK_CHARS = 8000

# Открытые Python-репозитории для расширения тестовой базы
# Выбраны: небольшие, хорошо структурированные FastAPI-экосистема
OPEN_SOURCE_REPOS = [
    {
        "name": "httpx",
        "url": "https://github.com/encode/httpx/archive/refs/heads/master.zip",
        "inner": "httpx-master/httpx",
        "desc": "HTTP client library (encode/httpx)",
    },
    {
        "name": "rich",
        "url": "https://github.com/Textualize/rich/archive/refs/heads/master.zip",
        "inner": "rich-master/rich",
        "desc": "Terminal formatting library (Textualize/rich)",
    },
    {
        "name": "starlette",
        "url": "https://github.com/encode/starlette/archive/refs/heads/master.zip",
        "inner": "starlette-master/starlette",
        "desc": "ASGI framework (encode/starlette) — основа FastAPI",
    },
]

EXTRA_REPOS_DIR = Path("./extra_repos")


# ─── Python-парсинг (через ast) ───────────────────────────────────────────────

def get_docstring(node: ast.AST) -> str:
    try:
        return (ast.get_docstring(node) or "")[:500]
    except Exception:
        return ""


def extract_source_lines(source_lines: list[str], node: ast.AST) -> str:
    code = "".join(source_lines[node.lineno - 1: node.end_lineno])
    return code[:MAX_CHUNK_CHARS] + ("\n# ... (truncated)" if len(code) > MAX_CHUNK_CHARS else "")


def iter_python_chunks(file_path: Path, root: Path) -> Iterator[dict]:
    """
    AST-парсер Python. Генерирует чанки вида:
      - function_name              (функция верхнего уровня)
      - ClassName                  (класс целиком)
      - ClassName.method_name      (метод — отдельный чанк)
    chunk_id = "rel/path.py:Name:lineno"
    """
    try:
        source = file_path.read_bytes().decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return

    source_lines = source.splitlines(keepends=True)
    norm_path = str(file_path.relative_to(root)).replace("\\", "/")

    def make(node, name: str, ctype: str) -> dict:
        code = extract_source_lines(source_lines, node)
        doc = get_docstring(node)
        return {
            "chunk_id": f"{norm_path}:{name}:{node.lineno}",
            "file_path": norm_path,
            "type": ctype,
            "name": name,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "docstring": doc,
            "source_code": code,
            "embed_text": f"{name}\n{doc}\n{code}",
            "language": "python",
        }

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ctype = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            yield make(node, node.name, ctype)
        elif isinstance(node, ast.ClassDef):
            yield make(node, node.name, "class")
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ctype = "async_method" if isinstance(child, ast.AsyncFunctionDef) else "method"
                    yield make(child, f"{node.name}.{child.name}", ctype)


# ─── JavaScript/TypeScript-парсинг (через регулярки) ─────────────────────────

# Паттерны для JS/TS:
_JS_FUNC_PATTERNS = [
    # function foo(...) { / async function foo(...) {
    re.compile(
        r'^(?P<indent>[ \t]*)(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)\s*\(',
        re.MULTILINE,
    ),
    # const foo = (...) => {   /   const foo = async (...) => {
    re.compile(
        r'^(?P<indent>[ \t]*)(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s*)?\(',
        re.MULTILINE,
    ),
    # foo(...) { — метод в классе (простой случай)
    re.compile(
        r'^(?P<indent>[ \t]*)(?:static\s+)?(?:async\s+)?(?P<name>\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE,
    ),
]

_JS_CLASS_PATTERN = re.compile(
    r'^(?P<indent>[ \t]*)(?:export\s+)?(?:default\s+)?class\s+(?P<name>\w+)',
    re.MULTILINE,
)


def _extract_block(lines: list[str], start_idx: int) -> tuple[int, str]:
    """
    Извлекает блок кода начиная со строки start_idx, считая фигурные скобки.
    Возвращает (end_line_1based, code_str).
    """
    depth = 0
    result = []
    found_open = False
    for i, line in enumerate(lines[start_idx:], start=start_idx):
        result.append(line)
        for ch in line:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
        if found_open and depth <= 0:
            return i + 1, "".join(result)
    return len(lines), "".join(result)


def iter_js_chunks(file_path: Path, root: Path) -> Iterator[dict]:
    """Регулярный парсер JS/TS. Находит функции, стрелочные функции, классы."""
    try:
        source = file_path.read_bytes().decode("utf-8", errors="replace")
    except Exception:
        return

    lines = source.splitlines(keepends=True)
    norm_path = str(file_path.relative_to(root)).replace("\\", "/")
    ext = file_path.suffix.lstrip(".")

    seen_lines: set[int] = set()

    def make_js(name: str, ctype: str, start_line: int, end_line: int, code: str) -> dict:
        if len(code) > MAX_CHUNK_CHARS:
            code = code[:MAX_CHUNK_CHARS] + "\n// ... (truncated)"
        return {
            "chunk_id": f"{norm_path}:{name}:{start_line}",
            "file_path": norm_path,
            "type": ctype,
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
            "docstring": "",
            "source_code": code,
            "embed_text": f"{name}\n{code}",
            "language": ext,
        }

    # Классы
    for m in _JS_CLASS_PATTERN.finditer(source):
        start_line = source.count("\n", 0, m.start()) + 1
        end_line, code = _extract_block(lines, start_line - 1)
        if start_line not in seen_lines:
            seen_lines.add(start_line)
            yield make_js(m.group("name"), "class", start_line, end_line, code)

    # Функции
    for pattern in _JS_FUNC_PATTERNS:
        for m in pattern.finditer(source):
            name = m.group("name")
            if name in ("if", "for", "while", "switch", "catch", "constructor",
                        "return", "else", "try", "do"):
                continue
            start_line = source.count("\n", 0, m.start()) + 1
            if start_line in seen_lines:
                continue
            end_line, code = _extract_block(lines, start_line - 1)
            seen_lines.add(start_line)
            ctype = "method" if m.group("indent") else "function"
            yield make_js(name, ctype, start_line, end_line, code)


# ─── Java-парсинг (через регулярки) ───────────────────────────────────────────

_JAVA_CLASS_PATTERN = re.compile(
    r'^(?P<indent>[ \t]*)(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+)*'
    r'(?:class|interface|enum)\s+(?P<name>\w+)',
    re.MULTILINE,
)

_JAVA_METHOD_PATTERN = re.compile(
    r'^(?P<indent>[ \t]*)(?:(?:public|private|protected|static|final|synchronized|abstract|'
    r'native|strictfp)\s+)*'
    r'(?:[\w<>\[\]]+\s+)+(?P<name>\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
    re.MULTILINE,
)


def iter_java_chunks(file_path: Path, root: Path) -> Iterator[dict]:
    """Регулярный парсер Java. Находит классы, интерфейсы и методы."""
    try:
        source = file_path.read_bytes().decode("utf-8", errors="replace")
    except Exception:
        return

    lines = source.splitlines(keepends=True)
    norm_path = str(file_path.relative_to(root)).replace("\\", "/")
    seen_lines: set[int] = set()

    def make_java(name: str, ctype: str, start_line: int, end_line: int, code: str) -> dict:
        if len(code) > MAX_CHUNK_CHARS:
            code = code[:MAX_CHUNK_CHARS] + "\n// ... (truncated)"
        return {
            "chunk_id": f"{norm_path}:{name}:{start_line}",
            "file_path": norm_path,
            "type": ctype,
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
            "docstring": "",
            "source_code": code,
            "embed_text": f"{name}\n{code}",
            "language": "java",
        }

    for m in _JAVA_CLASS_PATTERN.finditer(source):
        start_line = source.count("\n", 0, m.start()) + 1
        end_line, code = _extract_block(lines, start_line - 1)
        if start_line not in seen_lines:
            seen_lines.add(start_line)
            yield make_java(m.group("name"), "class", start_line, end_line, code)

    for m in _JAVA_METHOD_PATTERN.finditer(source):
        name = m.group("name")
        if name in ("if", "for", "while", "switch", "catch", "try", "else"):
            continue
        start_line = source.count("\n", 0, m.start()) + 1
        if start_line in seen_lines:
            continue
        end_line, code = _extract_block(lines, start_line - 1)
        seen_lines.add(start_line)
        yield make_java(name, "method", start_line, end_line, code)


# ─── Диспетчер по расширению ──────────────────────────────────────────────────

EXT_TO_LANG = {
    ".py":  "python",
    ".js":  "js",
    ".ts":  "js",
    ".jsx": "js",
    ".tsx": "js",
    ".java": "java",
}

LANG_TO_EXTS = {
    "python": [".py"],
    "js":     [".js", ".ts", ".jsx", ".tsx"],
    "java":   [".java"],
}


def iter_file_chunks(file_path: Path, root: Path) -> Iterator[dict]:
    ext = file_path.suffix.lower()
    lang = EXT_TO_LANG.get(ext)
    if lang == "python":
        yield from iter_python_chunks(file_path, root)
    elif lang == "js":
        yield from iter_js_chunks(file_path, root)
    elif lang == "java":
        yield from iter_java_chunks(file_path, root)


def collect_chunks(
    code_dirs: list[Path],
    languages: list[str],
) -> list[dict]:
    """Собирает все чанки из всех файлов в указанных директориях."""
    extensions = set()
    for lang in languages:
        extensions.update(LANG_TO_EXTS.get(lang, []))

    chunks: list[dict] = []
    total_files = 0

    for code_dir in code_dirs:
        files = [
            fp for fp in sorted(code_dir.rglob("*"))
            if fp.is_file() and fp.suffix.lower() in extensions
        ]
        total_files += len(files)
        lang_str = ", ".join(languages)
        print(f"  [{code_dir}] найдено файлов ({lang_str}): {len(files)}")

        for fp in files:
            try:
                file_chunks = list(iter_file_chunks(fp, code_dir))
                chunks.extend(file_chunks)
            except Exception as e:
                print(f"    ⚠ Ошибка при парсинге {fp}: {e}")

    print(f"Итого файлов: {total_files} | Итого чанков до дедупликации: {len(chunks)}")
    return chunks


def deduplicate(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for c in chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            result.append(c)
    return result


# ─── Загрузка open-source репозиториев ───────────────────────────────────────

def fetch_open_source_repos(dest: Path = EXTRA_REPOS_DIR) -> list[Path]:
    """
    Скачивает открытые Python-репозитории с GitHub для расширения тестовой базы.
    Возвращает список директорий с кодом.

    Выбранные репозитории:
    - httpx (encode/httpx) — популярный async HTTP client
    - rich (Textualize/rich) — terminal formatting
    - starlette (encode/starlette) — ASGI framework, основа FastAPI

    Все они имеют типовую FastAPI-подобную структуру и хорошо покрывают
    темы: middleware, routing, request/response handling — релевантные
    запросам eval_questions.json.
    """
    dest.mkdir(parents=True, exist_ok=True)
    result_dirs: list[Path] = []

    for repo in OPEN_SOURCE_REPOS:
        repo_dir = dest / repo["name"]
        if repo_dir.exists() and any(repo_dir.rglob("*.py")):
            print(f"  ✓ {repo['desc']} — уже скачан ({repo_dir})")
            result_dirs.append(repo_dir)
            continue

        print(f"  ⬇ Скачиваю {repo['desc']}...")
        try:
            with urllib.request.urlopen(repo["url"], timeout=30) as resp:
                data = resp.read()

            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                # Распаковываем только нужную внутреннюю директорию
                inner_prefix = repo["inner"] + "/"
                extracted = 0
                for member in zf.infolist():
                    if not member.filename.startswith(inner_prefix):
                        continue
                    # Переписываем путь: inner/foo.py → repo_dir/foo.py
                    rel = member.filename[len(inner_prefix):]
                    if not rel:
                        continue
                    target = repo_dir / rel
                    if member.filename.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(zf.read(member.filename))
                        extracted += 1

            py_count = len(list(repo_dir.rglob("*.py")))
            print(f"    ✓ Извлечено {extracted} файлов, Python-файлов: {py_count}")
            result_dirs.append(repo_dir)

        except Exception as e:
            print(f"    ✗ Не удалось скачать {repo['name']}: {e}")
            if repo_dir.exists():
                import shutil
                shutil.rmtree(repo_dir, ignore_errors=True)

    return result_dirs


# ─── Индексирование ───────────────────────────────────────────────────────────

def build_index(chunks: list[dict], db_path: str, reset: bool = False) -> None:
    """Строит эмбеддинги и сохраняет в ChromaDB."""

    print(f"\nЗагружаю модель {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Подключаюсь к ChromaDB: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Старая коллекция удалена.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"Новых чанков: {len(new_chunks)} (уже в БД: {len(existing_ids)})")

    if not new_chunks:
        print("Индекс актуален.")
        return

    texts = [c["embed_text"] for c in new_chunks]
    ids = [c["chunk_id"] for c in new_chunks]
    documents = [c["source_code"] for c in new_chunks]
    metadatas = [
        {
            "file_path": c["file_path"],
            "type": c["type"],
            "name": c["name"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "docstring": c["docstring"],
            "language": c.get("language", "python"),
        }
        for c in new_chunks
    ]

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        vecs = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(vecs.tolist())
        done = min(i + BATCH_SIZE, len(texts))
        print(f"\r  Эмбеддинги: {done}/{len(texts)}", end="", flush=True)
    print()

    for i in range(0, len(new_chunks), BATCH_SIZE):
        sl = slice(i, i + BATCH_SIZE)
        collection.add(
            ids=ids[sl],
            embeddings=all_embeddings[sl],
            documents=documents[sl],
            metadatas=metadatas[sl],
        )

    print(f"✓ Проиндексировано {len(new_chunks)} чанков. "
          f"Всего в коллекции: {collection.count()}")


# ─── Статистика ───────────────────────────────────────────────────────────────

def print_stats(chunks: list[dict]) -> None:
    from collections import Counter
    langs = Counter(c.get("language", "python") for c in chunks)
    types = Counter(c["type"] for c in chunks)
    files = len(set(c["file_path"] for c in chunks))
    print(f"\n📊 Статистика чанков:")
    print(f"   Файлов: {files}")
    print(f"   По языку: {dict(langs)}")
    print(f"   По типу: {dict(types)}")


# ─── Точка входа ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CodeLens — индексирование Python/JS/Java кодовой базы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python index.py ./codebase_python/gymhero
  python index.py ./codebase_python/gymhero --reset
  python index.py ./codebase_python/gymhero --fetch-repos
  python index.py ./codebase_python/gymhero --lang python js
  python index.py ./my_project --lang java
        """,
    )
    parser.add_argument(
        "code_dirs",
        nargs="+",
        help=(
            "Пути к директориям с кодом. "
            "Для чемпионата: ./codebase_python/gymhero"
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Удалить существующую коллекцию и переиндексировать заново",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Путь к ChromaDB (по умолчанию: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        choices=["python", "js", "java"],
        default=["python"],
        help="Языки для индексирования (по умолчанию: python)",
    )
    parser.add_argument(
        "--fetch-repos",
        action="store_true",
        help=(
            "Скачать открытые Python-репозитории (httpx, rich, starlette) "
            "для расширения тестовой базы. Сохраняются в ./extra_repos/"
        ),
    )
    args = parser.parse_args()

    # Проверяем директории
    code_dirs = []
    for d in args.code_dirs:
        p = Path(d)
        if not p.exists():
            print(f"✗ Директория не найдена: {p}", file=sys.stderr)
            sys.exit(1)
        code_dirs.append(p)

    # Загружаем открытые репозитории (бонус)
    if args.fetch_repos:
        print("\n📦 Загрузка open-source репозиториев для расширения тестовой базы...")
        extra_dirs = fetch_open_source_repos()
        code_dirs.extend(extra_dirs)
        if "python" not in args.lang:
            args.lang = list(args.lang) + ["python"]

    print(f"\n🔍 Директории: {[str(d) for d in code_dirs]}")
    print(f"🌐 Языки: {args.lang}")

    # Собираем чанки
    chunks = collect_chunks(code_dirs, args.lang)
    chunks = deduplicate(chunks)

    if not chunks:
        print("⚠ Не найдено ни одного чанка. Проверьте путь и расширения.")
        sys.exit(1)

    print_stats(chunks)

    # Индексируем
    build_index(chunks, args.db_path, reset=args.reset)


if __name__ == "__main__":
    main()
