import json
import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from documentor.config import load_config, resolve

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL = re.compile(r"[a-z][A-Z]")
FENCE = re.compile(r"^\s*(```|~~~)")
# "node" alone is left out: it is a core LangGraph word ("add a node"), so only node.js / nodejs count.
JS_WORDS = ("javascript", "typescript", "js", "ts", "nodejs", "node.js", "langgraph.js", "langgraphjs", "npm", "yarn",
            "pnpm", "deno", "bun")
PY_WORDS = ("python", "pip", "pypi", "conda", "poetry")
JS_TEXT = "@langchain/langgraph"


def is_identifier(word: str) -> bool:
    """Identifier-shaped: has `_`, a lower-to-upper step (addNode), or starts uppercase with another uppercase."""
    if not re.search(r"[A-Za-z]", word):
        return False
    return "_" in word or bool(CAMEL.search(word)) or (word[0].isupper() and any(c.isupper() for c in word[1:]))


def code_lines(text: str) -> list[str]:
    lines, in_fence = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        elif in_fence:
            lines.append(line)
    return lines


def identifier_counts(texts: list[str]) -> Counter:
    counts = Counter()
    for text in texts:
        for line in code_lines(text):
            counts.update(w for w in WORD.findall(line) if is_identifier(w))
    return counts


def split_vocab(python: Counter, js: Counter, min_count: int) -> dict:
    """Identifiers seen often enough in one view and never in the other; shared ones are neutral."""
    py_only = {w: c for w, c in python.items() if c >= min_count and w not in js}
    js_only = {w: c for w, c in js.items() if c >= min_count and w not in python}
    order = lambda d: dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))
    return {"python_only": order(py_only), "js_only": order(js_only)}


def view_texts(clean_dir: Path, language: str) -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted((clean_dir / language).rglob("*.md"))]


def corpus_commit() -> str:
    path = resolve("data/raw/manifest.json")
    return json.loads(path.read_text())["commit"] if path.exists() else ""


def build_language_vocab(cfg: dict, clean_dir: Path | None = None, out: Path | None = None) -> dict:
    clean_dir = clean_dir or resolve(cfg["corpus"]["clean_dir"])
    vocab = split_vocab(identifier_counts(view_texts(clean_dir, "python")), identifier_counts(view_texts(clean_dir, "js")),
                        cfg["language"]["min_identifier_count"])
    vocab |= {"corpus_commit": corpus_commit(), "built_at": datetime.now().isoformat(timespec="seconds")}
    out = out or resolve(cfg["language"]["vocab_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, indent=1, ensure_ascii=False), encoding="utf-8")
    load_vocab.cache_clear()
    return vocab


@lru_cache(maxsize=1)
def load_vocab() -> dict:
    path = resolve(load_config()["language"]["vocab_path"])
    if not path.exists():
        print(f"WARNING: no language vocabulary at {path}; identifiers won't route. Run scripts/build_language_vocab.py.")
        return {"python_only": {}, "js_only": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def explicit_word(question: str, words) -> str | None:
    for word in words:
        if re.search(rf"(?<![\w@/.-]){re.escape(word)}(?![\w-])", question, re.IGNORECASE):
            return word
    return None


def detect_language(question: str, previous: str | None = None, override: str | None = None,
                    vocab: dict | None = None, cfg: dict | None = None) -> tuple[str, str]:
    """Explicit words, then identifiers found in only one view, then the previous turn, then the default."""
    cfg = cfg or load_config()
    if override:
        return override, "override"
    js_word = JS_TEXT if JS_TEXT in question.lower() else explicit_word(question, JS_WORDS)
    py_word = explicit_word(question, PY_WORDS)
    if js_word and not py_word:
        return "js", f'word "{js_word}"'
    if py_word and not js_word:
        return "python", f'word "{py_word}"'
    vocab = vocab if vocab is not None else load_vocab()
    words = [w for w in dict.fromkeys(WORD.findall(question)) if is_identifier(w)]
    js_hits = [w for w in words if w in vocab["js_only"]]
    py_hits = [w for w in words if w in vocab["python_only"]]
    if len(js_hits) > len(py_hits):
        return "js", f"identifier {js_hits[0]}"
    if len(py_hits) > len(js_hits):
        return "python", f"identifier {py_hits[0]}"
    if previous and cfg["language"]["sticky"]:
        return previous, "previous turn"
    return cfg["corpus"]["default_language"], "default"
