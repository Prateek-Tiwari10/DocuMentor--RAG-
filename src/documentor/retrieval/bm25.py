import json
import re
import time
from datetime import datetime
from pathlib import Path

from rank_bm25 import BM25Okapi

from documentor.config import resolve
from documentor.ingest.chunker import make_embed_text

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+[0-9]*")
STOPWORDS = frozenset("""
    a an and are as at be but by can do does for from how i if in is it its me my of on or our should so that the
    their then there this to was we what when where which why will with would you your
""".split())


def word_parts(word: str) -> list[str]:
    """Lowercased parts of an identifier: split on `_`, then on camel-case boundaries."""
    return [part.lower() for piece in word.split("_") for part in CAMEL.findall(piece)]


def bm25_tokens(text: str) -> list[str]:
    """Each word lowercased, plus the parts of a compound identifier. Used for documents and queries alike."""
    tokens = []
    for word in WORD.findall(text):
        lower = word.lower()
        if lower not in STOPWORDS:
            tokens.append(lower)
        parts = word_parts(word)
        if parts != [lower]:
            tokens += [part for part in parts if part not in STOPWORDS]
    return tokens


class BM25Index:
    def __init__(self, k1: float, b: float):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.tokens: list[list[str]] = []
        self.meta: dict = {}
        self.bm25: BM25Okapi | None = None

    def _fit(self) -> None:
        self.bm25 = BM25Okapi(self.tokens, k1=self.k1, b=self.b) if self.tokens else None

    def build(self, ids: list[str], texts: list[str], collection: str = "", language: str = "",
              corpus_commit: str = "") -> "BM25Index":
        self.ids = list(ids)
        self.tokens = [bm25_tokens(text) for text in texts]
        self.meta = {"collection": collection, "language": language, "corpus_commit": corpus_commit,
                     "chunk_count": len(self.ids), "built_at": datetime.now().isoformat(timespec="seconds")}
        self._fit()
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ids": self.ids, "tokens": self.tokens, **self.meta}, ensure_ascii=False),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, k1: float, b: float) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(k1, b)
        index.ids = data.pop("ids")
        index.tokens = data.pop("tokens")
        index.meta = data
        index._fit()
        return index

    @property
    def vocabulary_size(self) -> int:
        return len(self.bm25.idf) if self.bm25 else 0

    def top_n(self, query: str, n: int) -> list[tuple[str, float]]:
        tokens = bm25_tokens(query)
        if not self.bm25 or not tokens:
            return []
        pairs = [(i, float(s)) for i, s in zip(self.ids, self.bm25.get_scores(tokens)) if s > 0]
        return sorted(pairs, key=lambda p: (-p[1], p[0]))[:n]


def index_path(cfg: dict, collection: str, language: str) -> Path:
    return resolve(cfg["bm25"]["path"]) / f"{collection}__{language}.json"


def warn_if_stale(index: BM25Index, count: int) -> bool:
    built = index.meta.get("chunk_count")
    if built == count:
        return False
    print(f"WARNING: BM25 index {index.meta.get('collection')} ({index.meta.get('language')}) has {built} chunks but "
          f"the collection has {count}. Rerun scripts/build_bm25.py.")
    return True


def build_from_store(store, language: str, cfg: dict, corpus_commit: str = "") -> BM25Index:
    """Index exactly what Chroma holds for one language, so ids match the vector store."""
    prepend = cfg["chunking"]["prepend_headings"]
    chunks = store.by_language(language)
    texts = [make_embed_text(c.get("title") or "", c.get("heading_path") or "", c["text"], prepend) for c in chunks]
    return BM25Index(cfg["bm25"]["k1"], cfg["bm25"]["b"]).build(
        [c["id"] for c in chunks], texts, collection=store.name, language=language, corpus_commit=corpus_commit)


def build_and_save(store, language: str, cfg: dict, corpus_commit: str = "") -> tuple[BM25Index, float]:
    start = time.perf_counter()
    index = build_from_store(store, language, cfg, corpus_commit)
    index.save(index_path(cfg, store.name, language))
    return index, time.perf_counter() - start
