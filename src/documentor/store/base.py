from typing import Protocol

from documentor.config import resolve

BACKENDS = ("chroma", "qdrant")


class VectorStore(Protocol):
    name: str

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None: ...
    def query(self, embedding, k: int, language: str | None = None) -> list[dict]: ...  # score = cosine similarity
    def get_all(self, language: str) -> list[dict]: ...      # id, text, metadata; used by BM25 and migration
    def by_ids(self, ids: list[str]) -> list[dict]: ...      # chunks BM25 found that dense search didn't fetch
    def delete_page(self, page_slug: str, language: str) -> int: ...
    def count(self, language: str | None = None) -> int: ...
    def reset(self) -> None: ...


def get_store(cfg: dict, collection: str, backend: str | None = None) -> VectorStore:
    """The configured backend; `backend` overrides it for migration and parity checks."""
    s = cfg["store"]
    backend = backend or s.get("backend", "chroma")
    batch = s.get("batch_size", 1000)
    if backend == "chroma":
        from documentor.store.chroma_store import ChromaStore
        return ChromaStore(resolve(s["chroma_path"]), collection, batch_size=batch)
    if backend == "qdrant":
        from documentor.store.qdrant_store import QdrantStore
        return QdrantStore(collection, path=None if s.get("qdrant_url") else resolve(s["qdrant_path"]),
                           url=s.get("qdrant_url"), batch_size=batch)
    raise ValueError(f"unknown store backend: {backend}")
