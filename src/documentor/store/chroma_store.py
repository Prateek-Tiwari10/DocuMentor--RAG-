import chromadb

SKIP_FIELDS = {"text", "embed_text"}


def to_metadata(chunk: dict) -> dict:
    meta = {}
    for key, value in chunk.items():
        if key in SKIP_FIELDS or value is None:
            continue
        if isinstance(value, list):
            value = " > ".join(value)
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
    return meta


class ChromaStore:
    def __init__(self, path: str | None, collection: str, client=None, batch_size: int = 1000):
        self.client = client or chromadb.PersistentClient(path=str(path))
        self.name = collection
        self.batch_size = batch_size
        self.collection = self._open()

    def _open(self):
        return self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.collection = self._open()

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[to_metadata(c) for c in batch],
                embeddings=embeddings[i:i + self.batch_size],
            )

    def query(self, embedding: list[float], k: int, language: str | None = None) -> list[dict]:
        where = {"language": language} if language else None
        res = self.collection.query(query_embeddings=[embedding], n_results=k, where=where,
                                    include=["documents", "metadatas", "distances"])
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({**meta, "text": doc, "score": 1 - dist})
        return hits

    def _rows(self, res: dict) -> list[dict]:
        return [{**meta, "id": chunk_id, "text": doc}
                for chunk_id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]

    def get_all(self, language: str) -> list[dict]:
        return self._rows(self.collection.get(where={"language": language}, include=["documents", "metadatas"]))

    def by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        return self._rows(self.collection.get(ids=ids, include=["documents", "metadatas"]))

    def page_ids(self, page_slug: str, language: str) -> list[str]:
        where = {"$and": [{"page_slug": page_slug}, {"language": language}]}
        return self.collection.get(where=where, include=[])["ids"]

    def delete_page(self, page_slug: str, language: str) -> int:
        ids = self.page_ids(page_slug, language)
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self, language: str | None = None) -> int:
        if language is None:
            return self.collection.count()
        return len(self.collection.get(where={"language": language}, include=[])["ids"])

    def batches(self, size: int = 1000):
        """Ids, embeddings, documents and metadata in batches, for the Qdrant migration."""
        total = self.collection.count()
        for offset in range(0, total, size):
            res = self.collection.get(include=["embeddings", "documents", "metadatas"], limit=size, offset=offset)
            chunks = self._rows(res)
            yield chunks, [list(map(float, e)) for e in res["embeddings"]]
