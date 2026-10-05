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

    def count(self) -> int:
        return self.collection.count()
