import atexit
import warnings

from qdrant_client import QdrantClient, models

from documentor.store.chroma_store import to_metadata

_clients: dict[str, QdrantClient] = {}


def get_client(path: str | None = None, url: str | None = None) -> QdrantClient:
    """One client per location: local mode locks its folder, so two clients on one path would clash."""
    key = url or str(path)
    if key not in _clients:
        client = QdrantClient(url=url) if url else QdrantClient(path=str(path))
        atexit.register(client.close)  # closing during interpreter shutdown fails noisily, so close first
        _clients[key] = client
    return _clients[key]


def point_id(chunk_id: str) -> int:
    """Chunk ids are 16 hex characters; Qdrant wants an unsigned integer or a UUID."""
    return int(chunk_id, 16)


def where(language: str | None = None, page_slug: str | None = None) -> models.Filter | None:
    must = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in (("language", language), ("page_slug", page_slug)) if v]
    return models.Filter(must=must) if must else None


class QdrantStore:
    def __init__(self, collection: str, path: str | None = None, url: str | None = None, client=None,
                 batch_size: int = 1000):
        self.client = client or get_client(path, url)
        self.name = collection
        self.batch_size = batch_size

    def exists(self) -> bool:
        return self.client.collection_exists(self.name)

    def _create(self, size: int) -> None:
        self.client.create_collection(self.name, vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE))
        with warnings.catch_warnings():  # local mode ignores payload indexes and says so; a server uses them
            warnings.simplefilter("ignore", UserWarning)
            for field in ("language", "page_slug"):
                self.client.create_payload_index(self.name, field, field_schema=models.PayloadSchemaType.KEYWORD)

    def reset(self) -> None:
        if self.exists():
            self.client.delete_collection(self.name)

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if not self.exists():
            self._create(len(embeddings[0]))
        for i in range(0, len(chunks), self.batch_size):
            points = [models.PointStruct(id=point_id(c["id"]), vector=[float(x) for x in e],
                                         payload=to_metadata(c) | {"id": c["id"], "text": c["text"]})
                      for c, e in zip(chunks[i:i + self.batch_size], embeddings[i:i + self.batch_size])]
            self.client.upsert(self.name, points=points)

    def query(self, embedding: list[float], k: int, language: str | None = None) -> list[dict]:
        if not self.exists():
            return []
        res = self.client.query_points(self.name, query=list(embedding), limit=k, query_filter=where(language),
                                       with_payload=True)
        return [{**p.payload, "score": p.score} for p in res.points]  # cosine similarity already, unlike Chroma's distance

    def _scroll(self, flt: models.Filter | None) -> list[dict]:
        rows, offset = [], None
        while True:
            points, offset = self.client.scroll(self.name, scroll_filter=flt, limit=self.batch_size, offset=offset,
                                                with_payload=True)
            rows += [dict(p.payload) for p in points]
            if offset is None:
                return rows

    def get_all(self, language: str) -> list[dict]:
        return self._scroll(where(language)) if self.exists() else []

    def by_ids(self, ids: list[str]) -> list[dict]:
        if not ids or not self.exists():
            return []
        return [dict(p.payload) for p in self.client.retrieve(self.name, ids=[point_id(i) for i in ids], with_payload=True)]

    def delete_page(self, page_slug: str, language: str) -> int:
        if not self.exists():
            return 0
        flt = where(language, page_slug)
        n = self.client.count(self.name, count_filter=flt, exact=True).count
        if n:
            self.client.delete(self.name, points_selector=models.FilterSelector(filter=flt))
        return n

    def count(self, language: str | None = None) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.name, count_filter=where(language), exact=True).count
