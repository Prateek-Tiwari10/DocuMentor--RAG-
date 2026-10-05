from functools import lru_cache

from documentor.config import collection_name, load_config, resolve
from documentor.embed.embedder import Embedder
from documentor.store.chroma_store import ChromaStore
from documentor.store.parent_store import ParentStore

RESULT_FIELDS = ("page_slug", "language", "url", "title", "heading_path", "tokens", "text")


@lru_cache(maxsize=2)
def get_embedder(model: str) -> Embedder:
    emb = load_config()["embedding"]
    return Embedder(model, emb["device"], emb["batch_size"])


@lru_cache(maxsize=4)
def get_store(strategy: str, model: str) -> ChromaStore:
    return ChromaStore(resolve(load_config()["store"]["chroma_path"]), collection_name(strategy, model))


@lru_cache(maxsize=4)
def get_parents(strategy: str, model: str) -> ParentStore:
    path = resolve(load_config()["store"]["parents_path"]) / f"{collection_name(strategy, model)}.json"
    return ParentStore(path)


def group_by_parent(hits: list[dict], k: int) -> list[dict]:
    """Walk child hits in rank order and keep the first k unique parents."""
    groups: dict[str, dict] = {}
    for hit in hits:
        pid = hit["parent_id"]
        if pid in groups:
            groups[pid]["matched_children"] += 1
        elif len(groups) < k:
            groups[pid] = {"parent_id": pid, "best_child_score": hit["score"],
                           "best_child_text": hit["text"], "matched_children": 1}
    return list(groups.values())


def _heading_str(path) -> str:
    return " > ".join(path) if isinstance(path, list) else path


def search(query: str, strategy: str, k: int | None = None, language: str | None = None,
           model: str | None = None, embedder=None, store=None, parents=None) -> list[dict]:
    cfg = load_config()
    k = k or cfg["retrieval"]["top_k"]
    language = language or cfg["corpus"]["default_language"]
    model = model or cfg["embedding"]["model"]
    embedder = embedder or get_embedder(model)
    store = store or get_store(strategy, model)
    query_vec = embedder.embed_query(query)

    if strategy == "recursive":
        hits = store.query(query_vec, k, language=language)
        results = [{f: h.get(f) for f in RESULT_FIELDS} | {"score": h["score"], "id": h.get("id")}
                   for h in hits]
    elif strategy == "parent_child":
        parents = parents or get_parents(strategy, model)
        hits = store.query(query_vec, cfg["chunking"]["parent_child"]["child_top_k"], language=language)
        results = []
        for group in group_by_parent(hits, k):
            parent = parents.get(group["parent_id"]) or {}
            results.append({f: parent.get(f) for f in RESULT_FIELDS}
                           | group | {"score": group["best_child_score"]})
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        r["heading_path"] = _heading_str(r["heading_path"])
    return results
