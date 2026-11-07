from functools import lru_cache

from documentor.config import collection_name, load_config, resolve
from documentor.embed.embedder import Embedder
from documentor.retrieval import TRACE_FIELDS
from documentor.store.base import VectorStore
from documentor.store.base import get_store as backend_store
from documentor.store.parent_store import ParentStore

RESULT_FIELDS = ("page_slug", "language", "url", "title", "heading_path", "tokens", "text")


@lru_cache(maxsize=2)
def get_embedder(model: str) -> Embedder:
    emb = load_config()["embedding"]
    return Embedder(model, emb["device"], emb["batch_size"])


@lru_cache(maxsize=4)
def get_store(strategy: str, model: str) -> VectorStore:
    return backend_store(load_config(), collection_name(strategy, model))


@lru_cache(maxsize=4)
def get_parents(strategy: str, model: str) -> ParentStore:
    path = resolve(load_config()["store"]["parents_path"]) / f"{collection_name(strategy, model)}.json"
    return ParentStore(path)


def trace(hit: dict) -> dict:
    return {f: hit.get(f) for f in TRACE_FIELDS}


def group_by_parent(hits: list[dict], k: int) -> list[dict]:
    """Walk child hits in rank order and keep the first k unique parents."""
    groups: dict[str, dict] = {}
    for hit in hits:
        pid = hit["parent_id"]
        if pid in groups:
            groups[pid]["matched_children"] += 1
        elif len(groups) < k:
            groups[pid] = {"parent_id": pid, "best_child_score": hit["score"], "best_child_text": hit["text"],
                           "matched_children": 1} | trace(hit)
    return list(groups.values())


def _heading_str(path) -> str:
    return " > ".join(path) if isinstance(path, list) else path


def dense_candidates(vector: list[float], n: int, language: str, store) -> list[dict]:
    hits = store.query(vector, n, language=language)
    for rank, hit in enumerate(hits, 1):
        hit["dense_rank"] = rank
    return hits


def recursive_results(hits: list[dict]) -> list[dict]:
    return [{f: h.get(f) for f in RESULT_FIELDS} | {"score": h["score"], "id": h.get("id")} | trace(h) for h in hits]


def group_parents(hits: list[dict], k: int, parents) -> list[dict]:
    results = []
    for group in group_by_parent(hits, k):
        parent = parents.get(group["parent_id"]) or {}
        results.append({f: parent.get(f) for f in RESULT_FIELDS} | group | {"score": group["best_child_score"]})
    return results


def number_results(results: list[dict]) -> list[dict]:
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        r["heading_path"] = _heading_str(r["heading_path"])
    return results


def search(query: str, strategy: str | None = None, k: int | None = None, language: str | None = None,
           mode: str | None = None, rerank: bool | None = None, *, reranker=None, timings: dict | None = None,
           model: str | None = None, embedder=None, store=None, parents=None, index=None,
           child_top_k: int | None = None, depth: int | None = None) -> list[dict]:
    """The public entry point. A setting left as None uses the config default."""
    from documentor.retrieval.hybrid import retrieve

    cfg = load_config()
    r = cfg["retrieval"]
    return retrieve(query, strategy or r["strategy"], k or r["top_k"], language or cfg["corpus"]["default_language"],
                    mode or r["mode"], r["rerank"] if rerank is None else rerank, reranker, timings, model=model,
                    embedder=embedder, store=store, parents=parents, index=index, child_top_k=child_top_k, depth=depth)
