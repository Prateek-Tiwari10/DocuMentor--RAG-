import time
from functools import lru_cache

from documentor.config import collection_name, load_config
from documentor.ingest.chunker import make_embed_text
from documentor.retrieval.bm25 import BM25Index, index_path, warn_if_stale
from documentor.retrieval.dense import (dense_candidates, get_embedder, get_parents, get_store, group_parents,
                                        number_results, recursive_results)
from documentor.retrieval.fusion import rrf
from documentor.retrieval.rerank import get_reranker

STRATEGIES = ("recursive", "parent_child")
MODES = ("dense", "bm25", "hybrid")


@lru_cache(maxsize=8)
def get_index(strategy: str, model: str, language: str) -> BM25Index:
    cfg = load_config()
    name = collection_name(strategy, model)
    path = index_path(cfg, name, language)
    if not path.exists():
        raise FileNotFoundError(f"No BM25 index at {path}. Run scripts/build_bm25.py first.")
    index = BM25Index.load(path, cfg["bm25"]["k1"], cfg["bm25"]["b"])
    warn_if_stale(index, get_store(strategy, model).count(language))
    return index


def ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def bm25_candidates(ranked: list[tuple[str, float]], known: dict[str, dict], store) -> dict[str, dict]:
    """Chunks for BM25's ids, adding what dense search didn't already fetch, with their BM25 rank and score."""
    fetched = {c["id"]: c for c in store.by_ids([i for i, _ in ranked if i not in known])}
    chunks = known | fetched
    for rank, (chunk_id, score) in enumerate(ranked, 1):
        chunks[chunk_id] |= {"bm25_rank": rank, "bm25_score": score}
    return chunks


def retrieve(query: str, strategy: str, k: int, language: str, mode: str, rerank: bool, reranker=None,
             timings: dict | None = None, *, model: str | None = None, embedder=None, store=None, parents=None,
             index=None, child_top_k: int | None = None, depth: int | None = None) -> list[dict]:
    """Candidates -> fuse -> cut -> rerank -> top k (parents for parent_child). `depth` is for deep-rank reports only."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    cfg = load_config()
    r = cfg["retrieval"]
    model = model or cfg["embedding"]["model"]
    store = store or get_store(strategy, model)
    spent = {"embed_ms": 0, "dense_ms": 0, "bm25_ms": 0, "rerank_ms": 0}
    start = time.perf_counter()

    if mode == "dense" and not rerank:
        want = k if strategy == "recursive" else child_top_k or cfg["chunking"]["parent_child"]["child_top_k"]
        cut = want
    else:
        cut = depth or r["rerank_top_n"]
        want = max(r["candidates"], cut)

    dense_hits, ranked = [], []
    if mode in ("dense", "hybrid"):
        t = time.perf_counter()
        vector = (embedder or get_embedder(model)).embed_query(query)
        spent["embed_ms"] = ms(t)
        t = time.perf_counter()
        dense_hits = dense_candidates(vector, want, language, store)
        spent["dense_ms"] = ms(t)
    chunks = {h["id"]: h for h in dense_hits}
    if mode in ("bm25", "hybrid"):
        t = time.perf_counter()
        ranked = (index or get_index(strategy, model, language)).top_n(query, want)
        chunks = bm25_candidates(ranked, chunks, store)
        spent["bm25_ms"] = ms(t)

    if mode == "dense":
        order = [h["id"] for h in dense_hits]
    elif mode == "bm25":
        order = [i for i, _ in ranked]
        for chunk_id, score in ranked:
            chunks[chunk_id]["score"] = score
    else:
        fused = rrf([[h["id"] for h in dense_hits], [i for i, _ in ranked]], r["rrf_k"])
        order = [i for i, _ in fused]
        for chunk_id, score in fused:
            chunks[chunk_id] |= {"rrf_score": score, "score": score}
    hits = [chunks[i] for i in order][:cut]

    if rerank and hits:
        t = time.perf_counter()
        reranker = reranker or get_reranker(cfg, "full")
        prepend = cfg["chunking"]["prepend_headings"]
        texts = [make_embed_text(h.get("title") or "", h.get("heading_path") or "", h["text"], prepend) for h in hits]
        for hit, score in zip(hits, reranker.score(query, texts)):
            hit |= {"rerank_score": score, "score": score}
        hits = sorted(hits, key=lambda h: -h["rerank_score"])  # stable, so ties keep the fused order
        spent["rerank_ms"] = ms(t)

    if strategy == "recursive":
        results = recursive_results(hits[:k])
    else:
        results = group_parents(hits, k, parents or get_parents(strategy, model))
    if timings is not None:
        timings.update(spent | {"total_ms": ms(start)})
    return number_results(results)


def trace_line(r: dict) -> str:
    """One result's path through the stages, for example: dense #31 · bm25 #1 · rrf 0.0275 · rerank 0.912"""
    rank = lambda v: f"#{v}" if v else "-"
    parts = [f"dense {rank(r.get('dense_rank'))}", f"bm25 {rank(r.get('bm25_rank'))}"]
    if r.get("rrf_score") is not None:
        parts.append(f"rrf {r['rrf_score']:.4f}")
    if r.get("rerank_score") is not None:
        parts.append(f"rerank {r['rerank_score']:.3f}")
    return " · ".join(parts)


def timing_line(t: dict) -> str:
    return " · ".join(f"{name} {t[f'{name}_ms']:,} ms" for name in ("embed", "dense", "bm25", "rerank"))
