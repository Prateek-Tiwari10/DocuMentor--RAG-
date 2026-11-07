import argparse
import json
from datetime import datetime

import chromadb

from documentor.config import collection_name, load_config, resolve
from documentor.evaluation import load_jsonl
from documentor.evaluation.metrics import recall_in_context, retrieval_metrics
from documentor.evaluation.report import set_config_value
from documentor.retrieval.dense import get_embedder, search
from documentor.store.base import get_store
from documentor.store.chroma_store import ChromaStore

STRATEGIES = ["recursive", "parent_child"]
MIN_OVERLAP = 0.95
MAX_SCORE_DIFF = 0.001


def migrate(cfg: dict) -> None:
    """Copy every Chroma collection into Qdrant: ids, vectors, text and metadata. Nothing is re-embedded."""
    path = resolve(cfg["store"]["chroma_path"])
    names = sorted(c.name for c in chromadb.PersistentClient(path=str(path)).list_collections())
    print(f"{'collection':<34} {'language':<8} {'chroma':>7} {'qdrant':>7}")
    for name in names:
        src = ChromaStore(path, name)
        dst = get_store(cfg, name, backend="qdrant")
        dst.reset()
        for chunks, embeddings in src.batches(cfg["store"].get("batch_size", 1000)):
            dst.add(chunks, embeddings)
        for language in cfg["corpus"]["languages"]:
            a, b = src.count(language), dst.count(language)
            print(f"{name:<34} {language:<8} {a:>7,} {b:>7,}{'' if a == b else '  MISMATCH'}")
            if a != b:
                raise SystemExit(f"Counts differ for {name} ({language}); Qdrant was not fully written.")


def dense_parity(questions: list[dict], cfg: dict, embedder, k: int) -> dict:
    model = cfg["embedding"]["model"]
    vectors = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
    out = {}
    for strategy in STRATEGIES:
        chroma = get_store(cfg, collection_name(strategy, model), backend="chroma")
        qdrant = get_store(cfg, collection_name(strategy, model), backend="qdrant")
        overlaps, worst, diffs = [], 0.0, []
        for q in questions:
            language = q.get("language") or cfg["corpus"]["default_language"]
            a = chroma.query(vectors[q["id"]], k, language=language)
            b = qdrant.query(vectors[q["id"]], k, language=language)
            ids_a, ids_b = [h["id"] for h in a], [h["id"] for h in b]
            overlaps.append(len(set(ids_a) & set(ids_b)) / k)
            scores_b = {h["id"]: h["score"] for h in b}
            for h in a:
                if h["id"] in scores_b:
                    worst = max(worst, abs(h["score"] - scores_b[h["id"]]))
            if ids_a != ids_b:
                diffs.append({"id": q["id"], "chroma": ids_a, "qdrant": ids_b})
        out[strategy] = {"overlap": sum(overlaps) / len(overlaps), "max_score_diff": worst, "order_differences": diffs}
    return out


def default_metrics(golden: list[dict], cfg: dict, embedder, k: int) -> dict:
    """The default retrieval config (hybrid, recursive) on each backend, scored with the Phase 4 eval code."""
    gen = cfg["generation"]
    strategy = cfg["retrieval"]["strategy"]
    out = {}
    for backend in ("chroma", "qdrant"):
        store = get_store(cfg, collection_name(strategy, cfg["embedding"]["model"]), backend=backend)
        results = {q["id"]: search(q["question"], strategy, k=k, language=q.get("language"), embedder=embedder, store=store)
                   for q in golden}
        m = retrieval_metrics(results, golden)
        ctx = recall_in_context(results, golden, gen["context_max_tokens"], gen["budget_encoding"])
        out[backend] = {"R@5": m["R@5"], "MRR": m["MRR"], "R@ctx": ctx["R@ctx"], "ranks": m["ranks"]}
    return out


def check(cfg: dict, k: int) -> None:
    ev = cfg["evaluation"]
    golden = load_jsonl(resolve(ev["golden_path"]))
    questions = golden + load_jsonl(resolve(ev["oos_path"]))
    embedder = get_embedder(cfg["embedding"]["model"])
    dense = dense_parity(questions, cfg, embedder, k)
    metrics = default_metrics(golden, cfg, embedder, k)
    for strategy, d in dense.items():
        print(f"dense top-{k} {strategy}: overlap {d['overlap']:.3f}, largest score difference {d['max_score_diff']:.6f}, "
              f"{len(d['order_differences'])} questions with a different list or order")
        for diff in d["order_differences"]:
            print(f"  {diff['id']}: chroma {diff['chroma']}\n  {'':{len(diff['id'])}}  qdrant {diff['qdrant']}")
    for backend, m in metrics.items():
        print(f"default config on {backend}: R@5 {m['R@5']:.3f}  R@ctx {m['R@ctx']:.3f}  MRR {m['MRR']:.3f}")
    rank_diffs = {q: (metrics["chroma"]["ranks"][q], metrics["qdrant"]["ranks"][q]) for q in metrics["chroma"]["ranks"]
                  if metrics["chroma"]["ranks"][q] != metrics["qdrant"]["ranks"][q]}
    if rank_diffs:
        print(f"  hit ranks that differ (chroma, qdrant): {rank_diffs}")

    same = all(metrics["chroma"][m] == metrics["qdrant"][m] for m in ("R@ctx", "MRR"))
    overlap_ok = all(d["overlap"] >= MIN_OVERLAP for d in dense.values())
    score_ok = all(d["max_score_diff"] <= MAX_SCORE_DIFF for d in dense.values())
    passed = same and overlap_ok and score_ok
    reasons = [r for r, ok in (("R@ctx or MRR differ", same), (f"overlap under {MIN_OVERLAP}", overlap_ok),
                               (f"score difference over {MAX_SCORE_DIFF}", score_ok)) if not ok]
    print(f"\nParity {'PASSED' if passed else 'FAILED: ' + '; '.join(reasons)}")
    if passed and cfg["store"]["backend"] != "qdrant":
        set_config_value(resolve("config/settings.yaml"), "store", "backend", "qdrant")
        print("Config written: store.backend: chroma -> qdrant")
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    out = resolve(f"eval/results/phase5_parity_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k": k, "dense": dense, "default_config": metrics, "passed": passed, "reasons": reasons},
                              indent=2))
    print(f"Saved {out.relative_to(resolve('.'))}")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Copy the Chroma collections into Qdrant, or check that both agree.")
    ap.add_argument("--check", action="store_true", help="compare retrieval on both backends instead of migrating")
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    args = ap.parse_args()
    check(cfg, args.k) if args.check else migrate(cfg)


if __name__ == "__main__":
    main()
