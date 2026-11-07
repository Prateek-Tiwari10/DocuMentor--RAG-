import argparse

from documentor.config import load_config
from documentor.retrieval.dense import search
from documentor.retrieval.hybrid import timing_line, trace_line
from documentor.retrieval.rerank import get_reranker


def show(results: list[dict], explain: bool) -> None:
    for r in results:
        snippet = " ".join((r["text"] or "").split())[:200]
        extra = f"  children={r['matched_children']}" if "matched_children" in r else ""
        print(f"{r['rank']}. {r['score']:.3f}  [{r['language']}]  {r['page_slug']}  |  "
              f"{r['heading_path']}  ({r['tokens']} tok){extra}")
        if explain:
            print(f"     {trace_line(r)}")
        print(f"     {snippet}")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Search the LangGraph docs.")
    ap.add_argument("query")
    ap.add_argument("--strategy", choices=["recursive", "parent_child", "both"], default="both")
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    ap.add_argument("--language", choices=cfg["corpus"]["languages"], default=cfg["corpus"]["default_language"])
    ap.add_argument("--fast", action="store_true", help="query the all-MiniLM-L6-v2 collections")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default=cfg["retrieval"]["mode"])
    ap.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=cfg["retrieval"]["rerank"])
    ap.add_argument("--reranker", choices=["full", "fast"], default="full")
    ap.add_argument("--explain", action="store_true", help="show each result's path through the stages, and timings")
    args = ap.parse_args()
    model = cfg["embedding"]["fast_model"] if args.fast else cfg["embedding"]["model"]
    reranker = get_reranker(cfg, args.reranker) if args.rerank else None

    strategies = ["recursive", "parent_child"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        print(f"\n=== {strategy}  mode={args.mode}  rerank={args.rerank}  language={args.language}  k={args.k}")
        timings: dict = {}
        show(search(args.query, strategy, k=args.k, language=args.language, mode=args.mode, rerank=args.rerank,
                    reranker=reranker, timings=timings, model=model), args.explain)
        if args.explain:
            print(timing_line(timings))


if __name__ == "__main__":
    main()
