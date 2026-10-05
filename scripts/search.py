import argparse

from documentor.config import load_config
from documentor.retrieval.dense import search


def show(results: list[dict]) -> None:
    for r in results:
        snippet = " ".join((r["text"] or "").split())[:200]
        extra = f"  children={r['matched_children']}" if "matched_children" in r else ""
        print(f"{r['rank']}. {r['score']:.3f}  [{r['language']}]  {r['page_slug']}  |  "
              f"{r['heading_path']}  ({r['tokens']} tok){extra}")
        print(f"     {snippet}")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Dense search over the LangGraph docs.")
    ap.add_argument("query")
    ap.add_argument("--strategy", choices=["recursive", "parent_child", "both"], default="both")
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    ap.add_argument("--language", choices=cfg["corpus"]["languages"], default=cfg["corpus"]["default_language"])
    ap.add_argument("--fast", action="store_true", help="query the all-MiniLM-L6-v2 collections")
    args = ap.parse_args()
    model = cfg["embedding"]["fast_model"] if args.fast else cfg["embedding"]["model"]

    strategies = ["recursive", "parent_child"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        print(f"\n=== {strategy}  language={args.language}  k={args.k}")
        show(search(args.query, strategy, k=args.k, language=args.language, model=model))


if __name__ == "__main__":
    main()
