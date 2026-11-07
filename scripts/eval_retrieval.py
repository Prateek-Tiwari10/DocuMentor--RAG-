import argparse
import json
from datetime import datetime

from documentor.config import collection_name, load_config, resolve
from documentor.evaluation import expected_entries, first_hit_rank, load_jsonl
from documentor.evaluation.metrics import summarize_ranks
from documentor.retrieval.dense import search

STRATEGIES = ["recursive", "parent_child"]


def deep_rank(q: dict, strategy: str, n: int, model: str) -> int | None:
    # For parent_child, fetch enough children to fill n de-duplicated parents.
    child_top_k = n * 10 if strategy == "parent_child" else None
    results = search(q["question"], strategy, k=n, language=q.get("language"), model=model,
                     child_top_k=child_top_k)
    return first_hit_rank(results, q)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Recall@k / MRR on the mini golden set.")
    ap.add_argument("--strategy", choices=STRATEGIES + ["both"], default="both")
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    ap.add_argument("--golden", default="eval/mini_golden.jsonl")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--deep", type=int, default=0,
                    help="for misses, show the rank of the first hit within the top N")
    args = ap.parse_args()
    model = cfg["embedding"]["fast_model"] if args.fast else cfg["embedding"]["model"]
    strategies = STRATEGIES if args.strategy == "both" else [args.strategy]

    golden = load_jsonl(resolve(args.golden))
    unverified = [q["id"] for q in golden if not q.get("verified")]
    if unverified:
        print(f"Note: {len(unverified)} of {len(golden)} questions have verified: false.\n")

    per_question = {q["id"]: {"question": q["question"], "expected": [e["slug"] for e in expected_entries(q)]}
                    for q in golden}
    summary = {}
    for strategy in strategies:
        ranks, tokens = [], []
        for q in golden:
            results = search(q["question"], strategy, k=args.k, language=q.get("language"), model=model)
            rank = first_hit_rank(results, q)
            ranks.append(rank)
            tokens.append(sum(r["tokens"] or 0 for r in results))
            per_question[q["id"]][strategy] = {
                "rank": rank,
                "top": [f"{r['page_slug']} | {r['heading_path']}" for r in results],
            }
            if args.deep and not rank:
                per_question[q["id"]][strategy]["deep_rank"] = deep_rank(q, strategy, args.deep, model)
        summary[strategy] = summarize_ranks(ranks, tokens)

    print(f"\n{'strategy':<14} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>6}  avg_tokens_returned")
    for strategy, s in summary.items():
        print(f"{strategy:<14} {s['R@1']:>5.2f} {s['R@3']:>5.2f} {s['R@5']:>5.2f} {s['MRR']:>6.3f}  "
              f"{s['avg_tokens_returned']:>8,}")

    print(f"\n{'id':<5} " + " ".join(f"{s:>13}" for s in strategies) + "  question")
    for qid, row in per_question.items():
        cells = " ".join(f"{row[s]['rank'] or '-':>13}" for s in strategies)
        print(f"{qid:<5} {cells}  {row['question'][:70]}")

    if args.deep:
        print(f"\nDeep rank of the first hit for misses (top {args.deep}):")
        for qid, row in per_question.items():
            for s in strategies:
                if "deep_rank" in row[s]:
                    print(f"  {qid:<5} {s:<14} {row[s]['deep_rank'] or f'>{args.deep}'}")

    manifest = json.loads(resolve("data/raw/manifest.json").read_text())
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "corpus_commit": manifest["commit"],
        "model": model,
        "k": args.k,
        "deep": args.deep,
        "collections": {s: collection_name(s, model) for s in strategies},
        "config": {"chunking": cfg["chunking"], "retrieval": cfg["retrieval"]},
        "summary": summary,
        "per_question": per_question,
    }
    path = resolve(f"eval/results/phase1_{datetime.now():%Y%m%d_%H%M}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path.relative_to(resolve('.'))}")


if __name__ == "__main__":
    main()
