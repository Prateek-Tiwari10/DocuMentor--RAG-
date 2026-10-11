import argparse
import json
import statistics
from datetime import datetime

from documentor.config import collection_name, load_config, resolve
from documentor.retrieval.dense import search

STRATEGIES = ["recursive", "parent_child"]


def load_golden(path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def text_headings(text: str) -> list[str]:
    """Markdown heading lines inside a chunk, skipping `#` comments in code fences."""
    headings, in_fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            headings.append(line)
    return headings


def is_hit(result: dict, q: dict) -> bool:
    if result["page_slug"] != q["expected_slug"]:
        return False
    section = q.get("expected_section", "").lower()
    if not section or section in (result["heading_path"] or "").lower():
        return True
    # A parent (or a recursive chunk spanning headings) can contain the expected subsection.
    return any(section in h.lower() for h in text_headings(result["text"] or ""))


def first_hit_rank(results: list[dict], q: dict) -> int | None:
    return next((r["rank"] for r in results if is_hit(r, q)), None)


def summarize(ranks: list[int | None], tokens: list[int]) -> dict:
    n = len(ranks)
    recall = lambda k: sum(1 for r in ranks if r and r <= k) / n
    return {"R@1": recall(1), "R@3": recall(3), "R@5": recall(5),
            "MRR": sum(1 / r for r in ranks if r) / n,
            "avg_tokens_returned": round(statistics.mean(tokens))}


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Recall@k / MRR on the mini golden set.")
    ap.add_argument("--strategy", choices=STRATEGIES + ["both"], default="both")
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    ap.add_argument("--golden", default="eval/mini_golden.jsonl")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    model = cfg["embedding"]["fast_model"] if args.fast else cfg["embedding"]["model"]
    strategies = STRATEGIES if args.strategy == "both" else [args.strategy]

    golden = load_golden(resolve(args.golden))
    unverified = [q["id"] for q in golden if not q.get("verified")]
    if unverified:
        print(f"WARNING: {len(unverified)} unverified questions: {', '.join(unverified)}\n")

    per_question = {q["id"]: {"question": q["question"], "expected": q["expected_slug"]} for q in golden}
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
        summary[strategy] = summarize(ranks, tokens)

    print(f"\n{'strategy':<14} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>6}  avg_tokens_returned")
    for strategy, s in summary.items():
        print(f"{strategy:<14} {s['R@1']:>5.2f} {s['R@3']:>5.2f} {s['R@5']:>5.2f} {s['MRR']:>6.3f}  "
              f"{s['avg_tokens_returned']:>8,}")

    print(f"\n{'id':<5} " + " ".join(f"{s:>13}" for s in strategies) + "  question")
    for qid, row in per_question.items():
        cells = " ".join(f"{row[s]['rank'] or '-':>13}" for s in strategies)
        print(f"{qid:<5} {cells}  {row['question'][:70]}")

    manifest = json.loads(resolve("data/raw/manifest.json").read_text())
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "corpus_commit": manifest["commit"],
        "model": model,
        "k": args.k,
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
