import argparse
import json
import os
import re
import subprocess
from datetime import datetime

from eval_answers import percentile

from documentor.config import ROOT, collection_name, load_config, resolve
from documentor.evaluation import expected_entries, first_hit_rank, load_jsonl
from documentor.evaluation.metrics import recall_in_context, retrieval_metrics, summarize_ranks
from documentor.evaluation.report import set_config_value
from documentor.retrieval.dense import search
from documentor.retrieval.rerank import get_reranker

STRATEGIES = ["recursive", "parent_child"]
SETUPS = [("dense", None), ("bm25", None), ("hybrid", None), ("hybrid", "fast"), ("hybrid", "full")]
# Cheapest to most expensive; choose_config relies on this order.
CONFIGS = [(strategy, mode, which) for strategy in STRATEGIES for mode, which in SETUPS]
IDENTIFIER = re.compile(r"`[^`]+`|\w\(\)|\b\w+_\w+\b|[a-z][A-Z]")
STAGES = ("embed", "dense", "bm25", "rerank", "total")


def deep_rank(q: dict, strategy: str, n: int, model: str, mode: str | None = None, rerank: bool | None = None,
              reranker=None) -> int | None:
    # For parent_child, fetch enough children to fill n de-duplicated parents.
    wide = n * 10 if strategy == "parent_child" else None
    results = search(q["question"], strategy, k=n, language=q.get("language"), model=model, mode=mode, rerank=rerank,
                     reranker=reranker, child_top_k=wide, depth=wide or n)
    return first_hit_rank(results, q)


def is_identifier(q: dict) -> bool:
    """A golden line can set "identifier" itself when the regex misses a name, such as a bare `Command`."""
    return bool(q["identifier"]) if "identifier" in q else bool(IDENTIFIER.search(q["question"]))


def config_label(strategy: str, mode: str, which: str | None) -> str:
    return f"{strategy} · {mode}" + (f" · {which}" if which else "")


def run_config(num: int, strategy: str, mode: str, which: str | None, golden: list[dict], k: int, cfg: dict) -> dict:
    gen = cfg["generation"]
    reranker = get_reranker(cfg, which) if which else None
    results, timings = {}, []
    for q in golden:
        t: dict = {}
        results[q["id"]] = search(q["question"], strategy, k=k, language=q.get("language"), mode=mode,
                                  rerank=bool(which), reranker=reranker, timings=t)
        timings.append(t)
    metrics = retrieval_metrics(results, golden)
    ctx = recall_in_context(results, golden, gen["context_max_tokens"], gen["budget_encoding"])
    warm = timings[1:]  # the first query of a config pays for loading
    stages = {s: {"p50": percentile([t[f"{s}_ms"] for t in warm], 0.5), "p95": percentile([t[f"{s}_ms"] for t in warm], 0.95)}
              for s in STAGES}
    return {"num": num, "label": config_label(strategy, mode, which), "strategy": strategy, "mode": mode,
            "reranker": which, "reranker_model": reranker.model_name if reranker else None,
            "R@1": metrics["R@1"], "R@3": metrics["R@3"], "R@5": metrics["R@5"], "R@ctx": ctx["R@ctx"],
            "ctx_hits": sum(ctx["hits"].values()), "MRR": metrics["MRR"], "tokens_returned": metrics["avg_tokens_returned"],
            "tokens_in_context": ctx["avg_tokens_in_context"], "p50_ms": stages["total"]["p50"], "p95_ms": stages["total"]["p95"],
            "stages": stages, "ranks": metrics["ranks"], "ctx": ctx["hits"], "results": results}


def choose_config(rows: list[dict], cfg: dict) -> tuple[dict, list[str]]:
    """Keep the simpler setup unless a more expensive one is clearly better. Each row is compared with the current best."""
    gain_needed = cfg["decision"]["min_question_gain"]
    mrr_needed = cfg["decision"]["min_mrr_gain"]
    cap = cfg["retrieval"]["rerank_max_p95_ms"]
    best = rows[0]
    reasons = [f"{best['label']}: starting point."]
    for row in rows[1:]:
        gain = row["ctx_hits"] - best["ctx_hits"]
        mrr = round(row["MRR"] - best["MRR"], 9)
        numbers = f"R@ctx {gain:+d} questions, MRR {mrr:+.3f} against {best['label']}"
        if row["reranker"] and row["p95_ms"] > cap:
            reasons.append(f"{row['label']}: rejected (CPU p95 {row['p95_ms']:,.0f} ms is over the {cap:,} ms cap; {numbers})")
        elif gain <= -gain_needed or mrr <= -mrr_needed:
            reasons.append(f"{row['label']}: rejected (lower; {numbers})")
        elif gain < gain_needed and mrr < mrr_needed:
            reasons.append(f"{row['label']}: rejected (no clear gain; {numbers})")
        else:
            reasons.append(f"{row['label']}: accepted ({numbers})")
            best = row
    return best, reasons


def decision_sentence(best: dict, base: dict, reasons: list[str]) -> str:
    rejected = [r.replace(": rejected (", " (", 1) for r in reasons if ": rejected (" in r]
    gain = best["ctx_hits"] - base["ctx_hits"]
    return (f"{best['label']}: R@ctx {base['R@ctx']:.2f} → {best['R@ctx']:.2f} ({gain:+d} questions), "
            f"MRR {base['MRR']:.3f} → {best['MRR']:.3f}, p95 {best['p95_ms']:,.0f} ms. Rejected: {'; '.join(rejected) or 'none'}.")


def tag_exists(tag: str) -> bool:
    try:
        return subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], cwd=ROOT,
                              capture_output=True).returncode == 0
    except OSError:  # no git here, for example on a remote runner
        return False


def write_decision(best: dict, cfg: dict) -> list[str]:
    """Write only the values that change."""
    r = cfg["retrieval"]
    wanted = {"strategy": best["strategy"], "mode": best["mode"], "rerank": bool(best["reranker"])}
    if best["reranker"]:
        wanted["reranker_model"] = best["reranker_model"]
    changed = []
    for key, value in wanted.items():
        if r.get(key) != value:
            set_config_value(resolve("config/settings.yaml"), "retrieval", key, str(value).lower() if isinstance(value, bool) else value)
            changed.append(f"retrieval.{key}: {r.get(key)} -> {value}")
    return changed


def apply_decision(best: dict, cfg: dict) -> str:
    if not tag_exists("phase-3-done"):
        return ("Config not written, because phase-3-done doesn't exist yet. After the tag, run: "
                "python scripts/eval_retrieval.py --apply-decision <this run's JSON>")
    changed = write_decision(best, cfg)
    return "Config written: " + "; ".join(changed) if changed else "Config already matches; nothing written."


def slice_rows(rows: list[dict], golden: list[dict]) -> list[dict]:
    flagged = {q["id"] for q in golden if is_identifier(q)}
    def recall(row, ids, measure):
        if not ids:
            return None
        return sum(bool(measure(row, q)) for q in ids) / len(ids)
    groups = {"identifier": sorted(flagged), "other": sorted({q["id"] for q in golden} - flagged)}
    out = []
    for row in rows:
        cells = {"num": row["num"], "label": row["label"]}
        for name, ids in groups.items():
            cells[f"{name}_R@5"] = recall(row, ids, lambda r, q: r["ranks"][q] and r["ranks"][q] <= 5)
            cells[f"{name}_R@ctx"] = recall(row, ids, lambda r, q: r["ctx"][q])
        out.append(cells)
    return out


def stage_ranks(q: dict, strategy: str, n: int) -> dict:
    """Where each retriever on its own puts the expected section, so a rescue can be credited."""
    wide = n * 10 if strategy == "parent_child" else None
    common = {"strategy": strategy, "k": n, "language": q.get("language"), "rerank": False}
    return {"dense_rank": first_hit_rank(search(q["question"], mode="dense", child_top_k=wide, **common), q),
            "bm25_rank": first_hit_rank(search(q["question"], mode="bm25", depth=wide or n, **common), q)}


def score_calibration(best: dict, golden: list[dict], oos: list[dict], k: int, cfg: dict) -> dict:
    reranker = get_reranker(cfg, best["reranker"]) if best["reranker"] else None
    top = lambda results: results[0]["score"] if results else None
    rows = [{"id": q["id"], "group": "golden", "score": top(best["results"][q["id"]])} for q in golden]
    for q in oos:
        results = search(q["question"], best["strategy"], k=k, language=q.get("language"), mode=best["mode"],
                         rerank=bool(best["reranker"]), reranker=reranker)
        rows.append({"id": q["id"], "group": "oos", "score": top(results)})
    rows.sort(key=lambda r: -(r["score"] or 0))
    low = min(r["score"] for r in rows if r["group"] == "golden")
    high = max(r["score"] for r in rows if r["group"] == "oos")
    note = (f"Golden and OOS top-1 scores separate (lowest golden {low:.3f}, highest OOS {high:.3f}). "
            f"Suggested generation.min_score: {(low + high) / 2:.3f}." if low > high else
            f"Golden and OOS top-1 scores overlap (lowest golden {low:.3f}, highest OOS {high:.3f}), so no min_score is suggested.")
    return {"rows": rows, "note": note + f" With only {len(oos)} OOS questions this is weak evidence. min_score is never written."}


def rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def render_report(run: dict) -> str:
    rows, best = run["rows"], run["chosen"]
    lines = ["# Phase 4 retrieval comparison", "",
             f"Corpus {run['corpus_commit'][:7]}, embeddings `{run['embedding_model']}`, device {run['device']} "
             f"({run['cpu_count']} CPU cores), {run['questions']} golden questions, k={run['k']}.", "", "## Summary", "",
             "| # | Config | R@1 | R@3 | R@5 | R@ctx | MRR | tokens returned | tokens in context | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['num']} | {r['label']} | {r['R@1']:.2f} | {r['R@3']:.2f} | {r['R@5']:.2f} | {r['R@ctx']:.2f} | "
                     f"{r['MRR']:.3f} | {r['tokens_returned']:,} | {r['tokens_in_context']:,} | {r['p50_ms']:,.0f} | {r['p95_ms']:,.0f} |")
    lines += ["", "## Decision", "", run["decision"], "", run["config_note"], ""]
    lines += [f"- {reason}" for reason in run["reasons"]]
    flagged = run["identifier_questions"]
    lines += ["", "## Identifier vs other", "", f"Identifier questions ({len(flagged)}): {', '.join(flagged)}.", "",
              "| # | Config | identifier R@5 | identifier R@ctx | other R@5 | other R@ctx |", "|---|---|---|---|---|---|"]
    for c in run["slices"]:
        lines.append(f"| {c['num']} | {c['label']} | {rate(c['identifier_R@5'])} | {rate(c['identifier_R@ctx'])} | "
                     f"{rate(c['other_R@5'])} | {rate(c['other_R@ctx'])} |")
    nums = [r["num"] for r in rows]
    lines += ["", "## Per question (hit rank, - for a miss)", "", "| ID | " + " | ".join(map(str, nums)) + " |",
              "|---|" + "---|" * len(nums)]
    for qid, q in run["per_question"].items():
        lines.append(f"| {qid} | " + " | ".join(str(q["ranks"][str(n)] or "-") for n in nums) + " |")
    lines += ["", f"## Changed questions: config 1 to {best['label']} (R@ctx)", ""]
    changed = [c for c in run["changed"]]
    lines += [f"- {c['id']} {c['direction']}: dense rank {c['dense_rank'] or '-'}, bm25 rank {c['bm25_rank'] or '-'}" for c in changed] or ["- None."]
    deep = run["deep"]
    lines += ["", f"## Deep ranks for {best['label']}'s misses (top {deep})", ""]
    lines += [f"- {qid}: {rank or f'>{deep}'}" for qid, rank in run["deep_ranks"].items()] or ["- No misses, or --deep not set."]
    lines += ["", "## CPU latency per stage (p50 / p95 ms)", "", "| # | Config | embed | dense | bm25 | rerank | total |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = " | ".join(f"{r['stages'][s]['p50']:,.0f} / {r['stages'][s]['p95']:,.0f}" for s in STAGES)
        lines.append(f"| {r['num']} | {r['label']} | {cells} |")
    lines += ["", "## Score calibration (top-1 score of the chosen config)", "", run["calibration"]["note"], "", "| ID | Group | Score |", "|---|---|---|"]
    lines += [f"| {r['id']} | {r['group']} | {r['score']:.3f} |" for r in run["calibration"]["rows"]]
    return "\n".join(lines) + "\n"


def run_matrix(args, cfg: dict, golden: list[dict]) -> None:
    rows = []
    print(f"Identifier questions: {', '.join(q['id'] for q in golden if is_identifier(q))}\n")
    for num, (strategy, mode, which) in enumerate(CONFIGS, 1):
        print(f"[{num}/{len(CONFIGS)}] {config_label(strategy, mode, which)}", flush=True)
        row = run_config(num, strategy, mode, which, golden, args.k, cfg)
        rows.append(row)
        print(f"  R@1 {row['R@1']:.2f} R@3 {row['R@3']:.2f} R@5 {row['R@5']:.2f} R@ctx {row['R@ctx']:.2f} "
              f"MRR {row['MRR']:.3f} tokens {row['tokens_returned']:,}/{row['tokens_in_context']:,} "
              f"p50 {row['p50_ms']:,.0f} p95 {row['p95_ms']:,.0f} ms", flush=True)

    best, reasons = choose_config(rows, cfg)
    base = rows[0]
    print("\nRetriever ranks per question...", flush=True)
    stages = {s: {q["id"]: stage_ranks(q, s, args.deep or cfg["retrieval"]["candidates"]) for q in golden} for s in STRATEGIES}
    changed = []
    for q in golden:
        before, after = base["ctx"][q["id"]], best["ctx"][q["id"]]
        if before != after:
            changed.append({"id": q["id"], "direction": "miss to hit" if after else "hit to miss", **stages[best["strategy"]][q["id"]]})
    deep_ranks = {}
    if args.deep:
        reranker = get_reranker(cfg, best["reranker"]) if best["reranker"] else None
        for q in golden:
            if not best["ctx"][q["id"]]:
                deep_ranks[q["id"]] = deep_rank(q, best["strategy"], args.deep, cfg["embedding"]["model"], best["mode"],
                                                bool(best["reranker"]), reranker)
    oos = load_jsonl(resolve(cfg["evaluation"]["oos_path"]))
    calibration = score_calibration(best, golden, oos, args.k, cfg)
    per_question = {q["id"]: {"question": q["question"], "identifier": is_identifier(q),
                              "ranks": {str(r["num"]): r["ranks"][q["id"]] for r in rows},
                              "in_context": {str(r["num"]): r["ctx"][q["id"]] for r in rows},
                              "retriever_ranks": {s: stages[s][q["id"]] for s in STRATEGIES},
                              "chosen_top": [f"{x['page_slug']} | {x['heading_path']}" for x in best["results"][q["id"]]]}
                    for q in golden}
    kept = [{k: v for k, v in r.items() if k != "results"} for r in rows]
    chosen = next(r for r in kept if r["num"] == best["num"])
    manifest = json.loads(resolve("data/raw/manifest.json").read_text())
    run = {"timestamp": datetime.now().isoformat(timespec="seconds"), "corpus_commit": manifest["commit"],
           "embedding_model": cfg["embedding"]["model"], "device": cfg["embedding"]["device"], "cpu_count": os.cpu_count(),
           "k": args.k, "deep": args.deep,
           "questions": len(golden), "config": {"retrieval": cfg["retrieval"], "bm25": cfg["bm25"], "decision": cfg["decision"]},
           "rows": kept, "chosen": chosen, "reasons": reasons, "decision": decision_sentence(best, base, reasons),
           "config_note": apply_decision(chosen, cfg), "identifier_questions": [q["id"] for q in golden if is_identifier(q)],
           "slices": slice_rows(rows, golden), "per_question": per_question, "changed": changed, "deep_ranks": deep_ranks,
           "calibration": calibration}
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    json_path, md_path = resolve(f"eval/results/phase4_{stamp}.json"), resolve(f"eval/results/phase4_{stamp}.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(run, indent=2))
    report = render_report(run)
    md_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved {json_path.relative_to(resolve('.'))} and {md_path.name}")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Recall@k / MRR on the mini golden set.")
    ap.add_argument("--strategy", choices=STRATEGIES + ["both"], default="both")
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    ap.add_argument("--golden", default="eval/mini_golden.jsonl")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--deep", type=int, default=0,
                    help="for misses, show the rank of the first hit within the top N")
    ap.add_argument("--matrix", action="store_true", help="compare the ten retrieval configs and pick the default")
    ap.add_argument("--device", help="force the embedder and rerankers onto this device, for example cpu")
    ap.add_argument("--apply-decision", metavar="PHASE4_JSON", help="write a saved run's decision to config")
    args = ap.parse_args()
    if args.apply_decision:
        print(apply_decision(json.loads(resolve(args.apply_decision).read_text())["chosen"], cfg))
        return
    if args.device:
        cfg["embedding"]["device"] = args.device
    model = cfg["embedding"]["fast_model"] if args.fast else cfg["embedding"]["model"]
    strategies = STRATEGIES if args.strategy == "both" else [args.strategy]

    golden = load_jsonl(resolve(args.golden))
    unverified = [q["id"] for q in golden if not q.get("verified")]
    if unverified:
        print(f"Note: {len(unverified)} of {len(golden)} questions have verified: false.\n")
    if args.matrix:
        run_matrix(args, cfg, golden)
        return

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
