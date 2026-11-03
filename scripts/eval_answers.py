import argparse
import json
import math
import re
import statistics
from dataclasses import asdict
from datetime import datetime

from documentor.config import load_config, resolve
from documentor.evaluation import expected_entries, is_hit, load_jsonl
from documentor.generation.llm import get_llm
from documentor.generation.prompt import language_name
from documentor.pipeline import answer

CLOSED_BOOK_SYSTEM = ("You are a helpful assistant for the LangGraph library. "
                      "Answer concisely and include {language_name} code.")
RESULTS_DIR = "eval/results"
GRADE_LINES = {
    "rag": "RAG correct? (Y / P / N):",
    "closed": "Closed-book correct? (Y / P / N):",
    "outdated": "Closed-book used an outdated or invented API? (yes / no, which):",
}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def cited_blocks(a: dict) -> list[dict]:
    cited = {s["n"] for s in a["sources"]}
    return [b for b in a["blocks"] if b["n"] in cited]


def compute_metrics(golden: list[dict], answers: dict, oos: list[dict]) -> dict:
    g = [answers[q["id"]] for q in golden]
    o = [answers[q["id"]] for q in oos]
    cited_hit = [q["id"] for q, a in zip(golden, g) if any(is_hit(b, q) for b in cited_blocks(a))]
    retrieved_not_cited = [q["id"] for q, a in zip(golden, g)
                           if q["id"] not in cited_hit and any(is_hit(b, q) for b in a["blocks"])]
    invalid = {a["question_id"]: a["invalid_citations"] for a in g + o if a["invalid_citations"]}
    live = [a for a in g + o if not a["cache_hit"] and a["refusal_reason"] != "low_score"]
    with_usage = [a for a in g + o if a["usage"]]
    mean = lambda xs: round(statistics.mean(xs)) if xs else None
    no_wait = [a["timings"]["total_ms"] - a["timings"]["wait_ms"] for a in live]
    return {
        "answered_rate": sum(not a["refused"] for a in g) / len(g),
        "expected_source_cited": len(cited_hit) / len(g),
        "retrieved_not_cited": retrieved_not_cited,
        "invalid_citations": sum(len(v) for v in invalid.values()),
        "invalid_citation_questions": sorted(invalid),
        "uncited_answers": sum(a["uncited"] and not a["refused"] for a in g + o),
        "oos_refusal_accuracy": sum(a["refused"] for a in o) / len(o) if o else None,
        "false_refusals": [q["id"] for q, a in zip(golden, g) if a["refused"]],
        "latency_p50_ms": percentile([a["timings"]["total_ms"] for a in live], 0.5),
        "latency_p95_ms": percentile([a["timings"]["total_ms"] for a in live], 0.95),
        "latency_no_wait_p50_ms": percentile(no_wait, 0.5),
        "latency_no_wait_p95_ms": percentile(no_wait, 0.95),
        "wait_ms_mean": mean([a["timings"]["wait_ms"] for a in live]),
        "retrieve_ms_mean": mean([a["timings"]["retrieve_ms"] for a in live]),
        "generate_ms_mean": mean([a["timings"]["generate_ms"] for a in live]),
        "latency_excluded_cache_hits": sum(a["cache_hit"] for a in g + o),
        "prompt_tokens_mean": mean([a["usage"]["prompt_tokens"] for a in with_usage]),
        "completion_tokens_mean": mean([a["usage"]["completion_tokens"] for a in with_usage]),
    }


def calibration(golden: list[dict], oos: list[dict], answers: dict) -> dict:
    top1 = lambda qid: answers[qid]["retrieved"][0]["score"] if answers[qid]["retrieved"] else 0.0
    rows = sorted([(top1(q["id"]), "in-scope", q["id"]) for q in golden]
                  + [(top1(q["id"]), "oos", q["id"]) for q in oos], reverse=True)
    low_in = min(s for s, grp, _ in rows if grp == "in-scope")
    high_oos = max((s for s, grp, _ in rows if grp == "oos"), default=None)
    separated = high_oos is not None and low_in > high_oos
    return {"rows": rows, "min_in_scope": low_in, "max_oos": high_oos, "separated": separated,
            "suggested_min_score": round((low_in + high_oos) / 2, 3) if separated else None}


def run_rag(items: list[dict], strategy: str, llm, use_cache: bool, search_fn=None) -> dict:
    answers = {}
    for i, q in enumerate(items, 1):
        a = asdict(answer(q["question"], language=q.get("language"), strategy=strategy, llm=llm,
                          search_fn=search_fn, use_cache=use_cache))
        a["question_id"] = q["id"]
        answers[q["id"]] = a
        status = "refused" if a["refused"] else f"cited {[s['n'] for s in a['sources']]}"
        print(f"  [{i}/{len(items)}] {q['id']:<6} {status}{'  (cache)' if a['cache_hit'] else ''}")
    return answers


def run_closed_book(golden: list[dict], llm, cfg: dict, use_cache: bool) -> dict:
    language = cfg["corpus"]["default_language"]
    answers = {}
    for i, q in enumerate(golden, 1):
        lang = q.get("language") or language
        messages = [{"role": "system", "content": CLOSED_BOOK_SYSTEM.format(language_name=language_name(lang))},
                    {"role": "user", "content": q["question"]}]
        out = llm.complete(messages, use_cache=use_cache)
        answers[q["id"]] = {"question_id": q["id"], "question": q["question"], "language": lang,
                            "text": out.text, "model": out.model, "cache_hit": out.cache_hit,
                            "usage": {"prompt_tokens": out.prompt_tokens,
                                      "completion_tokens": out.completion_tokens},
                            "latency_ms": out.latency_ms}
        print(f"  [{i}/{len(golden)}] {q['id']:<6} {len(out.text)} chars{'  (cache)' if out.cache_hit else ''}")
    return answers


def latest_run(mode: str) -> dict | None:
    for path in sorted(resolve(RESULTS_DIR).glob("phase2_*.json"), reverse=True):
        data = json.loads(path.read_text())
        if data.get("mode") == mode:
            return data
    return None


def tag(q: dict) -> str:
    value = q.get("post_cutoff")
    return "untagged" if value is None else str(value).lower()


def review_sheet(golden: list[dict], rag: dict | None, closed: dict | None) -> str:
    lines = ["# Phase 2 review sheet", "",
             "Y = correct and runnable · P = partly right or missing a key step · "
             "N = wrong, or uses an API that doesn't exist", ""]
    for q in golden:
        expected = "; ".join(f"{e['slug']} § {e.get('section') or '(page)'}" for e in expected_entries(q))
        lines += [f"## [{q['id']}] {q['question']}", "", f"post_cutoff: {tag(q)}", f"Expected: {expected}", ""]
        a = (rag or {}).get("answers", {}).get(q["id"])
        lines += ["### RAG answer", "", a["text"] if a else "_not run_", ""]
        if a and a["sources"]:
            lines += ["Cited:"] + [f"- [{s['n']}] {s['heading_path']} — {s['url']}" for s in a["sources"]] + [""]
        c = (closed or {}).get("answers", {}).get(q["id"])
        lines += ["### Closed-book answer", "", c["text"] if c else "_not run_", ""]
        lines += [GRADE_LINES["rag"], GRADE_LINES["closed"], GRADE_LINES["outdated"], "", "---", ""]
    return "\n".join(lines)


def summarize_review(path) -> dict:
    """Y/P/N counts for RAG and closed-book, overall and by post_cutoff, from a graded sheet."""
    groups = ["overall", "true", "false", "untagged"]
    counts = {g: {m: {"Y": 0, "P": 0, "N": 0, "ungraded": 0} for m in ("rag", "closed")} for g in groups}
    outdated, current = 0, None
    for line in open(path):
        line = line.rstrip("\n")
        if re.match(r"^## \[\w+\]", line):
            current = {"tag": "untagged"}
        elif current is None:
            continue
        elif line.startswith("post_cutoff: "):
            current["tag"] = line.split(": ", 1)[1].strip()
        else:
            for mode in ("rag", "closed"):
                if line.startswith(GRADE_LINES[mode]):
                    value = line[len(GRADE_LINES[mode]):].strip()[:1].upper()
                    grade = value if value in ("Y", "P", "N") else "ungraded"
                    for g in ("overall", current["tag"]):
                        counts[g][mode][grade] += 1
            if line.startswith(GRADE_LINES["outdated"]):
                outdated += line[len(GRADE_LINES["outdated"]):].strip().lower().startswith("yes")
    return {"counts": counts, "closed_book_outdated_api": outdated}


def print_summary(result: dict) -> None:
    print(f"\n{'group':<10} {'RAG Y/P/N (ungraded)':<24} {'closed-book Y/P/N (ungraded)'}")
    for g, modes in result["counts"].items():
        cell = lambda c: f"{c['Y']}/{c['P']}/{c['N']} ({c['ungraded']})"
        print(f"{g:<10} {cell(modes['rag']):<24} {cell(modes['closed'])}")
    print(f"\nClosed-book answers with an outdated or invented API: {result['closed_book_outdated_api']}")


def print_metrics(m: dict) -> None:
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    rows = [
        ("Answered rate (in-scope)", pct(m["answered_rate"])),
        ("Expected source cited", pct(m["expected_source_cited"])),
        ("Retrieved but not cited", ", ".join(m["retrieved_not_cited"]) or "none"),
        ("Invalid citations", f"{m['invalid_citations']} ({', '.join(m['invalid_citation_questions']) or 'none'})"),
        ("Uncited answers", str(m["uncited_answers"])),
        ("Refusal accuracy (OOS)", pct(m["oos_refusal_accuracy"])),
        ("False refusals", ", ".join(m["false_refusals"]) or "none"),
        ("Latency p50 / p95", f"{m['latency_p50_ms']} / {m['latency_p95_ms']} ms "
                              f"({m['latency_excluded_cache_hits']} cache hits excluded)"),
        ("Same, without waiting", f"{m['latency_no_wait_p50_ms']} / {m['latency_no_wait_p95_ms']} ms"),
        ("Mean retrieve / wait / gen", f"{m['retrieve_ms_mean']} / {m['wait_ms_mean']} / {m['generate_ms_mean']} ms"),
        ("Mean tokens in / out", f"{m['prompt_tokens_mean']} / {m['completion_tokens_mean']}"),
    ]
    print()
    for name, value in rows:
        print(f"{name:<28} {value}")


def print_calibration(cal: dict) -> None:
    print("\nTop-1 retrieval score (sorted):")
    for score, group, qid in cal["rows"]:
        print(f"  {score:.3f}  {group:<9} {qid}")
    if cal["separated"]:
        print(f"Groups separate: lowest in-scope {cal['min_in_scope']:.3f} > highest OOS {cal['max_oos']:.3f}. "
              f"Suggested min_score: {cal['suggested_min_score']} (not written to config).")
    else:
        print("Groups overlap, so keep generation.min_score: null.")


def save(mode: str, cfg: dict, golden: list[dict], answers: dict, extra: dict) -> str:
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    manifest = json.loads(resolve("data/raw/manifest.json").read_text())
    out = {"mode": mode, "timestamp": datetime.now().isoformat(timespec="seconds"),
           "model": cfg["llm"]["model"], "corpus_commit": manifest["commit"],
           "config": {k: cfg[k] for k in ("llm", "generation", "retrieval")}, "answers": answers} | extra
    path = resolve(f"{RESULTS_DIR}/phase2_{stamp}{'_closed' if mode == 'closed_book' else ''}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    rag = out if mode == "rag" else latest_run("rag")
    closed = out if mode == "closed_book" else latest_run("closed_book")
    review = resolve(f"{RESULTS_DIR}/phase2_{stamp}_review.md")
    review.write_text(review_sheet(golden, rag, closed))
    print(f"\nSaved {path}\nReview sheet {review}")
    return str(review)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Phase 2 answer evaluation.")
    ap.add_argument("--strategy", choices=["recursive", "parent_child"], default=cfg["retrieval"]["strategy"])
    ap.add_argument("--closed-book", action="store_true", help="ask the model with no retrieved context")
    ap.add_argument("--no-cache", action="store_true", help="skip reading the cache (still writes)")
    ap.add_argument("--summarize", metavar="REVIEW_MD", help="count grades in a filled-in review sheet")
    ap.add_argument("--golden", default="eval/mini_golden.jsonl")
    ap.add_argument("--oos", default="eval/oos.jsonl")
    args = ap.parse_args()

    if args.summarize:
        print_summary(summarize_review(args.summarize))
        return

    golden = load_jsonl(resolve(args.golden))
    unverified = [q["id"] for q in golden if not q.get("verified")]
    if unverified:
        print(f"WARNING: {len(unverified)} unverified questions: {', '.join(unverified)}\n")
    llm = get_llm(cfg)
    use_cache = not args.no_cache

    if args.closed_book:
        print(f"Closed-book run on {len(golden)} golden questions")
        answers = run_closed_book(golden, llm, cfg, use_cache)
        save("closed_book", cfg, golden, answers, {})
        return

    oos = load_jsonl(resolve(args.oos))
    print(f"RAG run ({args.strategy}) on {len(golden)} golden + {len(oos)} OOS questions")
    answers = run_rag(golden + oos, args.strategy, llm, use_cache)
    metrics = compute_metrics(golden, answers, oos)
    cal = calibration(golden, oos, answers)
    print_calibration(cal)
    save("rag", cfg, golden, answers, {"strategy": args.strategy, "metrics": metrics,
                                       "calibration": {k: v for k, v in cal.items() if k != "rows"}})
    print_metrics(metrics)


if __name__ == "__main__":
    main()
