import argparse
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace

from eval_answers import percentile, run_closed_book, summarize_review

from documentor.config import load_config, resolve
from documentor.evaluation import load_jsonl
from documentor.evaluation.agreement import to_grade
from documentor.evaluation.golden import load_golden
from documentor.evaluation.metrics import citation_coverage, citation_validity, refusal_correctness, retrieval_metrics
from documentor.evaluation.ragas_eval import (CLOSED_METRICS, RAG_METRICS, BudgetExhausted, RagasScorer, budget_message,
                                              budget_sleep, closed_samples, failed_metrics, is_daily_limit, mean_score,
                                              parse_metrics, rag_samples, reset_hint, score_sample)
from documentor.evaluation.report import (choose_strategy, compare_runs, render_compare, render_report,
                                          set_config_value)
from documentor.generation.llm import GroqLLM
from documentor.pipeline import answer
from documentor.retrieval.dense import search

STRATEGIES = ["recursive", "parent_child"]


def recording_llm(llm, latencies: list[int]):
    """Keeps the model's own latency, which the Phase 2 cache preserves, so reruns report the same number."""
    def complete(messages, **overrides):
        out = llm.complete(messages, **overrides)
        latencies.append(out.latency_ms)
        return out
    return SimpleNamespace(complete=complete)


def run_answers(items: list[dict], strategy: str, k: int, llm) -> tuple[dict, dict]:
    answers, latencies = {}, {}
    for i, q in enumerate(items, 1):
        seen: list[int] = []
        a = asdict(answer(q["question"], language=q.get("language"), strategy=strategy, k=k,
                          llm=recording_llm(llm, seen)))
        answers[q["id"]] = a | {"question_id": q["id"]}
        latencies[q["id"]] = seen[-1] if seen else None
        status = "refused" if a["refused"] else f"cited {[s['n'] for s in a['sources']]}"
        print(f"  [{i}/{len(items)}] {q['id']:<6} {status}{'  (cache)' if a['cache_hit'] else ''}")
    return answers, latencies


def load_plan(root) -> dict:
    path = root / "plan.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_plan(root, plan: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps(plan, indent=2))


def score_all(scorer, samples, metrics_for_sample, root, budget: dict, label: str) -> dict[str, dict]:
    rows = {}
    for i, sample in enumerate(samples, 1):
        row = score_sample(scorer, sample, metrics_for_sample, root)
        rows[sample.qid] = row
        fresh = [e for e in row.values() if not e["cached"]]
        budget["done"] += 1
        values = " ".join(f"{m[:6]}={e['value']:.2f}" if e["value"] is not None else f"{m[:6]}=failed" for m, e in row.items())
        print(f"  {label} [{i}/{len(samples)}] {sample.qid:<5} {values}{'' if fresh else '  (cache)'}")
        if fresh:
            budget["fresh_samples"] += 1
            budget["fresh_tokens"] += sum(e["tokens"] for e in fresh)
            budget["after_sample"]()
    return rows


def project(budget: dict, ev: dict, plan: dict, root, both: bool) -> None:
    if budget["projected"] or budget["fresh_samples"] < 3:
        return
    budget["projected"] = True
    per_sample = budget["fresh_tokens"] / budget["fresh_samples"]
    days = per_sample * max(budget["planned"] - budget["done"], 0) / ev["judge_tokens_per_day"]
    print(f"\nMeasured {per_sample:,.0f} judge tokens per sample; projected {days:.1f} more days "
          f"at {ev['judge_tokens_per_day']:,} tokens per day.\n")
    if days > ev["daily_budget_days_max"] and both and "precision_strategies" not in plan:
        plan["precision_strategies"] = ["recursive"]
        plan["reason"] = (f"Projected {days:.1f} days exceeded daily_budget_days_max "
                          f"({ev['daily_budget_days_max']}), so context precision ran for recursive only.")
        save_plan(root, plan)
        print(plan["reason"] + "\n")


def grade_rows(rows: dict, cutoffs: dict) -> dict[str, str]:
    if cutoffs.get("Y") is None:
        return {}
    grades = {}
    for qid, row in rows.items():
        value = row.get("answer_correctness", {}).get("value")
        if value is not None:
            grades[qid] = to_grade(value, cutoffs)
    return grades


def grade_counts(grades: dict[str, str]) -> dict[str, int]:
    return {g: sum(v == g for v in grades.values()) for g in ("Y", "P", "N")}


def strategy_summary(golden: list[dict], retrieval: dict, answers: dict, latencies: dict, rows: dict, grades: dict,
                     oos_ids: list[str], use_ragas: bool) -> dict:
    golden_ids = [q["id"] for q in golden]
    golden_answers = {q: answers[q] for q in golden_ids}
    validity = citation_validity({q: answers[q] for q in golden_ids + oos_ids})
    coverage = citation_coverage(golden_answers, golden)
    refusal = refusal_correctness(answers, golden_ids, oos_ids)
    usage = [a["usage"].get("prompt_tokens", 0) + a["usage"].get("completion_tokens", 0)
             for a in golden_answers.values() if a["usage"]]
    row_list = list(rows.values())
    out = {"questions": len(golden), "citation_validity": validity["validity"], "invalid_total": validity["invalid_total"],
           "citation_coverage": coverage["coverage"], "cited_expected": coverage["cited_expected"],
           "unit_coverage": coverage["unit_coverage"],
           "tokens_per_answer": round(statistics.mean(usage)) if usage else None,
           "p50_generate_ms": percentile([v for q, v in latencies.items() if q in golden_answers and v is not None], 0.5)}
    out |= {k: refusal[k] for k in ("oos_refusal_rate", "oos_refused", "oos_total", "false_refusal_rate", "false_refusals")}
    out |= {k: retrieval[k] for k in ("R@1", "R@3", "R@5", "MRR", "avg_tokens_returned")}
    if use_ragas:
        out |= {m: mean_score(row_list, m) for m in RAG_METRICS}
        refused = {q: "N" for q in refusal["false_refusals"]}
        out |= grade_counts(grades | refused) if grades else {}
    return out


def closed_summary(golden: list[dict], answers: dict, rows: dict, grades: dict) -> dict:
    usage = [a["usage"]["prompt_tokens"] + a["usage"]["completion_tokens"] for a in answers.values()]
    row_list = list(rows.values())
    out = {"questions": len(golden), "tokens_per_answer": round(statistics.mean(usage)),
           "p50_generate_ms": percentile([a["latency_ms"] for a in answers.values()], 0.5)}
    out |= {m: mean_score(row_list, m) for m in CLOSED_METRICS}
    return out | (grade_counts(grades) if grades else {})


def group_row(ids: list[str], rows: dict, grades: dict, refused: list[str]) -> dict:
    subset = {q: grades[q] for q in ids if q in grades} | {q: "N" for q in refused if q in ids}
    return {"answer_correctness": mean_score([rows[q] for q in ids if q in rows], "answer_correctness")} | grade_counts(subset)


def human_comparison(sheet) -> dict:
    """RAG vs closed-book from the human-graded Phase 2 sheet, which doesn't depend on the judge."""
    result = summarize_review(sheet)
    counts = result["counts"]
    return {"sheet": sheet.name, "closed_book_outdated_api": result["closed_book_outdated_api"],
            "groups": {label: {"RAG": counts[key]["rag"], "Closed-book": counts[key]["closed"]}
                       for label, key in {"All questions": "overall", "post_cutoff: true": "true"}.items()}}


def build_run(cfg, golden, configs: dict, skipped: list[str], decision: str, version: str | None) -> dict:
    ev = cfg["evaluation"]
    summary = {name: c["summary"] for name, c in configs.items()}
    post_cutoff = [q["id"] for q in golden if q.get("post_cutoff") is True]
    groups = {"All questions": [q["id"] for q in golden], "post_cutoff: true": post_cutoff}
    rag_vs_closed = {}
    if "closed-book" in configs:
        rag_vs_closed = {g: {name: group_row(ids, c["rows"], c["grades"], c["summary"].get("false_refusals", []))
                             for name, c in configs.items()} for g, ids in groups.items()}
    per_question = []
    for q in golden:
        row = {"id": q["id"], "post_cutoff": q.get("post_cutoff"), "configs": {}}
        for name, c in configs.items():
            scores = c["rows"].get(q["id"], {})
            refused = q["id"] in c["summary"].get("false_refusals", [])
            row["configs"][name] = {"grade": "N" if refused and c["grades"] else c["grades"].get(q["id"]),
                                    "refused": refused} | {m: e["value"] for m, e in scores.items()}
        per_question.append(row)
    failed = [f | {"config": name} for name, c in configs.items() for f in failed_metrics(c["rows"])]
    entries = [e for c in configs.values() for row in c["rows"].values() for e in row.values()]
    usage = [a["usage"] for c in configs.values() for a in c["answers"].values() if a["usage"]]
    manifest = json.loads(resolve("data/raw/manifest.json").read_text())
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "corpus_commit": manifest["commit"],
            "judge_model": ev["judge_model"], "calibrated": ev.get("calibrated"), "ragas_version": version,
            "config": {k: cfg[k] for k in ("llm", "generation", "retrieval", "evaluation")},
            "summary": summary, "rag_vs_closed": rag_vs_closed,
            "human_rag_vs_closed": human_comparison(resolve(ev["human_sheet"])), "per_question": per_question,
            "grades": {name: {**c["grades"], **{q: "N" for q in c["summary"].get("false_refusals", [])}} if c["grades"] else {}
                       for name, c in configs.items()},
            "answers": {name: c["answers"] for name, c in configs.items()}, "failed": failed, "skipped": skipped,
            "strategy_decision": decision, "judge_tokens": sum(e["tokens"] for e in entries),
            "answer_tokens": sum(u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) for u in usage),
            "scoring_days": len({e["created_at"][:10] for e in entries})}


def baseline_view(run: dict) -> dict:
    keys = ("timestamp", "corpus_commit", "judge_model", "calibrated", "ragas_version", "summary", "grades",
            "rag_vs_closed", "human_rag_vs_closed", "per_question", "failed", "skipped", "strategy_decision", "judge_tokens",
            "answer_tokens", "scoring_days")
    return {k: run[k] for k in keys}


def main() -> None:
    cfg = load_config()
    ev = cfg["evaluation"]
    ap = argparse.ArgumentParser(description="Score retrieval, answers and refusals on the golden set.")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--closed-book", action="store_true")
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--compare", metavar="BASELINE_JSON")
    ap.add_argument("--no-ragas", action="store_true", help="own metrics only, no judge calls")
    ap.add_argument("--metrics", default=",".join(RAG_METRICS), help="comma-separated RAGAS metric keys (default: all four)")
    args = ap.parse_args()
    strategies = args.strategies.split(",")
    use_ragas = not args.no_ragas
    try:
        selected = parse_metrics(args.metrics)
    except ValueError as e:
        ap.error(str(e))
    if use_ragas and not ev.get("judge_model"):
        sys.exit("No judge model set. Run scripts/calibrate_judge.py first, or use --no-ragas.")

    golden = load_golden(resolve(ev["golden_path"]))
    oos = load_jsonl(resolve(ev["oos_path"]))
    unverified = sum(not q.get("verified") for q in golden)
    print(f"{len(golden)} golden + {len(oos)} OOS questions; {unverified} have verified: false.")
    llm = GroqLLM(cfg, sleep=budget_sleep(ev["budget_max_wait_s"]))
    root = resolve(ev["ragas_cache_path"])
    scorer = RagasScorer(cfg, ev["judge_model"]) if use_ragas else None
    plan = load_plan(root)
    cutoffs = ev["cutoffs"] if ev.get("calibrated") is not None else {}
    both = set(STRATEGIES) <= set(strategies)
    planned = len(golden) * (len(strategies) + (1 if args.closed_book else 0))
    budget = {"done": 0, "fresh_samples": 0, "fresh_tokens": 0, "projected": False, "planned": planned}
    budget["after_sample"] = lambda: project(budget, ev, plan, root, both)

    configs, skipped = {}, []
    for strategy in strategies:
        print(f"\nRetrieval: {strategy}")
        by_q = {q["id"]: search(q["question"], strategy, k=args.k, language=q.get("language")) for q in golden}
        retrieval = retrieval_metrics(by_q, golden)
        print(f"Answers: {strategy}")
        answers, latencies = run_answers(golden + oos, strategy, args.k, llm)
        rows = {}
        if use_ragas:
            samples = rag_samples({q["id"]: answers[q["id"]] for q in golden}, golden)
            metrics = list(selected)
            allowed = plan.get("precision_strategies")
            if "context_precision" in metrics and allowed and strategy not in allowed:
                metrics.remove("context_precision")
                skipped.append(f"context precision for {strategy}: {plan['reason']}")
            print(f"Scoring: {strategy}")
            rows = score_all(scorer, samples, metrics, root, budget, strategy)
        grades = grade_rows(rows, cutoffs)
        summary = strategy_summary(golden, retrieval, answers, latencies, rows, grades, [q["id"] for q in oos], use_ragas)
        configs[strategy] = {"summary": summary, "rows": rows, "grades": grades, "answers": answers}

    if args.closed_book:
        print("\nAnswers: closed-book")
        closed = run_closed_book(golden, llm, cfg, True)
        rows = {}
        if use_ragas:
            print("Scoring: closed-book")
            closed_metrics = [m for m in CLOSED_METRICS if m in selected]
            rows = score_all(scorer, closed_samples(closed, golden), closed_metrics, root, budget, "closed-book")
        grades = grade_rows(rows, cutoffs)
        configs["closed-book"] = {"summary": closed_summary(golden, closed, rows, grades), "rows": rows,
                                  "grades": grades, "answers": closed}

    if use_ragas and len(selected) < len(RAG_METRICS):
        left_out = ", ".join(m for m in RAG_METRICS if m not in selected)
        skipped.append(f"RAGAS metrics not run (--metrics): {left_out}.")
    if not args.closed_book:
        skipped.append("Closed-book RAGAS: RAG vs closed-book comes from the Phase 2 human grades.")
    missing = [s for s in STRATEGIES if s not in strategies]
    if missing:
        skipped.append(f"{', '.join(missing)} answers and RAGAS: see the strategy decision.")

    decision = "Not decided: both strategies did not run with RAGAS."
    if ev.get("strategy_deferred"):
        decision = f"{ev['strategy_deferred']} Default stays {cfg['retrieval']['strategy']}."
    if both and use_ragas:
        current = cfg["retrieval"]["strategy"]
        choice, decision = choose_strategy(configs["recursive"]["summary"], configs["parent_child"]["summary"],
                                           ev["strategy_rule"], bool(ev.get("calibrated")))
        if choice != current:
            set_config_value(resolve("config/settings.yaml"), "retrieval", "strategy", choice)
            decision += f" Default changed from {current} to {choice}."
        else:
            decision += f" Default stays {current}."
    print(f"\n{decision}")

    run = build_run(cfg, golden, configs, skipped, decision, getattr(scorer, "version", None))
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    json_path = resolve(f"eval/results/phase3_{stamp}.json")
    md_path = resolve(f"eval/results/phase3_{stamp}.md")
    json_path.write_text(json.dumps(run, indent=2))
    md_path.write_text(render_report(run), encoding="utf-8")
    print(f"\nSaved {json_path}\nReport {md_path}")
    print(render_report(run).split("## RAG vs closed-book")[0])
    if args.save_baseline:
        path = resolve(ev["baseline_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(baseline_view(run), indent=2))
        print(f"Baseline saved to {path}")
    if args.compare:
        print(render_compare(compare_runs(json.loads(resolve(args.compare).read_text()), run)))


if __name__ == "__main__":
    try:
        main()
    except BudgetExhausted as e:
        print(budget_message(e))
        sys.exit(3)
    except Exception as e:
        if not is_daily_limit(str(e)):
            raise
        print(budget_message(BudgetExhausted(reset_hint(str(e)))))
        sys.exit(3)
