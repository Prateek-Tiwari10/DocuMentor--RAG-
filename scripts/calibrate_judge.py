import json
import sys
from datetime import datetime

from documentor.config import load_config, resolve
from documentor.evaluation.agreement import (LABELS, agreement, cohen_kappa, confusion, parse_grades, passes_trust,
                                             pick_judge, search_cutoffs, to_grade)
from documentor.evaluation.golden import load_golden
from documentor.evaluation.ragas_eval import (BudgetExhausted, RagasScorer, Sample, budget_message, is_daily_limit,
                                              reset_hint, score_sample)
from documentor.evaluation.report import set_config_value

# Who assigned each label on the 1508 sheet. Prateek graded the closed-book answers directly;
# the answered RAG labels are Prateek's review of Claude's draft grades (kept 12, changed 6).
LABEL_SOURCE = {"closed": "Prateek", "rag": "Prateek, reviewing Claude's drafts"}


def graded_answers(cfg: dict, golden: dict[str, dict]) -> list[dict]:
    sheet = resolve(cfg["evaluation"]["human_sheet"])
    grades = parse_grades(sheet)
    base = sheet.name.removesuffix("_review.md")
    rag = json.loads((sheet.parent / f"{base}.json").read_text())["answers"]
    closed = json.loads((sheet.parent / f"{base}_closed.json").read_text())["answers"]
    out = []
    for qid, grade in grades.items():
        q = golden[qid]
        if not rag[qid]["refused"]:
            out.append({"qid": qid, "mode": "rag", "human": grade["rag"], "text": rag[qid]["text"], "q": q})
        out.append({"qid": qid, "mode": "closed", "human": grade["closed"], "text": closed[qid]["text"], "q": q})
    return out


def score_candidate(model: str, graded: list[dict], cfg: dict) -> dict:
    scorer = RagasScorer(cfg, model)
    root = resolve(cfg["evaluation"]["ragas_cache_path"])
    rows, failed = [], []
    for i, g in enumerate(graded, 1):
        sample = Sample(g["qid"], g["q"]["question"], g["text"], [], g["q"]["reference"])
        entry = score_sample(scorer, sample, ["answer_correctness"], root)["answer_correctness"]
        print(f"  [{i}/{len(graded)}] {g['qid']:<4} {g['mode']:<6} {entry['value']}{'  (cache)' if entry['cached'] else ''}")
        if entry["value"] is None:
            failed.append({"qid": g["qid"], "mode": g["mode"], "error": entry["error"]})
        else:
            rows.append(g | {"score": entry["value"], "tokens": entry["tokens"]})
    human = [r["human"] for r in rows]
    best = search_cutoffs([r["score"] for r in rows], human)
    cutoffs = {"Y": best["Y"], "P": best["P"]}
    judge = [to_grade(r["score"], cutoffs) for r in rows]
    by_source = {}
    for mode, source in LABEL_SOURCE.items():
        h = [a for a, r in zip(human, rows) if r["mode"] == mode]
        j = [a for a, r in zip(judge, rows) if r["mode"] == mode]
        by_source[mode] = {"labels_by": source, "n": len(h), "agreement": agreement(h, j), "kappa": cohen_kappa(h, j)}
    disagreements = [{"qid": r["qid"], "mode": r["mode"], "human": r["human"], "score": r["score"], "judge": g}
                     for r, g in zip(rows, judge) if r["human"] != g]
    return {"model": model, "cutoffs": cutoffs, "agreement": best["agreement"], "kappa": best["kappa"], "n": len(rows),
            "confusion": confusion(human, judge), "disagreements": disagreements, "failed": failed,
            "by_label_source": by_source, "tokens": sum(r["tokens"] for r in rows),
            "scores": [{"qid": r["qid"], "mode": r["mode"], "human": r["human"], "score": r["score"]} for r in rows]}


def score_candidates(models: list[str], graded: list[dict], cfg: dict) -> dict:
    """Groq's daily limits are per model, so one judge running out doesn't hold up the other."""
    results, stopped = {}, []
    for model in models:
        print(f"\nScoring answer correctness with {model}")
        try:
            results[model] = score_candidate(model, graded, cfg)
        except BudgetExhausted as e:
            print(f"{model}: {budget_message(e)}")
            stopped.append(e)
    if stopped:
        raise max(stopped, key=lambda e: e.at)
    return results


def print_result(r: dict, trust_agreement: float, trust_kappa: float) -> None:
    verdict = "passes" if passes_trust(r, trust_agreement, trust_kappa) else "fails"
    print(f"\n{r['model']}: {verdict} the trust bar. Cut-offs P={r['cutoffs']['P']} Y={r['cutoffs']['Y']}, "
          f"agreement {r['agreement']:.3f}, kappa {r['kappa']:.3f}, n={r['n']}, tokens {r['tokens']:,}")
    print("human \\ judge  " + " ".join(f"{label:>3}" for label in LABELS))
    for h in LABELS:
        print(f"{h:<14} " + " ".join(f"{r['confusion'][h][j]:>3}" for j in LABELS))
    for mode, s in r["by_label_source"].items():
        print(f"  {mode} answers (labels by {s['labels_by']}, n={s['n']}): agreement {s['agreement']:.3f}, kappa {s['kappa']:.3f}")
    print("  Disagreements:")
    for d in sorted(r["disagreements"], key=lambda d: (d["mode"], d["qid"])):
        print(f"    {d['qid']:<4} {d['mode']:<6} human {d['human']}  judge score {d['score']:.3f} -> {d['judge']}")
    for f in r["failed"]:
        print(f"  Failed: {f['qid']} {f['mode']}: {f['error']}")


def main() -> None:
    cfg = load_config()
    ev = cfg["evaluation"]
    golden = {q["id"]: q for q in load_golden(resolve(ev["golden_path"]))}
    graded = graded_answers(cfg, golden)
    print(f"{sum(g['mode'] == 'rag' for g in graded)} answered RAG + {sum(g['mode'] == 'closed' for g in graded)} "
          f"closed-book = {len(graded)} graded answers")
    results = score_candidates(ev["judge_candidates"], graded, cfg)
    for r in results.values():
        print_result(r, ev["trust_agreement"], ev["trust_kappa"])

    small, large = ev["judge_candidates"]
    judge, calibrated = pick_judge(results, ev["trust_agreement"], ev["trust_kappa"], small, large)
    chosen = results[judge]
    settings = resolve("config/settings.yaml")
    set_config_value(settings, "evaluation", "judge_model", judge)
    set_config_value(settings, "evaluation", "cutoffs", f"{{Y: {chosen['cutoffs']['Y']}, P: {chosen['cutoffs']['P']}}}")
    set_config_value(settings, "evaluation", "calibrated", str(calibrated).lower())
    print(f"\nJudge: {judge}, calibrated: {calibrated}")
    print(f"{len(graded)} graded answers is a small sample; cut-offs fitted on it are optimistic.")
    print("All labels are Prateek's: closed-book graded directly, RAG by reviewing Claude's draft grades, "
          "so the two groups are reported separately.")
    out = {"timestamp": datetime.now().isoformat(timespec="seconds"), "judge": judge, "calibrated": calibrated,
           "label_sources": LABEL_SOURCE, "results": results}
    path = resolve(f"eval/results/calibration_{datetime.now():%Y%m%d_%H%M}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path.relative_to(resolve('.'))}")


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
