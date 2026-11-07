import re
from pathlib import Path

GRADE_ORDER = {"N": 0, "P": 1, "Y": 2}
COMPARE_METRICS = [("Y", "questions"), ("P", "questions"), ("N", "questions"),
                   ("answer_correctness", None), ("faithfulness", None), ("answer_relevancy", None),
                   ("context_precision", None), ("citation_validity", "rate"), ("citation_coverage", "rate"),
                   ("oos_refusal_rate", "rate"), ("false_refusal_rate", "rate"), ("R@5", "rate"), ("MRR", None),
                   ("tokens_per_answer", None), ("p50_generate_ms", None)]


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value)


def set_config_value(path, section: str, key: str, value: str) -> None:
    """Set one `key:` inside a top-level section, keeping the file's comments. Adds the key if missing."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").split("\n")
    start = lines.index(f"{section}:")
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith(" ")), len(lines))
    pattern = re.compile(rf"^  {re.escape(key)}:(\s*)[^#]*?(\s*#.*)?$")
    for i in range(start + 1, end):
        m = pattern.match(lines[i])
        if m:
            lines[i] = f"  {key}: {value}{m.group(2) or ''}"
            break
    else:
        lines.insert(start + 1, f"  {key}: {value}")
    p.write_text("\n".join(lines), encoding="utf-8")


def choose_strategy(recursive: dict, parent_child: dict, rule: dict, calibrated: bool) -> tuple[str, str]:
    """Default strategy from Step 5's rule, with the numbers behind the decision."""
    drop = (recursive["faithfulness"] or 0) - (parent_child["faithfulness"] or 0)
    if calibrated:
        gain, measure = parent_child["Y"] - recursive["Y"], "Y grades"
    else:
        gain = len(parent_child["cited_expected"]) - len(recursive["cited_expected"])
        measure = "questions citing an expected section (judge uncalibrated)"
    ok_gain, ok_drop = gain >= rule["min_question_gain"], drop <= rule["max_faithfulness_drop"]
    choice = "parent_child" if ok_gain and ok_drop else "recursive"
    reason = (f"parent_child vs recursive: {gain:+d} {measure} (needs at least +{rule['min_question_gain']}); "
              f"faithfulness drop {drop:+.3f} (allowed up to {rule['max_faithfulness_drop']}). Choice: {choice}.")
    return choice, reason


def compare_runs(old: dict, new: dict) -> dict:
    """Old vs new per config. A difference of one question is labelled within noise."""
    out = {}
    for config in new["summary"]:
        if config not in old["summary"]:
            continue
        before, after = old["summary"][config], new["summary"][config]
        n = after.get("questions") or before.get("questions") or 0
        rows = []
        for metric, kind in COMPARE_METRICS:
            a, b = before.get(metric), after.get(metric)
            diff = None if a is None or b is None else b - a
            noise = False
            if diff:
                questions = abs(diff) if kind == "questions" else abs(diff) * n if kind == "rate" else None
                noise = questions is not None and round(questions) == 1
            rows.append({"metric": metric, "old": a, "new": b, "diff": diff, "noise": noise})
        old_q, new_q = old["grades"].get(config, {}), new["grades"].get(config, {})
        shared = [q for q in new_q if q in old_q]
        better = [q for q in shared if GRADE_ORDER[new_q[q]] > GRADE_ORDER[old_q[q]]]
        worse = [q for q in shared if GRADE_ORDER[new_q[q]] < GRADE_ORDER[old_q[q]]]
        changed = better + worse
        out[config] = {"rows": rows, "better": len(better), "worse": len(worse), "same": len(shared) - len(changed),
                       "changed": sorted(changed), "grade_noise": len(changed) == 1,
                       "any_change": bool(changed) or any(r["diff"] not in (None, 0) for r in rows)}
    return out


def render_compare(result: dict) -> str:
    lines = []
    for config, c in result.items():
        lines += [f"\n{config}", f"{'metric':<22} {'old':>10} {'new':>10} {'diff':>10}"]
        for r in c["rows"]:
            flag = "  within noise" if r["noise"] else ""
            diff = "-" if r["diff"] is None else f"{r['diff']:+.3f}"
            lines.append(f"{r['metric']:<22} {fmt(r['old']):>10} {fmt(r['new']):>10} {diff:>10}{flag}")
        noise = " (within noise)" if c["grade_noise"] else ""
        lines.append(f"better {c['better']}, worse {c['worse']}, same {c['same']}{noise}; "
                     f"changed: {', '.join(c['changed']) or 'none'}")
    if not any(c["any_change"] for c in result.values()):
        lines.append("\nNo changes.")
    return "\n".join(lines)


def summary_row(config: str, s: dict) -> str:
    grades = f"{s['Y']} / {s['P']} / {s['N']}" if s.get("Y") is not None else "-"
    refusal = "-"
    if s.get("oos_total"):
        refusal = f"{s['oos_refused']}/{s['oos_total']} OOS, {len(s['false_refusals'])} false"
    retrieval = f"{s['R@5']:.2f} / {s['MRR']:.3f}" if s.get("R@5") is not None else "-"
    cells = [config, grades] + [fmt(s.get(m)) for m in ("answer_correctness", "faithfulness", "answer_relevancy",
                                                        "context_precision", "citation_validity", "citation_coverage")]
    cells += [refusal, retrieval, fmt(s.get("tokens_per_answer")), fmt(s.get("p50_generate_ms"))]
    return "| " + " | ".join(cells) + " |"


def summary_table(summary: dict) -> list[str]:
    return ["| Config | Y / P / N | Answer correctness | Faithfulness | Answer relevancy | Context precision | "
            "Citation validity | Citation coverage | Refusal correctness | R@5 / MRR | Tokens per answer | "
            "p50 generate ms |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"] + [summary_row(c, s) for c, s in summary.items()]


def human_section(human: dict) -> list[str]:
    lines = [f"From the Phase 2 human grades (`{human['sheet']}`), not from the judge.", "",
             "| Group | Config | Y / P / N |", "|---|---|---|"]
    for group, rows in human["groups"].items():
        for config, c in rows.items():
            lines.append(f"| {group} ({c['Y'] + c['P'] + c['N']}) | {config} | {c['Y']} / {c['P']} / {c['N']} |")
    return lines + ["", f"Closed-book answers with an outdated or invented API: {human['closed_book_outdated_api']}."]


def render_report(run: dict) -> str:
    lines = ["# Phase 3 evaluation", "",
             f"Judge: `{run['judge_model']}` (calibrated: {run['calibrated']}), RAGAS {run['ragas_version']}, "
             f"corpus {run['corpus_commit'][:7]}.", "", "## Summary", ""]
    lines += summary_table(run["summary"])
    lines += ["", "A refused in-scope question counts as N and has no RAGAS scores."]
    if not run["calibrated"]:
        lines += ["", "Answer correctness is uncalibrated: not used for decisions."]
    lines += ["", "## RAG vs closed-book", ""] + human_section(run["human_rag_vs_closed"])
    if run["rag_vs_closed"]:
        lines += ["", "Judge scores on this run's questions:", "",
                  "| Group | Config | Answer correctness | Y / P / N |", "|---|---|---|---|"]
        for group, rows in run["rag_vs_closed"].items():
            for config, r in rows.items():
                lines.append(f"| {group} | {config} | {fmt(r['answer_correctness'])} | {r['Y']} / {r['P']} / {r['N']} |")
    configs = list(run["summary"])
    lines += ["", "## Per question", "",
              "| ID | post_cutoff | " + " | ".join(f"{c} grade | {c} faithfulness" for c in configs) + " |",
              "|---|---|" + "---|---|" * len(configs)]
    for q in run["per_question"]:
        cells = [q["id"], str(q["post_cutoff"]).lower() if q["post_cutoff"] is not None else "-"]
        for c in configs:
            row = q["configs"].get(c, {})
            cells += [row.get("grade") or "-", fmt(row.get("faithfulness"))]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Failed metrics and skipped work", ""]
    lines += [f"- Failed: {f['config']} {f['qid']} {f['metric']}: {f['error']}" for f in run["failed"]] or ["- No failed metrics."]
    lines += [f"- Skipped: {s}" for s in run["skipped"]]
    lines += ["", "## Strategy decision", "", run["strategy_decision"], "", "## Cost", "",
              f"- Judge tokens: {run['judge_tokens']:,}", f"- Answer tokens: {run['answer_tokens']:,}",
              f"- Days the scoring ran on: {run['scoring_days']}"]
    return "\n".join(lines) + "\n"
