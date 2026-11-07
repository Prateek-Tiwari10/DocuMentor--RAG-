import argparse
import glob
import json
import statistics
from datetime import datetime
from pathlib import Path

from eval_answers import percentile
from eval_retrieval import choose_config

from documentor.config import load_config, resolve
from documentor.conversation import MODES, Conversation, Turn, default_rewrite_llm, make_standalone
from documentor.evaluation import load_jsonl
from documentor.evaluation.metrics import recall_in_context, retrieval_metrics
from documentor.evaluation.report import set_config_value
from documentor.retrieval.dense import search

TYPES = ("follow_up", "topic_switch")


def turn1_answers(items: list[dict], cfg: dict) -> tuple[dict, str]:
    """Turn 1's answer from the latest Phase 3 results file; generated only if it's missing there."""
    files = sorted(glob.glob(str(resolve("eval/results/phase3_*.json"))))
    stored = json.loads(open(files[-1]).read())["answers"]["recursive"] if files else {}
    out, generated = {}, 0
    for item in items:
        qid = item["turn1_golden"]
        if qid in stored:
            out[item["id"]] = stored[qid]
            continue
        from documentor.pipeline import answer
        result = answer(item["turns"][0], language=item["language"])
        out[item["id"]] = {"text": result.text, "sources": result.sources, "language": result.language}
        generated += 1
    source = f"{Path(files[-1]).name if files else 'none'} ({generated} generated)"
    return out, source


def conversation_for(item: dict, first: dict) -> Conversation:
    q = item["turns"][0]
    turn = Turn(q, q, item["language"], first["text"], first.get("sources", []))
    return Conversation(id=item["id"], turns=[turn], language=item["language"])


def score(items: list[dict], results: dict, cfg: dict) -> dict:
    gen = cfg["generation"]
    m = retrieval_metrics(results, items)
    ctx = recall_in_context(results, items, gen["context_max_tokens"], gen["budget_encoding"])
    return {"n": len(items), "R@ctx": ctx["R@ctx"], "ctx_hits": sum(ctx["hits"].values()), "MRR": m["MRR"],
            "ranks": m["ranks"], "hits": ctx["hits"]}


def run_mode(mode: str, items: list[dict], firsts: dict, cfg: dict, llm) -> dict:
    rewrites, infos, results = {}, [], {}
    for item in items:
        conv = conversation_for(item, firsts[item["id"]])
        standalone, info = make_standalone(item["turns"][1], conv, mode, llm=llm, cfg=cfg)
        rewrites[item["id"]] = standalone
        infos.append(info)
        results[item["id"]] = search(standalone, language=item["language"])
    overall = score(items, results, cfg)
    by_type = {t: score([i for i in items if i["type"] == t], {i["id"]: results[i["id"]] for i in items if i["type"] == t}, cfg)
               for t in TYPES}
    calls = [i for i in infos if i["latency_ms"]]
    latencies = [i["latency_ms"] for i in calls]
    return {"label": mode, "mode": mode, "overall": overall, "by_type": by_type, "rewrites": rewrites,
            "ctx_hits": overall["ctx_hits"], "MRR": overall["MRR"], "R@ctx": overall["R@ctx"], "unit": "conversations",
            "p50_ms": percentile(latencies, 0.5) or 0, "p95_ms": percentile(latencies, 0.95) or 0,
            "cap_ms": cfg["conversation"]["rewrite_max_p95_ms"] if mode == "llm" else None,
            "calls": len(calls), "cache_hits": sum(i["cache_hit"] for i in calls),
            "mean_tokens": round(statistics.mean([i["prompt_tokens"] + i["completion_tokens"] for i in calls])) if calls else 0,
            "fallbacks": sum(i["fallback"] for i in infos)}


def decision_sentence(best: dict, base: dict, reasons: list[str]) -> str:
    rejected = [r.replace(": rejected (", " (", 1) for r in reasons if ": rejected (" in r]
    gain = best["ctx_hits"] - base["ctx_hits"]
    latency = f", rewrite p95 {best['p95_ms']:,.0f} ms" if best["mode"] == "llm" else ""
    return (f"{best['mode']}: turn-2 R@ctx {base['R@ctx']:.2f} → {best['R@ctx']:.2f} ({gain:+d} conversations), "
            f"MRR {base['MRR']:.3f} → {best['MRR']:.3f}{latency}. Rejected: {'; '.join(rejected) or 'none'}.")


def render(run: dict) -> str:
    lines = ["# Phase 5 conversation eval", "",
             f"{run['n']} two-turn conversations ({run['counts']}), drafted by an LLM and checked automatically against "
             f"the docs; none verified by a person. Retrieval only, default config ({run['retrieval']}). "
             f"Turn 1 answers: {run['turn1_source']}.", "", "## Summary", "",
             "| Mode | R@ctx | MRR | follow_up R@ctx | follow_up MRR | topic_switch R@ctx | topic_switch MRR | "
             "rewrite p50 ms | rewrite p95 ms | mean tokens | fallbacks |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in run["rows"]:
        f, t = row["by_type"]["follow_up"], row["by_type"]["topic_switch"]
        lines.append(f"| {row['mode']} | {row['R@ctx']:.2f} ({row['ctx_hits']}/{run['n']}) | {row['MRR']:.3f} | "
                     f"{f['R@ctx']:.2f} ({f['ctx_hits']}/{f['n']}) | {f['MRR']:.3f} | {t['R@ctx']:.2f} ({t['ctx_hits']}/{t['n']}) | "
                     f"{t['MRR']:.3f} | {row['p50_ms']:,.0f} | {row['p95_ms']:,.0f} | {row['mean_tokens']:,} | {row['fallbacks']} |")
    lines += ["", "Rewrite latency is the model call's own time (kept when a call is served from the cache).", "",
              "## Decision", "", run["decision"], ""] + [f"- {r}" for r in run["reasons"]]
    lines += ["", "## Rewrites (hit rank per mode, - for a miss)", ""]
    for item in run["items"]:
        lines.append(f"**{item['id']}** ({item['type']}) turn 1: {item['turns'][0]}  ")
        lines.append(f"turn 2: {item['turns'][1]}")
        for row in run["rows"]:
            rank = row["overall"]["ranks"][item["id"]]
            ctx = "in context" if row["overall"]["hits"][item["id"]] else "not in context"
            lines.append(f"- {row['mode']} (rank {rank or '-'}, {ctx}): {row['rewrites'][item['id']]}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Compare follow-up rewrite modes on two-turn conversations (retrieval only).")
    ap.add_argument("--path", default="eval/conversations.jsonl")
    ap.add_argument("--no-write", action="store_true", help="report the decision without writing it to config")
    args = ap.parse_args()
    items = load_jsonl(resolve(args.path))
    firsts, source = turn1_answers(items, cfg)
    llm = default_rewrite_llm(cfg)
    rows = []
    for mode in MODES:
        row = run_mode(mode, items, firsts, cfg, llm)
        rows.append(row)
        print(f"\n== {mode}: R@ctx {row['R@ctx']:.2f} ({row['ctx_hits']}/{len(items)}), MRR {row['MRR']:.3f}, "
              f"p95 {row['p95_ms']:,.0f} ms, fallbacks {row['fallbacks']}")
        if mode != "none":
            for item in items:
                print(f"  {item['id']} {item['type']:<12} {item['turns'][1]}\n  {'':16}→ {row['rewrites'][item['id']]}")

    best, reasons = choose_config(rows, cfg)
    decision = decision_sentence(best, rows[0], reasons)
    current = cfg["conversation"]["rewrite"]
    if best["mode"] != current and not args.no_write:
        set_config_value(resolve("config/settings.yaml"), "conversation", "rewrite", best["mode"])
        decision += f" Config written: conversation.rewrite: {current} -> {best['mode']}."
    print(f"\n{decision}")
    r = cfg["retrieval"]
    counts = ", ".join(f"{sum(i['type'] == t for i in items)} {t}" for t in TYPES)
    run = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n": len(items), "counts": counts,
           "retrieval": f"{r['strategy']} · {r['mode']}{' · rerank' if r['rerank'] else ''}, k={r['top_k']}",
           "turn1_source": source, "rewrite_model": cfg["conversation"]["rewrite_model"], "rows": rows,
           "decision": decision, "reasons": reasons, "items": items}
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    for ext, text in (("json", json.dumps(run, indent=2, default=str)), ("md", render(run))):
        path = resolve(f"eval/results/phase5_conversations_{stamp}.{ext}")
        path.write_text(text, encoding="utf-8")
        print(f"Saved {path.relative_to(resolve('.'))}")


if __name__ == "__main__":
    main()
