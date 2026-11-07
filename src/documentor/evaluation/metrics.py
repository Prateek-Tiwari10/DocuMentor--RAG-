import re
import statistics

from documentor.evaluation import first_hit_rank, is_hit
from documentor.generation.citations import CITATION, FULLWIDTH_CITATION, split_fences
from documentor.generation.prompt import build_context

CITATION_ONLY = re.compile(r"^(?:\s*\[\d+(?:\s*,\s*\d+)*\]\s*)+$")
LABEL = re.compile(r"^(?:#+\s.*|\*\*[^*]+\*\*:?)$")


def summarize_ranks(ranks: list[int | None], tokens: list[int]) -> dict:
    n = len(ranks)
    recall = lambda k: sum(1 for r in ranks if r and r <= k) / n
    return {"R@1": recall(1), "R@3": recall(3), "R@5": recall(5),
            "MRR": sum(1 / r for r in ranks if r) / n,
            "avg_tokens_returned": round(statistics.mean(tokens))}


def retrieval_metrics(results_by_q: dict[str, list[dict]], golden: list[dict]) -> dict:
    ranks = [first_hit_rank(results_by_q[q["id"]], q) for q in golden]
    tokens = [sum(r.get("tokens") or 0 for r in results_by_q[q["id"]]) for q in golden]
    return summarize_ranks(ranks, tokens) | {"ranks": {q["id"]: r for q, r in zip(golden, ranks)}}


def recall_in_context(results_by_q: dict[str, list[dict]], golden: list[dict], max_tokens: int, encoding: str) -> dict:
    """R@ctx: a hit counts only if it survives the context budget, because a block that is cut never reaches the model."""
    hits, kept = {}, []
    for q in golden:
        _, blocks = build_context(results_by_q[q["id"]], max_tokens, encoding)
        hits[q["id"]] = any(is_hit(b, q) for b in blocks)
        kept.append(sum(b["tokens"] for b in blocks))
    return {"R@ctx": sum(hits.values()) / len(golden), "hits": hits, "avg_tokens_in_context": round(statistics.mean(kept))}


def cited_blocks(a: dict) -> list[dict]:
    cited = {s["n"] for s in a["sources"]}
    return [b for b in a["blocks"] if b["n"] in cited]


def citation_validity(answers: dict[str, dict]) -> dict:
    answered = [a for a in answers.values() if not a["refused"]]
    clean = sum(not a["invalid_citations"] for a in answered)
    return {"validity": clean / len(answered) if answered else None,
            "invalid_total": sum(len(a["invalid_citations"]) for a in answered)}


def has_citation(text: str) -> bool:
    return bool(CITATION.search(text) or FULLWIDTH_CITATION.search(text))


def answer_units(text: str) -> list[tuple[str, bool]]:
    """Answer split into (unit, is_code); units are code blocks and prose paragraphs, labels skipped."""
    units = []
    for segment, is_code in split_fences(text):
        if is_code:
            units.append((segment, True))
            continue
        for para in re.split(r"\n\s*\n", segment):
            if para.strip() and not LABEL.match(para.strip()):
                units.append((para.strip(), False))
    return units


def unit_coverage(text: str) -> tuple[int, int]:
    """(covered, total): prose carrying or followed by a citation, code followed by one."""
    units = answer_units(text)
    covered = total = 0
    for i, (unit, is_code) in enumerate(units):
        if not is_code and CITATION_ONLY.match(unit):
            continue
        total += 1
        after = units[i + 1] if i + 1 < len(units) else None
        if is_code:
            ok = bool(after) and not after[1] and has_citation(after[0].splitlines()[0])
        else:
            ok = has_citation(unit) or (bool(after) and not after[1] and bool(CITATION_ONLY.match(after[0])))
        covered += ok
    return covered, total


def citation_coverage(answers: dict[str, dict], golden: list[dict]) -> dict:
    answered = [q for q in golden if not answers[q["id"]]["refused"]]
    hits = [q["id"] for q in answered if any(is_hit(b, q) for b in cited_blocks(answers[q["id"]]))]
    pairs = [unit_coverage(answers[q["id"]]["text"]) for q in answered]
    total = sum(t for _, t in pairs)
    return {"coverage": len(hits) / len(answered) if answered else None,
            "cited_expected": hits,
            "unit_coverage": sum(c for c, _ in pairs) / total if total else None}


def refusal_correctness(answers: dict[str, dict], golden_ids: list[str], oos_ids: list[str]) -> dict:
    false_refusals = [q for q in golden_ids if answers[q]["refused"]]
    oos_refused = [q for q in oos_ids if answers[q]["refused"]]
    return {"oos_refusal_rate": len(oos_refused) / len(oos_ids) if oos_ids else None,
            "oos_refused": len(oos_refused), "oos_total": len(oos_ids),
            "false_refusal_rate": len(false_refusals) / len(golden_ids) if golden_ids else None,
            "false_refusals": false_refusals}
