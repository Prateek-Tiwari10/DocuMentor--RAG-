import math
import re

LABELS = ["Y", "P", "N"]
GRADE_LINES = {"rag": "RAG correct? (Y / P / N):", "closed": "Closed-book correct? (Y / P / N):"}


def to_grade(score: float, cutoffs: dict) -> str:
    if score >= cutoffs["Y"]:
        return "Y"
    return "P" if score >= cutoffs["P"] else "N"


def agreement(human: list[str], judge: list[str]) -> float:
    return sum(h == j for h, j in zip(human, judge)) / len(human)


def cohen_kappa(human: list[str], judge: list[str]) -> float:
    n = len(human)
    observed = agreement(human, judge)
    expected = sum(human.count(label) * judge.count(label) for label in LABELS) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def confusion(human: list[str], judge: list[str]) -> dict[str, dict[str, int]]:
    return {h: {j: sum(a == h and b == j for a, b in zip(human, judge)) for j in LABELS} for h in LABELS}


def search_cutoffs(scores: list[float], human: list[str], step: float = 0.05,
                   prefer: tuple[float, float] = (0.4, 0.7)) -> dict:
    """Best (P, Y): highest agreement, then kappa, then closest to `prefer`. Deterministic."""
    steps = round(1 / step)
    best, best_rank = None, None
    for p_i in range(1, steps):
        for y_i in range(p_i + 1, steps):
            cutoffs = {"P": round(p_i * step, 2), "Y": round(y_i * step, 2)}
            judge = [to_grade(s, cutoffs) for s in scores]
            agree, kappa = agreement(human, judge), cohen_kappa(human, judge)
            distance = math.hypot(cutoffs["P"] - prefer[0], cutoffs["Y"] - prefer[1])
            rank = (round(agree, 9), round(kappa, 9), -round(distance, 9))
            if best_rank is None or rank > best_rank:
                best, best_rank = cutoffs | {"agreement": agree, "kappa": kappa}, rank
    return best


def passes_trust(result: dict, trust_agreement: float, trust_kappa: float) -> bool:
    return result["agreement"] >= trust_agreement and result["kappa"] >= trust_kappa


def pick_judge(results: dict[str, dict], trust_agreement: float, trust_kappa: float,
               small: str = "openai/gpt-oss-20b", large: str = "openai/gpt-oss-120b",
               margin: float = 0.05) -> tuple[str, bool]:
    """(judge, calibrated). The kappa margin and the model pair come from Step 4's rule."""
    passing = [m for m, r in results.items() if passes_trust(r, trust_agreement, trust_kappa)]
    if len(passing) == 2:
        return (large if results[large]["kappa"] - results[small]["kappa"] > margin else small), True
    if len(passing) == 1:
        return passing[0], True
    return max(results, key=lambda m: results[m]["kappa"]), False


def parse_grades(path) -> dict[str, dict[str, str]]:
    """Per-question grades from a review sheet: {qid: {"rag": "Y", "closed": "N"}}. Bold is accepted."""
    grades: dict[str, dict[str, str]] = {}
    current = None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^## \[(\w+)\]", line)
        if m:
            current = m.group(1)
            grades[current] = {}
            continue
        if current is None:
            continue
        for mode, prefix in GRADE_LINES.items():
            if line.startswith(prefix):
                value = line[len(prefix):].strip().lstrip("*").strip()[:1].upper()
                if value in LABELS:
                    grades[current][mode] = value
    return grades
