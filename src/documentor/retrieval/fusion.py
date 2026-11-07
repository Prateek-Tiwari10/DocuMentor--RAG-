def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: sum of 1 / (k + rank) over the lists holding an id, with ranks starting at 1."""
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], best[pair[0]], pair[0]))
