from __future__ import annotations

from typing import Mapping


def normalize_scores(scored: Mapping[int, float]) -> dict[int, float]:
    """Min-max normalize to [0, 1]. Empty input → empty output."""
    if not scored:
        return {}
    vals = list(scored.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {int(k): 0.5 for k in scored}
    rng = hi - lo
    return {int(k): (float(v) - lo) / rng for k, v in scored.items()}


def topk(scored: Mapping[int, float], k: int) -> list[tuple[int, float]]:
    return sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:k]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def mmr_rerank(
    scored: list[tuple[int, float]],
    item_features: dict[int, set],
    k: int,
    lambda_: float = 0.55,
) -> list[tuple[int, float]]:
    """Maximal Marginal Relevance reranking using Jaccard similarity on item feature sets.

    Higher ``lambda_`` favors relevance; lower favors diversity.
    """
    selected: list[tuple[int, float]] = []
    pool = list(scored)
    while pool and len(selected) < k:
        best_idx = 0
        best_val = float("-inf")
        for i, (mid, rel) in enumerate(pool):
            if not selected:
                mmr = rel
            else:
                feats = item_features.get(mid, set())
                max_sim = max(
                    _jaccard(feats, item_features.get(s_mid, set())) for s_mid, _ in selected
                )
                mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_val:
                best_val = mmr
                best_idx = i
        selected.append(pool.pop(best_idx))
    return selected
