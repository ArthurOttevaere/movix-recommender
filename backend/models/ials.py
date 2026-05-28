"""
iALS (WRMF) — ALS closed-form folding-in for new users.
Carousel: Top Picks For You.

Interface:
    load() → None
    recommend(user_ratings, n) → list[tuple[int, float]]

Strategy — ALS closed-form folding-in:
    The trained iALS model contains:
      item_factors  → item latent factors  (n_items × n_factors)   (implicit.als)
      trainset      → raw_id ↔ inner_id mapping                    (Surprise)
      alpha         → confidence scaling parameter (c_ui = 1 + alpha * r_ui)

    Gram matrix YTY = item_factors.T @ item_factors, precomputed at load time.

    For a new user with ratings {iid: r}:
    1. Convert raw movie_ids → inner_ids (skip unknown films).
    2. Confidence weights: c_rated = 1 + alpha * r_ui
    3. Solve for pu (ALS closed-form):
         A = YTY + Y_rated.T @ diag(alpha * ratings) @ Y_rated + λI
         b = Y_rated.T @ c_rated
         pu = solve(A, b)
    4. Score all unrated films:
         raw_scores = item_factors @ pu
    5. Normalize to [0.5, 5.0] via min-max over candidates.
    6. Sort descending, return top-n.

Artifact generation (from notebook):
    import pickle
    model = ModeliALS(factors=50, iterations=20, regularization=0.01, alpha=40)
    model.fit(full_trainset)
    with open("backend/artifacts/ials_model.pkl", "wb") as f:
        pickle.dump(model, f)

Source: Hu Y. et al. (2008). "Collaborative Filtering for Implicit Feedback Datasets." ICDM, pp. 263-272.
"""

import pickle
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
LAMBDA = 1.0  # regularization for folding-in

_ials_model = None
_item_factors = None  # (n_items × n_factors), float64
_YTY = None           # precomputed gram matrix (n_factors × n_factors)


def load() -> None:
    """Load ials_model.pkl from artifacts/."""
    global _ials_model, _item_factors, _YTY
    path = ARTIFACTS_DIR / "ials_model.pkl"
    with open(path, "rb") as f:
        _ials_model = pickle.load(f)
    # implicit may store factors as float32 (GPU) — cast for numerical stability
    _item_factors = np.array(_ials_model._als.item_factors, dtype=np.float64)
    _YTY = _item_factors.T @ _item_factors
    print(f"[ials] Model loaded — {_ials_model.trainset.n_items} films, {_item_factors.shape[1]} factors.")


def recommend(user_ratings: dict, n: int = 20) -> list[tuple[int, float]]:
    """Return top-n (movie_id, raw_score) via ALS closed-form folding-in."""
    if _ials_model is None:
        return []

    ts = _ials_model.trainset
    alpha = float(_ials_model.alpha)

    # 1. Convert raw movie_ids → inner_ids (skip unknown films)
    rated = []
    for raw_iid, r in user_ratings.items():
        try:
            inner = ts.to_inner_iid(raw_iid)
            rated.append((inner, float(r)))
        except ValueError:
            pass

    if not rated:
        return []

    inner_ids = np.array([x[0] for x in rated])
    ratings = np.array([x[1] for x in rated])

    # 2. Confidence weights: c_rated = 1 + alpha * r_ui
    c_rated = 1.0 + alpha * ratings

    # 3. ALS closed-form folding-in:
    #    A = YTY + Y_rated.T @ diag(alpha * r) @ Y_rated + λI
    #    b = Y_rated.T @ c_rated
    Y_rated = _item_factors[inner_ids]  # (n_rated × n_factors)
    n_factors = _item_factors.shape[1]

    weighted_Y = Y_rated * (alpha * ratings)[:, np.newaxis]
    A = _YTY + Y_rated.T @ weighted_Y + LAMBDA * np.eye(n_factors)
    b = Y_rated.T @ c_rated
    pu = np.linalg.solve(A, b)

    # 4. Score all unrated films
    rated_inner = set(inner_ids.tolist())
    raw_scores = _item_factors @ pu  # (n_items,)

    candidates_inner = [i for i in range(ts.n_items) if i not in rated_inner]
    if not candidates_inner:
        return []

    # 5. Normalize to [0.5, 5.0] via min-max over candidates
    candidate_scores = raw_scores[candidates_inner]
    s_min, s_max = float(candidate_scores.min()), float(candidate_scores.max())
    if s_max > s_min:
        normalized = 0.5 + (candidate_scores - s_min) / (s_max - s_min) * 4.5
    else:
        normalized = np.full(len(candidates_inner), 2.75)

    candidates = [
        (int(ts.to_raw_iid(candidates_inner[k])), float(normalized[k]))
        for k in range(len(candidates_inner))
    ]

    # 6. Sort descending, return top-n
    candidates.sort(key=lambda x: -x[1])
    return candidates[:n]
