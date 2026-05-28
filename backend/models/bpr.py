"""
BPR-MF + Popularity Penalty — novelty-enhanced ranking for new users.
Carousel: Discover Something New.

Interface:
    load() → None
    recommend(user_ratings, n) → list[tuple[int, float]]

Strategy — Weighted ridge folding-in + popularity penalty:
    The trained BPR model contains:
      item_factors  → item latent factors  (n_items × n_factors)   (implicit.bpr)
      trainset      → raw_id ↔ inner_id mapping + popularity info  (Surprise)

    Step 1 — Folding-in (estimate user vector from onboarding ratings):
         w_i = r_ui / 5.0
         A = Y_rated.T @ diag(w) @ Y_rated + λI
         b = Y_rated.T @ w
         pu = solve(A, b)

    Step 2 — Popularity penalty (boost novelty):
         norm_score(i)  = (score(i) - min) / (max - min)
         penalty(i)     = log(1 + pop(i)) / log(1 + pop_max)
         adjusted(i)    = (1 - β) * norm_score(i) - β * penalty(i)

    Step 3 — Sort by adjusted score, return top-n.

    β = 0.2 : 80% relevance, 20% novelty — calibrated from evaluation results.

Artifact generation (from notebook):
    import pickle
    model = ModelBPRNovelty(factors=64, learning_rate=0.01, regularization=0.01,
                            iterations=100, beta=0.2)
    model.fit(full_trainset)
    with open("backend/artifacts/bpr_model.pkl", "wb") as f:
        pickle.dump(model, f)

Sources:
  Rendle et al. (2009). "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI '09.
  Abdollahpouri et al. (2019). "Managing Popularity Bias in Recommender Systems
      with Personalized Re-ranking." FLAIRS'19.
"""

import pickle
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
LAMBDA = 0.1   # regularization for folding-in
BETA   = 0.2   # novelty weight: 0.0 = pure BPR, 1.0 = pure novelty

_bpr_model  = None
_item_factors = None  # (n_items × n_factors), float64
_log_pop      = None  # log(1 + n_ratings_per_item), indexed by inner_id
_log_pop_max  = 1.0


def load() -> None:
    """Load bpr_model.pkl from artifacts/."""
    global _bpr_model, _item_factors, _log_pop, _log_pop_max
    path = ARTIFACTS_DIR / "bpr_model.pkl"
    with open(path, "rb") as f:
        _bpr_model = pickle.load(f)
    # implicit may store factors as float32 (GPU) — cast for numerical stability
    _item_factors = np.array(_bpr_model._bpr.item_factors, dtype=np.float64)
    # Compute log-popularity from trainset (number of ratings per item)
    ts = _bpr_model.trainset
    pop = np.array([len(ts.ir[i]) for i in range(ts.n_items)], dtype=np.float64)
    _log_pop = np.log1p(pop)
    _log_pop_max = float(_log_pop.max()) if _log_pop.max() > 0 else 1.0
    print(f"[bpr] Model loaded — {ts.n_items} films, {_item_factors.shape[1]} factors, beta={BETA}.")


def recommend(user_ratings: dict, n: int = 20) -> list[tuple[int, float]]:
    """Return top-n (movie_id, score) via BPR folding-in + popularity penalty."""
    if _bpr_model is None:
        return []

    ts = _bpr_model.trainset

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
    ratings   = np.array([x[1] for x in rated])

    # 2. Weighted ridge folding-in: pu ← argmin ||Y_rated pu - w||² + λ||pu||²
    weights  = ratings / 5.0
    Y_rated  = _item_factors[inner_ids]
    n_factors = _item_factors.shape[1]
    A  = Y_rated.T @ (Y_rated * weights[:, np.newaxis]) + LAMBDA * np.eye(n_factors)
    b  = Y_rated.T @ weights
    pu = np.linalg.solve(A, b)

    # 3. Score all unrated films
    rated_inner       = set(inner_ids.tolist())
    candidates_inner  = [i for i in range(ts.n_items) if i not in rated_inner]
    if not candidates_inner:
        return []

    raw_scores = (_item_factors @ pu)[candidates_inner]

    # 4. Popularity penalty: adjust scores to favour less-seen items
    s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
    norm_scores = (raw_scores - s_min) / (s_max - s_min) if s_max > s_min else np.full(len(candidates_inner), 0.5)
    pop_penalty = _log_pop[candidates_inner] / _log_pop_max
    adjusted    = (1 - BETA) * norm_scores - BETA * pop_penalty

    # 5. Map adjusted scores back to [0.5, 5.0] for API consistency
    a_min, a_max = float(adjusted.min()), float(adjusted.max())
    if a_max > a_min:
        final_scores = 0.5 + (adjusted - a_min) / (a_max - a_min) * 4.5
    else:
        final_scores = np.full(len(candidates_inner), 2.75)

    candidates = [
        (int(ts.to_raw_iid(candidates_inner[k])), float(final_scores[k]))
        for k in range(len(candidates_inner))
    ]

    # 6. Sort descending, return top-n
    candidates.sort(key=lambda x: -x[1])
    return candidates[:n]
