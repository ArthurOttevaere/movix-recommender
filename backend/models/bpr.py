"""
BPR-MF + Popularity Penalty — novelty-enhanced ranking for new users.
Carousel: Discover Something New.

Interface:
    load() → None
    recommend(user_ratings, n) → list[tuple[int, float]]

Strategy — Weighted ridge folding-in + relevance-gated novelty re-ranking:
    The trained BPR model contains:
      item_factors  → item latent factors  (n_items × n_factors)   (implicit.bpr)
      trainset      → raw_id ↔ inner_id mapping + popularity info  (Surprise)

    Step 1 — Folding-in (estimate user vector from onboarding ratings):
         w_i = r_ui / 5.0
         A = Y_rated.T @ diag(w) @ Y_rated + λI
         b = Y_rated.T @ w
         pu = solve(A, b)

    Step 2 — Relevance gate (keep only items that match the user's taste):
         Rank all unrated films by pure relevance (item_factors @ pu) and keep
         the TOP_K most relevant as the candidate pool. This guarantees every
         recommendation stays within the user's taste — the worst item we can
         surface is the K-th most relevant film.

    Step 3 — Novelty re-ranking (within the relevant pool only):
         norm_score(i)  = minmax(score(i))   over the pool   → [0, 1]
         penalty(i)     = minmax(log1p(pop))  over the pool   → [0, 1]
         adjusted(i)    = (1 - β) * norm_score(i) - β * penalty(i)
         Both terms are min-max normalized *within the pool* so β trades them off
         on equal footing. (Normalizing popularity globally would make it nearly
         constant across a pool of popular blockbusters → β would have no effect.)

    Step 4 — Sort by adjusted score, return top-n.

    Why the gate matters: applying the popularity penalty over the WHOLE catalog
    makes obscure-but-irrelevant films (relevance≈0, popularity≈0) win as soon as
    β grows, producing random-looking picks. Gating to the top-K relevant items
    first (personalized re-ranking, Abdollahpouri 2019) keeps novelty *inside*
    the user's taste, and makes β behave smoothly instead of cliff-like.

    TOP_K = 200 : size of the relevant candidate pool re-ranked for novelty.
    β = 0.5 : balanced — with the gate, distinct from iALS (≈0-35% top-20 overlap)
              while every pick stays within the user's top-200 most relevant films.

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
BETA   = 0.5   # novelty weight within the relevant pool: 0.0 = pure relevance, 1.0 = pure novelty
TOP_K  = 200   # size of the relevance-gated candidate pool re-ranked for novelty

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

    # 3. Score all unrated films by pure relevance
    rated_inner       = set(inner_ids.tolist())
    candidates_inner  = [i for i in range(ts.n_items) if i not in rated_inner]
    if not candidates_inner:
        return []

    candidates_inner = np.array(candidates_inner)
    relevance = (_item_factors @ pu)[candidates_inner]

    # 4. Relevance gate: keep only the TOP_K most relevant films as the pool.
    #    Novelty is applied *within* this pool, so we never surface items that
    #    fall outside the user's taste — the worst pick is the K-th most relevant.
    k = min(TOP_K, len(candidates_inner))
    top_idx = np.argpartition(-relevance, k - 1)[:k]
    candidates_inner = candidates_inner[top_idx]
    raw_scores = relevance[top_idx]

    # 5. Novelty re-ranking within the relevant pool: favour the less mainstream
    #    films *among the relevant ones*. Both relevance and popularity are
    #    min-max normalized over the pool so β trades them off on equal footing.
    #    (Normalizing popularity globally would make it ≈constant across a pool of
    #     popular blockbusters, leaving β with no effect.)
    s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
    norm_scores = (raw_scores - s_min) / (s_max - s_min) if s_max > s_min else np.full(len(raw_scores), 0.5)
    pool_pop = _log_pop[candidates_inner]
    p_min, p_max = float(pool_pop.min()), float(pool_pop.max())
    pop_penalty = (pool_pop - p_min) / (p_max - p_min) if p_max > p_min else np.zeros(len(pool_pop))
    adjusted    = (1 - BETA) * norm_scores - BETA * pop_penalty

    # 6. Taste-match score = cosine(pu, item) ∈ [0, 1] for the pool. This is what
    #    the green "% match" shows: how well each film fits the user's taste,
    #    INDEPENDENT of the novelty re-ranking (a fresh pick can still report its
    #    true match). Ranking stays by `adjusted` (relevance + novelty); only the
    #    displayed score is the match. Encoded into [0.5, 5.0] for normalize_score().
    pu_norm = float(np.linalg.norm(pu)) + 1e-9
    pool_factors = _item_factors[candidates_inner]
    pool_norms = np.linalg.norm(pool_factors, axis=1) + 1e-9
    match = np.clip(raw_scores / (pool_norms * pu_norm), 0.0, 1.0)
    display = 0.5 + match * 4.5

    # 7. Rank by novelty-adjusted score (descending), return top-n with match score
    order = np.argsort(-adjusted)[:n]
    return [
        (int(ts.to_raw_iid(int(candidates_inner[j]))), float(display[j]))
        for j in order
    ]
