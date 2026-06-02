"""
BPR-MF + Popularity Penalty — novelty-enhanced ranking for new users.
Carousel: Discover Something New.

Interface:
    load() → None
    recommend(user_ratings, n) → list[tuple[int, float]]

Strategy — Weighted ridge folding-in + novelty re-ranking:
    Faithful to the evaluated ModelBPRNovelty.test() (configs.BPR_Novelty): the
    novelty re-ranking is applied over the WHOLE candidate catalogue with a global
    popularity penalty — no relevance gate, no pool-local normalization — so the
    served carousel matches the model whose metrics (coverage, miuf, …) are reported.

    The trained BPR model contains:
      item_factors  → item latent factors  (n_items × n_factors)   (implicit.bpr)
      trainset      → raw_id ↔ inner_id mapping + popularity info  (Surprise)

    Step 1 — Folding-in (estimate user vector from onboarding ratings):
         w_i = r_ui / 5.0
         A = Y_rated.T @ diag(w) @ Y_rated + λI
         b = Y_rated.T @ w
         pu = solve(A, b)

    Step 2 — Novelty re-ranking over all unrated films:
         norm_score(i)  = minmax(relevance(i))  over all candidates  → [0, 1]
         penalty(i)     = log1p(pop_i) / log_pop_max  (global)       → [0, 1]
         adjusted(i)    = (1 - β) * norm_score(i) - β * penalty(i)

    Step 3 — Sort by adjusted score, return top-n.

    β = 0.2 : aligned with configs.BPR_Novelty (the evaluated model). At this weight
              relevance dominates (0.8) and novelty only adjusts at the margin, which
              is why dropping the old relevance gate is safe here.

    Note: serving uses folding-in (the new user is not in the trainset), so pu is a
    closed-form approximation — the re-ranking algorithm matches test() exactly, but
    the scores are not bit-identical to an offline retrain. This is inherent to
    real-time serving.

Artifact note:
    The pickle `bpr_model.pkl` stores a plain `ModelBPR` (BPR-MF) — it only carries
    the trained latent factors, NOT a beta. This is equivalent to ModelBPRNovelty for
    serving: ModelBPRNovelty subclasses ModelBPR and beta only affects the re-ranking
    (test()), never the factor training, so the item_factors are identical either way.
    The novelty weight beta=0.2 is therefore applied HERE at serving (the BETA constant
    below), which together with the trained factors reproduces configs.BPR_Novelty.

    Generation (from notebook):
        import pickle
        model = ModelBPR(factors=64, learning_rate=0.01, regularization=0.01,
                         iterations=100)
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
BETA   = 0.2   # novelty weight — aligned with configs.BPR_Novelty (beta=0.2), the evaluated model

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

    # 4. Novelty re-ranking — faithful to the evaluated ModelBPRNovelty.test():
    #    applied over the WHOLE candidate catalogue (no relevance gate), relevance
    #    min-max normalized over all candidates, and popularity penalised GLOBALLY
    #    (log_pop / log_pop_max) rather than within a pool. This is what produces the
    #    reported coverage / miuf metrics, so the served carousel matches them.
    s_min, s_max = float(relevance.min()), float(relevance.max())
    norm_scores = (relevance - s_min) / (s_max - s_min) if s_max > s_min else np.full(len(relevance), 0.5)
    pop_penalty = _log_pop[candidates_inner] / _log_pop_max
    adjusted    = (1 - BETA) * norm_scores - BETA * pop_penalty

    # 5. Taste-match score = cosine(pu, item) ∈ [0, 1] over all candidates. This is
    #    what the green "% match" shows: how well each film fits the user's taste,
    #    INDEPENDENT of the novelty re-ranking (a fresh pick can still report its
    #    true match). Ranking stays by `adjusted`; only the displayed score is the
    #    match. Encoded into [0.5, 5.0] so normalize_score() maps it back to a %.
    pu_norm = float(np.linalg.norm(pu)) + 1e-9
    cand_factors = _item_factors[candidates_inner]
    cand_norms = np.linalg.norm(cand_factors, axis=1) + 1e-9
    match = np.clip(relevance / (cand_norms * pu_norm), 0.0, 1.0)
    display = 0.5 + match * 4.5

    # 6. Rank by novelty-adjusted score (descending), return top-n with match score
    order = np.argsort(-adjusted)[:n]
    return [
        (int(ts.to_raw_iid(int(candidates_inner[j]))), float(display[j]))
        for j in order
    ]
