"""
SVD++ model — folding-in adapté pour LatentFactorPP (SVDpp Surprise).

Interface :
    load() → None
    recommend(user_ratings, n) → list[tuple[int, float]]

Stratégie — Folding-in SVD++ :
    Le modèle SVD++ (LatentFactorPP) contient :
      qi  → facteurs items        (n_items × n_factors)
      bi  → biais items           (n_items,)
      yj  → facteurs implicites   (n_items × n_factors)
      trainset.global_mean → moyenne globale

    Pour un nouvel utilisateur avec ratings {iid: r} :
    1. Convertir raw movie_ids → inner_ids (skip les inconnus).
    2. Terme implicite (feedback implicite SVD++) :
         implicit = |I_u|^{-0.5} * sum(yj[j] for j in rated_inner_ids)
    3. Résidu pour le terme explicite :
         residuals = ratings - global_mean - bi[rated]
    4. Résoudre pour pu (folding-in ridge) :
         A = Q_rated.T @ Q_rated + lambda * I
         b = Q_rated.T @ residuals
         pu = solve(A, b)
    5. Vecteur utilisateur effectif :
         effective_pu = pu + implicit
    6. Scorer tous les films non notés :
         score_i = global_mean + bi[i] + qi[i] · effective_pu
    7. Clipper à [0.5, 5.0], trier décroissant, retourner top-n.

Source : Koren, Y. (2008). Factorization meets the neighborhood. KDD '08, pp. 426-434.
"""

import pickle
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
LAMBDA = 0.1  # régularisation pour le folding-in

_svd_model = None  # ModelBaseline4 entraîné, chargé par load()


def load() -> None:
    """Charge svd_model.pkl depuis artifacts/."""
    global _svd_model
    path = ARTIFACTS_DIR / "svd_model.pkl"
    with open(path, "rb") as f:
        _svd_model = pickle.load(f)
    print(f"[svd] Modèle chargé — {_svd_model.trainset.n_items} films.")


def recommend(user_ratings: dict, n: int = 20) -> list[tuple[int, float]]:
    """Retourne top-n (movie_id, raw_score) via folding-in SVD++."""
    if _svd_model is None:
        return []

    ts = _svd_model.trainset
    mu = ts.global_mean

    # 1. Convertir raw movie_ids → inner_ids (ignorer les films inconnus)
    rated = []
    for raw_iid, r in user_ratings.items():
        try:
            inner = ts.to_inner_iid(raw_iid)
            rated.append((inner, r))
        except ValueError:
            pass

    if not rated:
        return []

    inner_ids = [x[0] for x in rated]
    ratings = np.array([x[1] for x in rated])

    # 2. Terme implicite SVD++ : |I_u|^{-0.5} * sum(yj for j in I_u)
    implicit = np.sum(_svd_model.yj[inner_ids], axis=0) / np.sqrt(len(inner_ids))

    # 3. Résidu pour le terme explicite
    Q_rated = _svd_model.qi[inner_ids]
    b_rated = _svd_model.bi[inner_ids]
    residuals = ratings - mu - b_rated

    # 4. Résoudre pour pu (folding-in ridge)
    n_factors = Q_rated.shape[1]
    A = Q_rated.T @ Q_rated + LAMBDA * np.eye(n_factors)
    bv = Q_rated.T @ residuals
    pu = np.linalg.solve(A, bv)

    # 5. Vecteur utilisateur effectif = terme explicite + terme implicite
    effective_pu = pu + implicit

    # 6. Scorer tous les films non notés
    rated_inner = set(inner_ids)
    all_scores = mu + _svd_model.bi + _svd_model.qi @ effective_pu
    candidates = [
        (int(ts.to_raw_iid(i)), float(np.clip(all_scores[i], 0.5, 5.0)))
        for i in range(ts.n_items) if i not in rated_inner
    ]

    # 7. Trier et retourner top-n
    candidates.sort(key=lambda x: -x[1])
    return candidates[:n]
