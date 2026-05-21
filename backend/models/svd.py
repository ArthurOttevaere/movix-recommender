"""
SVD model — Personne 2 implémente ce fichier.

Interface à respecter :
    load() → None
    recommend(user_ratings, n) → list[tuple[int, float]]

Entrées de recommend() :
    user_ratings : dict {movie_id (int) → rating (float, 0.5–5.0)}
    n            : nombre de recommandations à retourner

Sortie de recommend() :
    Liste de (movie_id, raw_score) triée par score décroissant.
    Les scores doivent être clampés dans [0.5, 5.0].
    Retourner [] si impossible (artefact manquant, pas assez de données).

Stratégie — Folding-in :
    Après entraînement, le modèle SVD (ModelBaseline4) contient :
      _svd_model.qi      → facteurs items  (n_items × n_factors)
      _svd_model.bi      → biais items     (n_items,)
      _svd_model.trainset.global_mean → moyenne globale

    Pour un nouvel utilisateur avec ratings {iid: r} :
    1. Convertir raw movie_ids → inner_ids via trainset.to_inner_iid() (skip les inconnus).
    2. Collecter Q_rated = qi[inner_ids] et b_rated = bi[inner_ids].
    3. Résidu : residuals = ratings - global_mean - b_rated
    4. Résoudre pour pu (folding-in, ridge) :
         A = Q_rated.T @ Q_rated + lambda * I   (lambda ≈ 0.1)
         b = Q_rated.T @ residuals
         pu = np.linalg.solve(A, b)
    5. Scorer tous les films non notés :
         score_i = global_mean + bi[i] + pu · qi[i]
    6. Clipper à [0.5, 5.0], trier décroissant, retourner top-n.

Génération de l'artefact (à ajouter dans ton notebook) :
    import pickle
    with open("backend/artifacts/svd_model.pkl", "wb") as f:
        pickle.dump(svd_algo, f)
    # svd_algo est le ModelBaseline4 entraîné sur le trainset complet
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
    """Retourne top-n (movie_id, raw_score) via folding-in SVD."""
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

    # 2. Récupérer les facteurs et biais des films notés
    Q_rated = _svd_model.qi[inner_ids]       # (k_rated × n_factors)
    b_rated = _svd_model.bi[inner_ids]       # (k_rated,)
    residuals = ratings - mu - b_rated

    # 3. Résoudre pour pu (folding-in ridge)
    n_factors = Q_rated.shape[1]
    A = Q_rated.T @ Q_rated + LAMBDA * np.eye(n_factors)
    bv = Q_rated.T @ residuals
    pu = np.linalg.solve(A, bv)             # (n_factors,)

    # 4. Scorer tous les films non notés
    rated_inner = set(inner_ids)
    all_scores = mu + _svd_model.bi + _svd_model.qi @ pu  # (n_items,)
    candidates = [
        (int(ts.to_raw_iid(i)), float(np.clip(all_scores[i], 0.5, 5.0)))
        for i in range(ts.n_items) if i not in rated_inner
    ]

    # 5. Trier et retourner top-n
    candidates.sort(key=lambda x: -x[1])
    return candidates[:n]
