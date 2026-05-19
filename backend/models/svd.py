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

    # TODO : implémenter le folding-in (voir docstring du fichier)
    return []
