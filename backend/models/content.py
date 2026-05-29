"""
Content-Based model — Personne 1 implémente ce fichier.

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

Stratégie suggérée :
    1. Filtrer user_ratings aux films présents dans content_features.index.
    2. Si moins de 2 films ont des features → retourner [].
    3. Construire X (n_rated × n_features) et y (n_rated,) depuis les ratings.
    4. Entraîner RidgeCV(alphas=[0.1, 1, 10, 100, 1000]) sur X, y.
    5. Scorer TOUS les films : scores = model.predict(content_features.values).
    6. Exclure les films déjà notés.
    7. Retourner top-n triés, scores clampés [0.5, 5.0].

Génération de l'artefact (à ajouter dans ton notebook) :
    import pickle
    with open("backend/artifacts/content_features.pkl", "wb") as f:
        pickle.dump(content_model.content_features, f)
    # content_model.content_features est le pd.DataFrame du modèle entraîné
"""

import pickle
from pathlib import Path
from sklearn.linear_model import RidgeCV

import numpy as np

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_content_features = None  # pd.DataFrame chargé par load()


def load() -> None:
    """Charge content_features.pkl depuis artifacts/."""
    global _content_features
    path = ARTIFACTS_DIR / "content_features.pkl"
    with open(path, "rb") as f:
        _content_features = pickle.load(f)
    print(f"[content] {len(_content_features)} films chargés.")


def similar_items(movie_id: int, n: int = 12) -> list[tuple[int, float]]:
    """Films les plus proches de `movie_id` dans l'espace de features content-based.

    Similarité cosinus entre le vecteur de features du film et tous les autres
    (les features content = genome/genres, ≥ 0 → cosinus dans [0, 1]). Sert la
    section « More Like This » du modal. Score encodé en [0.5, 5.0] pour que le
    normalize_score de l'API le remappe en % de match, comme les autres modèles.
    """
    if _content_features is None or n <= 0:
        return []
    mid = int(movie_id)
    if mid not in _content_features.index:
        return []

    X = _content_features.values
    idx = _content_features.index.get_loc(mid)
    target = X[idx]
    norms = np.linalg.norm(X, axis=1)
    denom = norms * (np.linalg.norm(target) + 1e-9) + 1e-9
    sims = (X @ target) / denom

    order = np.argsort(-sims)
    out: list[tuple[int, float]] = []
    for j in order:
        if j == idx:
            continue
        match = float(np.clip(sims[j], 0.0, 1.0))
        out.append((int(_content_features.index[j]), round(0.5 + match * 4.5, 4)))
        if len(out) >= n:
            break
    return out


def recommend(user_ratings: dict, n: int = 20) -> list[tuple[int, float]]:
    """Retourne top-n (movie_id, raw_score) pour un nouvel utilisateur."""

    if _content_features is None or n <= 0:
        return []

    clean_ratings = {}
    for movie_id, rating in user_ratings.items():
        try:
            clean_ratings[int(movie_id)] = float(rating)
        except (ValueError, TypeError):
            continue

    valid_ids = [
        movie_id
        for movie_id in clean_ratings
        if movie_id in _content_features.index
    ]

    if len(valid_ids) < 2:
        return []

    X_train = _content_features.loc[valid_ids].values
    y_train = np.array([clean_ratings[movie_id] for movie_id in valid_ids])

    model = RidgeCV(
        alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
        scoring="neg_root_mean_squared_error",
    )

    model.fit(X_train, y_train)

    rated_ids = set(clean_ratings.keys())

    candidate_ids = [
        movie_id
        for movie_id in _content_features.index
        if int(movie_id) not in rated_ids
    ]

    if not candidate_ids:
        return []

    X_candidates = _content_features.loc[candidate_ids].values
    scores = np.clip(model.predict(X_candidates), 0.5, 5.0)

    order = np.argsort(-scores)

    return [
        (int(candidate_ids[i]), float(scores[i]))
        for i in order[:n]
    ]
