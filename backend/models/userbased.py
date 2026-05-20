"""
User-Based model — Personne 3 implémente ce fichier.

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

Stratégie — similarité MSD à la volée :
    1. Construire le vecteur du nouvel utilisateur aligné sur les inner_ids
       (utiliser _item_index pour la conversion raw_movieId → colonne).
    2. Pour chaque utilisateur existant u dans _rating_matrix :
         intersection = items notés par les deux
         si support == 0 → sim = 0
         sinon : msd = mean((r_new - r_u)²) sur l'intersection
                 sim = 1 / (msd + 1)
    3. Garder les K meilleurs voisins (K = 50 suggéré).
    4. Pour chaque film non noté j :
         pred = mean_new + Σ(sim_u * (r_uj - mean_u)) / Σ|sim_u|
    5. Clipper à [0.5, 5.0], trier décroissant, retourner top-n.

    Astuce performance : utiliser les opérations matricielles scipy sparse
    plutôt que densifier toute la matrice.

Génération des artefacts (à ajouter dans ton notebook) :
    import pickle, numpy as np
    from scipy.sparse import csr_matrix, save_npz

    ts = userbased_algo.trainset
    rows, cols, vals = [], [], []
    for u in range(ts.n_users):
        for iid, r in ts.ur[u]:
            rows.append(u); cols.append(iid); vals.append(r)
    R = csr_matrix((vals, (rows, cols)), shape=(ts.n_users, ts.n_items))
    save_npz("backend/artifacts/rating_matrix.npz", R)

    means = np.array([np.mean([r for _, r in ts.ur[u]]) for u in range(ts.n_users)])
    np.save("backend/artifacts/user_means.npy", means)

    with open("backend/artifacts/userbased_model.pkl", "wb") as f:
        pickle.dump(userbased_algo, f)
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
K_NEIGHBORS = 50          # nombre de voisins à considérer
MIN_ITEM_SUPPORT = 3      # un film doit avoir été noté par ≥ N voisins pour être recommandé
MIN_USER_SUPPORT = 2      # un voisin doit avoir noté ≥ N films en commun avec le nouvel utilisateur
SIG_WEIGHTING_CAP = 20    # plafond pour la pondération par significance
SHRINKAGE = 5.0           # régularisation : tire la prédiction vers new_mean quand peu de voisins soutiennent l'item

_userbased_model = None  # modèle entraîné (pour accéder au trainset)
_rating_matrix = None    # scipy.sparse.csr_matrix (n_users × n_items)
_user_means = None       # np.ndarray (n_users,)
_item_index: dict = {}   # {raw_movie_id (int) → inner_item_id (int)}


def load() -> None:
    """Charge les artefacts user-based depuis artifacts/."""
    global _userbased_model, _rating_matrix, _user_means, _item_index
    with open(ARTIFACTS_DIR / "userbased_model.pkl", "rb") as f:
        _userbased_model = pickle.load(f)
    _rating_matrix = load_npz(ARTIFACTS_DIR / "rating_matrix.npz")
    _user_means = np.load(ARTIFACTS_DIR / "user_means.npy")
    ts = _userbased_model.trainset
    _item_index = {int(ts.to_raw_iid(iid)): iid for iid in range(ts.n_items)}
    print(f"[userbased] {ts.n_users} utilisateurs, {ts.n_items} films chargés.")


def recommend(user_ratings: dict, n: int = 20) -> list[tuple[int, float]]:
    """Retourne top-n (movie_id, raw_score) via similarité MSD à la volée."""
    if _userbased_model is None or _rating_matrix is None:
        return []

    ts = _userbased_model.trainset
    n_items = ts.n_items

    # 1. Items du nouvel utilisateur (inner ids + ratings)
    new_inner = {_item_index[mid]: r for mid, r in user_ratings.items() if mid in _item_index}
    if not new_inner:
        return []

    inner_ids = np.array(list(new_inner.keys()), dtype=int)
    new_ratings = np.array([new_inner[i] for i in inner_ids], dtype=float)
    new_mean = float(np.mean(new_ratings))

    # 2. Similarité MSD — vectorisée sur les colonnes des items notés uniquement
    # sub shape : (n_users, n_rated_items) — peu de colonnes, extraction rapide
    sub = _rating_matrix[:, inner_ids].toarray().astype(float)  # (n_users, n_rated)
    rated_mask = sub != 0                                         # items notés par chaque user
    support = rated_mask.sum(axis=1)                              # (n_users,)

    # MSD pure (1/(msd+1)) pondérée par significance plafonnée pour rester
    # robuste : un voisin qui partage 1 seul film ne pèse pas autant qu'un voisin
    # qui en partage 5.
    sims = np.zeros(ts.n_users)
    valid = support >= MIN_USER_SUPPORT
    if valid.any():
        diff_sq = (sub - new_ratings) ** 2          # (n_users, n_rated)
        diff_sq[~rated_mask] = 0.0                  # ignorer items non notés
        msd = diff_sq.sum(axis=1) / np.where(support > 0, support, 1)
        msd_sim = 1.0 / (msd + 1.0)                                  # ∈ (0, 1]
        sig_weight = np.minimum(support, SIG_WEIGHTING_CAP) / SIG_WEIGHTING_CAP
        sims[valid] = msd_sim[valid] * sig_weight[valid]

    # 3. K meilleurs voisins (seulement ceux avec sim > 0)
    candidate_idx = np.where(sims > 0)[0]
    if candidate_idx.size == 0:
        return []
    order = candidate_idx[np.argsort(-sims[candidate_idx])][:K_NEIGHBORS]
    top_k = order
    top_k_sims = sims[top_k]

    # 4. Prédire — sous-matrice (K × n_items) des voisins
    nb_matrix = _rating_matrix[top_k].toarray().astype(float)   # (K, n_items)
    nb_means = _user_means[top_k]                                 # (K,)

    rated_inner = set(new_inner.keys())
    results = []
    for i in range(n_items):
        if i in rated_inner:
            continue
        col = nb_matrix[:, i]
        nb_mask = col != 0
        n_supporting = int(nb_mask.sum())
        # Exiger qu'un nombre minimum de voisins similaires aient noté l'item :
        # évite qu'une note isolée d'un voisin atypique propulse un film en top
        # (cause principale des recommandations « hors-sujet »).
        if n_supporting < MIN_ITEM_SUPPORT:
            continue
        sims_i = top_k_sims[nb_mask]
        denom = float(sims_i.sum())
        if denom == 0:
            continue
        num = float(np.dot(sims_i, col[nb_mask] - nb_means[nb_mask]))
        # Shrinkage : un denom faible (peu de voisins, faibles sims) tire
        # la prédiction vers new_mean au lieu de produire des écarts extrêmes
        # basés sur un seul voisin atypique.
        pred = new_mean + num / (denom + SHRINKAGE)
        results.append((int(ts.to_raw_iid(i)), pred, n_supporting))

    if not results:
        return []

    # Trier sur la prédiction brute (pas clippée) pour préserver l'ordre :
    # deux films à pred=6.2 et 5.4 ne sont pas équivalents même si tous deux
    # dépassent le plafond 5.0. Le n° de voisins départage les ex-aequo.
    results.sort(key=lambda x: (-x[1], -x[2]))
    top = [(mid, score) for mid, score, _ in results[:n]]

    # Normalisation min-max sur le top-n pour étaler le pourcentage affiché
    # de 10 % à 100 % (sinon tout est collé à 100 % car les bonnes prédictions
    # sautent souvent au plafond 5.0).
    raw_scores = [s for _, s in top]
    lo, hi = min(raw_scores), max(raw_scores)
    if hi > lo:
        spread = [(mid, 0.5 + 4.5 * (s - lo) / (hi - lo)) for mid, s in top]
    else:
        spread = [(mid, float(lo)) for mid, _ in top]
    return [(mid, round(float(np.clip(s, 0.5, 5.0)), 4)) for mid, s in spread]
