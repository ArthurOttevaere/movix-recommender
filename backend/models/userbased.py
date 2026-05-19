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
K_NEIGHBORS = 50  # nombre de voisins à considérer

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

    # TODO : implémenter la similarité MSD (voir docstring du fichier)
    return []
