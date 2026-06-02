"""
User-Based model — Modèle KNNBaseline (Pearson Baseline) à la volée.

Interface à respecter :
    load() → None
    recommend(user_ratings, n) → list[tuple[int, float]]

Stratégie — Pearson Baseline dynamique :
    1. Charger les biais précalculés par Surprise (mu, bu, bi).
    2. Estimer le biais du nouvel utilisateur (b_new) selon la formule ALS de Surprise.
    3. Calculer les déviations (note_réelle - baseline_estimée) pour le nouvel utilisateur.
    4. Trouver l'intersection avec la matrice backend et calculer les déviations des voisins.
    5. Appliquer la similarité de Pearson avec le Shrinkage standard (100).
    6. Prédire les notes manquantes en pondérant les déviations des K meilleurs voisins.

Génération des artefacts (à ajouter dans ton notebook) :
    import pickle
    from scipy.sparse import csr_matrix, save_npz
    from surprise import Dataset, Reader, KNNBaseline
    from loaders import load_ratings
    from constants import Constant as C

    df = load_ratings(surprise_format=False)
    reader = Reader(rating_scale=C.RATINGS_SCALE)
    data = Dataset.load_from_df(df[[C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]], reader)
    trainset = data.build_full_trainset()

    # Le modèle exact validé lors de ton évaluation Two-Stage
    ub_algo = KNNBaseline(k=40, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True, 'min_support': 3})
    ub_algo.fit(trainset)

    # 1. Sauvegarde de la matrice sparse pour les calculs rapides
    ts = ub_algo.trainset
    rows, cols, vals = [], [], []
    for u in range(ts.n_users):
        for iid, r in ts.ur[u]:
            rows.append(u); cols.append(iid); vals.append(r)
    R = csr_matrix((vals, (rows, cols)), shape=(ts.n_users, ts.n_items))
    save_npz("backend/artifacts/rating_matrix.npz", R)

    # 2. Sauvegarde du modèle (qui contient mu, bu, et bi)
    with open("backend/artifacts/userbased_model.pkl", "wb") as f:
        pickle.dump(ub_algo, f)

    print(f"Artefacts générés : {ts.n_users} users, {ts.n_items} items")
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
K_NEIGHBORS = 40          # k (le voisinage max)
MIN_ITEM_SUPPORT = 2      # min_k (combien de voisins doivent avoir vu le film)
MIN_USER_SUPPORT = 3      # min_support (intersection minimale avec un voisin)
SHRINKAGE = 100.0         # Régularisation standard de Pearson dans Surprise
REG_U = 15.0              # Régularisation ALS pour estimer le biais du nouvel utilisateur

# Reranking popularité — identique au pipeline d'évaluation (models.get_top_n_reranked,
# configs.UserBased_Pearson_Natif: rerank_popularity=True, rerank_alpha=0.1).
# score_classement = note_prédite + alpha * log1p(popularité_item).
RERANK_POPULARITY = True
RERANK_ALPHA = 0.5

_userbased_model = None
_rating_matrix = None
_mu = 0.0                # Global mean
_bu = None               # User biases (backend)
_bi = None               # Item biases (backend)
_item_index: dict = {}
_item_logpop = None      # log1p(nb users ayant noté l'item), indexé par inner_id


def load() -> None:
    """Charge les artefacts user-based depuis artifacts/."""
    global _userbased_model, _rating_matrix, _mu, _bu, _bi, _item_index, _item_logpop
    with open(ARTIFACTS_DIR / "userbased_model.pkl", "rb") as f:
        _userbased_model = pickle.load(f)

    _rating_matrix = load_npz(ARTIFACTS_DIR / "rating_matrix.npz")

    # Extraction directe des biais calculés par le C++ de Surprise
    ts = _userbased_model.trainset
    _mu = ts.global_mean
    _bu = _userbased_model.bu
    _bi = _userbased_model.bi

    _item_index = {int(ts.to_raw_iid(iid)): iid for iid in range(ts.n_items)}

    # Popularité = nb d'utilisateurs ayant noté chaque item (= item_freq de l'éval),
    # précalculée en log1p pour le reranking. Indexé par inner_id.
    item_pop = np.asarray((_rating_matrix > 0).sum(axis=0)).ravel().astype(float)
    _item_logpop = np.log1p(item_pop)
    print(f"[userbased_pearson] {_rating_matrix.shape[0]} utilisateurs, {_rating_matrix.shape[1]} films chargés.")


def recommend(user_ratings: dict, n: int = 20) -> list[tuple[int, float]]:
    """Retourne top-n (movie_id, raw_score) via similarité Pearson Baseline à la volée."""
    if _userbased_model is None or _rating_matrix is None:
        return []

    ts = _userbased_model.trainset
    n_items = ts.n_items

    # 1. Extraction des items connus par l'algo
    new_inner = {_item_index[mid]: r for mid, r in user_ratings.items() if mid in _item_index}
    if not new_inner:
        return []

    inner_ids = np.array(list(new_inner.keys()), dtype=int)
    new_ratings = np.array([new_inner[i] for i in inner_ids], dtype=float)
    
    # 2. Calcul du biais du nouvel utilisateur (b_new) via l'équation ALS
    b_i_new = _bi[inner_ids]
    b_new = np.sum(new_ratings - _mu - b_i_new) / (len(inner_ids) + REG_U)
    
    # Déviations du nouvel utilisateur par rapport à sa propre baseline
    new_devs = new_ratings - (_mu + b_new + b_i_new)
    norm_new = np.linalg.norm(new_devs)

    # 3. Calcul de la similarité de Pearson avec tous les utilisateurs existants
    sub = _rating_matrix[:, inner_ids].toarray().astype(float)
    mask = sub > 0
    support = mask.sum(axis=1)
    
    valid = support >= MIN_USER_SUPPORT
    sims = np.zeros(ts.n_users)

    if valid.any() and norm_new > 0:
        sub_devs = np.zeros_like(sub)
        
        # Calcul de la baseline des voisins (broadcast optimisé)
        baseline_v = _mu + _bu[:, None] + b_i_new[None, :]
        sub_devs[mask] = sub[mask] - baseline_v[mask]
        
        # Pearson numérateur et dénominateur
        dot_products = np.dot(sub_devs, new_devs)
        norm_users = np.sqrt((sub_devs ** 2).sum(axis=1))
        
        denominators = norm_users * norm_new
        raw_sims = np.where(denominators > 0, dot_products / denominators, 0.0)
        
        # Application du Shrinkage de Pearson (Surprise Formula)
        shrunk_sims = raw_sims * (support - 1) / (support - 1 + SHRINKAGE)
        sims[valid] = shrunk_sims[valid]

    # 4. Sélection des K meilleurs voisins (seulement les corrélations positives)
    candidate_idx = np.where(sims > 0)[0]
    if candidate_idx.size == 0:
        return []
    
    order = candidate_idx[np.argsort(-sims[candidate_idx])][:K_NEIGHBORS]
    top_k = order
    
    nb_matrix = _rating_matrix[top_k].toarray().astype(float)
    sims_k = sims[top_k]
    bu_k = _bu[top_k]

    # 5. Prédiction des items non notés
    rated_inner = set(new_inner.keys())
    results = []
    
    for i in range(n_items):
        if i in rated_inner:
            continue
            
        col = nb_matrix[:, i]
        nb_mask = col > 0
        n_supp = nb_mask.sum()
        
        if n_supp < MIN_ITEM_SUPPORT:
            continue
            
        sims_active = sims_k[nb_mask]
        denom = np.abs(sims_active).sum()
        if denom == 0:
            continue
            
        # On pondère la déviation des voisins (r_vi - baseline_vi) par la similarité
        dev_vj = col[nb_mask] - (_mu + bu_k[nb_mask] + _bi[i])
        num = np.dot(sims_active, dev_vj)
        
        # Note = Baseline_nouvel_utilisateur + déviation pondérée
        pred = _mu + b_new + _bi[i] + (num / denom)

        # Score de classement = reranking popularité (= models.get_top_n_reranked).
        # On le sépare de la note affichée : il sert UNIQUEMENT à ordonner le carrousel,
        # exactement comme dans le pipeline d'évaluation où rerank_popularity=True.
        rank_score = pred
        if RERANK_POPULARITY and _item_logpop is not None:
            rank_score = pred + RERANK_ALPHA * _item_logpop[i]

        results.append((int(ts.to_raw_iid(i)), pred, n_supp, rank_score))

    if not results:
        return []

    # 6. Tri (score de reranking desc, puis support) et sortie.
    #    On TRIE par le score reranké (note + alpha*log popularité) pour aligner
    #    l'ordre du carrousel sur le modèle évalué (UserBased_Pearson_Natif,
    #    rerank_alpha=0.1), mais on RENVOIE la NOTE PRÉDITE clampée [0.5, 5.0] —
    #    comme content.py. Ainsi le "% match" vert reste une estimation
    #    absolue du goût (cf. green-match-score) et n'est pas gonflé par la popularité.
    results.sort(key=lambda x: (-x[3], -x[2]))
    return [
        (mid, round(float(np.clip(score, 0.5, 5.0)), 4))
        for mid, score, _, _ in results[:n]
    ]