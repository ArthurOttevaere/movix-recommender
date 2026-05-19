# Guide d'intégration — Ajouter ton modèle au site Cinematch

Ce guide s'adresse aux trois personnes qui vont chacune implémenter un modèle de recommandation dans le backend FastAPI. Lis-le en entier avant de commencer.

---

## Vue d'ensemble

L'application fonctionne ainsi :

```
Navigateur (frontend HTML/JS)
        │  appels HTTP
        ▼
Serveur FastAPI  ←──  tes modèles chargés en mémoire
        │
        ▼
  backend/artifacts/  ←──  fichiers .pkl/.npz générés depuis ton notebook
```

Chacun de vous travaille sur **une seule branche** et **un seul fichier** :

| Personne | Branche | Fichier à implémenter |
|----------|---------|----------------------|
| 1 — Content-Based | `feature/model-content` | `backend/models/content.py` |
| 2 — SVD | `feature/model-svd` | `backend/models/svd.py` |
| 3 — User-Based | `feature/model-userbased` | `backend/models/userbased.py` |

Les fichiers `main.py`, `store.py` et `utils.py` sont **gelés** — ne pas les modifier.

---

## Étape 1 — Créer ta branche

Depuis la branche `integration` :

```bash
git checkout integration
git pull origin integration
git checkout -b feature/model-content   # adapter selon ton modèle
```

---

## Étape 2 — Installer les dépendances

```bash
pip install -r requirements.txt
```

Les dépendances importantes pour le backend : `fastapi`, `uvicorn`, `scipy`.

---

## Étape 3 — Générer les artefacts partagés

Ces fichiers sont nécessaires pour que le serveur démarre. **Une seule personne le fait**, commit le script (pas les CSV — ils sont gitignorés), et les autres le lancent aussi en local.

```bash
python backend/generate_artifacts.py
```

Résultat dans `backend/artifacts/` :
- `movies.csv` — titre et genres de chaque film
- `links.csv` — correspondance `movieId` ↔ `tmdbId`
- `popularity.csv` — nombre de ratings par film (pour le fallback)

> Si tu vois `Dossier data/ introuvable`, lance d'abord `python unzip_data.py`.

---

## Étape 4 — Comprendre l'interface à respecter

Ton fichier modèle doit exposer exactement deux fonctions :

```python
def load() -> None:
    """Charge tes artefacts (.pkl, .npz) depuis backend/artifacts/."""

def recommend(user_ratings: dict[int, float], n: int) -> list[tuple[int, float]]:
    """
    Paramètres :
        user_ratings  — {movie_id (int): rating (float, entre 0.5 et 5.0)}
                        Ce sont les films notés par l'utilisateur pendant l'onboarding.
        n             — nombre de recommandations à retourner

    Retourne :
        Liste de tuples (movie_id, score) triée par score DÉCROISSANT.
        Les scores doivent être dans l'intervalle [0.5, 5.0].
        Si tu ne peux pas produire de recommandations → retourner [].
        Le serveur utilisera alors les films les plus populaires comme fallback.
    """
```

Le serveur appelle `load()` une seule fois au démarrage, puis appelle `recommend()` à chaque requête utilisateur.

**Important :** l'utilisateur est toujours un **nouvel utilisateur** qui n'est pas dans le trainset. Il a noté au minimum 5 films pendant l'onboarding. Ton modèle doit s'adapter à ça.

---

## Étape 5 — Implémenter ton modèle

### Personne 1 — Content-Based (`backend/models/content.py`)

**Stratégie :** entraîner un `RidgeCV` à la volée sur les 5+ films notés par l'utilisateur, en utilisant les features de contenu pré-chargées.

**Artefact à générer dans ton notebook :**

```python
import pickle

# content_model est ton ContentBased entraîné sur le trainset complet
# (utilise build_full_trainset())
content_features = content_model.content_features  # pd.DataFrame

with open("backend/artifacts/content_features.pkl", "wb") as f:
    pickle.dump(content_features, f)

print(f"Sauvegardé : {content_features.shape}")
# Ex : (9742, 2800) → 9742 films, 2800 features
```

**Logique de `recommend()` à implémenter :**

```python
from sklearn.linear_model import RidgeCV
import numpy as np

def recommend(user_ratings, n=20):
    # 1. Garder uniquement les films qui ont des features connues
    valid_ids = [mid for mid in user_ratings if mid in _content_features.index]
    if len(valid_ids) < 2:
        return []  # pas assez de données

    # 2. Construire X et y depuis les films notés
    X_train = _content_features.loc[valid_ids].values
    y_train = np.array([user_ratings[mid] for mid in valid_ids])

    # 3. Entraîner RidgeCV
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
                    scoring="neg_root_mean_squared_error")
    model.fit(X_train, y_train)

    # 4. Scorer tous les films non encore notés
    candidate_ids = [mid for mid in _content_features.index if mid not in user_ratings]
    X_all = _content_features.loc[candidate_ids].values
    scores = np.clip(model.predict(X_all), 0.5, 5.0)

    # 5. Trier et retourner top-n
    order = np.argsort(-scores)
    return [(int(candidate_ids[i]), float(scores[i])) for i in order[:n]]
```

---

### Personne 2 — SVD (`backend/models/svd.py`)

**Stratégie :** folding-in — résoudre pour les facteurs latents du nouvel utilisateur en utilisant les facteurs items déjà appris.

**Artefact à générer dans ton notebook :**

```python
import pickle
from models import ModelBaseline4
from loaders import load_ratings

# Entraîner SVD sur le trainset complet
sp_ratings = load_ratings(surprise_format=True)
trainset = sp_ratings.build_full_trainset()

svd_algo = ModelBaseline4()
svd_algo.fit(trainset)

with open("backend/artifacts/svd_model.pkl", "wb") as f:
    pickle.dump(svd_algo, f)

print("SVD sauvegardé.")
print(f"  n_factors : {svd_algo.n_factors}")
print(f"  n_items   : {trainset.n_items}")
```

**Logique de `recommend()` — le folding-in :**

Après entraînement, le modèle SVD contient :
- `_svd_model.qi` → matrice des facteurs items `(n_items × n_factors)`
- `_svd_model.bi` → biais par item `(n_items,)`
- `_svd_model.trainset.global_mean` → moyenne globale

```python
import numpy as np

LAMBDA = 0.1  # régularisation

def recommend(user_ratings, n=20):
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
    ratings   = np.array([x[1] for x in rated])

    # 2. Récupérer les facteurs et biais des films notés
    Q_rated   = _svd_model.qi[inner_ids]        # (k_rated × n_factors)
    b_rated   = _svd_model.bi[inner_ids]        # (k_rated,)
    residuals = ratings - mu - b_rated          # ce qu'il reste à expliquer

    # 3. Résoudre pour pu (folding-in)
    n_factors = Q_rated.shape[1]
    A  = Q_rated.T @ Q_rated + LAMBDA * np.eye(n_factors)
    bv = Q_rated.T @ residuals
    pu = np.linalg.solve(A, bv)                # (n_factors,)

    # 4. Scorer tous les films non notés
    rated_inner = set(inner_ids)
    all_scores  = mu + _svd_model.bi + _svd_model.qi @ pu  # (n_items,)
    candidates  = [
        (int(ts.to_raw_iid(i)), float(np.clip(all_scores[i], 0.5, 5.0)))
        for i in range(ts.n_items) if i not in rated_inner
    ]

    # 5. Trier et retourner top-n
    candidates.sort(key=lambda x: -x[1])
    return candidates[:n]
```

---

### Personne 3 — User-Based (`backend/models/userbased.py`)

**Stratégie :** calculer la similarité MSD entre le nouvel utilisateur et tous les utilisateurs existants, puis faire une prédiction par moyenne pondérée.

**Artefacts à générer dans ton notebook :**

```python
import pickle
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from models import UserBased
from loaders import load_ratings

# Entraîner UserBased sur le trainset complet
sp_ratings = load_ratings(surprise_format=True)
trainset   = sp_ratings.build_full_trainset()

ub_algo = UserBased(k=20, min_k=1, sim_options={'name': 'msd', 'min_support': 1})
ub_algo.fit(trainset)

# 1. Matrice de ratings sparse (n_users × n_items)
rows, cols, vals = [], [], []
for u in range(trainset.n_users):
    for iid, r in trainset.ur[u]:
        rows.append(u); cols.append(iid); vals.append(r)
R = csr_matrix((vals, (rows, cols)), shape=(trainset.n_users, trainset.n_items))
save_npz("backend/artifacts/rating_matrix.npz", R)

# 2. Moyenne de rating par utilisateur
means = np.array([
    np.mean([r for _, r in trainset.ur[u]]) for u in range(trainset.n_users)
])
np.save("backend/artifacts/user_means.npy", means)

# 3. Modèle (pour accéder au trainset)
with open("backend/artifacts/userbased_model.pkl", "wb") as f:
    pickle.dump(ub_algo, f)

print(f"Sauvegardé : {trainset.n_users} users, {trainset.n_items} items")
```

**Logique de `recommend()` :**

```python
import numpy as np

K_NEIGHBORS = 50

def recommend(user_ratings, n=20):
    ts = _userbased_model.trainset
    mu = ts.global_mean
    n_items = ts.n_items

    # 1. Construire le vecteur de ratings du nouvel utilisateur (inner_ids)
    new_vec = np.full(n_items, np.nan)
    for raw_iid, r in user_ratings.items():
        if raw_iid in _item_index:
            new_vec[_item_index[raw_iid]] = r

    new_mean = float(np.nanmean(new_vec)) if not np.all(np.isnan(new_vec)) else mu

    # 2. Calculer la similarité MSD avec chaque utilisateur existant
    sims = np.zeros(ts.n_users)
    for u in range(ts.n_users):
        u_vec = _rating_matrix[u].toarray().flatten().astype(float)
        u_vec[u_vec == 0] = np.nan
        mask    = ~np.isnan(new_vec) & ~np.isnan(u_vec)
        support = int(mask.sum())
        if support == 0:
            continue
        msd       = float(np.sum((new_vec[mask] - u_vec[mask]) ** 2)) / support
        sims[u]   = 1.0 / (msd + 1.0)

    # 3. Garder les K meilleurs voisins
    top_k = set(np.argsort(-sims)[:K_NEIGHBORS])

    # 4. Prédire pour chaque film non noté
    rated_inner = {_item_index[mid] for mid in user_ratings if mid in _item_index}
    results = []
    for i in range(n_items):
        if i in rated_inner:
            continue
        neighbors = [(u, sims[u]) for u in top_k if _rating_matrix[u, i] != 0]
        if not neighbors:
            continue
        num   = sum(sim * (_rating_matrix[u, i] - _user_means[u]) for u, sim in neighbors)
        denom = sum(sim for _, sim in neighbors)
        est   = float(np.clip(new_mean + (num / denom if denom > 0 else 0), 0.5, 5.0))
        results.append((int(ts.to_raw_iid(i)), est))

    results.sort(key=lambda x: -x[1])
    return results[:n]
```

---

## Étape 6 — Tester son modèle en isolation

Avant de lancer le serveur complet, vérifie que ton modèle fonctionne :

```bash
# Depuis la racine du repo
python - <<'EOF'
from backend.models.content import load, recommend   # adapter selon ton modèle

load()
# 5 films notés (simulation de l'onboarding)
ratings = {1: 4.0, 2: 3.5, 3: 5.0, 4: 3.0, 5: 4.5}
recs = recommend(ratings, n=10)
print(f"{len(recs)} recommandations reçues")
for movie_id, score in recs[:5]:
    print(f"  movie_id={movie_id}  score={score:.3f}")
EOF
```

Résultat attendu :
```
[content] 9742 films chargés.
10 recommandations reçues
  movie_id=318   score=4.712
  movie_id=527   score=4.689
  ...
```

Si tu vois `0 recommandations reçues` : vérifie que les `movie_id` du dict de test correspondent bien à des films présents dans ton artefact.

---

## Étape 7 — Lancer le serveur complet

```bash
uvicorn backend.main:app --reload --port 8000
```

Au démarrage, tu dois voir quelque chose comme :

```
[utils] 9742 films, 9734 liens TMDB chargés.
[content] 9742 films chargés.
[svd] WARNING: artefact manquant — ...   ← normal si l'autre personne n'a pas encore livré
[userbased] WARNING: artefact manquant — ...
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Ouvre ensuite **http://localhost:8000** dans ton navigateur. Le frontend se charge directement.

### Tester les endpoints manuellement

La doc interactive est disponible sur **http://localhost:8000/docs**.

Ou en ligne de commande :

```bash
# 1. Films pour l'onboarding
curl http://localhost:8000/onboarding/movies

# 2. Soumettre des ratings (simule l'onboarding)
curl -X POST http://localhost:8000/onboarding/submit \
  -H "Content-Type: application/json" \
  -d '{"ratings": {"1": 4.0, "2": 3.5, "3": 5.0, "4": 3.0, "5": 4.5}}'
# → {"user_token": "AbCdEfGh...", "status": "ok"}

# 3. Récupérer les recommandations (remplace TOKEN)
curl http://localhost:8000/recommendations/AbCdEfGh...
# → {"hero": {...}, "carousels": [...]}
```

---

## Étape 8 — Faire sa Pull Request

Une fois ton modèle testé :

```bash
git add backend/models/content.py   # seulement ton fichier
git commit -m "feat: implement content-based recommend()"
git push origin feature/model-content
```

Puis ouvre une **Pull Request vers `integration`** (pas vers `main`).

**Checklist avant de soumettre la PR :**

- [ ] `load()` ne plante pas si l'artefact est présent
- [ ] `recommend()` retourne une liste non vide avec 5+ ratings en entrée
- [ ] Les scores retournés sont bien dans `[0.5, 5.0]`
- [ ] Les `movie_id` retournés sont des `int` (pas des `numpy.int64`)
- [ ] Le serveur démarre sans erreur avec ton artefact
- [ ] Le carousel correspondant s'affiche dans le frontend

---

## Structure des dossiers

```
backend/
├── main.py                  ← NE PAS MODIFIER
├── store.py                 ← NE PAS MODIFIER
├── utils.py                 ← NE PAS MODIFIER
├── generate_artifacts.py    ← lancer une seule fois
├── models/
│   ├── __init__.py          ← NE PAS MODIFIER
│   ├── content.py           ← Personne 1 implémente ici
│   ├── svd.py               ← Personne 2 implémente ici
│   └── userbased.py         ← Personne 3 implémente ici
└── artifacts/               ← gitignorés, générés en local
    ├── movies.csv            (generate_artifacts.py)
    ├── links.csv             (generate_artifacts.py)
    ├── popularity.csv        (generate_artifacts.py)
    ├── content_features.pkl  (Personne 1, depuis notebook)
    ├── svd_model.pkl         (Personne 2, depuis notebook)
    ├── userbased_model.pkl   (Personne 3, depuis notebook)
    ├── rating_matrix.npz     (Personne 3, depuis notebook)
    └── user_means.npy        (Personne 3, depuis notebook)
```

---

## Questions fréquentes

**Mon `load()` plante avec `FileNotFoundError`.**
→ Tu n'as pas encore généré tes artefacts. Lance la cellule de sauvegarde dans ton notebook (voir Étape 5).

**`recommend()` retourne `[]` alors que j'ai des artefacts.**
→ Les `movie_id` du dict de test ne correspondent peut-être pas à ceux de ton trainset. Vérifie avec `ts.to_inner_iid(movie_id)` — si ça lève `ValueError`, le film n'est pas dans le trainset.

**Le serveur démarre mais le carousel de mon modèle est vide.**
→ `recommend()` retourne `[]` → le serveur utilise le fallback popularité. Teste ton modèle en isolation (Étape 6) pour diagnostiquer.

**J'ai une erreur `ModuleNotFoundError: No module named 'backend'`.**
→ Lance `uvicorn` depuis la racine du repo, pas depuis le dossier `backend/`.

**Puis-je modifier `main.py` pour ajouter un carousel ?**
→ Non. Si tu veux un carousel supplémentaire (ex : genre spécifique), discutes-en avec le responsable de l'intégration.
