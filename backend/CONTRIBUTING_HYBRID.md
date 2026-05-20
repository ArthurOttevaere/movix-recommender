# Intégration du modèle hybride — Guide partenaire

Ce backend FastAPI sert les recommandations au frontend Movix. **Tout est déjà
câblé** (routes, schémas, ranker, MMR, filtres, enrichissement, profils,
watchlist, onboarding). Ton seul travail : produire le **modèle hybride**.

## Architecture en une image

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Frontend (vanilla JS)                                                   │
│  ─ POST /onboarding/submit (>=5 ratings)                                 │
│  ─ GET  /recommendations/{token}     ← rendu Netflix-like                │
└──────────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ JSON
┌──────────────────────────────────────────────────────────────────────────┐
│  FastAPI app (backend/app/main.py)                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Orchestrator (services/orchestrator.py)                            │  │
│  │   ─ exclut les films déjà notés                                    │  │
│  │   ─ APPEL UNIQUE → recommender.score_candidates(user_ratings, …)   │  │
│  │   ─ normalize_scores → topk → MMR (diversité) → filtre genre       │  │
│  │   ─ enrichit (titre, year, tmdb_id, genres, watchlist…)            │  │
│  │   ─ assemble 8 carrousels + hero                                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Recommender package (backend/app/recommender/)                    │  │
│  │   ─ base.py        ← TON contrat (HybridRecommender ABC)           │  │
│  │   ─ dummy.py       ← fallback popularité (à ne PAS utiliser en   │  │
│  │                       prod, juste pour avoir un serveur qui tourne)│  │
│  │   ─ loader.py      ← charge artifacts/hybrid_model.pkl ou fallback │  │
│  │   ─ ranker.py      ← normalize, topk, MMR  ← NE TOUCHE PAS         │  │
│  │   ─ filters.py     ← exclude_ids…           ← NE TOUCHE PAS        │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Ce que tu reçois

- Le package `backend.app.recommender` avec :
  - `HybridRecommender` (classe abstraite à hériter)
  - `DummyHybrid` (exemple de référence, popularité pure — pas un template)
- Les datasets :
  - `data/hackathon/evidence/ratings.csv` (colonnes : userId, movieId, rating, timestamp)
  - `data/hackathon/content/movies.csv` (colonnes : movieId, title, genres)
  - `data/hackathon/content/links.csv` (movieId → tmdbId) — déjà géré par le backend
  - `data/hackathon/content/genome-scores.csv`, `tmdb_cache.json`, `tmdb_embeddings.npz` (utile si tu veux content-based avancé)
- La référence : `models.py` à la racine, qui contient les implémentations
  content-based / user-based / SVD du cours. **Tu peux t'en inspirer** mais
  ne le réimporte pas tel quel — réécris une version propre dans ton fichier.

## Ce que tu dois produire

1. **Un fichier** `backend/app/recommender/impl_<nom>.py` contenant **une seule
   classe** qui hérite de `HybridRecommender`.
2. **Un pickle** `backend/artifacts/hybrid_model.pkl` contenant une instance
   déjà entraînée de cette classe.

Le serveur le charge automatiquement au démarrage. Si le pickle est absent ou
invalide, il tombe en fallback sur `DummyHybrid` (popularité) et logue un
warning — donc tu peux travailler en parallèle, le frontend ne casse pas.

## Surface obligatoire

Lis le contrat complet dans `backend/app/recommender/base.py`. Résumé :

```python
class HybridRecommender(ABC):
    name: str = "hybrid"
    version: str = "0.1.0"

    @abstractmethod
    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "HybridRecommender":
        """Entraînement hors-ligne. Retourne self (picklable après)."""

    @abstractmethod
    def score_candidates(
        self,
        user_ratings: Mapping[int, float],
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[int, float]:
        """Inférence : {movieId: score brut} — plus haut = mieux."""

    def recommend_similar(self, movie_id: int, k: int = 20) -> Mapping[int, float]:
        """OPTIONNEL — utilisé pour la sélection du hero."""
        return {}
```

## Garanties que le backend te donne

- `user_ratings` aura **toujours ≥ 5 entrées**, ratings dans `[0.5, 5.0]`,
  `movieId` = MovieLens ID.
- `candidate_ids` **ne contient pas** les films déjà notés par l'utilisateur.
- **Tu ne normalises pas tes scores** : le backend fait min-max → [0,1].
- **Tu ne diversifies pas** : le backend applique MMR pour le carrousel
  "Découverte".
- **Tu ne filtres pas par genre** : le backend croise tes scores avec les sets
  de genres pour les carrousels "Top Drame", "Sci-Fi", etc.
- **Tu n'enrichis pas** : le backend ajoute titre, year, tmdb_id, poster, etc.
- **Tu ne tries pas** : le backend fait topK.
- `fit()` est appelé **une seule fois**, hors-ligne. À l'inférence seul
  `score_candidates()` est appelé.

## Ce que tu NE DOIS PAS faire

- Toucher à `ranker.py`, `filters.py`, `schemas.py`, `orchestrator.py`, ou
  aux routers : tu casses la liaison frontend.
- Mettre `print()` partout : utilise `logging.getLogger(__name__)`.
- Faire des chemins absolus dans `__init__` ou dans le pickle.
- Sérialiser des `lambda` ou des fonctions locales comme attributs (pickle
  cassera).
- Ouvrir des fichiers / connexions au moment du pickle (la déserialisation
  doit fonctionner sans réseau).

## Règles du pickle

- **Une seule instance** de ta sous-classe.
- Attributs autorisés : numpy arrays, pandas DataFrame/Series, dict, set, list,
  `surprise.AlgoBase` (fit), `sklearn` (fit), tes propres dataclasses
  picklables.
- Ta classe doit être **importable** par le serveur. Place-la dans
  `backend/app/recommender/impl_<nom>.py`.

## Test local — workflow complet

```bash
# 1. Installer
pip install -r backend/requirements.txt

# 2. (Sans ton modèle) Le serveur tourne déjà avec DummyHybrid
uvicorn backend.app.main:app --reload --port 8000
curl localhost:8000/healthz
# → {"status":"ok","model":"dummy_popularity","version":"0.0.1","catalog_size":8737}

# 3. Entraîner ton modèle et sauver le pickle
#    (édite scripts/train_hybrid.py pour importer + instancier ta classe)
python -m backend.scripts.train_hybrid

# 4. Relancer uvicorn (Ctrl+C puis up-arrow)
# /healthz doit maintenant reporter TON name + version
curl localhost:8000/healthz
# → {"status":"ok","model":"ton_nom","version":"0.1.0", ...}

# 5. Lancer la suite de tests
pytest backend/tests -v
```

Si les tests `test_contract.py` passent avec ton pickle en place, la liaison
frontend est garantie de fonctionner.

## Template minimal — copie-colle pour démarrer

`backend/app/recommender/impl_svd.py` :

```python
"""Exemple simple : SVD Surprise + repli popularité.
Le partenaire est ENCOURAGÉ à faire mieux (hybride content+user+latent)."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

from .base import HybridRecommender


class SvdHybrid(HybridRecommender):
    name = "svd_baseline"
    version = "0.1.0"

    def __init__(self, n_factors: int = 50, random_state: int = 0):
        self._n_factors = n_factors
        self._random_state = random_state
        self._svd: SVD | None = None
        self._pop_mean: dict[int, float] = {}
        self._all_movies: list[int] = []

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "SvdHybrid":
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
        self._svd = SVD(n_factors=self._n_factors, random_state=self._random_state)
        self._svd.fit(data.build_full_trainset())
        self._pop_mean = ratings.groupby("movieId")["rating"].mean().to_dict()
        self._all_movies = movies["movieId"].tolist()
        return self

    def score_candidates(
        self,
        user_ratings: Mapping[int, float],
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[int, float]:
        assert self._svd is not None
        ids = list(candidate_ids) if candidate_ids is not None else self._all_movies
        # SVD cold-start: on prend un "pseudo user" qui matche la moyenne des notes hautes
        user_mean = float(np.mean(list(user_ratings.values())))
        # Fold-in approximatif : moyenne des facteurs des films notés (>=4 stars)
        # — c'est un placeholder ; remplace par ton vrai mécanisme de hybride.
        scores: dict[int, float] = {}
        for mid in ids:
            try:
                est = self._svd.predict(uid="__cold__", iid=mid, r_ui=user_mean).est
            except Exception:
                est = self._pop_mean.get(mid, user_mean)
            scores[int(mid)] = float(est)
        return scores
```

Et l'entraînement (`backend/scripts/train_hybrid.py` — déjà fourni en
template) :

```python
from backend.app.recommender.impl_svd import SvdHybrid

def build_model() -> HybridRecommender:
    return SvdHybrid(n_factors=50)
```

## Recette pour un vrai hybride

Trois ingrédients à blender dans `score_candidates` :

1. **Content-based** : similarité cosine entre les embeddings/genome-scores
   des films notés par l'utilisateur et tous les candidats. Score = somme
   pondérée par les ratings utilisateur.
2. **User-based (KNN)** : trouver les K users les plus proches en cosine sur
   le vecteur de ratings → score d'un film = moyenne pondérée des notes
   qu'ils lui ont données.
3. **Latent factor (SVD ou ALS)** : fold-in du nouvel utilisateur dans
   l'espace latent.

Stratégie simple : normaliser chaque composante en [0,1], puis combinaison
linéaire pondérée. Tester plusieurs poids ; rapporter le meilleur sur un
hold-out.

Bonus possibles (sans casser le contrat) :
- Boost popularité légère (anti-niche) : `score += 0.05 * popularity[i]`
- Pénalité fraîcheur : pondérer par `decay(year)` pour favoriser les sorties
  récentes ou vice-versa.
- Cap : `score = np.tanh(score)` pour limiter les outliers.

Tout le reste (MMR diversity, filtre genre, watchlist, top-N) est déjà géré.

## Contact / questions

Si quelque chose dans le contrat est ambigu, ne devine pas — demande. Mieux
vaut clarifier l'interface tôt qu'avoir à re-pickler à la dernière minute.
