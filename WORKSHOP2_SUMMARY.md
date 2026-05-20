# Workshop 2 — Résumé des changements & Guide d'utilisation

> Document destiné aux membres de l'équipe.  
> Branche : `feature/model-svd`

---

## 1. `latent_factor.ipynb` — Optimisation du modèle SVD

### Ce qui a changé

**Section 4 — GridSearchCV**  
Remplacement des hyperparamètres fixes par une recherche systématique via `GridSearchCV` de la librairie Surprise :

```python
param_grid = {
    "n_factors": [50, 100, 150],
    "n_epochs":  [20, 30],
    "lr_all":    [0.005, 0.01],
    "reg_all":   [0.02, 0.05, 0.1],
}
gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3)
gs.fit(data)
```

54 combinaisons × 3 folds = **162 entraînements**.  
Meilleurs paramètres obtenus : `n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.05`  
**RMSE = 0.8012** (meilleur modèle collaboratif du projet).

**Section 8** — Le modèle final utilise ces paramètres optimisés au lieu de valeurs arbitraires.

---

## 2. `models.py` — Ajout de `LatentFactor`

Nouvelle classe héritant de `SVD` (Surprise), pré-configurée avec les meilleurs paramètres issus du GridSearchCV :

```python
class LatentFactor(SVD):
    def __init__(self, random_state=1):
        SVD.__init__(self, n_factors=150, n_epochs=30,
                     lr_all=0.01, reg_all=0.05, random_state=random_state)
```

Avantage : le modèle est réutilisable directement dans le pipeline d'évaluation sans avoir à répéter les paramètres.

---

## 3. `configs.py` — Mise à jour de la configuration d'évaluation

### Modèles actifs (run courant, optimisé pour le temps de calcul)

```python
models = [
    ("baseline_1",          ModelBaseline1, {}),
    ("baseline_2",          ModelBaseline2, {}),
    ("baseline_3",          ModelBaseline3, {}),
    ("baseline_4",          ModelBaseline4, {"random_state": 1}),
    ("UserBased_Manual",    UserBased,      {"k": 3, "min_k": 2, ...}),
    ("ContentBased_ridge_cv", ContentBased, {"features_method": "all_content_tmdb_tags2000",
                                             "regressor_method": "ridge_cv"}),
    ("LatentFactor",        LatentFactor,   {}),
]
```

Les autres modèles (KNNwithMeans, RandomSample, LinearRegression, ContentBased_ridge) sont commentés pour réduire le temps de calcul. Ils peuvent être réactivés pour un run complet.

### Métriques

```python
split_metrics = ["rmse", "mae"]          # prédiction de note
loo_metrics   = ["hit_rate", "ndcg"]     # pertinence top-N
full_metrics  = ["novelty", "miuf", "ild"]  # beyond-accuracy
```

---

## 4. `evaluator.ipynb` — Nouvelles métriques

### Métriques ajoutées

| Métrique | Type | Formule | Interprétation |
|---|---|---|---|
| **MAE** | split | `mean(|r̂ - r|)` | Erreur absolue moyenne |
| **NDCG@K** | loo | `(1/N) Σ 1/log2(rank+1)` | Qualité du classement (position-aware) |
| **MIUF** | full | `(1/\|R\|) Σ -log2(\|U_i\|/\|U\|)` | Nouveauté via popularité inverse |
| **ILD** | full | `mean_{i<j}(1 - cos(gi, gj))` | Diversité intra-liste par genre |

**Référence NDCG :** Cremonesi et al. (2010) — *Performance of Recommender Algorithms on Top-N Recommendation Tasks*

### Fonctions ajoutées dans la cellule métriques

- `get_ndcg(anti_testset_top_n, testset)` — NDCG@K standard LOO
- `get_novelty_miuf(anti_testset_top_n, item_freq, n_users)` — MIUF
- `build_genre_vectors(df_items)` — vecteurs binaires de genres
- `get_diversity_ild(anti_testset_top_n, genre_vectors)` — ILD cosinus

### `precompute_information` mis à jour

```python
def precompute_information(df_ratings, df_items):
    # item_to_rank  : popularité relative de chaque film
    # item_freq     : nombre d'utilisateurs ayant noté chaque film
    # n_users       : total utilisateurs distincts
    # genre_vectors : vecteur binaire de genres par film
```

---

## 5. `recommender_building.py` + `library_lenny.csv` — Workshop 2 : Ratings implicites

### Contexte

Le Workshop 2 (section 4.5.1 du cours) consiste à **enrichir le dataset MovieLens** avec des ratings implicites dérivés du comportement de visionnage personnel, puis à entraîner le modèle SVD sur ce dataset augmenté.

### Formule des ratings implicites (section 4.5.1)

```
IR = w_top10 × top10 + w_watched × min(n_watched, 5) + w_wishlist × wishlist + w_recent × recent
```

Normalisé dans [0.5, 5.0].

| Signal | Poids | Description |
|---|---|---|
| `top10` | 100 | Film dans le top-10 personnel |
| `n_watched` | 50 | Nombre de visionnages (capé à 5) |
| `wishlist` | 30 | Film dans la liste de souhaits |
| `recent` | 15 | Vu dans les 2 dernières années |

### `library_lenny.csv`

Fichier CSV contenant les 200 films les plus populaires du dataset MovieLens. Pour chaque film, les colonnes `n_watched`, `wishlist`, `recent`, `top10` ont été remplies manuellement. 38 films ont au moins un signal non nul.

### Comment utiliser `recommender_building.py`

**Étape 1 — Vérifier / mettre à jour `library_lenny.csv`**

Ouvrir le fichier dans Excel et remplir les colonnes :
- `n_watched` : nombre de fois que tu as vu le film (0, 1, 2, ...)
- `wishlist` : 1 si tu veux le voir, 0 sinon
- `recent` : 1 si vu dans les 2 dernières années, 0 sinon
- `top10` : 1 si c'est dans ton top-10 personnel, 0 sinon

Sauvegarder en CSV (UTF-8).

**Étape 2 — Générer le modèle augmenté**

Depuis la racine du projet :

```bash
python recommender_building.py
```

Le script :
1. Lit `library_lenny.csv` et calcule les ratings implicites
2. Concatène ces ratings avec les 381 181 ratings MovieLens (dataset augmenté : ~381 219 ratings)
3. Entraîne `SVD(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.05)` sur le dataset complet
4. Sauvegarde le modèle dans **`backend/artifacts/svd_model.pkl`**

**Étape 3 — Redémarrer le backend**

```bash
uvicorn backend.main:app --reload
```

Le backend charge automatiquement `svd_model.pkl` au démarrage. Le carousel **"In Your Style — SVD Model"** utilise désormais le modèle enrichi par les ratings implicites.

> **Note :** `recommender_building.py` doit être relancé à chaque fois que `library_lenny.csv` est mis à jour.

---

## 6. Architecture de recommandation — Quel système adopter ?

### Option A — Un modèle par carousel (architecture actuelle)

Chaque carousel du site est alimenté par un modèle différent :

| Carousel | Modèle |
|---|---|
| "Recommandé pour vous" | Content-based (RidgeCV) |
| "In Your Style" | SVD / LatentFactor |
| "Viewers Like You" | User-based collaboratif |
| "Genres" | Filtrage par genre |

### Option B — Système hybride à deux étapes (retrieval + ranker)

1. **Retrieval** : plusieurs modèles génèrent ~500 candidats chacun
2. **Ranker** : un modèle unique (ex. LightGBM, MLP) re-score et trie les candidats finaux

---

### Recommandation : **Option A — Un modèle par carousel**

C'est le choix le plus adapté à votre projet, et c'est **l'architecture décrite par Netflix** dans le papier de référence du cours :

> *Gomez-Uribe & Hunt (2015) — The Netflix Recommender System: Algorithms, Business Value, and Innovation*

Netflix utilise explicitement des carousels différents pour des objectifs différents : personnalisation (collaboratif), similarité (content-based), tendances (popularité), genres, etc. Chaque carousel a sa propre logique.

**Pourquoi l'Option A est plus adaptée ici :**

| Critère | Option A (carousel) | Option B (ranker) |
|---|---|---|
| Complexité d'implémentation | Faible — déjà en place | Élevée — ranker à entraîner |
| Données nécessaires | Ratings explicites | Signaux d'engagement (clics, watch time) que vous n'avez pas |
| Explicabilité | Haute — chaque carousel a une logique claire | Faible — boîte noire |
| Diversité des recommandations | Naturelle — modèles différents = angles différents | Doit être forcée explicitement |
| Référence académique | Netflix 2015, Cinematch | YouTube 2016, LinkedIn, Spotify à grande échelle |

L'Option B (deux étapes) est adoptée par YouTube, Spotify et LinkedIn **uniquement parce qu'ils ont des dizaines de millions d'utilisateurs et des signaux implicites riches** (durée de visionnage, skip, clics). Avec un dataset MovieLens de 1 000 utilisateurs, entraîner un ranker supervisé n'est pas réalisable correctement.

**Conclusion :** Votre architecture actuelle (un modèle par carousel) est la bonne pratique pour votre échelle, elle est validée par la littérature, et elle est déjà implémentée.
