"""
Workshop 2 — Implicit Library
==============================
Fonctions :
  1. compute_implicit_ratings(library_path)  → DataFrame {movie_id, implicit_rating}
  2. build_augmented_trainset(implicit_df)   → Surprise Trainset
  3. train_and_save(trainset, path)          → algo entraîné + pickle sauvegardé

Formule (section 4.5.1) :
  IR_{i,u} = w_top10 * top10 + w_watched * min(n_watched, MAX_WATCHED)
           + w_wishlist * wishlist + w_recent * recent
  Normalisé dans [0.5, 5.0].
"""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from models import LatentFactorPP

from constants import Constant as C
from loaders import load_ratings, load_items

# ── Identifiant utilisateur (négatif pour ne pas entrer en conflit avec MovieLens) ──
MY_USER_ID = -1

# ── Poids des événements (section 4.5.1) ─────────────────────────────────────────
# top10   → signal le plus fort, comme "buy"        → w = 100
# n_watched → nombre de visionnages (capé), comme "moreDetails" → w = 50
# wishlist  → intention de voir, comme "details + more" → w = 30
# recent    → vu récemment, comme "details"          → w = 15
W_TOP10    = 100
W_WATCHED  = 50
W_WISHLIST = 30
W_RECENT   = 15
MAX_WATCHED = 5   # cap : au-delà de 5 visions, le signal n'augmente plus


# ─────────────────────────────────────────────────────────────────────────────────
# UTILITAIRE : générer le template CSV à remplir
# ─────────────────────────────────────────────────────────────────────────────────

def generate_library_template(output_path: str = "library_lenny.csv", n: int = 200) -> None:
    """
    Génère un fichier CSV trié par popularité décroissante (n films les plus populaires).
    L'utilisateur remplit ensuite les colonnes n_watched, wishlist, recent, top10.
    """
    df_ratings = load_ratings(surprise_format=False)
    df_items   = load_items()

    # Popularité = nombre de ratings par film
    popularity = df_ratings.groupby(C.ITEM_ID_COL)[C.RATING_COL].count().rename("n_ratings")
    df_popular = (
        popularity
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    df_popular = df_popular.merge(
        df_items[[C.LABEL_COL, C.GENRES_COL]].reset_index(),
        on=C.ITEM_ID_COL,
        how="left"
    )
    df_popular = df_popular.rename(columns={C.ITEM_ID_COL: "movie_id", C.LABEL_COL: "title"})

    # Colonnes à remplir — valeurs par défaut à 0
    df_popular["n_watched"] = 0   # nombre de fois visionné {0, 1, 2, ...}
    df_popular["wishlist"]  = 0   # 1 si tu veux le voir, 0 sinon
    df_popular["recent"]    = 0   # 1 si vu dans les 2 dernières années, 0 sinon
    df_popular["top10"]     = 0   # 1 si dans ton top-10 personnel, 0 sinon

    cols = ["movie_id", "title", "genres", "n_ratings", "n_watched", "wishlist", "recent", "top10"]
    df_popular[cols].to_csv(output_path, index=False)
    print(f"Template généré : {output_path}  ({len(df_popular)} films)")
    print("→ Remplis les colonnes n_watched / wishlist / recent / top10 dans Excel, puis exporte en CSV.")


# ─────────────────────────────────────────────────────────────────────────────────
# FONCTION 1 : calculer les ratings implicites depuis la bibliothèque
# ─────────────────────────────────────────────────────────────────────────────────

def compute_implicit_ratings(library_path: str) -> pd.DataFrame:
    """
    Lit la bibliothèque personnelle et calcule un rating implicite ∈ [0.5, 5.0]
    pour chaque film, selon la formule de la section 4.5.1.

    Paramètre
    ---------
    library_path : chemin vers le CSV rempli (library_lenny.csv)

    Retourne
    --------
    DataFrame avec colonnes [movie_id, implicit_rating]
    """
    df = pd.read_csv(library_path)

    # ── Formule section 4.5.1 : IR = somme(w_k * #event_k) ──────────────────
    df["raw_ir"] = (
        W_TOP10    * df["top10"] +
        W_WATCHED  * df["n_watched"].clip(upper=MAX_WATCHED) +
        W_WISHLIST * df["wishlist"] +
        W_RECENT   * df["recent"]
    )

    # Forcer movie_id en entier (Excel peut convertir en float : 356 → 356.0)
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df = df.dropna(subset=["movie_id"])
    df["movie_id"] = df["movie_id"].astype(int)

    # Garder seulement les films avec au moins un événement
    df = df[df["raw_ir"] > 0].copy()

    # ── Normalisation dans [0.5, 5.0] ────────────────────────────────────────
    ir_min = df["raw_ir"].min()
    ir_max = df["raw_ir"].max()
    if ir_max > ir_min:
        df["implicit_rating"] = 0.5 + (df["raw_ir"] - ir_min) / (ir_max - ir_min) * 4.5
    else:
        df["implicit_rating"] = 3.0

    df["implicit_rating"] = df["implicit_rating"].clip(0.5, 5.0).round(2)

    print(f"[1] {len(df)} films avec rating implicite calculé")
    print(f"    Min IR : {df['implicit_rating'].min():.2f}  |  Max IR : {df['implicit_rating'].max():.2f}")

    return df[["movie_id", "implicit_rating"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────────
# FONCTION 2 : construire le trainset augmenté (MovieLens + ratings implicites)
# ─────────────────────────────────────────────────────────────────────────────────

def build_augmented_trainset(implicit_ratings_df: pd.DataFrame):
    """
    Ajoute les ratings implicites au dataset MovieLens et construit un Surprise Trainset.
    L'utilisateur reçoit l'ID MY_USER_ID (négatif pour éviter les conflits).

    Paramètre
    ---------
    implicit_ratings_df : sortie de compute_implicit_ratings()

    Retourne
    --------
    Surprise Trainset entraînable
    """
    df_ratings = load_ratings(surprise_format=False)

    # Construire les lignes correspondant à l'utilisateur personnel
    implicit_rows = pd.DataFrame({
        C.USER_ID_COL: MY_USER_ID,
        C.ITEM_ID_COL: implicit_ratings_df["movie_id"].values,
        C.RATING_COL:  implicit_ratings_df["implicit_rating"].values,
    })

    # Concaténer avec les ratings MovieLens existants
    df_augmented = pd.concat(
        [df_ratings[C.USER_ITEM_RATINGS], implicit_rows],
        ignore_index=True
    )

    print(f"[2] Dataset augmenté : {len(df_augmented)} ratings  "
          f"({len(df_ratings)} MovieLens + {len(implicit_rows)} implicites)")
    print(f"    User ID personnel : {MY_USER_ID}")

    reader   = Reader(rating_scale=C.RATINGS_SCALE)
    dataset  = Dataset.load_from_df(df_augmented[C.USER_ITEM_RATINGS], reader)
    trainset = dataset.build_full_trainset()
    return trainset


# ─────────────────────────────────────────────────────────────────────────────────
# FONCTION 3 : entraîner le modèle et sauvegarder le pickle
# ─────────────────────────────────────────────────────────────────────────────────

def train_and_save(trainset, artifact_path: str = "backend/artifacts/svd_model.pkl"):
    """
    Entraîne un SVD++ (LatentFactorPP) sur le trainset augmenté et sauvegarde le modèle en pickle.
    Paramètres issus de la littérature : Koren, Y. (2008). KDD '08, pp. 426–434.

    Paramètres
    ----------
    trainset      : Surprise Trainset (sortie de build_augmented_trainset)
    artifact_path : chemin de sauvegarde du pickle

    Retourne
    --------
    algo entraîné
    """
    # Koren, Y. (2008). Factorization meets the neighborhood. KDD '08, pp. 426–434.
    algo = LatentFactorPP()
    algo.fit(trainset)

    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "wb") as f:
        pickle.dump(algo, f)

    print(f"[3] Modèle sauvegardé : {artifact_path}")
    print(f"    n_items  : {trainset.n_items}")
    print(f"    n_users  : {trainset.n_users}")
    return algo


# ─────────────────────────────────────────────────────────────────────────────────
# MAIN : pipeline complet
# ─────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LIBRARY_PATH  = "library_lenny.csv"
    ARTIFACT_PATH = "backend/artifacts/svd_model.pkl"

    print("=== Workshop 2 — Implicit Library ===\n")

    print("Étape 1 — Calcul des ratings implicites...")
    implicit_df = compute_implicit_ratings(LIBRARY_PATH)

    print("\nÉtape 2 — Construction du trainset augmenté...")
    trainset = build_augmented_trainset(implicit_df)

    print("\nÉtape 3 — Entraînement et sauvegarde...")
    train_and_save(trainset, ARTIFACT_PATH)

    print("\nTerminé !")
