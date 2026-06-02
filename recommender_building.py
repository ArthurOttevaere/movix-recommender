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

# ── Negative User ID not to conflict with MovieLens ──
MY_USER_ID = -1

# ── Events weights (section 4.5.1) ─────────────────────────────────────────
# top10   → strong signal, like "buy"        → w = 100
# n_watched → number of viewings (capped), like "moreDetails" → w = 50
# wishlist  → intention to watch, like "details + more" → w = 30
# recent    → recently watched, like "details"          → w = 15
W_TOP10    = 100
W_WATCHED  = 50
W_WISHLIST = 30
W_RECENT   = 15
MAX_WATCHED = 5   # cap : above 5viewings, the signal does not increase further


# ─────────────────────────────────────────────────────────────────────────────────
# UTILS : generate the template
# ─────────────────────────────────────────────────────────────────────────────────

def generate_library_template(output_path: str = "library_lenny.csv", n: int = 200) -> None:
    """
    Generates a CSV file sorted by decreasing popularity (n most popular films).
    The user then fills in the columns n_watched, wishlist, recent, top10.
    """
    df_ratings = load_ratings(surprise_format=False)
    df_items   = load_items()

    # Popularity = number of ratings
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

    # Columns to fill
    df_popular["n_watched"] = 0   # number of times watched {0, 1, 2, ...}
    df_popular["wishlist"]  = 0   # 1 if you want to see it, 0 otherwise
    df_popular["recent"]    = 0   # 1 if watched in the last 2 years, 0 otherwise
    df_popular["top10"]     = 0   # 1 if in your personal top-10, 0 otherwise

    cols = ["movie_id", "title", "genres", "n_ratings", "n_watched", "wishlist", "recent", "top10"]
    df_popular[cols].to_csv(output_path, index=False)
    print(f"Template generated : {output_path}  ({len(df_popular)} films)")
    print("→ Fill the columns n_watched / wishlist / recent / top10 in Excel, then export as CSV.")


# ─────────────────────────────────────────────────────────────────────────────────
# FONCTION 1 : compute the implicit ratings from the filled library
# ─────────────────────────────────────────────────────────────────────────────────

def compute_implicit_ratings(library_path: str) -> pd.DataFrame:
    """
    Read the personal library and calculate an implicit rating ∈ [0.5, 5.0]
    for each film, according to the formula of section 4.5.1.

    Parameter
    ---------
    library_path : path to the filled CSV (library_lenny.csv)

    Returns
    --------
    DataFrame with columns [movie_id, implicit_rating]
    """
    df = pd.read_csv(library_path)

    # ── Formula section 4.5.1 : IR = sum(w_k * #event_k) ──────────────────
    df["raw_ir"] = (
        W_TOP10    * df["top10"] +
        W_WATCHED  * df["n_watched"].clip(upper=MAX_WATCHED) +
        W_WISHLIST * df["wishlist"] +
        W_RECENT   * df["recent"]
    )

    # Force movie_id to integer (Excel can convert to float : 356 → 356.0)
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df = df.dropna(subset=["movie_id"])
    df["movie_id"] = df["movie_id"].astype(int)

    # Keep only the films with at least one event
    df = df[df["raw_ir"] > 0].copy()

    # ── Normalization in [0.5, 5.0] ────────────────────────────────────────
    ir_min = df["raw_ir"].min()
    ir_max = df["raw_ir"].max()
    if ir_max > ir_min:
        df["implicit_rating"] = 0.5 + (df["raw_ir"] - ir_min) / (ir_max - ir_min) * 4.5
    else:
        df["implicit_rating"] = 3.0

    df["implicit_rating"] = df["implicit_rating"].clip(0.5, 5.0).round(2)

    print(f"[1] {len(df)} films with implicit rating calculated")
    print(f"    Min IR : {df['implicit_rating'].min():.2f}  |  Max IR : {df['implicit_rating'].max():.2f}")

    return df[["movie_id", "implicit_rating"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 : build the augmented trainset (MovieLens + implicit ratings)
# ─────────────────────────────────────────────────────────────────────────────────

def build_augmented_trainset(implicit_ratings_df: pd.DataFrame):
    """
    Add the implicit ratings to the MovieLens dataset and build a Surprise Trainset.
    The user receives the ID MY_USER_ID (negative to avoid conflicts).

    Parameter
    ---------
    implicit_ratings_df : output of compute_implicit_ratings()

    Returns
    --------
    Surprise Trainset trainable
    """
    df_ratings = load_ratings(surprise_format=False)

    # Construct the rows corresponding to the personal user
    implicit_rows = pd.DataFrame({
        C.USER_ID_COL: MY_USER_ID,
        C.ITEM_ID_COL: implicit_ratings_df["movie_id"].values,
        C.RATING_COL:  implicit_ratings_df["implicit_rating"].values,
    })

    # Concatenate with the existing MovieLens ratings
    df_augmented = pd.concat(
        [df_ratings[C.USER_ITEM_RATINGS], implicit_rows],
        ignore_index=True
    )

    print(f"[2] Augmented dataset : {len(df_augmented)} ratings  "
          f"({len(df_ratings)} MovieLens + {len(implicit_rows)} implicit)")
    print(f"    Personal User ID : {MY_USER_ID}")

    reader   = Reader(rating_scale=C.RATINGS_SCALE)
    dataset  = Dataset.load_from_df(df_augmented[C.USER_ITEM_RATINGS], reader)
    trainset = dataset.build_full_trainset()
    return trainset


# ─────────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 : train the model and save the pickle
# ─────────────────────────────────────────────────────────────────────────────────

def train_and_save(trainset, artifact_path: str = "backend/artifacts/svd_model.pkl"):
    """
    Trains an SVD++ (LatentFactorPP) on the augmented trainset and saves the model as a pickle.
    Parameters
    ----------
    trainset      : Surprise Trainset (output of build_augmented_trainset)
    artifact_path : path to save the model  pickle

    Returns
    --------
    algo trained
    """
    # Koren, Y. (2008). Factorization meets the neighborhood. KDD '08, pp. 426–434.
    algo = LatentFactorPP()
    algo.fit(trainset)

    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "wb") as f:
        pickle.dump(algo, f)

    print(f"[3] Model saved : {artifact_path}")
    print(f"    n_items  : {trainset.n_items}")
    print(f"    n_users  : {trainset.n_users}")
    return algo


# ─────────────────────────────────────────────────────────────────────────────────
# MAIN : entire pipeline
# ─────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LIBRARY_PATH  = "library_lenny.csv"
    ARTIFACT_PATH = "backend/artifacts/svd_model.pkl"

    print("=== Workshop 2 — Implicit Library ===\n")

    print("Step 1 — Computing implicit ratings...")
    implicit_df = compute_implicit_ratings(LIBRARY_PATH)

    print("\nStep 2 — Building the augmented trainset...")
    trainset = build_augmented_trainset(implicit_df)

    print("\nStep 3 — Training and saving...")
    train_and_save(trainset, ARTIFACT_PATH)

    print("\nCompleted!")
