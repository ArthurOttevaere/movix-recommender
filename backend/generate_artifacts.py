"""
Script for the generation of shared artifacts from the raw data.

Run from the root of the repo :
    python backend/generate_artifacts.py

Generates in backend/artifacts/ :
    movies.csv      — movieId, title, genres
    links.csv       — movieId, tmdbId (NaN exclus, tmdbId converted in int)
    popularity.csv  — movieId, rating_count (sorted in descending order)

Each person then generates their own model artifacts
from their notebook (see the docstrings in backend/models/).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
HACKATHON = ROOT / "data" / "hackathon"
OUT = Path(__file__).parent / "artifacts"


def main():
    if not HACKATHON.exists():
        raise SystemExit(
            f"Data folder/ not found : {HACKATHON}\n"
            "Make sure you have decompressed the data (python unzip_data.py)."
        )

    OUT.mkdir(exist_ok=True)

    # movies.csv
    movies = pd.read_csv(HACKATHON / "content" / "movies.csv")
    movies.to_csv(OUT / "movies.csv", index=False)
    print(f"movies.csv      : {len(movies):>6} films")

    # links.csv — keep only valid movieId and tmdbId
    links = pd.read_csv(HACKATHON / "content" / "links.csv")[["movieId", "tmdbId"]]
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)
    links.to_csv(OUT / "links.csv", index=False)
    print(f"links.csv       : {len(links):>6} films with tmdb_id")

    # popularity.csv — number of ratings per film
    ratings = pd.read_csv(HACKATHON / "evidence" / "ratings.csv")
    popularity = (
        ratings.groupby("movieId")
        .size()
        .reset_index(name="rating_count")
        .sort_values("rating_count", ascending=False)
    )
    popularity.to_csv(OUT / "popularity.csv", index=False)
    print(f"popularity.csv  : {len(popularity):>6} films")

    print(f"\nShared artifacts generated in {OUT.resolve()}/")
    print("Each person then generates their own model artifacts from their notebook.")


if __name__ == "__main__":
    main()
