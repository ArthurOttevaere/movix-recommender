"""
Script de génération des artefacts partagés.

Lancez depuis la racine du repo :
    python backend/generate_artifacts.py

Génère dans backend/artifacts/ :
    movies.csv      — movieId, title, genres
    links.csv       — movieId, tmdbId (NaN exclus, tmdbId converti en int)
    popularity.csv  — movieId, rating_count (trié décroissant)

Chaque personne génère ensuite ses propres artefacts de modèle
depuis son notebook (voir les docstrings dans backend/models/).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
HACKATHON = ROOT / "data" / "hackathon"
OUT = Path(__file__).parent / "artifacts"


def main():
    if not HACKATHON.exists():
        raise SystemExit(
            f"Dossier data/ introuvable : {HACKATHON}\n"
            "Assurez-vous d'avoir décompressé les données (python unzip_data.py)."
        )

    OUT.mkdir(exist_ok=True)

    # movies.csv
    movies = pd.read_csv(HACKATHON / "content" / "movies.csv")
    movies.to_csv(OUT / "movies.csv", index=False)
    print(f"movies.csv      : {len(movies):>6} films")

    # links.csv — ne garder que movieId et tmdbId valides
    links = pd.read_csv(HACKATHON / "content" / "links.csv")[["movieId", "tmdbId"]]
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)
    links.to_csv(OUT / "links.csv", index=False)
    print(f"links.csv       : {len(links):>6} films avec tmdb_id")

    # popularity.csv — nombre de ratings par film
    ratings = pd.read_csv(HACKATHON / "evidence" / "ratings.csv")
    popularity = (
        ratings.groupby("movieId")
        .size()
        .reset_index(name="rating_count")
        .sort_values("rating_count", ascending=False)
    )
    popularity.to_csv(OUT / "popularity.csv", index=False)
    print(f"popularity.csv  : {len(popularity):>6} films")

    print(f"\nArtefacts partagés générés dans {OUT.resolve()}/")
    print("Chaque personne génère ses artefacts de modèle depuis son notebook.")


if __name__ == "__main__":
    main()
