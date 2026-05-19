import re
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
RATING_MIN = 0.5
RATING_MAX = 5.0

# Populated once at startup by load_shared_artifacts()
_movies_df: pd.DataFrame | None = None   # index=movieId, cols=[title, genres]
_links_df: pd.DataFrame | None = None    # index=movieId, cols=[tmdbId]
_popularity: pd.Series | None = None     # index=movieId, values=rating_count


def load_shared_artifacts() -> None:
    global _movies_df, _links_df, _popularity

    for fname in ("movies.csv", "links.csv", "popularity.csv"):
        if not (ARTIFACTS_DIR / fname).exists():
            raise RuntimeError(
                f"Artifact manquant : backend/artifacts/{fname}\n"
                "Lancez d'abord : python backend/generate_artifacts.py"
            )

    _movies_df = pd.read_csv(ARTIFACTS_DIR / "movies.csv").set_index("movieId")
    _links_df = pd.read_csv(ARTIFACTS_DIR / "links.csv").set_index("movieId")
    _popularity = pd.read_csv(
        ARTIFACTS_DIR / "popularity.csv", index_col="movieId"
    )["rating_count"]
    print(f"[utils] {len(_movies_df)} films, {len(_links_df)} liens TMDB chargés.")


def normalize_score(raw: float) -> float:
    return round(float(np.clip((raw - RATING_MIN) / (RATING_MAX - RATING_MIN), 0.0, 1.0)), 4)


def parse_title_year(raw_title: str) -> tuple[str, int | None]:
    match = re.search(r"\((\d{4})\)\s*$", raw_title)
    if match:
        return raw_title[: match.start()].strip(), int(match.group(1))
    return raw_title, None


def _get_genres(movie_id: int) -> list[str]:
    if _movies_df is None or movie_id not in _movies_df.index:
        return []
    raw = _movies_df.loc[movie_id, "genres"]
    if not raw or raw == "(no genres listed)":
        return []
    return [g for g in str(raw).split("|") if g]


def movie_to_dict(
    movie_id: int,
    raw_score: float | None,
    user_ratings: dict,
    user_watchlist: list,
) -> dict:
    movie_id = int(movie_id)

    raw_title = (
        _movies_df.loc[movie_id, "title"]
        if _movies_df is not None and movie_id in _movies_df.index
        else "Unknown"
    )
    title, year = parse_title_year(raw_title)
    genres = _get_genres(movie_id)

    tmdb_id = None
    if _links_df is not None and movie_id in _links_df.index:
        val = _links_df.loc[movie_id, "tmdbId"]
        tmdb_id = int(val) if pd.notna(val) else None

    score = normalize_score(raw_score) if raw_score is not None else None

    return {
        "movie_id": movie_id,
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "genres": genres,
        "poster_url": None,
        "score": score,
        "in_watchlist": movie_id in user_watchlist,
        "user_rating": user_ratings.get(movie_id),
    }


def popular_movies(n: int, exclude_ids: set[int] | None = None) -> list[tuple[int, float]]:
    """Return list of (movie_id, raw_score) for top-n most-rated movies."""
    if _popularity is None:
        return []
    pop = _popularity if not exclude_ids else _popularity[~_popularity.index.isin(exclude_ids)]
    top = pop.nlargest(n)
    if top.empty:
        return []
    max_c, min_c = float(top.max()), float(top.min())
    results = []
    for mid, count in top.items():
        if max_c > min_c:
            raw = RATING_MIN + (RATING_MAX - RATING_MIN) * (float(count) - min_c) / (max_c - min_c)
        else:
            raw = (RATING_MIN + RATING_MAX) / 2
        results.append((int(mid), round(raw, 4)))
    return results
