from __future__ import annotations

from typing import Mapping

from ..schemas import MovieOut
from .catalog import MovieCatalog


def to_movie_out(
    movie_id: int,
    score: float,
    catalog: MovieCatalog,
    user_ratings: Mapping[int, float],
    watchlist: set[int],
    *,
    rank: int | None = None,
) -> MovieOut | None:
    row = catalog.get(movie_id)
    if row is None:
        return None
    return MovieOut(
        movie_id=row.movie_id,
        tmdb_id=row.tmdb_id,
        title=row.title,
        year=row.year,
        genres=list(row.genres),
        poster_url=None,  # the frontend resolves via TMDB
        score=round(float(score), 4),
        in_watchlist=row.movie_id in watchlist,
        user_rating=user_ratings.get(row.movie_id),
        rank=rank,
    )
