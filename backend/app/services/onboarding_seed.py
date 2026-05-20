"""Curated onboarding film list.

We seed with the same TMDB IDs the frontend mock currently uses (so posters resolve
identically pre/post backend switch). At runtime they are translated to MovieLens IDs
via the catalog's links.csv mapping, so what the API returns has ``movie_id`` =
MovieLens ID and ``tmdb_id`` = TMDB ID.
"""

from __future__ import annotations

from .catalog import MovieCatalog

# TMDB IDs of broadly popular, well-rated films covering several genres.
ONBOARDING_SEED_TMDB_IDS: list[int] = [
    550,     # Fight Club
    27205,   # Inception
    238,     # The Godfather
    278,     # The Shawshank Redemption
    13,      # Forrest Gump
    680,     # Pulp Fiction
    157336,  # Interstellar
    129,     # Spirited Away
    11,      # Star Wars
    244786,  # Whiplash
    372058,  # Your Name
    637,     # Life Is Beautiful
    539,     # Psycho
    769,     # GoodFellas
    508442,  # Soul
    324857,  # Spider-Man: Into the Spider-Verse
    289,     # Casablanca
    598,     # City of God
    429,     # The Good, the Bad and the Ugly
    346364,  # It
    603,     # The Matrix
    155,     # The Dark Knight
    807,     # Se7en
    274,     # The Silence of the Lambs
    77,      # Memento
]


def resolve_seed(catalog: MovieCatalog) -> list[dict]:
    """Resolve TMDB seed list to MovieLens-indexed onboarding entries."""
    out: list[dict] = []
    for tmdb_id in ONBOARDING_SEED_TMDB_IDS:
        movie_id = catalog.resolve_tmdb_id(tmdb_id)
        if movie_id is None:
            continue
        row = catalog.get(movie_id)
        if row is None:
            continue
        out.append(
            {
                "movie_id": row.movie_id,
                "tmdb_id": row.tmdb_id,
                "title": row.title,
                "year": row.year,
                "genres": list(row.genres),
            }
        )
    return out
