from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


@dataclass(frozen=True)
class MovieRow:
    movie_id: int
    tmdb_id: int | None
    title: str
    year: int | None
    genres: tuple[str, ...]
    tagline: str | None = None
    overview: str | None = None


def _parse_title(raw: str) -> tuple[str, int | None]:
    m = _YEAR_RE.search(raw or "")
    if not m:
        return raw, None
    year = int(m.group(1))
    title = _YEAR_RE.sub("", raw).strip()
    return title, year


def _parse_genres(raw: str) -> tuple[str, ...]:
    if not raw or raw == "(no genres listed)":
        return tuple()
    return tuple(g for g in raw.split("|") if g)


# Map the MovieLens genre labels onto the frontend's preferred short form when needed.
_GENRE_REMAP = {"Science Fiction": "Sci-Fi"}


def _normalize_genre(g: str) -> str:
    return _GENRE_REMAP.get(g, g)


class MovieCatalog:
    def __init__(self, movies_df: pd.DataFrame, links_df: pd.DataFrame, tmdb_cache: dict | None = None):
        tmdb_cache = tmdb_cache or {}

        links = links_df.set_index("movieId")["tmdbId"].dropna().astype(int)
        tmdb_by_movie: dict[int, int] = links.to_dict()
        movie_by_tmdb: dict[int, int] = {v: k for k, v in tmdb_by_movie.items()}

        rows: dict[int, MovieRow] = {}
        for movie_id, title_raw, genres_raw in movies_df[["movieId", "title", "genres"]].itertuples(index=False):
            title, year = _parse_title(title_raw)
            genres = tuple(_normalize_genre(g) for g in _parse_genres(genres_raw))
            tmdb_id = tmdb_by_movie.get(int(movie_id))
            extra = tmdb_cache.get(str(tmdb_id)) if tmdb_id else None
            extra = extra or {}
            rows[int(movie_id)] = MovieRow(
                movie_id=int(movie_id),
                tmdb_id=tmdb_id,
                title=title,
                year=year,
                genres=genres,
                tagline=(extra.get("tagline") or None),
                overview=(extra.get("overview") or None),
            )

        self._rows = rows
        self._tmdb_to_movie = movie_by_tmdb
        self._by_genre: dict[str, set[int]] = {}
        for mid, row in rows.items():
            for g in row.genres:
                self._by_genre.setdefault(g, set()).add(mid)

    @classmethod
    def from_paths(cls, movies_path: Path, links_path: Path, tmdb_cache_path: Path | None = None) -> "MovieCatalog":
        movies_df = pd.read_csv(movies_path)
        links_df = pd.read_csv(links_path)
        tmdb_cache = {}
        if tmdb_cache_path and tmdb_cache_path.exists():
            tmdb_cache = json.loads(tmdb_cache_path.read_text())
        return cls(movies_df, links_df, tmdb_cache)

    def get(self, movie_id: int) -> MovieRow | None:
        return self._rows.get(int(movie_id))

    def all_ids(self) -> list[int]:
        return list(self._rows.keys())

    def ids_with_genre(self, genre: str) -> set[int]:
        return self._by_genre.get(genre, set())

    def resolve_tmdb_id(self, tmdb_id: int) -> int | None:
        """Translate a TMDB id back to a MovieLens id (used for legacy onboarding payloads)."""
        return self._tmdb_to_movie.get(int(tmdb_id))

    def filter_existing(self, ids: Iterable[int]) -> list[int]:
        return [i for i in ids if i in self._rows]

    def __len__(self) -> int:
        return len(self._rows)
