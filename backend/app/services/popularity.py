from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import MovieCatalog


class PopularityIndex:
    """Bayesian-averaged popularity over the training ratings.

    score(i) = (C * m + sum_r_i) / (C + n_i)   then normalized to [0, 1]
    where m = global mean rating, C = prior strength (50 ratings).
    """

    def __init__(self, ratings_df: pd.DataFrame, *, prior_C: int = 50):
        m = float(ratings_df["rating"].mean())
        agg = ratings_df.groupby("movieId")["rating"].agg(["sum", "count"])
        bayes = (prior_C * m + agg["sum"]) / (prior_C + agg["count"])

        lo, hi = float(bayes.min()), float(bayes.max())
        if hi > lo:
            normed = (bayes - lo) / (hi - lo)
        else:
            normed = pd.Series(np.full(len(bayes), 0.5), index=bayes.index)

        self._scores: dict[int, float] = {int(i): float(v) for i, v in normed.items()}
        self._counts: dict[int, int] = {int(i): int(v) for i, v in agg["count"].items()}
        self._global_mean = m

    @classmethod
    def from_path(cls, ratings_path: Path) -> "PopularityIndex":
        df = pd.read_csv(ratings_path, usecols=["movieId", "rating"])
        return cls(df)

    def score(self, movie_id: int) -> float:
        return self._scores.get(int(movie_id), 0.0)

    def count(self, movie_id: int) -> int:
        return self._counts.get(int(movie_id), 0)

    def top(self, k: int, *, min_count: int = 50) -> list[tuple[int, float]]:
        items = (
            (mid, s) for mid, s in self._scores.items() if self._counts.get(mid, 0) >= min_count
        )
        return sorted(items, key=lambda x: x[1], reverse=True)[:k]

    def top_in_genre(self, catalog: MovieCatalog, genre: str, k: int, *, min_count: int = 20) -> list[tuple[int, float]]:
        in_genre = catalog.ids_with_genre(genre)
        items = (
            (mid, self._scores[mid])
            for mid in in_genre
            if mid in self._scores and self._counts.get(mid, 0) >= min_count
        )
        return sorted(items, key=lambda x: x[1], reverse=True)[:k]

    def scores_for(self, ids):
        return {int(i): self._scores.get(int(i), 0.0) for i in ids}
