from __future__ import annotations

from typing import Mapping, Optional, Sequence

import pandas as pd

from ..services.popularity import PopularityIndex
from .base import HybridRecommender


class DummyHybrid(HybridRecommender):
    """Fallback popularité quand aucun pickle partenaire n'est présent.

    NE PAS utiliser comme template d'implémentation — le partenaire doit produire
    son propre modèle hybride (cf. CONTRIBUTING_HYBRID.md).
    """

    name = "dummy_popularity"
    version = "0.0.1"

    def __init__(self, popularity: PopularityIndex):
        self._pop = popularity

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "DummyHybrid":
        return self

    def score_candidates(
        self,
        user_ratings: Mapping[int, float],
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[int, float]:
        if candidate_ids is not None:
            return self._pop.scores_for(candidate_ids)
        return dict(self._pop.top(2000, min_count=20))
