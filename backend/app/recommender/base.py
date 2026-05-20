"""Contrat partenaire pour le modèle hybride.

Le partenaire DOIT :
  - hériter de HybridRecommender
  - implémenter fit(ratings, movies) et score_candidates(user_ratings, candidate_ids)
  - sérialiser une instance fittée vers backend/artifacts/hybrid_model.pkl

Le partenaire NE doit PAS :
  - réimplémenter de ranker, MMR, filtre watchlist, ou enrichissement (déjà gérés)
  - dépendre de paths absolus dans __init__
  - utiliser de lambdas / fonctions locales non picklables comme attributs

Voir backend/CONTRIBUTING_HYBRID.md pour les détails.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence

import pandas as pd


class HybridRecommender(ABC):
    """Interface minimale du modèle de recommandation hybride.

    Garanties fournies par le backend (à l'inférence) :
      - ``user_ratings`` contient au moins 5 entrées, ratings dans [0.5, 5.0],
        movieId = MovieLens ID (int) tel qu'il apparaît dans movies.csv.
      - ``candidate_ids`` est déjà filtré des films notés par l'utilisateur.
      - Le backend normalise tes scores en [0, 1] par carrousel.
      - Le backend applique MMR (diversité), filtre watchlist, dedup, top-N,
        et enrichit (titre, year, tmdb_id, genres).
      - ``fit()`` est appelé hors-ligne ; à l'inférence seul ``score_candidates`` est appelé.
    """

    #: Identifiant lisible reporté par /healthz
    name: str = "hybrid"
    #: Version libre, reportée par /healthz (utile pour vérifier le bon pickle chargé)
    version: str = "0.1.0"

    @abstractmethod
    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "HybridRecommender":
        """Entraîne le modèle hors-ligne.

        Args:
            ratings: DataFrame avec colonnes ['userId', 'movieId', 'rating', 'timestamp'].
                     Source : data/hackathon/evidence/ratings.csv.
            movies: DataFrame avec colonnes ['movieId', 'title', 'genres'].
                    Source : data/hackathon/content/movies.csv.

        Returns:
            self — l'objet doit être picklable après fit.
        """

    @abstractmethod
    def score_candidates(
        self,
        user_ratings: Mapping[int, float],
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[int, float]:
        """Score les candidats pour un utilisateur (potentiellement nouveau).

        Args:
            user_ratings: {movieId: rating} — historique de l'utilisateur, len >= 5.
            candidate_ids: si fourni, ne scorer QUE ces movieIds. Sinon, scorer tout
                           le catalogue (le backend filtrera ensuite si nécessaire).

        Returns:
            {movieId: score} — scores arbitraires, plus haut = mieux. Les candidats
            absents du dict sont ignorés par le backend. Aucune borne requise.
        """

    def recommend_similar(self, movie_id: int, k: int = 20) -> Mapping[int, float]:
        """Optionnel : similarité item-item (utilisée pour la sélection du hero).

        Si non surchargé, le backend tombe en fallback popularité.
        """
        return {}
