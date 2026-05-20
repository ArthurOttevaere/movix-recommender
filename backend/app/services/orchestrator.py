from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..recommender.base import HybridRecommender
from ..recommender.filters import exclude_ids, restrict_to
from ..recommender.ranker import mmr_rerank, normalize_scores, topk
from ..schemas import Carousel, Hero, MovieOut, RecommendationsResponse
from .catalog import MovieCatalog
from .enrichment import to_movie_out
from .popularity import PopularityIndex
from .state import UserState


@dataclass(frozen=True)
class CarouselSpec:
    id: str
    label: str
    model: str
    explanation: str
    builder: str  # name of the builder function on the orchestrator
    genres: tuple[str, ...] | None = None


CAROUSELS: list[CarouselSpec] = [
    CarouselSpec(
        id="pour_toi",
        label="Pour Toi",
        model="ensemble",
        explanation="Notre meilleur mix rien que pour vous",
        builder="_build_top",
    ),
    CarouselSpec(
        id="discovery",
        label="Découverte",
        model="ensemble",
        explanation="Films à fort potentiel hors de votre zone de confort",
        builder="_build_discovery",
    ),
    CarouselSpec(
        id="trending",
        label="Tendances",
        model="trending",
        explanation="Ce que tout le monde regarde",
        builder="_build_trending",
    ),
    CarouselSpec(
        id="genre_drama",
        label="Top Drame",
        model="genre:Drama",
        explanation="Les meilleurs films dramatiques",
        builder="_build_genre",
        genres=("Drama",),
    ),
    CarouselSpec(
        id="genre_thriller",
        label="Thriller & Suspense",
        model="genre:Thriller",
        explanation="Suspense et tension à couper le souffle",
        builder="_build_genre",
        genres=("Thriller",),
    ),
    CarouselSpec(
        id="genre_scifi",
        label="Science-Fiction",
        model="genre:Sci-Fi",
        explanation="Voyages dans d'autres mondes, d'autres temps",
        builder="_build_genre",
        genres=("Sci-Fi",),
    ),
    CarouselSpec(
        id="genre_animation",
        label="Animation",
        model="genre:Animation",
        explanation="Animation et fantastique",
        builder="_build_genre",
        genres=("Animation",),
    ),
    CarouselSpec(
        id="genre_crime",
        label="Crime & Mystère",
        model="genre:Crime",
        explanation="Enquêtes, mafias et histoires de détectives",
        builder="_build_genre",
        genres=("Crime",),
    ),
]


class Orchestrator:
    def __init__(
        self,
        recommender: HybridRecommender,
        catalog: MovieCatalog,
        popularity: PopularityIndex,
        settings: Settings,
    ):
        self.recommender = recommender
        self.catalog = catalog
        self.popularity = popularity
        self.settings = settings

    # ----- public entry -----

    def build(self, user: UserState) -> RecommendationsResponse:
        seen = set(user.ratings.keys())
        candidate_ids = exclude_ids(self.catalog.all_ids(), seen)

        raw_scores = self.recommender.score_candidates(user.ratings, candidate_ids=candidate_ids) or {}
        # Defensive coercion: ensure int keys and float values
        raw_scores = {int(k): float(v) for k, v in raw_scores.items() if v is not None}
        # Drop any seen items the recommender accidentally returned
        raw_scores = {k: v for k, v in raw_scores.items() if k not in seen}

        scored = normalize_scores(raw_scores)

        carousels = [self._build_one(spec, user, scored) for spec in CAROUSELS]
        # Drop carousels that ended up empty (e.g. genre with no candidates)
        carousels = [c for c in carousels if c.movies]
        hero = self._build_hero(scored, user)
        return RecommendationsResponse(hero=hero, carousels=carousels)

    # ----- per-carousel builders -----

    def _build_one(self, spec: CarouselSpec, user: UserState, scored: dict[int, float]) -> Carousel:
        builder = getattr(self, spec.builder)
        movies = builder(spec, user, scored)
        return Carousel(
            id=spec.id,
            label=spec.label,
            model=spec.model,
            explanation=spec.explanation,
            genres=list(spec.genres) if spec.genres else None,
            movies=movies,
        )

    def _enrich_pairs(self, pairs: list[tuple[int, float]], user: UserState, *, ranked: bool = False) -> list[MovieOut]:
        out: list[MovieOut] = []
        for idx, (mid, score) in enumerate(pairs):
            mo = to_movie_out(
                movie_id=mid,
                score=score,
                catalog=self.catalog,
                user_ratings=user.ratings,
                watchlist=user.watchlist,
                rank=(idx + 1) if ranked else None,
            )
            if mo is not None:
                out.append(mo)
        return out

    def _build_top(self, spec: CarouselSpec, user: UserState, scored: dict[int, float]) -> list[MovieOut]:
        if not scored:
            return self._popularity_fallback(self.settings.top_n_default, user)
        pairs = topk(scored, self.settings.top_n_default)
        return self._enrich_pairs(pairs, user)

    def _build_discovery(self, spec: CarouselSpec, user: UserState, scored: dict[int, float]) -> list[MovieOut]:
        if not scored:
            return []
        # Operate on a wider pool, then rerank for diversity.
        pool = topk(scored, max(50, self.settings.top_n_discovery * 5))
        item_features = {mid: set(self.catalog.get(mid).genres) for mid, _ in pool if self.catalog.get(mid)}
        reranked = mmr_rerank(pool, item_features, self.settings.top_n_discovery, self.settings.mmr_lambda)
        return self._enrich_pairs(reranked, user)

    def _build_trending(self, spec: CarouselSpec, user: UserState, scored: dict[int, float]) -> list[MovieOut]:
        seen = set(user.ratings.keys())
        # Pull a few extra in case some overlap user.ratings.
        pairs = [
            (mid, score)
            for mid, score in self.popularity.top(self.settings.top_n_trending + 20, min_count=100)
            if mid not in seen
        ][: self.settings.top_n_trending]
        return self._enrich_pairs(pairs, user, ranked=True)

    def _build_genre(self, spec: CarouselSpec, user: UserState, scored: dict[int, float]) -> list[MovieOut]:
        assert spec.genres
        primary = spec.genres[0]
        in_genre = self.catalog.ids_with_genre(primary)
        if not in_genre:
            return []
        seen = set(user.ratings.keys())

        # Prefer the hybrid scores intersected with the genre, fall back to popularity if too few.
        if scored:
            genre_scored = {mid: scored[mid] for mid in restrict_to(scored.keys(), in_genre) if mid not in seen}
            pairs = topk(genre_scored, self.settings.top_n_genre)
            if len(pairs) >= max(5, self.settings.top_n_genre // 2):
                return self._enrich_pairs(pairs, user)

        # Popularity fallback (still relevant: shows top of the genre globally).
        pop_pairs = [
            (mid, score)
            for mid, score in self.popularity.top_in_genre(self.catalog, primary, self.settings.top_n_genre + 10, min_count=30)
            if mid not in seen
        ][: self.settings.top_n_genre]
        return self._enrich_pairs(pop_pairs, user)

    # ----- hero -----

    def _build_hero(self, scored: dict[int, float], user: UserState) -> Hero:
        # Prefer hybrid top-1 with a tagline; else top-1; else popularity top-1.
        if scored:
            for mid, score in topk(scored, 25):
                row = self.catalog.get(mid)
                if row and row.tagline:
                    return Hero(movie_id=row.movie_id, tmdb_id=row.tmdb_id, title=row.title, score=round(score, 4), tagline=row.tagline)
            mid, score = topk(scored, 1)[0]
            row = self.catalog.get(mid)
            if row is not None:
                return Hero(movie_id=row.movie_id, tmdb_id=row.tmdb_id, title=row.title, score=round(score, 4), tagline=row.tagline)
        # Last resort: popularity.
        for mid, score in self.popularity.top(50, min_count=200):
            row = self.catalog.get(mid)
            if row is not None:
                return Hero(movie_id=row.movie_id, tmdb_id=row.tmdb_id, title=row.title, score=round(score, 4), tagline=row.tagline)
        raise RuntimeError("Catalog is empty — cannot build hero")

    # ----- helpers -----

    def _popularity_fallback(self, n: int, user: UserState) -> list[MovieOut]:
        seen = set(user.ratings.keys())
        pairs = [
            (mid, score) for mid, score in self.popularity.top(n + 20, min_count=50) if mid not in seen
        ][:n]
        return self._enrich_pairs(pairs, user)
