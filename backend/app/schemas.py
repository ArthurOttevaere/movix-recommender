from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class MovieOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    movie_id: int
    tmdb_id: int | None = None
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    poster_url: str | None = None
    score: float
    in_watchlist: bool = False
    user_rating: float | None = None
    rank: int | None = None


class Hero(BaseModel):
    model_config = ConfigDict(extra="ignore")

    movie_id: int
    tmdb_id: int | None = None
    title: str
    score: float
    backdrop_url: str | None = None
    tagline: str | None = None


class Carousel(BaseModel):
    id: str
    label: str
    model: str
    explanation: str
    genres: list[str] | None = None
    movies: list[MovieOut]


class RecommendationsResponse(BaseModel):
    hero: Hero
    carousels: list[Carousel]


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------


class OnboardingMovie(BaseModel):
    movie_id: int
    tmdb_id: int | None = None
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)


class OnboardingMoviesResponse(BaseModel):
    movies: list[OnboardingMovie]


class OnboardingRating(BaseModel):
    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)


class OnboardingSubmitRequest(BaseModel):
    ratings: list[OnboardingRating] = Field(min_length=5)


class OnboardingSubmitResponse(BaseModel):
    user_token: str
    status: str = "ok"


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


class RateRequest(BaseModel):
    user_token: str
    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)


class RateResponse(BaseModel):
    status: str = "ok"
    profile_updated: bool = True


class WatchlistRequest(BaseModel):
    user_token: str
    movie_id: int
    action: Literal["add", "remove"]


class WatchlistResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Profile (mirrors frontend/mock/profile.json shape — keep field names verbatim)
# ---------------------------------------------------------------------------


class RatingHistoryItem(BaseModel):
    movie_id: int
    tmdb_id: int | None = None
    title: str
    rating: float
    timestamp: str


class TopGenre(BaseModel):
    genre: str
    count: int
    avg_rating: float


class Badge(BaseModel):
    title: str
    description: str
    icon: str


class ProfileResponse(BaseModel):
    total_ratings: int
    mean_rating: float
    hours_watched: int = 0
    streak_days: int = 0
    completion_rate: float = 0.0
    member_since: str | None = None
    badge: Badge | None = None
    top_genres: list[TopGenre] = Field(default_factory=list)
    rating_distribution: dict[str, int] = Field(default_factory=dict)
    feature_profile: dict[str, float] = Field(default_factory=dict)
    rating_history: list[RatingHistoryItem] = Field(default_factory=list)
    watchlist: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    version: str
    catalog_size: int
