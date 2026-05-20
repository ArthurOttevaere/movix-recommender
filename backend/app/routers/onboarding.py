from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_catalog, get_state_store
from ..schemas import (
    OnboardingMovie,
    OnboardingMoviesResponse,
    OnboardingSubmitRequest,
    OnboardingSubmitResponse,
)
from ..services.catalog import MovieCatalog
from ..services.onboarding_seed import resolve_seed
from ..services.state import UserStateStore

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


@router.get("/movies", response_model=OnboardingMoviesResponse)
def get_onboarding_movies(catalog: MovieCatalog = Depends(get_catalog)) -> OnboardingMoviesResponse:
    items = [OnboardingMovie(**row) for row in resolve_seed(catalog)]
    if not items:
        raise HTTPException(500, "Onboarding seed could not be resolved against the catalog")
    return OnboardingMoviesResponse(movies=items)


@router.post("/submit", response_model=OnboardingSubmitResponse)
def submit_onboarding(
    body: OnboardingSubmitRequest,
    catalog: MovieCatalog = Depends(get_catalog),
    users: UserStateStore = Depends(get_state_store),
) -> OnboardingSubmitResponse:
    ratings: dict[int, float] = {}
    for entry in body.ratings:
        mid = entry.movie_id
        if catalog.get(mid) is None:
            # Maybe the frontend is still sending the legacy TMDB id — try to remap.
            resolved = catalog.resolve_tmdb_id(mid)
            if resolved is None:
                # Skip unknown — we still need >=5 valid; checked below.
                continue
            mid = resolved
        ratings[mid] = entry.rating

    if len(ratings) < 5:
        raise HTTPException(422, "Need at least 5 ratings on known movies (after id resolution)")

    token = users.create(ratings)
    return OnboardingSubmitResponse(user_token=token, status="ok")
