from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_catalog, get_state_store
from ..schemas import RateRequest, RateResponse, WatchlistRequest, WatchlistResponse
from ..services.catalog import MovieCatalog
from ..services.state import UserStateStore

router = APIRouter(tags=["interactions"])


def _resolve_movie_id(catalog: MovieCatalog, raw_id: int) -> int | None:
    if catalog.get(raw_id) is not None:
        return raw_id
    return catalog.resolve_tmdb_id(raw_id)


@router.post("/rate", response_model=RateResponse)
def rate_movie(
    body: RateRequest,
    catalog: MovieCatalog = Depends(get_catalog),
    users: UserStateStore = Depends(get_state_store),
) -> RateResponse:
    movie_id = _resolve_movie_id(catalog, body.movie_id)
    if movie_id is None:
        raise HTTPException(404, "Unknown movie_id")
    if not users.add_rating(body.user_token, movie_id, body.rating):
        raise HTTPException(404, "Unknown user_token")
    return RateResponse(status="ok", profile_updated=True)


@router.post("/watchlist", response_model=WatchlistResponse)
def update_watchlist(
    body: WatchlistRequest,
    catalog: MovieCatalog = Depends(get_catalog),
    users: UserStateStore = Depends(get_state_store),
) -> WatchlistResponse:
    movie_id = _resolve_movie_id(catalog, body.movie_id)
    if movie_id is None:
        raise HTTPException(404, "Unknown movie_id")
    if not users.update_watchlist(body.user_token, movie_id, body.action):
        raise HTTPException(404, "Unknown user_token")
    return WatchlistResponse(status="ok")
