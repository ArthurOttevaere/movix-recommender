from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_catalog, get_state_store
from ..schemas import Badge, ProfileResponse, RatingHistoryItem, TopGenre
from ..services.catalog import MovieCatalog
from ..services.state import UserState, UserStateStore

router = APIRouter(prefix="/profile", tags=["profile"])


def _build_profile(user: UserState, catalog: MovieCatalog) -> ProfileResponse:
    total = len(user.ratings)
    mean = round(sum(user.ratings.values()) / total, 2) if total else 0.0

    # Bucket distribution by integer rating bin (1..5).
    distribution: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    genre_counts: dict[str, int] = defaultdict(int)
    genre_sums: dict[str, float] = defaultdict(float)

    for mid, r in user.ratings.items():
        bucket = max(1, min(5, int(round(r))))
        distribution[str(bucket)] = distribution.get(str(bucket), 0) + 1
        row = catalog.get(mid)
        if row:
            for g in row.genres:
                genre_counts[g] += 1
                genre_sums[g] += r

    top_genres = [
        TopGenre(genre=g, count=genre_counts[g], avg_rating=round(genre_sums[g] / genre_counts[g], 2))
        for g in sorted(genre_counts, key=lambda x: (-genre_counts[x], -genre_sums[x] / max(1, genre_counts[x])))
    ][:5]

    feature_profile = {}
    if top_genres:
        total_count = sum(g.count for g in top_genres)
        feature_profile = {g.genre: round(g.count / total_count, 2) for g in top_genres}

    history: list[RatingHistoryItem] = []
    for mid, r, ts in reversed(user.rating_history[-30:]):
        row = catalog.get(mid)
        history.append(
            RatingHistoryItem(
                movie_id=mid,
                tmdb_id=row.tmdb_id if row else None,
                title=row.title if row else f"Film #{mid}",
                rating=r,
                timestamp=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            )
        )

    badge: Badge | None = None
    if top_genres and top_genres[0].count >= 3:
        top = top_genres[0]
        badge = Badge(
            title=f"{top.genre} Fan",
            description=f"Vous avez noté {top.count} films de {top.genre}",
            icon="🎬",
        )

    member_since = datetime.fromtimestamp(user.created_at, tz=timezone.utc).date().isoformat()

    return ProfileResponse(
        total_ratings=total,
        mean_rating=mean,
        hours_watched=total * 2,  # stub
        completion_rate=0.0,
        streak_days=0,
        member_since=member_since,
        badge=badge,
        top_genres=top_genres,
        rating_distribution=distribution,
        feature_profile=feature_profile,
        rating_history=history,
        watchlist=sorted(user.watchlist),
    )


@router.get("/{user_token}", response_model=ProfileResponse)
def get_profile(
    user_token: str,
    catalog: MovieCatalog = Depends(get_catalog),
    users: UserStateStore = Depends(get_state_store),
) -> ProfileResponse:
    user = users.get(user_token)
    if user is None:
        raise HTTPException(404, "Unknown user_token")
    return _build_profile(user, catalog)
