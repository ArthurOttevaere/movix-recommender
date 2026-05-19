import secrets
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UserProfile:
    token: str
    ratings: dict = field(default_factory=dict)            # {int(movie_id): float(rating 0.5–5.0)}
    watchlist: list = field(default_factory=list)          # [int(movie_id)]
    rating_timestamps: dict = field(default_factory=dict)  # {int(movie_id): datetime}
    created_at: datetime = field(default_factory=datetime.utcnow)


_STORE: dict[str, UserProfile] = {}


def create_user(initial_ratings: dict[int, float]) -> UserProfile:
    token = secrets.token_urlsafe(16)
    now = datetime.utcnow()
    user = UserProfile(
        token=token,
        ratings=initial_ratings,
        rating_timestamps={mid: now for mid in initial_ratings},
        created_at=now,
    )
    _STORE[token] = user
    return user


def get_user(token: str) -> UserProfile | None:
    return _STORE.get(token)


def update_rating(profile: UserProfile, movie_id: int, rating: float) -> bool:
    """Returns True if this is a new rating (not an update)."""
    is_new = movie_id not in profile.ratings
    profile.ratings[movie_id] = rating
    profile.rating_timestamps[movie_id] = datetime.utcnow()
    return is_new


def update_watchlist(profile: UserProfile, movie_id: int, action: str) -> None:
    if action == "add" and movie_id not in profile.watchlist:
        profile.watchlist.append(movie_id)
    elif action == "remove" and movie_id in profile.watchlist:
        profile.watchlist.remove(movie_id)
