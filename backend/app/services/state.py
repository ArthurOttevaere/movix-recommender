from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class UserState:
    ratings: dict[int, float] = field(default_factory=dict)
    watchlist: set[int] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    rating_history: list[tuple[int, float, float]] = field(default_factory=list)
    """List of (movie_id, rating, timestamp) in insertion order."""


class UserStateStore:
    """In-memory token → UserState store. Sufficient for the MVP single-process backend."""

    def __init__(self):
        self._users: dict[str, UserState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _new_token() -> str:
        return "u_" + secrets.token_urlsafe(12)

    def create(self, ratings: dict[int, float]) -> str:
        token = self._new_token()
        now = time.time()
        history = [(int(mid), float(r), now) for mid, r in ratings.items()]
        with self._lock:
            self._users[token] = UserState(
                ratings={int(mid): float(r) for mid, r in ratings.items()},
                created_at=now,
                last_seen_at=now,
                rating_history=history,
            )
        return token

    def get(self, token: str) -> UserState | None:
        user = self._users.get(token)
        if user is not None:
            user.last_seen_at = time.time()
        return user

    def add_rating(self, token: str, movie_id: int, rating: float) -> bool:
        with self._lock:
            user = self._users.get(token)
            if user is None:
                return False
            mid = int(movie_id)
            user.ratings[mid] = float(rating)
            user.rating_history.append((mid, float(rating), time.time()))
            user.last_seen_at = time.time()
            return True

    def update_watchlist(self, token: str, movie_id: int, action: Literal["add", "remove"]) -> bool:
        with self._lock:
            user = self._users.get(token)
            if user is None:
                return False
            mid = int(movie_id)
            if action == "add":
                user.watchlist.add(mid)
            else:
                user.watchlist.discard(mid)
            user.last_seen_at = time.time()
            return True
