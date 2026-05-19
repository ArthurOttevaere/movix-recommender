"""
Serveur FastAPI — ne pas modifier ce fichier.
Implémentez votre modèle dans backend/models/content.py, svd.py ou userbased.py.
"""

import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import utils
from backend.models import content, svd, userbased, load_all
from backend.store import (
    UserProfile,
    create_user,
    get_user,
    update_rating,
    update_watchlist,
)


# ─── Démarrage ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    utils.load_shared_artifacts()
    load_all()
    yield


app = FastAPI(title="Cinematch API", lifespan=lifespan)


# ─── Helpers internes ────────────────────────────────────────────────────────

def _require_user(token: str) -> UserProfile:
    user = get_user(token)
    if user is None:
        raise HTTPException(status_code=404, detail="Token invalide ou inconnu.")
    return user


def _pairs_to_movies(
    pairs: list[tuple[int, float]],
    user_ratings: dict,
    watchlist: list,
    limit: int,
) -> list[dict]:
    return [
        utils.movie_to_dict(mid, score, user_ratings, watchlist)
        for mid, score in pairs[:limit]
    ]


def _build_genre_carousels(
    all_pairs: list[tuple[int, float]],
    user_ratings: dict,
    watchlist: list,
) -> list[dict]:
    GENRES = [
        ("genre_drama",     "Drama",            "Top Drama Films"),
        ("genre_thriller",  "Thriller",         "Thriller & Suspense"),
        ("genre_scifi",     "Science Fiction",  "Science Fiction"),
        ("genre_animation", "Animation",        "Animation & Fantasy"),
        ("genre_crime",     "Crime",            "Crime & Mystery"),
    ]
    # Average score per movie across all model outputs
    score_acc: dict[int, list[float]] = {}
    for mid, score in all_pairs:
        score_acc.setdefault(mid, []).append(score)
    avg_scores = {mid: float(np.mean(scores)) for mid, scores in score_acc.items()}

    carousels = []
    for carousel_id, genre, label in GENRES:
        genre_pairs = [
            (mid, score) for mid, score in avg_scores.items()
            if genre in utils._get_genres(mid)
        ]
        genre_pairs.sort(key=lambda x: -x[1])
        if not genre_pairs:
            continue
        carousels.append({
            "id": carousel_id,
            "label": label,
            "model": f"genre:{genre}",
            "explanation": f"Best {genre} films for your taste",
            "genres": [genre],
            "movies": _pairs_to_movies(genre_pairs, user_ratings, watchlist, 15),
        })
    return carousels


def _build_profile_stats(user: UserProfile) -> dict:
    ratings = user.ratings
    if not ratings:
        return {
            "total_ratings": 0, "mean_rating": 0.0, "hours_watched": 0,
            "completion_rate": 0, "streak_days": 0,
            "member_since": user.created_at.strftime("%Y-%m-%d"),
            "top_genres": [], "rating_distribution": {str(k): 0 for k in range(1, 6)},
            "feature_profile": {}, "badge": None,
            "rating_history": [], "watchlist": user.watchlist,
        }

    values = list(ratings.values())
    mean_r = float(np.mean(values))

    dist = {str(k): 0 for k in range(1, 6)}
    for r in values:
        dist[str(min(5, max(1, round(r))))] += 1

    genre_counts: Counter = Counter()
    genre_rating_sums: dict[str, list[float]] = {}
    for mid, r in ratings.items():
        for g in utils._get_genres(mid):
            genre_counts[g] += 1
            genre_rating_sums.setdefault(g, []).append(r)

    top_genres = [
        {"genre": g, "count": c, "avg_rating": round(float(np.mean(genre_rating_sums[g])), 2)}
        for g, c in genre_counts.most_common(5)
    ]
    total_genre = sum(genre_counts.values()) or 1
    feature_profile = {
        g: round(c / total_genre, 3)
        for g, c in genre_counts.most_common(8)
    }

    history = []
    for mid, r in list(ratings.items())[-20:]:
        entry = utils.movie_to_dict(mid, None, {}, [])
        entry["rating"] = r
        ts = user.rating_timestamps.get(mid, user.created_at)
        entry["timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%S")
        history.append(entry)

    return {
        "total_ratings": len(ratings),
        "mean_rating": round(mean_r, 2),
        "hours_watched": len(ratings) * 2,
        "completion_rate": min(95, 60 + len(ratings) * 2),
        "streak_days": 1,
        "member_since": user.created_at.strftime("%Y-%m-%d"),
        "top_genres": top_genres,
        "rating_distribution": dist,
        "feature_profile": feature_profile,
        "badge": None,
        "rating_history": history,
        "watchlist": user.watchlist,
    }


# ─── Onboarding ──────────────────────────────────────────────────────────────

@app.get("/onboarding/movies")
def onboarding_movies():
    """Retourne 20 films populaires couvrant des genres diversifiés."""
    candidates = utils.popular_movies(100, exclude_ids=None)
    # Sélection greedy pour maximiser la diversité des genres
    selected: list[tuple[int, float]] = []
    seen_genres: set[str] = set()
    for mid, score in candidates:
        genres = utils._get_genres(mid)
        if any(g not in seen_genres for g in genres) or len(selected) < 5:
            selected.append((mid, score))
            seen_genres.update(genres)
        if len(selected) == 20:
            break
    # Compléter si besoin
    if len(selected) < 20:
        selected_ids = {mid for mid, _ in selected}
        for mid, score in candidates:
            if mid not in selected_ids:
                selected.append((mid, score))
            if len(selected) == 20:
                break

    return {"movies": [utils.movie_to_dict(mid, score, {}, []) for mid, score in selected]}


class OnboardingSubmitBody(BaseModel):
    ratings: dict[str, float]  # clés en string (JSON)


@app.post("/onboarding/submit")
def onboarding_submit(body: OnboardingSubmitBody):
    if len(body.ratings) < 5:
        raise HTTPException(status_code=400, detail="Minimum 5 films à noter.")
    initial = {int(k): v for k, v in body.ratings.items()}
    user = create_user(initial)
    return {"user_token": user.token, "status": "ok"}


# ─── Recommandations ─────────────────────────────────────────────────────────

@app.get("/recommendations/{token}")
async def get_recommendations(token: str):
    user = _require_user(token)
    ratings = user.ratings
    rated_ids = set(ratings.keys())
    N = 25

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        loop.run_in_executor(None, content.recommend, ratings, N),
        loop.run_in_executor(None, svd.recommend, ratings, N),
        loop.run_in_executor(None, userbased.recommend, ratings, N),
        return_exceptions=True,
    )

    fallback = utils.popular_movies(N, exclude_ids=rated_ids)

    def safe(res) -> list[tuple[int, float]]:
        if isinstance(res, Exception) or not res:
            return fallback
        return res

    content_recs  = safe(results[0])
    svd_recs      = safe(results[1])
    ub_recs       = safe(results[2])

    # Ensemble : interleave les 3 modèles, dédupliqué
    seen: set[int] = set()
    ensemble: list[tuple[int, float]] = []
    for pairs in (content_recs, svd_recs, ub_recs):
        for mid, score in pairs:
            if mid not in seen:
                seen.add(mid)
                ensemble.append((mid, score))

    # Hero : meilleur film du content-based
    hero_mid, hero_score = content_recs[0]
    hero = utils.movie_to_dict(hero_mid, hero_score, ratings, user.watchlist)
    hero["backdrop_url"] = None
    hero["tagline"] = ""

    all_pairs = content_recs + svd_recs + ub_recs
    genre_carousels = _build_genre_carousels(all_pairs, ratings, user.watchlist)

    return {
        "hero": hero,
        "carousels": [
            {
                "id": "content_based",
                "label": "Recommandé pour vous",
                "model": "content_based",
                "explanation": "Based on your taste profile",
                "movies": _pairs_to_movies(content_recs[1:], ratings, user.watchlist, 20),
            },
            {
                "id": "latent_factor",
                "label": "In Your Style — SVD Model",
                "model": "svd",
                "explanation": "Hidden patterns in your ratings",
                "movies": _pairs_to_movies(svd_recs, ratings, user.watchlist, 20),
            },
            {
                "id": "user_based",
                "label": "Viewers Like You Also Watched",
                "model": "user_based",
                "explanation": "Users with similar taste profiles",
                "movies": _pairs_to_movies(ub_recs, ratings, user.watchlist, 20),
            },
            {
                "id": "discovery",
                "label": "Step Outside Your Comfort Zone",
                "model": "ensemble",
                "explanation": "High-diversity mix from all models",
                "movies": _pairs_to_movies(ensemble, ratings, user.watchlist, 20),
            },
            *genre_carousels,
        ],
    }


# ─── Notation ────────────────────────────────────────────────────────────────

class RateBody(BaseModel):
    user_token: str
    movie_id: int
    rating: float


@app.post("/rate")
def rate_movie(body: RateBody):
    user = _require_user(body.user_token)
    if not (0.5 <= body.rating <= 5.0):
        raise HTTPException(status_code=422, detail="Rating doit être entre 0.5 et 5.0.")
    is_new = update_rating(user, body.movie_id, body.rating)
    return {"status": "ok", "profile_updated": is_new}


# ─── Watchlist ───────────────────────────────────────────────────────────────

class WatchlistBody(BaseModel):
    user_token: str
    movie_id: int
    action: Literal["add", "remove"]


@app.post("/watchlist")
def update_watchlist_endpoint(body: WatchlistBody):
    user = _require_user(body.user_token)
    update_watchlist(user, body.movie_id, body.action)
    return {"status": "ok"}


# ─── Profil ──────────────────────────────────────────────────────────────────

@app.get("/profile/{token}")
def get_profile(token: str):
    user = _require_user(token)
    return _build_profile_stats(user)


# ─── Serving du frontend (doit rester EN DERNIER) ────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
