import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import utils
from backend.models import content, userbased, ials, bpr, load_all
from backend.store import (
    UserProfile,
    create_user,
    get_user,
    update_rating,
    update_watchlist,
)


# ─── Launching ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    utils.load_shared_artifacts()
    load_all()
    yield


app = FastAPI(title="Cinematch API", lifespan=lifespan)


# ─── Anti-cache (dev) ────────────────────────────────────────────────────────
# StaticFiles doesn't send any Cache-Control headers: the browser then applies a heuristic cache
# and serves an old HTML/JS without revalidating. After a restart of the
# server (or a code modification), the old JS in cache remains loaded → page that
# "loads infinitely" until the cache is manually cleared. We therefore force the
# browser to always fetch the fresh version.

@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ─── Helpers ────────────────────────────────────────────────────────

def _require_user(token: str) -> UserProfile:
    user = get_user(token)
    if user is None:
        raise HTTPException(status_code=404, detail="Invalid Token or unknown.")
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
            "watchlist_movies": [
                utils.movie_to_dict(mid, None, {}, user.watchlist)
                for mid in user.watchlist
            ],
            "most_watched": [
                {**utils.movie_to_dict(mid, None, {}, user.watchlist), "n_watched": int(n)}
                for mid, n in user.most_watched
            ],
            "recently_watched": [
                utils.movie_to_dict(mid, None, {}, user.watchlist)
                for mid in user.recent_ids
            ],
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

    # Hours watched: based on the actual number of films viewed (CSV n_watched,
    # rewatches included) when available — falls back to the rated count otherwise.
    # ~2h average runtime per film.
    total_watched = sum(int(n) for _, n in user.most_watched)
    hours_watched = (total_watched * 2) if total_watched else (len(ratings) * 2)

    return {
        "total_ratings": len(ratings),
        "mean_rating": round(mean_r, 2),
        "hours_watched": hours_watched,
        "completion_rate": min(95, 60 + len(ratings) * 2),
        "streak_days": 1,
        "member_since": user.created_at.strftime("%Y-%m-%d"),
        "top_genres": top_genres,
        "rating_distribution": dist,
        "feature_profile": feature_profile,
        "badge": None,
        "rating_history": history,
        "watchlist": user.watchlist,
        "watchlist_movies": [
            utils.movie_to_dict(mid, None, ratings, user.watchlist)
            for mid in user.watchlist
        ],
        "most_watched": [
            {**utils.movie_to_dict(mid, None, ratings, user.watchlist), "n_watched": int(n)}
            for mid, n in user.most_watched
        ],
        "recently_watched": [
            utils.movie_to_dict(mid, None, ratings, user.watchlist)
            for mid in user.recent_ids
        ],
    }


# ─── Onboarding ──────────────────────────────────────────────────────────────

# Onboarding : vivier de films reconnaissables PAR époque, puis sélection
# farthest-first (espace latent iALS) SOUS QUOTAS d'époque → the seeds
# cover both taste diversity and a realistic mix of years (otherwise popularity = accumulated ratings bury
# recent films and the median of seeds drops to ~1993).

ONBOARDING_POOL_PER_ERA = 150   # most rated movies per era (recognizable)
ONBOARDING_ERAS = [             # (label, year_min, year_max_exclue, quota)
    ("Pre-1980", 0,    1980, 4),
    ("1980s",    1980, 1990, 6),
    ("1990s",    1990, 2000, 8),
    ("2000s",    2000, 2010, 11),
    ("2010s",    2010, 9999, 11),
]
ONBOARDING_N = sum(q for *_, q in ONBOARDING_ERAS)  # 40
ONBOARDING_BRANCH = 6           # at each step, we randomly select from the N most distant films
                                # → onboarding varies from one session to another
                                # without sacrificing the dispersion (so it remains optimal)


def _select_onboarding_seeds() -> list[tuple[int, float]]:
    """Onboarding seeds balanced per era and diverse in taste.

    1. Recognizable pool = the `POOL_PER_ERA` most rated movies of each
       era (global popularity favoring old films, we apply it
       *within* each era to bring up recent films).
    2. Farthest-first traversal (k-center) in the latent iALS space with a quota
       per era : we choose at each step the film the most distant (taste direction,
       cosine distance) from the seeds already taken, while disabling an era
       as soon as its quota is reached. → maximum taste dispersion *under* the
       constraint of a realistic mix of years.

    Fallback to [] if the iALS factors are unavailable (the caller then completes
    with popularity).
    """
    all_pop = utils.popular_movies(20000, exclude_ids=None)  # all the catalog, sorted by popularity

    # 1. Recognizable pool per era + movie_id → era index + quotas
    candidates: list[tuple[int, float]] = []
    era_of: dict[int, int] = {}
    quota: dict[int, int] = {}
    for ei, (_, lo, hi, q) in enumerate(ONBOARDING_ERAS):
        quota[ei] = q
        taken = 0
        for mid, score in all_pop:
            y = utils.movie_year(mid)
            if y is None or not (lo <= y < hi):
                continue
            candidates.append((mid, score))
            era_of[mid] = ei
            taken += 1
            if taken >= ONBOARDING_POOL_PER_ERA:
                break

    score_by_id = {mid: score for mid, score in candidates}
    found_ids, X = ials.item_factor_vectors([mid for mid, _ in candidates])
    if X is None:
        return []

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    eidx = np.array([era_of[mid] for mid in found_ids])
    scores = np.array([score_by_id[mid] for mid in found_ids])
    n = len(found_ids)

    # 2. Farthest-first randomised under era quotas.
    #    Randomisation (rng without seed → varies at each call) : departure chosen from
    #    the most popular films, then at each step we randomly select from
    #    the ONBOARDING_BRANCH most distant films from the already selected seeds. The
    #    dispersion remains high (we always choose from the most distant)
    #    but the selection varies from one session to another.
    rng = np.random.default_rng()
    avail = np.ones(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    filled = {ei: 0 for ei in quota}
    selected: list[int] = []

    def _take(i: int) -> None:
        selected.append(i)
        avail[i] = False
        e = int(eidx[i])
        filled[e] += 1
        if filled[e] >= quota[e]:        # quota reached → the era is removed
            avail[eidx == e] = False

    # Departure : at random among the ~15 most popular candidates (recognizable)
    pop_order = np.argsort(-scores)
    start = int(rng.choice(pop_order[:min(15, n)]))
    _take(start)
    min_dist = np.minimum(min_dist, 1.0 - Xn @ Xn[start])

    while len(selected) < ONBOARDING_N and avail.any():
        avail_idx = np.where(avail)[0]
        farthest = avail_idx[np.argsort(-min_dist[avail_idx])[:ONBOARDING_BRANCH]]
        nxt = int(rng.choice(farthest))
        _take(nxt)
        min_dist = np.minimum(min_dist, 1.0 - Xn @ Xn[nxt])

    return [(found_ids[i], score_by_id[found_ids[i]]) for i in selected]


@app.get("/onboarding/movies")
def onboarding_movies():
    """Return 40 recognizable films, covering diverse tastes AND a realistic mix
    of eras (latent iALS diversity under era quotas)."""
    selected = _select_onboarding_seeds()

    
    if len(selected) < ONBOARDING_N:
        seen = {mid for mid, _ in selected}
        for mid, score in utils.popular_movies(ONBOARDING_N * 4, exclude_ids=None):
            if mid not in seen:
                selected.append((mid, score))
                seen.add(mid)
            if len(selected) == ONBOARDING_N:
                break

    return {"movies": [utils.movie_to_dict(mid, score, {}, []) for mid, score in selected[:ONBOARDING_N]]}


class OnboardingSubmitBody(BaseModel):
    ratings: dict[str, float]  # keys as strings (JSON)


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
    # N is generous so the Discover page can filter each model's picks by genre
    # and still fill a row; the home page slices each row to ~20 client-side.
    N = 100

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        loop.run_in_executor(None, content.recommend, ratings, N),
        loop.run_in_executor(None, userbased.recommend, ratings, N),
        loop.run_in_executor(None, ials.recommend, ratings, N),
        loop.run_in_executor(None, bpr.recommend, ratings, N),
        return_exceptions=True,
    )

    fallback = utils.popular_movies(N, exclude_ids=rated_ids)

    def safe(res) -> list[tuple[int, float]]:
        if isinstance(res, Exception) or not res:
            return fallback
        return res

    content_recs  = safe(results[0])
    ub_recs       = safe(results[1])
    ials_recs     = safe(results[2])
    bpr_recs      = safe(results[3])

    # Ensemble : interleave the 2 models, deduplicated
    seen: set[int] = set()
    ensemble: list[tuple[int, float]] = []
    for pairs in (content_recs, ub_recs):
        for mid, score in pairs:
            if mid not in seen:
                seen.add(mid)
                ensemble.append((mid, score))

    # Hero : best film from the content-based model
    hero_mid, hero_score = content_recs[0]
    hero = utils.movie_to_dict(hero_mid, hero_score, ratings, user.watchlist)
    hero["backdrop_url"] = None
    hero["tagline"] = ""

    all_pairs = content_recs + ub_recs + ials_recs + bpr_recs
    genre_carousels = _build_genre_carousels(all_pairs, ratings, user.watchlist)

    return {
        "hero": hero,
        "carousels": [
            {
                "id": "content_based",
                "label": "Recommandé pour vous",
                "model": "content_based",
                "explanation": "Based on your taste profile",
                "movies": _pairs_to_movies(content_recs[1:], ratings, user.watchlist, N),
            },
            {
                "id": "user_based",
                "label": "Viewers Like You Also Watched",
                "model": "user_based",
                "explanation": "Users with similar taste profiles",
                "movies": _pairs_to_movies(ub_recs, ratings, user.watchlist, N),
            },
            {
                "id": "top_picks",
                "label": "Top Picks For You",
                "model": "ials",
                "explanation": "Confidence-weighted implicit ALS",
                "movies": _pairs_to_movies(ials_recs, ratings, user.watchlist, N),
            },
            {
                "id": "discover_new",
                "label": "Discover Something New",
                "model": "bpr",
                "explanation": "Ranking-based pairwise model",
                "movies": _pairs_to_movies(bpr_recs, ratings, user.watchlist, N),
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


# ─── Similar movies (More Like This) ───────────────────────────────────────

@app.get("/similar/{movie_id}")
def similar_movies(movie_id: int):
    """Similar movies to `movie_id` based on the content-based (item-item) model."""
    pairs = content.similar_items(movie_id, 12)
    return {"movies": [utils.movie_to_dict(mid, score, {}, []) for mid, score in pairs]}


# ─── Rating ────────────────────────────────────────────────────────────────

class RateBody(BaseModel):
    user_token: str
    movie_id: int
    rating: float


@app.post("/rate")
def rate_movie(body: RateBody):
    user = _require_user(body.user_token)
    if not (0.5 <= body.rating <= 5.0):
        raise HTTPException(status_code=422, detail="Rating must be between 0.5 and 5.0.")
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


# ─── Profile ──────────────────────────────────────────────────────────────────

@app.get("/profile/{token}")
def get_profile(token: str):
    user = _require_user(token)
    return _build_profile_stats(user)


# ─── Serving of frontend (must be last) ────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
