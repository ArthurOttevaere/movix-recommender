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
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import utils
from backend.models import content, svd, userbased, ials, bpr, load_all
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


# ─── Anti-cache (dev) ────────────────────────────────────────────────────────
# StaticFiles n'envoie aucun Cache-Control : le navigateur applique alors un cache
# heuristique et ressert un vieux HTML/JS sans revalider. Après un redémarrage du
# serveur (ou une modif de code), l'ancien JS en cache reste chargé → page qui
# "charge à l'infini" tant qu'on n'a pas vidé le cache à la main. On force donc le
# navigateur à toujours récupérer la version fraîche.
@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


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

# Onboarding : vivier de films reconnaissables PAR époque, puis sélection
# farthest-first (espace latent iALS) SOUS QUOTAS d'époque → les graines
# couvrent à la fois la diversité de goût (informatif pour les modèles) ET un
# mélange d'années réaliste (sinon la popularité = notes cumulées enterre les
# films récents et le médian des graines retombe vers ~1993).
ONBOARDING_POOL_PER_ERA = 150   # films les plus notés retenus par époque (reconnaissables)
ONBOARDING_ERAS = [             # (label, année_min, année_max_exclue, quota)
    ("Pre-1980", 0,    1980, 4),
    ("1980s",    1980, 1990, 6),
    ("1990s",    1990, 2000, 8),
    ("2000s",    2000, 2010, 11),
    ("2010s",    2010, 9999, 11),
]
ONBOARDING_N = sum(q for *_, q in ONBOARDING_ERAS)  # 40


def _select_onboarding_seeds() -> list[tuple[int, float]]:
    """Graines d'onboarding équilibrées par époque ET diverses en goût.

    1. Vivier reconnaissable = les `POOL_PER_ERA` films les plus notés de chaque
       époque (la popularité globale favorisant les vieux films, on l'applique
       *à l'intérieur* de chaque époque pour faire remonter des films récents).
    2. Traversée farthest-first (k-center) dans l'espace latent iALS avec un quota
       par époque : on choisit à chaque pas le film le plus éloigné (direction de
       goût, distance cosinus) des graines déjà prises, en désactivant une époque
       dès que son quota est atteint. → dispersion maximale du goût *sous* la
       contrainte d'un mélange d'années.

    Repli sur [] si les facteurs iALS sont indisponibles (le caller complète alors
    par popularité).
    """
    all_pop = utils.popular_movies(20000, exclude_ids=None)  # tout le catalogue, trié par popularité

    # 1. Vivier par époque + table movie_id → index d'époque + quotas
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

    # 2. Farthest-first sous quotas d'époque
    avail = np.ones(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    filled = {ei: 0 for ei in quota}
    selected: list[int] = []

    def _take(i: int) -> None:
        selected.append(i)
        avail[i] = False
        e = int(eidx[i])
        filled[e] += 1
        if filled[e] >= quota[e]:        # quota atteint → on retire toute l'époque
            avail[eidx == e] = False

    _take(int(np.argmax(scores)))        # départ : le film le plus connu
    min_dist = np.minimum(min_dist, 1.0 - Xn @ Xn[selected[0]])

    while len(selected) < ONBOARDING_N and avail.any():
        nxt = int(np.argmax(np.where(avail, min_dist, -np.inf)))
        _take(nxt)
        min_dist = np.minimum(min_dist, 1.0 - Xn @ Xn[nxt])

    return [(found_ids[i], score_by_id[found_ids[i]]) for i in selected]


@app.get("/onboarding/movies")
def onboarding_movies():
    """Retourne 40 films reconnaissables, couvrant des goûts variés ET un mélange
    d'époques (diversité latente iALS sous quotas d'époque)."""
    selected = _select_onboarding_seeds()

    # Repli / complétion par popularité si iALS indisponible ou pas assez de films.
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
    # N is generous so the Discover page can filter each model's picks by genre
    # and still fill a row; the home page slices each row to ~20 client-side.
    N = 100

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        loop.run_in_executor(None, content.recommend, ratings, N),
        loop.run_in_executor(None, svd.recommend, ratings, N),
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
    svd_recs      = safe(results[1])
    ub_recs       = safe(results[2])
    ials_recs     = safe(results[3])
    bpr_recs      = safe(results[4])

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

    all_pairs = content_recs + svd_recs + ub_recs + ials_recs + bpr_recs
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
