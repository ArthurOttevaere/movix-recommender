import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class UserProfile:
    token: str
    ratings: dict = field(default_factory=dict)            # {int(movie_id): float(rating 0.5–5.0)}
    watchlist: list = field(default_factory=list)          # [int(movie_id)]
    rating_timestamps: dict = field(default_factory=dict)  # {int(movie_id): datetime}
    most_watched: list = field(default_factory=list)       # [(int(movie_id), int(n_watched))] desc — démo Lenny
    recent_ids: list = field(default_factory=list)         # [int(movie_id)] vus récemment — démo Lenny
    created_at: datetime = field(default_factory=datetime.utcnow)


_STORE: dict[str, UserProfile] = {}


# ─── Profil démo « Lenny » ────────────────────────────────────────────────────
# Pour la démo (live ou enregistrée), l'onboarding « nouvel utilisateur » à 5 films
# donne un signal trop faible. On expose donc un profil pré-chargé avec de vraies
# préférences : la bibliothèque personnelle de Lenny (library_lenny.csv) convertie
# en ratings implicites via recommender_building.compute_implicit_ratings (cf. PDF,
# section 5). Le front propose ce profil dans l'écran « Who's watching? » sous un
# token fixe ; tous les modèles backend (content/svd/userbased/ials/bpr) calculent
# alors leurs recommandations à la volée à partir de ces ratings riches.
#
# main.py n'est PAS modifié : le profil est matérialisé paresseusement à la
# première résolution de get_user(LENNY_TOKEN). Si la lib est introuvable, on
# échoue silencieusement (le profil n'apparaît simplement pas côté serveur).
LENNY_TOKEN = "lenny-demo-token"
_LIBRARY_LENNY = Path(__file__).resolve().parent.parent / "library_lenny.csv"


def _build_lenny() -> "UserProfile | None":
    """Construit le profil Lenny depuis library_lenny.csv (idempotent, défensif)."""
    try:
        from recommender_building import compute_implicit_ratings

        df = compute_implicit_ratings(str(_LIBRARY_LENNY))
        ratings = {
            int(mid): float(r)
            for mid, r in zip(df["movie_id"], df["implicit_rating"])
        }
    except Exception as exc:  # CSV manquant, import indisponible, etc.
        print(f"[store] profil démo Lenny indisponible : {exc}")
        return None

    if not ratings:
        return None

    # Lecture des colonnes signalétiques du CSV (une seule passe) :
    #   wishlist  → pré-remplit « My List »          (binaire 1/0)
    #   n_watched → top « Most Watched »             (compteur de visionnages)
    #   recent    → carrousel « Recently Watched »   (binaire 1/0)
    watchlist: list[int] = []
    most_watched: list[tuple[int, int]] = []
    recent_ids: list[int] = []
    try:
        import pandas as pd

        lib = pd.read_csv(_LIBRARY_LENNY)
        lib["movie_id"] = pd.to_numeric(lib["movie_id"], errors="coerce")
        lib = lib.dropna(subset=["movie_id"])
        lib["movie_id"] = lib["movie_id"].astype(int)

        watchlist = [int(m) for m in lib.loc[lib["wishlist"] == 1, "movie_id"]]

        mw = lib[lib["n_watched"] > 0].sort_values("n_watched", ascending=False)
        most_watched = [(int(m), int(n)) for m, n in zip(mw["movie_id"], mw["n_watched"])]

        recent_ids = [int(m) for m in lib.loc[lib["recent"] == 1, "movie_id"]]
    except Exception as exc:
        print(f"[store] colonnes biblio Lenny indisponibles : {exc}")

    now = datetime.utcnow()
    user = UserProfile(
        token=LENNY_TOKEN,
        ratings=ratings,
        watchlist=watchlist,
        most_watched=most_watched,
        recent_ids=recent_ids,
        rating_timestamps={mid: now for mid in ratings},
        created_at=now,
    )
    _STORE[LENNY_TOKEN] = user
    print(
        f"[store] profil démo Lenny chargé ({len(ratings)} notés, "
        f"{len(watchlist)} wishlist, {len(most_watched)} vus, {len(recent_ids)} récents)."
    )
    return user


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
    user = _STORE.get(token)
    if user is None and token == LENNY_TOKEN:
        user = _build_lenny()  # matérialisation paresseuse du profil démo
    return user


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
