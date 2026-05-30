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

    now = datetime.utcnow()
    user = UserProfile(
        token=LENNY_TOKEN,
        ratings=ratings,
        rating_timestamps={mid: now for mid in ratings},
        created_at=now,
    )
    _STORE[LENNY_TOKEN] = user
    print(f"[store] profil démo Lenny chargé ({len(ratings)} films notés).")
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
