from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    movies_path: Path
    links_path: Path
    ratings_path: Path
    tmdb_cache_path: Path
    artifact_path: Path

    top_n_default: int = 12
    top_n_genre: int = 10
    top_n_trending: int = 10
    top_n_discovery: int = 10
    mmr_lambda: float = 0.55

    rating_min: float = 0.5
    rating_max: float = 5.0
    onboarding_min_ratings: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    data = repo_root / "data" / "hackathon"
    return Settings(
        repo_root=repo_root,
        movies_path=data / "content" / "movies.csv",
        links_path=data / "content" / "links.csv",
        ratings_path=data / "evidence" / "ratings.csv",
        tmdb_cache_path=data / "content" / "tmdb_cache.json",
        artifact_path=repo_root / "backend" / "artifacts" / "hybrid_model.pkl",
    )
