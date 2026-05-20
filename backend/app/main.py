from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .recommender.dummy import DummyHybrid
from .recommender.loader import load_recommender
from .routers import interactions, onboarding, profile, recommendations
from .schemas import HealthResponse
from .services.catalog import MovieCatalog
from .services.orchestrator import Orchestrator
from .services.popularity import PopularityIndex
from .services.state import UserStateStore

log = logging.getLogger("movix.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("Loading catalog from %s", settings.movies_path)
    catalog = MovieCatalog.from_paths(settings.movies_path, settings.links_path, settings.tmdb_cache_path)
    log.info("Catalog loaded: %d movies", len(catalog))

    log.info("Loading popularity index from %s", settings.ratings_path)
    popularity = PopularityIndex.from_path(settings.ratings_path)

    log.info("Loading recommender from %s", settings.artifact_path)
    recommender = load_recommender(
        settings.artifact_path,
        fallback_factory=lambda: DummyHybrid(popularity),
    )

    app.state.settings = settings
    app.state.catalog = catalog
    app.state.popularity = popularity
    app.state.users = UserStateStore()
    app.state.recommender = recommender
    app.state.orchestrator = Orchestrator(recommender, catalog, popularity, settings)
    log.info("Backend ready (model=%s v=%s)", recommender.name, recommender.version)
    try:
        yield
    finally:
        log.info("Shutting down")


app = FastAPI(title="Movix Recommender API", version="0.1.0", lifespan=lifespan)

# CORS — the frontend runs from file:// or a local static server. allow_origins=["*"]
# is fine for dev; tighten before production. allow_credentials must be False with "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


app.include_router(onboarding.router)
app.include_router(recommendations.router)
app.include_router(interactions.router)
app.include_router(profile.router)


@app.get("/healthz", response_model=HealthResponse)
def healthz(request: Request) -> HealthResponse:
    rec = request.app.state.recommender
    catalog = request.app.state.catalog
    return HealthResponse(
        status="ok",
        model=rec.name,
        version=rec.version,
        catalog_size=len(catalog),
    )
