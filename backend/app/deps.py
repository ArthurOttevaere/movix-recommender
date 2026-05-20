from __future__ import annotations

from fastapi import Request

from .recommender.base import HybridRecommender
from .services.catalog import MovieCatalog
from .services.popularity import PopularityIndex
from .services.state import UserStateStore
from .services.orchestrator import Orchestrator


def get_catalog(request: Request) -> MovieCatalog:
    return request.app.state.catalog


def get_popularity(request: Request) -> PopularityIndex:
    return request.app.state.popularity


def get_state_store(request: Request) -> UserStateStore:
    return request.app.state.users


def get_recommender(request: Request) -> HybridRecommender:
    return request.app.state.recommender


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator
