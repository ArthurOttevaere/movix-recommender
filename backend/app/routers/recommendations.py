from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_orchestrator, get_state_store
from ..schemas import RecommendationsResponse
from ..services.orchestrator import Orchestrator
from ..services.state import UserStateStore

router = APIRouter(tags=["recommendations"])


@router.get("/recommendations/{user_token}", response_model=RecommendationsResponse)
def get_recommendations(
    user_token: str,
    orch: Orchestrator = Depends(get_orchestrator),
    users: UserStateStore = Depends(get_state_store),
) -> RecommendationsResponse:
    user = users.get(user_token)
    if user is None:
        raise HTTPException(404, "Unknown user_token")
    return orch.build(user)
