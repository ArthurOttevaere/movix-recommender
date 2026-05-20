from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable

from .base import HybridRecommender

log = logging.getLogger(__name__)


def load_recommender(
    artifact_path: Path,
    *,
    fallback_factory: Callable[[], HybridRecommender],
) -> HybridRecommender:
    """Charge un HybridRecommender picklé, ou retourne le fallback en cas d'absence/erreur."""
    if artifact_path.exists():
        try:
            with open(artifact_path, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, HybridRecommender):
                raise TypeError(
                    f"Pickle at {artifact_path} is not a HybridRecommender (got {type(obj).__name__})"
                )
            log.info("Loaded hybrid model name=%s version=%s", obj.name, obj.version)
            return obj
        except Exception:
            log.exception("Failed to load %s — falling back to default recommender", artifact_path)
    else:
        log.warning("No artifact at %s — falling back to default recommender", artifact_path)
    return fallback_factory()
