from __future__ import annotations

from typing import Iterable


def exclude_ids(candidates: Iterable[int], excluded: set[int]) -> list[int]:
    return [c for c in candidates if c not in excluded]


def restrict_to(candidates: Iterable[int], allowed: set[int]) -> list[int]:
    return [c for c in candidates if c in allowed]
