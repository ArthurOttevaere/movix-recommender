"""End-to-end contract tests against the FastAPI app with DummyHybrid."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["catalog_size"] > 0
    assert body["model"] == "dummy_popularity"


def test_onboarding_movies(client):
    r = client.get("/onboarding/movies")
    assert r.status_code == 200
    body = r.json()
    assert len(body["movies"]) >= 20
    sample = body["movies"][0]
    assert {"movie_id", "tmdb_id", "title", "year", "genres"} <= sample.keys()


def _onboarded_token(client) -> str:
    seed = client.get("/onboarding/movies").json()["movies"]
    payload = {"ratings": [{"movie_id": m["movie_id"], "rating": 5.0} for m in seed[:5]]}
    r = client.post("/onboarding/submit", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["user_token"]


def test_onboarding_submit_requires_5(client):
    r = client.post("/onboarding/submit", json={"ratings": [{"movie_id": 1, "rating": 5.0}]})
    assert r.status_code == 422


def test_recommendations_shape(client):
    token = _onboarded_token(client)
    r = client.get(f"/recommendations/{token}")
    assert r.status_code == 200, r.text
    body = r.json()

    # Hero shape
    hero = body["hero"]
    assert {"movie_id", "tmdb_id", "title", "score"} <= hero.keys()

    # Carousels structure & required keys per movie
    carousels = body["carousels"]
    assert len(carousels) >= 4
    required_movie_keys = {
        "movie_id", "tmdb_id", "title", "year", "genres",
        "poster_url", "score", "in_watchlist", "user_rating",
    }
    for c in carousels:
        assert {"id", "label", "model", "explanation", "movies"} <= c.keys()
        assert len(c["movies"]) > 0
        for m in c["movies"]:
            assert required_movie_keys <= m.keys(), f"missing keys in {c['id']}: {required_movie_keys - m.keys()}"

    # Trending carousel should expose `rank`
    trending = next((c for c in carousels if c["model"] == "trending"), None)
    assert trending is not None
    assert all(m.get("rank") for m in trending["movies"])

    # tmdb_id coverage on recommended movies
    all_movies = [m for c in carousels for m in c["movies"]]
    with_tmdb = sum(1 for m in all_movies if m["tmdb_id"] is not None)
    assert with_tmdb / max(1, len(all_movies)) >= 0.9, "tmdb_id coverage <90%"


def test_rate_and_watchlist(client):
    token = _onboarded_token(client)
    seed = client.get("/onboarding/movies").json()["movies"]
    target = seed[7]["movie_id"]

    r = client.post("/rate", json={"user_token": token, "movie_id": target, "rating": 4.0})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"

    r = client.post("/watchlist", json={"user_token": token, "movie_id": target, "action": "add"})
    assert r.status_code == 200, r.text

    # The rated+watchlisted movie should NOT show up in /recommendations.
    recs = client.get(f"/recommendations/{token}").json()
    all_ids = {m["movie_id"] for c in recs["carousels"] for m in c["movies"]}
    assert target not in all_ids


def test_profile(client):
    token = _onboarded_token(client)
    r = client.get(f"/profile/{token}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_ratings"] >= 5
    assert 0.5 <= body["mean_rating"] <= 5.0


def test_unknown_token_404(client):
    r = client.get("/recommendations/bogus")
    assert r.status_code == 404
    r = client.get("/profile/bogus")
    assert r.status_code == 404
