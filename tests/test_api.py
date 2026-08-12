"""Tests for the FastAPI serving layer.

Uses a fake model + fake top_categories/feature_columns injected into
model_service, so tests don't depend on the real trained artifact or
reports/feature_summary.json being present in CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import data_service as ds
from app import model_service as ms
from app.main import app

FAKE_TOP_CATEGORIES = ["Italian", "Mexican", "Cafe"]
# Mirrors what feature_engineering.py would produce: numeric cols first,
# then cat_ dummies (alphabetical, as pandas.get_dummies emits them),
# target column excluded (serving never includes it).
FAKE_FEATURE_COLUMNS = [
    "Latitude",
    "Longitude",
    "has_website",
    "has_phone",
    "has_images",
    "title_length",
    "address_length",
    "cat_Cafe",
    "cat_Italian",
    "cat_Mexican",
    "cat_Other",
]

FAKE_RESTAURANTS_DF = pd.DataFrame(
    {
        "Title": ["Close Diner", "Far Bistro"],
        "Address": ["1 Near St", "2 Far Ave"],
        "Latitude": [37.7750, 38.5000],
        "Longitude": [-122.4195, -121.5000],
        "Website": [None, "http://x.com"],
        "Phone": ["555-0000", "555-1111"],
        "Images": ["img.jpg", None],
        "Category": ["Mexican", "Mexican"],
    }
)


class FakeModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        # Deterministic fake: higher probability when has_website == 1
        row = X.iloc[0]
        base = 0.8 if row["has_website"] == 1 else 0.3
        return np.array([[1 - base, base]])


@pytest.fixture(autouse=True)
def fake_model():
    ms.model_service._model = FakeModel()
    ms.model_service._top_categories = FAKE_TOP_CATEGORIES
    ms.model_service._feature_columns = FAKE_FEATURE_COLUMNS
    yield
    ms.model_service._model = None
    ms.model_service._top_categories = None
    ms.model_service._feature_columns = None


@pytest.fixture(autouse=True)
def fake_data_service():
    ds.data_service._df = FAKE_RESTAURANTS_DF.copy()
    yield
    ds.data_service._df = None


@pytest.fixture
def client():
    return TestClient(app)


def base_restaurant(**overrides):
    payload = {
        "restaurant_id": "r1",
        "title": "Tony's Pizza",
        "address": "123 Main St",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "has_website": True,
        "has_phone": True,
        "has_images": False,
        "category": "Italian",
    }
    payload.update(overrides)
    return payload


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["data_loaded"] is True


def test_predict(client):
    resp = client.post("/predict", json={"restaurant": base_restaurant()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["restaurant_id"] == "r1"
    assert 0.0 <= body["probability_highly_rated"] <= 1.0


def test_predict_unseen_category_falls_back_to_other(client):
    # "Bubble Tea" isn't in FAKE_TOP_CATEGORIES, so it should map to
    # the cat_Other column rather than erroring or being dropped.
    resp = client.post(
        "/predict",
        json={"restaurant": base_restaurant(category="Bubble Tea")},
    )
    assert resp.status_code == 200


def test_predict_null_category_becomes_unknown_then_other(client):
    resp = client.post(
        "/predict",
        json={"restaurant": base_restaurant(category=None)},
    )
    assert resp.status_code == 200


def test_recommend_ranks_by_score(client):
    payload = {
        "user_latitude": 37.7749,
        "user_longitude": -122.4194,
        "lambda_decay": 0.1,
        "top_k": 5,
        "candidates": [
            base_restaurant(
                restaurant_id="close_no_site",
                latitude=37.7750,
                longitude=-122.4195,
                has_website=False,
                category="Mexican",
            ),
            base_restaurant(
                restaurant_id="far_with_site",
                latitude=38.5000,
                longitude=-121.5000,
                has_website=True,
                category="Mexican",
            ),
        ],
    }
    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[0]["restaurant_id"] == "close_no_site"


def test_predict_returns_503_when_model_missing(client):
    ms.model_service._model = None
    resp = client.post("/predict", json={"restaurant": base_restaurant()})
    assert resp.status_code == 503


def test_nearby_returns_ranked_real_restaurants(client):
    # User located exactly at "Close Diner"'s coordinates. "Far Bistro" is
    # ~130km away with a much higher base probability (has_website), but at
    # the default lambda_decay the distance penalty should still put the
    # nearby option first.
    resp = client.get("/nearby", params={"lat": 37.7750, "lon": -122.4195, "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    results = body["results"]
    assert len(results) == 2
    assert results[0]["title"] == "Close Diner"
    assert results[0]["distance_km"] == 0.0
    # Close Diner has no website in the fixture -> should be null, not "nan"
    assert results[0]["website_url"] is None
    far_bistro = next(r for r in results if r["title"] == "Far Bistro")
    assert far_bistro["website_url"] == "http://x.com"


def test_nearby_category_filter(client):
    resp = client.get(
        "/nearby",
        params={"lat": 37.7750, "lon": -122.4195, "category": "Italian"},
    )
    assert resp.status_code == 200
    # Neither fake restaurant is Italian, so this should come back empty
    # rather than erroring.
    assert resp.json()["results"] == []


def test_nearby_returns_503_when_data_missing(client):
    ds.data_service._df = None
    resp = client.get("/nearby", params={"lat": 37.7750, "lon": -122.4195})
    assert resp.status_code == 503
