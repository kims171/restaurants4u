"""Tests for src/features.py — the shared training/serving transform."""
from __future__ import annotations

import pandas as pd

from src.features import encode_for_inference, engineer_features


def test_engineer_features_matches_training_shape():
    df = pd.DataFrame(
        {
            "Title": ["Tony's Pizza", "Cafe Luna"],
            "Address": ["123 Main St", "456 Oak Ave"],
            "Latitude": [37.77, 34.05],
            "Longitude": [-122.42, -118.24],
            "Website": ["http://x.com", None],
            "Phone": ["555-1234", "555-5678"],
            "Images": [None, "http://img.com/1.jpg"],
            "Category": ["Italian", "Cafe"],
            "is_highly_rated": [1, 0],
        }
    )
    features, top_categories = engineer_features(df, top_categories=None, top_category_count=20)

    assert "has_website" in features.columns
    assert features.loc[0, "has_website"] == 1
    assert features.loc[1, "has_website"] == 0
    assert features.loc[0, "title_length"] == len("Tony's Pizza")
    assert set(top_categories) == {"Italian", "Cafe"}
    assert "cat_Italian" in features.columns
    assert "is_highly_rated" in features.columns


def test_encode_for_inference_aligns_to_training_columns():
    top_categories = ["Italian", "Mexican"]
    feature_columns = [
        "Latitude",
        "Longitude",
        "has_website",
        "has_phone",
        "has_images",
        "title_length",
        "address_length",
        "cat_Italian",
        "cat_Mexican",
        "cat_Other",
    ]
    raw_row = {
        "Title": "New Spot",
        "Address": "1 Test St",
        "Latitude": 40.0,
        "Longitude": -74.0,
        "Website": None,
        "Phone": "555-0000",
        "Images": None,
        "Category": "Thai",  # not in top_categories -> should map to Other
    }

    encoded = encode_for_inference(raw_row, top_categories, feature_columns)

    assert list(encoded.columns) == feature_columns
    assert encoded.loc[0, "cat_Other"] == 1
    assert encoded.loc[0, "cat_Italian"] == 0
    assert encoded.loc[0, "has_website"] == 0
    assert encoded.loc[0, "has_phone"] == 1
