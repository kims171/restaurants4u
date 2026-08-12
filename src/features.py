"""Shared feature engineering logic.
  - feature_engineering.py (training-time: src/features.py replaces the
    engineer_features() function that used to live directly in that
    file — see the diff in docs/DEPLOYMENT.md for the one-line change
    needed there)
  - app/model_service.py (serving-time: encodes a single incoming
    request the exact same way)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

TARGET = "is_highly_rated"

NUMERIC_SOURCE_COLUMNS = [
    "Latitude",
    "Longitude",
    "has_website",
    "has_phone",
    "has_images",
    "title_length",
    "address_length",
]


def engineer_features(
    df: pd.DataFrame,
    top_categories: Optional[List[str]] = None,
    top_category_count: int = 20,
    include_target: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Transform raw restaurant rows into the model's numeric feature frame.
    """
    df = df.copy()

    for source_col, new_col in [
        ("Website", "has_website"),
        ("Phone", "has_phone"),
        ("Images", "has_images"),
    ]:
        if source_col in df.columns:
            df[new_col] = df[source_col].notna().astype(int)
        else:
            df[new_col] = 0

    if "Title" in df.columns:
        df["title_length"] = df["Title"].fillna("").astype(str).str.len()
    else:
        df["title_length"] = 0

    if "Address" in df.columns:
        df["address_length"] = df["Address"].fillna("").astype(str).str.len()
    else:
        df["address_length"] = 0

    if "Category" in df.columns:
        df["Category"] = df["Category"].fillna("Unknown").astype(str)

        if top_categories is None:
            top_categories = (
                df["Category"].value_counts().head(top_category_count).index.tolist()
            )

        df["clean_category"] = df["Category"].apply(
            lambda x: x if x in top_categories else "Other"
        )
    else:
        df["clean_category"] = "Unknown"
        if top_categories is None:
            top_categories = ["Unknown"]

    category_encoded = pd.get_dummies(df["clean_category"], prefix="cat")

    numeric_columns = [c for c in NUMERIC_SOURCE_COLUMNS if c in df.columns]

    frames = [df[numeric_columns], category_encoded]
    if include_target:
        frames.append(df[[TARGET]])

    feature_df = pd.concat(frames, axis=1)

    return feature_df, top_categories


def encode_for_inference(
    raw_row: dict,
    top_categories: List[str],
    feature_columns: List[str],
    top_category_count: int = 20,
) -> pd.DataFrame:
    """Encode a single raw request payload into the exact column layout
    the trained model expects (same names, same order, zero-filled for
    any category dummy not present in this single row).
    """
    df = pd.DataFrame([raw_row])
    encoded, _ = engineer_features(
        df,
        top_categories=top_categories,
        top_category_count=top_category_count,
        include_target=False,
    )
    return encoded.reindex(columns=feature_columns, fill_value=0)
