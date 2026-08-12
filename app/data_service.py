"""In-memory restaurant lookup for candidate retrieval.

Loads data/validated/restaurants_validated.parquet (output of
validate_data.py — cleaned but still in raw column form) once at
startup and answers "which restaurants are near this point" via a
vectorized haversine distance over the whole table.

This is a linear scan, not a spatial index. Fine for local dev and
demo purposes at this dataset's size; NOT what you'd want at real
production scale — that's exactly the KDTree-based spatial filtering
your README calls out as future/next-phase work. Swap this out for a
real spatial index (scipy.spatial.KDTree, or a proper geospatial DB)
before this needs to handle real traffic.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("data_service")

VALIDATED_DATA_PATH = Path(
    os.getenv("VALIDATED_DATA_PATH", "data/validated/restaurants_validated.parquet")
)


def haversine_km_vectorized(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def row_to_feature_dict(row: pd.Series) -> dict:
    """Same raw-column shape model_service.ModelService expects."""
    return {
        "title": row.get("Title"),
        "address": row.get("Address"),
        "latitude": float(row["Latitude"]),
        "longitude": float(row["Longitude"]),
        "has_website": bool(pd.notna(row.get("Website"))),
        "has_phone": bool(pd.notna(row.get("Phone"))),
        "has_images": bool(pd.notna(row.get("Images"))),
        "category": row.get("Category"),
    }


class DataService:
    def __init__(self, path: Path = VALIDATED_DATA_PATH):
        self.path = path
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Validated restaurant data not found at {self.path}. "
                "Run `python src/validate_data.py` (or `dvc repro`) first."
            )
        self._df = pd.read_parquet(self.path)
        logger.info("Loaded %d restaurants from %s", len(self._df), self.path)

    @property
    def is_loaded(self) -> bool:
        return self._df is not None

    def find_nearby(
        self, lat: float, lon: float, category: Optional[str] = None, limit: int = 30
    ) -> pd.DataFrame:
        if not self.is_loaded:
            raise RuntimeError("Data not loaded")

        df = self._df
        if category:
            df = df[df["Category"].str.contains(category, case=False, na=False)]

        if df.empty:
            return df

        distances = haversine_km_vectorized(lat, lon, df["Latitude"].values, df["Longitude"].values)
        df = df.assign(distance_km=distances)
        return df.nsmallest(limit, "distance_km")


data_service = DataService()
