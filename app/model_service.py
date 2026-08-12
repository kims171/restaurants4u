from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import joblib

from src.features import TARGET, encode_for_inference

logger = logging.getLogger("model_service")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/restaurant_classifier.pkl"))
FEATURE_REPORT_PATH = Path(os.getenv("FEATURE_REPORT_PATH", "reports/feature_summary.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")


class ModelService:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_report_path: Path = FEATURE_REPORT_PATH,
    ):
        self.model_path = model_path
        self.feature_report_path = feature_report_path
        self._model = None
        self._top_categories: Optional[List[str]] = None
        self._feature_columns: Optional[List[str]] = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Run `dvc pull` or check MODEL_PATH env var."
            )
        if not self.feature_report_path.exists():
            raise FileNotFoundError(
                f"Feature report not found at {self.feature_report_path}. "
                "Run `dvc pull` or check FEATURE_REPORT_PATH env var. "
                "This file is required to reproduce training-time category "
                "encoding and column order at serving time."
            )

        self._model = joblib.load(self.model_path)

        report = json.loads(self.feature_report_path.read_text())
        self._top_categories = report["top_categories"]
        # feature_columns in the report includes the TARGET column
        # (is_highly_rated); serving-time inputs never include it.
        self._feature_columns = [c for c in report["feature_columns"] if c != TARGET]

        logger.info(
            "Loaded model from %s (version=%s), %d categories, %d feature columns",
            self.model_path,
            MODEL_VERSION,
            len(self._top_categories),
            len(self._feature_columns),
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _to_raw_row(self, feature_dict: dict) -> dict:
        # Website/Phone/Images: the model only ever saw notna() on these,
        # so a truthy sentinel or None both replicate that exactly.
        return {
            "Title": feature_dict.get("title"),
            "Address": feature_dict.get("address"),
            "Latitude": feature_dict["latitude"],
            "Longitude": feature_dict["longitude"],
            "Website": "present" if feature_dict.get("has_website") else None,
            "Phone": "present" if feature_dict.get("has_phone") else None,
            "Images": "present" if feature_dict.get("has_images") else None,
            "Category": feature_dict.get("category"),
        }

    def predict_proba(self, feature_dict: dict) -> float:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        raw_row = self._to_raw_row(feature_dict)
        X = encode_for_inference(
            raw_row,
            top_categories=self._top_categories,
            feature_columns=self._feature_columns,
        )
        proba = self._model.predict_proba(X)[0]
        classes = list(getattr(self._model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else 1
        return float(proba[idx])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/long points."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def recommendation_score(p_highly_rated: float, distance_km: float, lam: float) -> float:
    """Recommendation Score = P(highly_rated) * exp(-lambda * distance)."""
    return p_highly_rated * math.exp(-lam * distance_km)


model_service = ModelService()
