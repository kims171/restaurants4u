"""FastAPI serving layer for the Restaurants4U recommendation model."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from app.data_service import data_service, row_to_feature_dict
from app.model_service import (
    MODEL_VERSION,
    haversine_km,
    model_service,
    recommendation_score,
)
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RankedRestaurant,
    RecommendRequest,
    RecommendResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("restaurants4u-api")

app = FastAPI(
    title="Restaurants4U Recommendation API",
    description="Two-stage content-based restaurant recommendation engine",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        model_service.load()
    except FileNotFoundError as exc:
        # Don't crash the process on startup so /health can report the issue;
        # requests will fail with 503 until the model is available.
        logger.error("Model failed to load: %s", exc)

    try:
        data_service.load()
    except FileNotFoundError as exc:
        logger.error("Restaurant data failed to load: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if (model_service.is_loaded and data_service.is_loaded) else "degraded",
        model_loaded=model_service.is_loaded,
        data_loaded=data_service.is_loaded,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    proba = model_service.predict_proba(request.restaurant.model_dump())
    return PredictResponse(
        restaurant_id=request.restaurant.restaurant_id,
        probability_highly_rated=proba,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    candidates = request.candidates
    if request.category_filter:
        candidates = [
            c for c in candidates if c.category.lower() == request.category_filter.lower()
        ]

    ranked: list[RankedRestaurant] = []
    for candidate in candidates:
        proba = model_service.predict_proba(candidate.model_dump())
        distance = haversine_km(
            request.user_latitude,
            request.user_longitude,
            candidate.latitude,
            candidate.longitude,
        )
        score = recommendation_score(proba, distance, request.lambda_decay)
        ranked.append(
            RankedRestaurant(
                restaurant_id=candidate.restaurant_id,
                title=candidate.title,
                website_url=candidate.website_url,
                probability_highly_rated=proba,
                distance_km=round(distance, 3),
                recommendation_score=score,
            )
        )

    ranked.sort(key=lambda r: r.recommendation_score, reverse=True)
    return RecommendResponse(results=ranked[: request.top_k])


@app.get("/nearby", response_model=RecommendResponse)
def nearby(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    category: Optional[str] = Query(None, description="Optional category substring filter"),
    top_k: int = Query(10, ge=1, le=50),
    candidate_pool: int = Query(
        30, ge=1, le=200, description="How many nearby rows to score before ranking"
    ),
    lambda_decay: float = Query(0.1),
) -> RecommendResponse:
    """Find and rank real restaurants near a point, with no candidate list
    required from the caller
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not data_service.is_loaded:
        raise HTTPException(status_code=503, detail="Restaurant data not loaded")

    candidates_df = data_service.find_nearby(lat, lon, category, candidate_pool)
    if candidates_df.empty:
        return RecommendResponse(results=[])

    ranked: list[RankedRestaurant] = []
    for _, row in candidates_df.iterrows():
        feature_dict = row_to_feature_dict(row)
        proba = model_service.predict_proba(feature_dict)
        distance = float(row["distance_km"])
        score = recommendation_score(proba, distance, lambda_decay)
        website_value = row.get("Website")
        ranked.append(
            RankedRestaurant(
                restaurant_id=str(row.name),
                title=row.get("Title"),
                website_url=website_value if isinstance(website_value, str) else None,
                probability_highly_rated=proba,
                distance_km=round(distance, 3),
                recommendation_score=score,
            )
        )

    ranked.sort(key=lambda r: r.recommendation_score, reverse=True)
    return RecommendResponse(results=ranked[:top_k])

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
