"""Pydantic schemas for the Restaurants4U inference API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, confloat


class RestaurantFeatures(BaseModel):
    """Raw feature payload for a single restaurant candidate.
    """

    restaurant_id: str = Field(..., description="Unique identifier for the restaurant")
    title: Optional[str] = Field(None, description="Restaurant name/title text")
    address: Optional[str] = Field(None, description="Restaurant address text")
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)
    has_website: bool = Field(..., description="True if the restaurant has a website listed")
    has_phone: bool = Field(..., description="True if the restaurant has a phone number listed")
    has_images: bool = Field(..., description="True if the restaurant has images listed")
    category: Optional[str] = Field(
        None, description="Raw restaurant category, pre-capping (e.g. 'Italian'); null becomes 'Unknown'"
    )
    website_url: Optional[str] = Field(
        None,
        description="Actual website URL, for display/linking only — has_website (not this) is what "
        "feeds the model, so this field can't cause train/serve skew.",
    )


class PredictRequest(BaseModel):
    restaurant: RestaurantFeatures


class PredictResponse(BaseModel):
    restaurant_id: str
    probability_highly_rated: float


class RecommendRequest(BaseModel):
    user_latitude: confloat(ge=-90, le=90)
    user_longitude: confloat(ge=-180, le=180)
    candidates: List[RestaurantFeatures]
    lambda_decay: float = Field(
        0.1, description="Distance penalty hyperparameter for exponential decay"
    )
    top_k: int = Field(10, ge=1, le=100)
    category_filter: Optional[str] = None


class RankedRestaurant(BaseModel):
    restaurant_id: str
    title: Optional[str] = None
    website_url: Optional[str] = None
    probability_highly_rated: float
    distance_km: float
    recommendation_score: float


class RecommendResponse(BaseModel):
    results: List[RankedRestaurant]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    model_version: Optional[str] = None
