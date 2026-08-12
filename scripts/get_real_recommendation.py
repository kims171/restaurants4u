"""Get a real recommendation from real data, against your running API.

This is a demo/dev-only script, not part of the production pipeline. It
pulls actual restaurant rows out of data/validated/restaurants_validated.parquet

Usage:
    pip install requests pandas pyarrow
    python scripts/get_real_recommendation.py --lat 37.7749 --lon -122.4194
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import requests

VALIDATED_DATA_PATH = Path("data/validated/restaurants_validated.parquet")


def haversine_km(lat1: float, lon1: float, lat2, lon2) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def load_candidates(
    user_lat: float,
    user_lon: float,
    num_candidates: int,
    category: str | None,
) -> pd.DataFrame:
    if not VALIDATED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{VALIDATED_DATA_PATH} not found. Run `python src/validate_data.py` "
            "(or the full `dvc repro`) first so real restaurant rows exist locally."
        )

    df = pd.read_parquet(VALIDATED_DATA_PATH)

    if category:
        df = df[df["Category"].str.contains(category, case=False, na=False)]
        if df.empty:
            raise ValueError(f"No restaurants found matching category '{category}'")

    df = df.copy()
    df["distance_km"] = df.apply(
        lambda row: haversine_km(user_lat, user_lon, row["Latitude"], row["Longitude"]),
        axis=1,
    )

    return df.nsmallest(num_candidates, "distance_km")


def to_candidate_payload(row: pd.Series) -> dict:
    return {
        "restaurant_id": str(row.name),
        "title": row.get("Title"),
        "address": row.get("Address"),
        "latitude": float(row["Latitude"]),
        "longitude": float(row["Longitude"]),
        "has_website": pd.notna(row.get("Website")),
        "has_phone": pd.notna(row.get("Phone")),
        "has_images": pd.notna(row.get("Images")),
        "category": row.get("Category"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True, help="Your latitude")
    parser.add_argument("--lon", type=float, required=True, help="Your longitude")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=15,
        help="How many nearby restaurants to send as candidates",
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Optional category substring filter"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="How many ranked results to request back"
    )
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    print(f"Loading real restaurants near ({args.lat}, {args.lon})...")
    candidates_df = load_candidates(args.lat, args.lon, args.num_candidates, args.category)
    print(f"Found {len(candidates_df)} nearby candidates, calling /recommend...")

    payload = {
        "user_latitude": args.lat,
        "user_longitude": args.lon,
        "top_k": args.top_k,
        "candidates": [to_candidate_payload(row) for _, row in candidates_df.iterrows()],
    }

    response = requests.post(f"{args.api_url}/recommend", json=payload, timeout=30)
    response.raise_for_status()
    results = response.json()["results"]

    print("\nTop recommendations:")
    for i, r in enumerate(results, start=1):
        print(
            f"{i}. {r.get('title') or r['restaurant_id']}  "
            f"(score={r['recommendation_score']:.3f}, "
            f"P(highly rated)={r['probability_highly_rated']:.3f}, "
            f"distance={r['distance_km']:.2f} km)"
        )


if __name__ == "__main__":
    main()
