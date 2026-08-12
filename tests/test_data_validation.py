from pathlib import Path

import pandas as pd


FIXTURE_PATH = Path("tests/fixtures/sample_restaurants.csv")


def test_sample_restaurant_file_exists():
    assert FIXTURE_PATH.exists(), "Sample restaurant fixture is missing."


def test_sample_restaurant_data_is_not_empty():
    df = pd.read_csv(FIXTURE_PATH)
    assert not df.empty, "Sample restaurant data should not be empty."


def test_required_columns_exist():
    df = pd.read_csv(FIXTURE_PATH)

    required_columns = {
        "Title",
        "Category",
        "Rating",
        "Latitude",
        "Longitude",
        "Website",
        "Phone",
    }

    missing_columns = required_columns - set(df.columns)

    assert not missing_columns, f"Missing required columns: {missing_columns}"


def test_rating_values_are_valid():
    df = pd.read_csv(FIXTURE_PATH)

    assert df["Rating"].between(0, 5).all(), "Ratings must be between 0 and 5."


def test_latitude_values_are_valid():
    df = pd.read_csv(FIXTURE_PATH)

    assert df["Latitude"].between(-90, 90).all(), "Latitude values must be between -90 and 90."


def test_longitude_values_are_valid():
    df = pd.read_csv(FIXTURE_PATH)

    assert df["Longitude"].between(-180, 180).all(), (
        "Longitude values must be between -180 and 180."
    )


def test_no_missing_core_prediction_fields():
    df = pd.read_csv(FIXTURE_PATH)

    core_columns = ["Category", "Rating", "Latitude", "Longitude"]

    assert not df[core_columns].isnull().any().any(), (
        "Core prediction fields should not contain missing values."
    )
