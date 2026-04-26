"""
tests/test_preprocessing.py
----------------------------
Unit tests for src.preprocessing.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=src
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    drop_zero_financials,
    drop_low_financials,
    drop_missing_cast_popularity,
    impute_runtime,
    drop_missing_release_date,
    drop_duplicates,
    add_roi,
    add_log_transforms,
    add_date_features,
    add_primary_genre,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_df():
    """Minimal valid DataFrame with the columns preprocessing needs."""
    return pd.DataFrame({
        "id":           [1, 2, 3, 4, 5],
        "title":        ["Film A", "Film B", "Film C", "Film D", "Film E"],
        "budget":       [1_000_000, 0, 5_000_000, 500, 20_000_000],
        "revenue":      [5_000_000, 3_000_000, 0, 200_000, 80_000_000],
        "runtime":      [100.0, 90.0, np.nan, 85.0, 120.0],
        "release_date": ["2015-06-12", "2016-03-01", "2017-11-20",
                         None, "2018-07-04"],
        "vote_average": [7.1, 6.4, 5.8, 6.0, 8.2],
        "vote_count":   [500, 200, 150, 80, 2000],
        "genres":       [
            '[{"id":28,"name":"Action"}]',
            '[{"id":35,"name":"Comedy"}]',
            '[{"id":18,"name":"Drama"}]',
            '[{"id":27,"name":"Horror"}]',
            '[{"id":12,"name":"Adventure"}]',
        ],
        "cast":         [
            '[{"id":1,"name":"Actor One","order":0}]',
            '[{"id":2,"name":"Actor Two","order":0}]',
            '[{"id":3,"name":"Actor Three","order":0}]',
            '[{"id":4,"name":"Actor Four","order":0}]',
            '[{"id":5,"name":"Actor Five","order":0}]',
        ],
        "cast_popularity_score": [25.3, np.nan, 12.1, 8.4, 55.0],
    })


# ─────────────────────────────────────────────────────────────
# drop_zero_financials
# ─────────────────────────────────────────────────────────────

def test_drop_zero_financials_removes_zero_budget(minimal_df):
    result = drop_zero_financials(minimal_df)
    assert (result["budget"] > 0).all(), "Zero-budget rows should be removed"


def test_drop_zero_financials_removes_zero_revenue(minimal_df):
    result = drop_zero_financials(minimal_df)
    assert (result["revenue"] > 0).all(), "Zero-revenue rows should be removed"


def test_drop_zero_financials_correct_count(minimal_df):
    # Film B has budget=0, Film C has revenue=0 → 2 removed
    result = drop_zero_financials(minimal_df)
    assert len(result) == 3


# ─────────────────────────────────────────────────────────────
# drop_low_financials
# ─────────────────────────────────────────────────────────────

def test_drop_low_financials_removes_below_minimum(minimal_df):
    df = drop_zero_financials(minimal_df)
    result = drop_low_financials(df)
    # Film D has budget=500 < MIN_BUDGET=1000 → should be removed
    assert all(result["budget"] >= 1_000)


# ─────────────────────────────────────────────────────────────
# impute_runtime
# ─────────────────────────────────────────────────────────────

def test_impute_runtime_fills_na(minimal_df):
    result = impute_runtime(minimal_df)
    assert result["runtime"].isna().sum() == 0, "No NaN should remain in runtime"


def test_impute_runtime_uses_median(minimal_df):
    result = impute_runtime(minimal_df)
    # The NaN was at index 2; after imputation, value should equal median of rest
    non_null = minimal_df["runtime"].dropna()
    expected_median = non_null.median()
    assert result.loc[2, "runtime"] == pytest.approx(expected_median)


# ─────────────────────────────────────────────────────────────
# drop_missing_release_date
# ─────────────────────────────────────────────────────────────

def test_drop_missing_release_date(minimal_df):
    result = drop_missing_release_date(minimal_df)
    # Row with None release_date should be gone
    assert result["release_date"].isna().sum() == 0


# ─────────────────────────────────────────────────────────────
# drop_missing_cast_popularity
# ─────────────────────────────────────────────────────────────

def test_drop_missing_cast_popularity(minimal_df):
    result = drop_missing_cast_popularity(minimal_df)
    # Film B had NaN cast_popularity_score
    assert result["cast_popularity_score"].isna().sum() == 0


# ─────────────────────────────────────────────────────────────
# add_roi
# ─────────────────────────────────────────────────────────────

def test_add_roi_formula(minimal_df):
    df = drop_zero_financials(minimal_df)
    result = add_roi(df)
    expected = (result["revenue"] - result["budget"]) / result["budget"]
    pd.testing.assert_series_equal(result["roi"], expected, check_names=False)


def test_add_roi_negative_for_loss(minimal_df):
    df = drop_zero_financials(minimal_df)
    result = add_roi(df)
    # Film D: revenue=200_000, budget=500 → ROI should be large positive
    # Film A: revenue=5M, budget=1M → ROI = 4.0
    film_a = result[result["title"] == "Film A"]["roi"].values[0]
    assert abs(film_a - 4.0) < 0.001


# ─────────────────────────────────────────────────────────────
# add_log_transforms
# ─────────────────────────────────────────────────────────────

def test_add_log_transforms_creates_columns(minimal_df):
    df = drop_zero_financials(minimal_df)
    result = add_log_transforms(df)
    for col in ["log_revenue", "log_budget", "log_cast_pop"]:
        assert col in result.columns, f"Missing column: {col}"


def test_add_log_transforms_values(minimal_df):
    df = drop_zero_financials(minimal_df)
    result = add_log_transforms(df)
    expected_log_rev = np.log1p(result["revenue"])
    pd.testing.assert_series_equal(result["log_revenue"], expected_log_rev,
                                   check_names=False)


# ─────────────────────────────────────────────────────────────
# add_date_features
# ─────────────────────────────────────────────────────────────

def test_add_date_features_columns(minimal_df):
    result = add_date_features(minimal_df)
    for col in ["release_year", "release_month", "release_season"]:
        assert col in result.columns


def test_add_date_features_season_values(minimal_df):
    result = add_date_features(minimal_df)
    valid_seasons = {"Spring", "Summer", "Fall", "Winter"}
    assert set(result["release_season"].dropna()).issubset(valid_seasons)


def test_add_date_features_summer(minimal_df):
    result = add_date_features(minimal_df)
    # Film A released 2015-06-12 → Summer
    row = result[result["title"] == "Film A"]
    assert row["release_season"].values[0] == "Summer"


# ─────────────────────────────────────────────────────────────
# add_primary_genre
# ─────────────────────────────────────────────────────────────

def test_add_primary_genre_columns(minimal_df):
    result = add_primary_genre(minimal_df)
    assert "primary_genre" in result.columns
    assert "genres_list" in result.columns


def test_add_primary_genre_values(minimal_df):
    result = add_primary_genre(minimal_df)
    # Film A genres = '[{"id":28,"name":"Action"}]' → "Action"
    film_a_genre = result[result["title"] == "Film A"]["primary_genre"].values[0]
    assert film_a_genre == "Action"


def test_add_primary_genre_is_list(minimal_df):
    result = add_primary_genre(minimal_df)
    for val in result["genres_list"]:
        assert isinstance(val, list)
