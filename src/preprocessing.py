"""
preprocessing.py
----------------
Full cleaning and feature-engineering pipeline for the Star Power Index.

Steps performed (matching Section 3 of the Week 3 EDA document):
  1. Drop rows where budget or revenue == 0 (not genuine values)
  2. Drop rows where cast_popularity_score is missing
  3. Impute the 2 missing runtime values with the median (107 min)
  4. Drop the 1 row with a missing release_date
  5. Remove rows below minimum budget / revenue thresholds
  6. Remove duplicate films (same title + release year)
  7. Derive ROI = (revenue - budget) / budget
  8. Apply log1p transformation to revenue, budget, cast_popularity_score
  9. Extract release_year, release_month, release_season
 10. Extract primary genre from the JSON genres field
 11. Parse cast column → top-N actor names

Usage
-----
    from src.preprocessing import run_pipeline
    df_clean = run_pipeline(df_raw)
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import MIN_BUDGET, MIN_REVENUE, KEEP_GENRES
from src.loader import extract_genre_names, extract_cast_names, fetch_cast_popularity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _season(month: int) -> str:
    """Map calendar month to a four-way season label."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


# ─────────────────────────────────────────────────────────────
# Individual cleaning steps
# ─────────────────────────────────────────────────────────────


def drop_zero_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where budget or revenue is 0 (missing disclosure)."""
    before = len(df)
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    logger.info(
        "drop_zero_financials: %d → %d rows (-%d)", before, len(df), before - len(df)
    )
    return df


def drop_low_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Remove likely data-entry errors (budget or revenue below floor)."""
    before = len(df)
    df = df[(df["budget"] >= MIN_BUDGET) & (df["revenue"] >= MIN_REVENUE)].copy()
    logger.info(
        "drop_low_financials: %d → %d rows (-%d)", before, len(df), before - len(df)
    )
    return df


def drop_missing_cast_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the TMDB API returned no cast popularity data."""
    before = len(df)
    df = df.dropna(subset=["cast_popularity_score"]).copy()
    logger.info(
        "drop_missing_cast_popularity: %d → %d rows (-%d)",
        before,
        len(df),
        before - len(df),
    )
    return df


def impute_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing runtime values with the dataset median (≈107 min)."""
    n_missing = df["runtime"].isna().sum()
    if n_missing > 0:
        median_rt = df["runtime"].median()
        df["runtime"] = df["runtime"].fillna(median_rt)
        logger.info(
            "impute_runtime: filled %d missing values with %.0f min",
            n_missing,
            median_rt,
        )
    return df


def drop_missing_release_date(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the single row with a null release_date."""
    before = len(df)
    df = df.dropna(subset=["release_date"]).copy()
    logger.info(
        "drop_missing_release_date: %d → %d rows (-%d)",
        before,
        len(df),
        before - len(df),
    )
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates on (title, release_year) after year extraction."""
    # release_year may not yet exist — use release_date if available
    if "release_year" not in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["_year_tmp"] = df["release_date"].dt.year
        key = ["title", "_year_tmp"]
    else:
        key = ["title", "release_year"]

    before = len(df)
    df = df.drop_duplicates(subset=key).copy()
    df = df.drop(columns=["_year_tmp"], errors="ignore")
    logger.info(
        "drop_duplicates: %d → %d rows (-%d)", before, len(df), before - len(df)
    )
    return df


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────


def add_cast_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch and add cast_popularity_score if not present."""
    if "cast_popularity_score" not in df.columns:
        df = df.copy()
        df["cast_popularity_score"] = fetch_cast_popularity(df)
    return df


def add_roi(df: pd.DataFrame) -> pd.DataFrame:
    """ROI = (revenue - budget) / budget"""
    df = df.copy()
    df["roi"] = (df["revenue"] - df["budget"]) / df["budget"]
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transformation to revenue, budget, and cast_popularity_score.
    These right-skewed variables need this transformation before OLS regression.
    """
    df = df.copy()
    df["log_revenue"] = np.log1p(df["revenue"])
    df["log_budget"] = np.log1p(df["budget"])
    df["log_cast_pop"] = np.log1p(df["cast_popularity_score"])
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract release_year, release_month, and release_season."""
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_season"] = df["release_month"].map(_season)
    return df


def add_primary_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the JSON genres column and extract:
      - genres_list : Python list of all genre names
      - primary_genre : the first listed genre (or 'Other')
    """
    df = df.copy()
    df["genres_list"] = df["genres"].apply(extract_genre_names)
    df["primary_genre"] = df["genres_list"].apply(lambda gl: gl[0] if gl else "Other")
    # Consolidate rare genres into 'Other'
    df["primary_genre"] = df["primary_genre"].where(
        df["primary_genre"].isin(KEEP_GENRES), other="Other"
    )
    return df


def add_cast_names(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Parse the JSON cast column and store top-N actor names as a list."""
    df = df.copy()
    if "cast" in df.columns:
        df["top_cast_names"] = df["cast"].apply(
            lambda s: extract_cast_names(s, top_n=top_n)
        )
    return df


# ─────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────


def run_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline and return the clean DataFrame.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Output of src.loader.load_raw()

    Returns
    -------
    pd.DataFrame
        Clean, feature-engineered dataset (~2,641 rows).
    """
    logger.info("Starting preprocessing pipeline on %d rows", len(df_raw))

    df = (
        df_raw.pipe(drop_zero_financials)
        .pipe(drop_missing_release_date)
        .pipe(impute_runtime)
        .pipe(drop_duplicates)
        .pipe(drop_low_financials)
        .pipe(add_cast_popularity_score)
        .pipe(add_roi)
        .pipe(add_log_transforms)
        .pipe(add_date_features)
        .pipe(add_primary_genre)
        .pipe(add_cast_names)
        .pipe(drop_missing_cast_popularity)
    )

    logger.info("Pipeline complete: %d rows retained", len(df))
    return df.reset_index(drop=True)


# -------------------------------------------------------------
# Entry point when run as: python -m src.preprocessing
# -------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="%(levelname)s  %(message)s"
    )

    from src.loader import load_raw
    from src.config import CLEAN_CSV

    # Check if raw data exists
    movies_path = Path("data/tmdb_5000_movies.csv")
    credits_path = Path("data/tmdb_5000_credits.csv")

    if not movies_path.exists():
        logger.error("Raw data not found! Place tmdb_5000_movies.csv in data/ folder")
        logger.error(
            "Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"
        )
        sys.exit(1)

    logger.info("Loading raw data...")
    df_raw = load_raw()

    logger.info("Running preprocessing pipeline...")
    df_clean = run_pipeline(df_raw)

    # Save to disk
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False)
    logger.info("✅ Clean dataset saved to: %s", CLEAN_CSV)
    logger.info("Shape: %d rows × %d columns", df_clean.shape[0], df_clean.shape[1])
