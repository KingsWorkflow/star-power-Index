"""
loader.py
---------
Load, merge, and optionally enrich the TMDB 5000 dataset.

Usage
-----
    from src.loader import load_raw, fetch_cast_popularity

The raw merge produces a single DataFrame with one row per film.
fetch_cast_popularity() queries the TMDB API for each cast member's
current popularity score and computes the mean of the top-N actors.
"""

import ast
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.config import (
    MOVIES_CSV, CREDITS_CSV, TMDB_API_KEY,
    TMDB_BASE_URL, TOP_N_CAST,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Raw load & merge
# ─────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """
    Load tmdb_5000_movies.csv and tmdb_5000_credits.csv,
    merge on movie id, and return a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged dataset with 4,803 rows (pre-cleaning).
    """
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(
            f"Movies file not found: {MOVIES_CSV}\n"
            "Download from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata "
            "and place it in the data/ folder."
        )
    if not CREDITS_CSV.exists():
        raise FileNotFoundError(
            f"Credits file not found: {CREDITS_CSV}\n"
            "Download from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata "
            "and place it in the data/ folder."
        )

    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    # Credits file uses 'movie_id' as the key; movies uses 'id'
    credits = credits.rename(columns={"movie_id": "id"})

    df = movies.merge(credits, on="id", how="left", suffixes=("", "_credits"))

    logger.info("Loaded raw dataset: %d rows, %d columns", len(df), df.shape[1])
    return df


# ─────────────────────────────────────────────────────────────
# JSON column parsers
# ─────────────────────────────────────────────────────────────

def _safe_parse(value) -> list:
    """Safely parse a JSON-encoded string column into a Python list."""
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def extract_genre_names(genres_str) -> list[str]:
    """Return a list of genre name strings from the JSON genres field."""
    parsed = _safe_parse(genres_str)
    return [g["name"] for g in parsed if "name" in g]


def extract_cast_ids(cast_str, top_n: int = TOP_N_CAST) -> list[int]:
    """Return the TMDB person IDs of the top-N billed cast members."""
    parsed = _safe_parse(cast_str)
    sorted_cast = sorted(parsed, key=lambda x: x.get("order", 999))
    return [c["id"] for c in sorted_cast[:top_n] if "id" in c]


def extract_cast_names(cast_str, top_n: int = TOP_N_CAST) -> list[str]:
    """Return the names of the top-N billed cast members."""
    parsed = _safe_parse(cast_str)
    sorted_cast = sorted(parsed, key=lambda x: x.get("order", 999))
    return [c["name"] for c in sorted_cast[:top_n] if "name" in c]


# ─────────────────────────────────────────────────────────────
# TMDB API — cast popularity
# ─────────────────────────────────────────────────────────────

def _get_person_popularity(person_id: int, api_key: str) -> float | None:
    """
    Query the TMDB /person/{person_id} endpoint and return
    the popularity score, or None on failure.
    """
    url = f"{TMDB_BASE_URL}/person/{person_id}"
    try:
        resp = requests.get(
            url,
            params={"api_key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("popularity")
        if resp.status_code == 429:          # rate limit
            time.sleep(10)
    except requests.RequestException as exc:
        logger.warning("API call failed for person %d: %s", person_id, exc)
    return None


def fetch_cast_popularity(
    df: pd.DataFrame,
    api_key: str | None = None,
    top_n: int = TOP_N_CAST,
    delay: float = 0.25,
) -> pd.Series:
    """
    For every film, compute the mean TMDB popularity of the top-N
    billed cast members.  Requires a valid TMDB API key.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'cast' column with JSON-encoded cast lists.
    api_key : str, optional
        Falls back to TMDB_API_KEY from config / .env.
    top_n : int
        Number of top-billed actors to average (default 3).
    delay : float
        Seconds to pause between requests (rate limiting).

    Returns
    -------
    pd.Series
        Float series indexed like df, named 'cast_popularity_score'.
    """
    key = api_key or TMDB_API_KEY
    if not key:
        raise ValueError(
            "No TMDB API key found.  Set TMDB_API_KEY in your .env file."
        )

    scores = []
    for cast_str in tqdm(df["cast"], desc="Fetching cast popularity"):
        person_ids = extract_cast_ids(cast_str, top_n=top_n)
        pops = []
        for pid in person_ids:
            pop = _get_person_popularity(pid, key)
            if pop is not None:
                pops.append(pop)
            time.sleep(delay)
        scores.append(float(np.mean(pops)) if pops else np.nan)

    return pd.Series(scores, index=df.index, name="cast_popularity_score")
