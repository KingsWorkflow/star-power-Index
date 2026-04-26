"""
config.py
---------
Central configuration for the Star Power Index project.
All paths, constants, and environment variables live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────
load_dotenv()

# ── Project root (this file lives in src/, root is one level up) ──
ROOT_DIR = Path(__file__).parent.parent

# ── Directories ───────────────────────────────────────────────
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Data file paths ───────────────────────────────────────────
MOVIES_CSV  = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA_DIR / "tmdb_5000_credits.csv"
CLEAN_CSV   = OUTPUT_DIR / "tmdb_cleaned.csv"

# ── TMDB API ──────────────────────────────────────────────────
TMDB_API_KEY   = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL  = "https://api.themoviedb.org/3"
TMDB_PERSON_URL = f"{TMDB_BASE_URL}/person/{{person_id}}"

# ── Modelling constants ───────────────────────────────────────
TOP_N_CAST     = 3          # Number of top-billed actors to average for SPI
MIN_BUDGET     = 1_000      # Drop films below this budget (USD) — likely errors
MIN_REVENUE    = 1_000      # Drop films below this revenue (USD)
RANDOM_STATE   = 42

# ── Genre mapping — consolidate rare genres ───────────────────
KEEP_GENRES = [
    "Drama", "Comedy", "Thriller", "Action",
    "Romance", "Adventure", "Crime", "Horror",
    "Animation", "Science Fiction",
]

# ── Plot style ────────────────────────────────────────────────
PLOT_STYLE   = "seaborn-v0_8-whitegrid"
FIGURE_DPI   = 150
PALETTE      = "Set2"
