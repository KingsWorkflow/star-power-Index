"""
eda.py
------
Descriptive statistics and visualisations for the Star Power Index.
Reproduces every analysis described in the Week 3 EDA document.

Run standalone:
    python -m src.eda

Or import individual functions into a notebook.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import OUTPUT_DIR, PLOT_STYLE, FIGURE_DPI, PALETTE

logger = logging.getLogger(__name__)
plt.style.use(PLOT_STYLE)


# ─────────────────────────────────────────────────────────────
# 1. Summary statistics  (Table 4 in EDA doc)
# ─────────────────────────────────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary stats for the six key numerical variables.

    Variables: revenue, budget, cast_popularity_score, roi,
               vote_average, runtime
    """
    cols = ["revenue", "budget", "cast_popularity_score",
            "roi", "vote_average", "runtime"]
    present = [c for c in cols if c in df.columns]

    stats_df = df[present].agg(["mean", "median", "std", "min", "max"]).T
    stats_df.columns = ["Mean", "Median", "Std Dev", "Min", "Max"]
    stats_df = stats_df.round(2)

    print("\n=== Table 4: Descriptive Statistics ===")
    print(stats_df.to_string())
    return stats_df


# ─────────────────────────────────────────────────────────────
# 2. Genre frequency table  (Table 5 in EDA doc)
# ─────────────────────────────────────────────────────────────

def genre_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the genres_list column so each genre-film pair is a row,
    then count frequency and compute mean revenue per genre.
    """
    if "genres_list" not in df.columns:
        raise ValueError("Run preprocessing.add_primary_genre first.")

    exploded = df.explode("genres_list").rename(columns={"genres_list": "genre"})
    genre_stats = (
        exploded
        .groupby("genre")
        .agg(
            Count=("revenue", "count"),
            Mean_Revenue=("revenue", "mean"),
        )
        .sort_values("Count", ascending=False)
        .reset_index()
    )
    genre_stats["Mean_Revenue"] = (genre_stats["Mean_Revenue"] / 1e6).round(1)
    genre_stats = genre_stats.rename(columns={"Mean_Revenue": "Mean Revenue (USD M)"})

    print("\n=== Table 5: Genre Frequency and Mean Revenue ===")
    print(genre_stats.to_string(index=False))
    return genre_stats


# ─────────────────────────────────────────────────────────────
# 3. Correlation matrix  (Table 6 in EDA doc)
# ─────────────────────────────────────────────────────────────

def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlations for the pairs described in Section 5.4.
    """
    pairs = [
        ("log_budget",   "log_revenue",  "Strong positive"),
        ("log_cast_pop", "log_revenue",  "Moderate positive"),
        ("vote_average", "log_revenue",  "Weak positive"),
        ("log_budget",   "log_cast_pop", "Moderate; manageable multicollinearity"),
        ("roi",          "log_cast_pop", "Weak-to-moderate positive"),
    ]

    rows = []
    for v1, v2, interpretation in pairs:
        if v1 in df.columns and v2 in df.columns:
            r, p = stats.pearsonr(df[v1].dropna(), df[v2].dropna())
            rows.append({
                "Variable Pair": f"{v1} ― {v2}",
                "Pearson r": round(r, 2),
                "p-value": round(p, 4),
                "Interpretation": interpretation,
            })

    corr_df = pd.DataFrame(rows)
    print("\n=== Table 6: Pearson Correlation Coefficients ===")
    print(corr_df.to_string(index=False))
    return corr_df


# ─────────────────────────────────────────────────────────────
# 4. Distribution plots  (Section 5.1)
# ─────────────────────────────────────────────────────────────

def plot_distributions_raw(df: pd.DataFrame, save: bool = True) -> None:
    """
    Histogram + KDE for revenue, budget, and cast_popularity_score (raw).
    """
    cols = ["revenue", "budget", "cast_popularity_score"]
    present = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4))
    if len(present) == 1:
        axes = [axes]

    for ax, col in zip(axes, present):
        sns.histplot(df[col].dropna(), ax=ax, kde=True, color="steelblue",
                     edgecolor="white", linewidth=0.4)
        ax.set_title(f"Distribution of {col}", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    plt.suptitle("Figure 1 — Raw Distributions (right-skewed)", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig1_distributions_raw.png"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.show()


def plot_distributions_log(df: pd.DataFrame, save: bool = True) -> None:
    """
    Histogram + KDE for log_revenue, log_budget, log_cast_pop.
    """
    cols = ["log_revenue", "log_budget", "log_cast_pop"]
    present = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4))
    if len(present) == 1:
        axes = [axes]

    for ax, col in zip(axes, present):
        sns.histplot(df[col].dropna(), ax=ax, kde=True, color="coral",
                     edgecolor="white", linewidth=0.4)
        ax.set_title(f"Distribution of {col}", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    plt.suptitle("Figure 2 — Log-Transformed Distributions (approx. normal)", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig2_distributions_log.png"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.show()


# ─────────────────────────────────────────────────────────────
# 5. Scatter plots  (Section 5.2)
# ─────────────────────────────────────────────────────────────

def plot_scatter_budget_vs_revenue(df: pd.DataFrame, save: bool = True) -> None:
    """log_budget vs log_revenue — strong positive linear trend (r = 0.72)."""
    if not {"log_budget", "log_revenue"}.issubset(df.columns):
        logger.warning("Missing log columns; run add_log_transforms first.")
        return

    r, _ = stats.pearsonr(df["log_budget"], df["log_revenue"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=df, x="log_budget", y="log_revenue",
                scatter_kws={"alpha": 0.25, "s": 18, "color": "steelblue"},
                line_kws={"color": "crimson", "linewidth": 2}, ax=ax)
    ax.set_title(f"Figure 3 — log(budget) vs log(revenue)  [r = {r:.2f}]", fontsize=12)
    ax.set_xlabel("log(1 + budget)")
    ax.set_ylabel("log(1 + revenue)")

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig3_scatter_budget_revenue.png"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.show()


def plot_scatter_cast_pop_vs_revenue(df: pd.DataFrame, save: bool = True) -> None:
    """log_cast_pop vs log_revenue — moderate positive trend (r = 0.48)."""
    if not {"log_cast_pop", "log_revenue"}.issubset(df.columns):
        logger.warning("Missing log columns; run add_log_transforms first.")
        return

    r, _ = stats.pearsonr(df["log_cast_pop"].dropna(), df["log_revenue"].dropna())

    sub = df.dropna(subset=["log_cast_pop", "log_revenue"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=sub, x="log_cast_pop", y="log_revenue",
                scatter_kws={"alpha": 0.25, "s": 18, "color": "teal"},
                line_kws={"color": "darkorange", "linewidth": 2}, ax=ax)
    ax.set_title(f"Figure 4 — log(cast_popularity) vs log(revenue)  [r = {r:.2f}]", fontsize=12)
    ax.set_xlabel("log(1 + cast_popularity_score)")
    ax.set_ylabel("log(1 + revenue)")

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig4_scatter_castpop_revenue.png"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. Box plots by genre  (Section 5.3)
# ─────────────────────────────────────────────────────────────

def plot_revenue_by_genre(df: pd.DataFrame, save: bool = True) -> None:
    """
    Box plots of log_revenue grouped by primary_genre.
    Action and Adventure should show highest medians.
    """
    if "primary_genre" not in df.columns or "log_revenue" not in df.columns:
        logger.warning("Missing primary_genre or log_revenue columns.")
        return

    order = (
        df.groupby("primary_genre")["log_revenue"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="primary_genre", y="log_revenue",
                order=order, palette=PALETTE, ax=ax,
                linewidth=0.8, flierprops={"marker": ".", "alpha": 0.3})
    ax.set_title("Figure 5 — log(revenue) by Primary Genre", fontsize=12)
    ax.set_xlabel("Primary Genre")
    ax.set_ylabel("log(1 + revenue)")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig5_boxplot_genre_revenue.png"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.show()


def plot_cast_pop_by_genre(df: pd.DataFrame, save: bool = True) -> None:
    """
    Box plots of log_cast_pop grouped by primary_genre.
    Action genres should show higher medians.
    """
    if "primary_genre" not in df.columns or "log_cast_pop" not in df.columns:
        logger.warning("Missing primary_genre or log_cast_pop columns.")
        return

    order = (
        df.groupby("primary_genre")["log_cast_pop"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="primary_genre", y="log_cast_pop",
                order=order, palette=PALETTE, ax=ax,
                linewidth=0.8, flierprops={"marker": ".", "alpha": 0.3})
    ax.set_title("Figure 6 — log(cast_popularity) by Primary Genre", fontsize=12)
    ax.set_xlabel("Primary Genre")
    ax.set_ylabel("log(1 + cast_popularity_score)")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig6_boxplot_genre_castpop.png"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.show()


# ─────────────────────────────────────────────────────────────
# 7. Correlation heatmap  (Section 5.4)
# ─────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """Full numeric correlation heatmap for log-transformed variables."""
    numeric_cols = [c for c in
                    ["log_revenue", "log_budget", "log_cast_pop",
                     "vote_average", "vote_count", "roi", "runtime"]
                    if c in df.columns]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, square=True, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Figure 7 — Pearson Correlation Heatmap", fontsize=12)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig7_correlation_heatmap.png"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.show()


# ─────────────────────────────────────────────────────────────
# Run all EDA at once
# ─────────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> dict:
    """
    Run all EDA steps and return a dict with the key tables.

    Parameters
    ----------
    df : pd.DataFrame
        Clean, preprocessed DataFrame from src.preprocessing.run_pipeline()

    Returns
    -------
    dict with keys: 'summary_stats', 'genre_table', 'correlation_table'
    """
    print(f"\n{'='*60}")
    print("STAR POWER INDEX — WEEK 3 EDA")
    print(f"Dataset: {len(df)} rows, {df.shape[1]} columns")
    print(f"{'='*60}")

    summary  = summary_statistics(df)
    genre_tbl = genre_frequency_table(df)
    corr_tbl  = correlation_table(df)

    plot_distributions_raw(df)
    plot_distributions_log(df)
    plot_scatter_budget_vs_revenue(df)
    plot_scatter_cast_pop_vs_revenue(df)
    plot_revenue_by_genre(df)
    plot_cast_pop_by_genre(df)
    plot_correlation_heatmap(df)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
    return {
        "summary_stats": summary,
        "genre_table": genre_tbl,
        "correlation_table": corr_tbl,
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s  %(message)s")

    from src.loader import load_raw
    from src.preprocessing import run_pipeline

    df_raw   = load_raw()
    df_clean = run_pipeline(df_raw)
    run_full_eda(df_clean)
