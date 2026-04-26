"""
app_utils.py
------------
Utility functions for the Streamlit Greenlight Tool.
Provides helper functions for data processing, predictions, and visualizations.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.analysis import run_ols, run_anova, check_vif
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Formatting Utilities
# ─────────────────────────────────────────────────────────────


def format_currency(value):
    """Format number as currency in millions/billions."""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    else:
        return f"${value:,.0f}"


def format_percentage(value):
    """Format number as percentage."""
    return f"{value:.1%}"


def format_number(value, decimals=1):
    """Format number with specific decimals."""
    return f"{value:,.{decimals}f}"


# ─────────────────────────────────────────────────────────────
# Prediction Engine
# ─────────────────────────────────────────────────────────────


class MoviePredictor:
    """Movie revenue prediction using OLS regression models."""

    def __init__(self, df):
        """Initialize predictor with dataset."""
        self.df = df
        self.model_basic = None
        self.model_full = None
        self._fit_models()

    def _fit_models(self):
        """Fit regression models on the full dataset."""
        # Basic model: revenue ~ budget
        self.model_basic = run_ols(self.df, formula="log_revenue ~ log_budget")

        # Full model: revenue ~ budget + cast_pop + ratings + genre
        self.model_full = run_ols(
            self.df,
            formula="log_revenue ~ log_budget + log_cast_pop + vote_average + C(primary_genre)",
        )

    def predict(self, budget, cast_pop, vote_avg, genre):
        """
        Predict movie revenue using the full model.

        Parameters
        ----------
        budget : float
            Production budget in USD
        cast_pop : float
            Average TMDB popularity score of top 3 cast members
        vote_avg : float
            Expected audience rating (0-10)
        genre : str
            Primary movie genre

        Returns
        -------
        dict
            Prediction results including revenue, ROI, confidence intervals
        """
        # Prepare input data
        input_data = pd.DataFrame(
            {
                "log_budget": [np.log1p(budget)],
                "log_cast_pop": [np.log1p(cast_pop)],
                "vote_average": [vote_avg],
                "primary_genre": [genre],
            }
        )

        # Make prediction
        try:
            prediction_log = self.model_full.predict(input_data)[0]
            predicted_revenue = np.expm1(prediction_log)

            # Get prediction intervals
            pred_results = self.model_full.get_prediction(input_data)
            ci_log = pred_results.conf_int(alpha=0.05)
            ci_lower = np.expm1(ci_log[0][0])
            ci_upper = np.expm1(ci_log[0][1])

            # Calculate ROI
            roi = (predicted_revenue - budget) / budget if budget > 0 else 0

            return {
                "predicted_revenue": predicted_revenue,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "roi": roi,
                "budget": budget,
                "cast_popularity": cast_pop,
                "vote_average": vote_avg,
                "genre": genre,
                "success_probability": self._estimate_success_prob(
                    predicted_revenue, budget
                ),
                "key_factors": self._analyze_key_factors(budget, cast_pop, vote_avg),
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

    def _estimate_success_prob(self, predicted_revenue, budget):
        """Estimate probability of financial success (ROI > 0)."""
        if predicted_revenue > budget:
            return (
                "High (>"
                + format_percentage((predicted_revenue - budget) / predicted_revenue)
                + " margin)"
            )
        else:
            return "Low (expected loss)"

    def _analyze_key_factors(self, budget, cast_pop, vote_avg):
        """Analyze which factors drive the prediction."""
        factors = []
        if cast_pop > 50:
            factors.append("⭐ Very High Star Power")
        elif cast_pop > 30:
            factors.append("⭐ High Star Power")
        elif cast_pop > 15:
            factors.append("⭐ Moderate Star Power")
        else:
            factors.append("⭐ Low Star Power")

        if budget > 100_000_000:
            factors.append("💰 Blockbuster Budget")
        elif budget > 50_000_000:
            factors.append("💰 Mid-High Budget")
        elif budget > 20_000_000:
            factors.append("💰 Mid Budget")
        else:
            factors.append("💰 Limited Budget")

        if vote_avg > 8:
            factors.append("🎯 Excellent Ratings Expected")
        elif vote_avg > 6.5:
            factors.append("🎯 Good Ratings Expected")
        elif vote_avg > 5:
            factors.append("🎯 Average Ratings Expected")
        else:
            factors.append("🎯 Poor Ratings Expected")

        return factors

    def get_model_summary(self):
        """Return model summary statistics."""
        return {
            "basic_model": {
                "r_squared": self.model_basic.rsquared,
                "adj_r_squared": self.model_basic.rsquared_adj,
                "aic": self.model_basic.aic,
                "bic": self.model_basic.bic,
                "params": self.model_basic.params.to_dict(),
            },
            "full_model": {
                "r_squared": self.model_full.rsquared,
                "adj_r_squared": self.model_full.rsquared_adj,
                "aic": self.model_full.aic,
                "bic": self.model_full.bic,
                "params": self.model_full.params.to_dict(),
            },
        }


# ─────────────────────────────────────────────────────────────
# Visualization Builders
# ─────────────────────────────────────────────────────────────


def create_budget_revenue_scatter(df, log_scale=False):
    """Create interactive budget vs revenue scatter plot."""
    if log_scale:
        x_col, y_col = "log_budget", "log_revenue"
        x_title, y_title = "Log(Budget)", "Log(Revenue)"
    else:
        x_col, y_col = "budget", "revenue"
        x_title, y_title = "Budget ($)", "Revenue ($)"

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="primary_genre",
        size="cast_popularity_score",
        hover_data=["title", "vote_average", "roi"],
        title=f"Budget vs Revenue {'(Log-Log)' if log_scale else ''}",
        labels={x_col: x_title, y_col: y_title},
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_genre_comparison_chart(df, metric="revenue"):
    """Create bar chart comparing genres."""
    genre_stats = (
        df.groupby("primary_genre")[["budget", "revenue", "roi"]].mean().reset_index()
    )

    fig = go.Figure()

    if metric == "revenue":
        fig.add_trace(
            go.Bar(
                x=genre_stats["primary_genre"],
                y=genre_stats["budget"],
                name="Avg Budget",
                marker_color="#1f77b4",
            )
        )
        fig.add_trace(
            go.Bar(
                x=genre_stats["primary_genre"],
                y=genre_stats["revenue"],
                name="Avg Revenue",
                marker_color="#ff7f0e",
            )
        )
        fig.update_layout(barmode="group", title="Average Budget vs Revenue by Genre")
    elif metric == "roi":
        fig = px.bar(
            genre_stats,
            x="primary_genre",
            y="roi",
            title="Average ROI by Genre",
            color="roi",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
        )

    fig.update_layout(height=500, xaxis_title="Genre", yaxis_title="Amount ($)")
    return fig


def create_distribution_plots(df):
    """Create grid of distribution plots."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Budget Distribution",
            "Revenue Distribution",
            "Cast Popularity",
            "Vote Average",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "histogram"}],
        ],
    )

    # Budget
    fig.add_trace(
        go.Histogram(x=df["budget"], name="Budget", marker_color="#1f77b4"),
        row=1,
        col=1,
    )
    # Revenue
    fig.add_trace(
        go.Histogram(x=df["revenue"], name="Revenue", marker_color="#ff7f0e"),
        row=1,
        col=2,
    )
    # Cast Popularity
    fig.add_trace(
        go.Histogram(
            x=df["cast_popularity_score"], name="Cast Pop", marker_color="#2ca02c"
        ),
        row=2,
        col=1,
    )
    # Vote Average
    fig.add_trace(
        go.Histogram(x=df["vote_average"], name="Votes", marker_color="#d62728"),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, showlegend=False, title_text="Variable Distributions")
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap."""
    numeric_cols = [
        "budget",
        "revenue",
        "cast_popularity_score",
        "vote_average",
        "runtime",
        "roi",
    ]
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
    )

    fig.update_layout(height=500)
    return fig


def create_top_movies_chart(df, n=10):
    """Create horizontal bar chart of top-grossing movies."""
    top_movies = df.nlargest(n, "revenue")[["title", "revenue", "primary_genre"]]

    fig = px.bar(
        top_movies,
        x="revenue",
        y="title",
        color="primary_genre",
        orientation="h",
        title=f"Top {n} Movies by Revenue",
        labels={"revenue": "Revenue ($)", "title": "Movie"},
        hover_data=["primary_genre"],
    )

    fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
    return fig


# ─────────────────────────────────────────────────────────────
# Statistical Analysis Helpers
# ─────────────────────────────────────────────────────────────


def run_full_analysis(df):
    """Run complete statistical analysis and return results."""
    results = {}

    # Summary statistics
    results["summary"] = df[
        ["budget", "revenue", "cast_popularity_score", "vote_average", "roi"]
    ].describe()

    # Correlation matrix
    results["correlation"] = df[
        ["budget", "revenue", "cast_popularity_score", "vote_average", "roi"]
    ].corr()

    # Model comparisons
    model_basic = run_ols(df, "log_revenue ~ log_budget")
    model_full = run_ols(
        df, "log_revenue ~ log_budget + log_cast_pop + vote_average + C(primary_genre)"
    )

    results["model_basic"] = {
        "rsquared": model_basic.rsquared,
        "adj_rsquared": model_basic.rsquared_adj,
        "aic": model_basic.aic,
        "bic": model_basic.bic,
        "params": model_basic.params,
    }

    results["model_full"] = {
        "rsquared": model_full.rsquared,
        "adj_rsquared": model_full.rsquared_adj,
        "aic": model_full.aic,
        "bic": model_full.bic,
        "params": model_full.params,
    }

    # ANOVA
    anova_result = run_anova(df)
    results["anova"] = anova_result

    # VIF
    vif_df = check_vif(df, ["log_budget", "log_cast_pop", "vote_average"])
    results["vif"] = vif_df

    return results


# ─────────────────────────────────────────────────────────────
# Data Filters & Slicers
# ─────────────────────────────────────────────────────────────


def filter_dataframe(
    df, genre=None, year_range=None, min_budget=None, max_revenue=None
):
    """Apply filters to DataFrame."""
    filtered = df.copy()

    if genre and genre != "All":
        filtered = filtered[filtered["primary_genre"] == genre]

    if year_range:
        filtered = filtered[
            (filtered["release_year"] >= year_range[0])
            & (filtered["release_year"] <= year_range[1])
        ]

    if min_budget:
        filtered = filtered[filtered["budget"] >= min_budget]

    if max_revenue:
        filtered = filtered[filtered["revenue"] <= max_revenue]

    return filtered


def get_genre_stats(df):
    """Get statistics by genre."""
    return (
        df.groupby("primary_genre")
        .agg(
            {
                "budget": ["mean", "median", "count"],
                "revenue": ["mean", "median"],
                "roi": ["mean", "median"],
                "cast_popularity_score": "mean",
                "vote_average": "mean",
            }
        )
        .round(2)
    )


# ─────────────────────────────────────────────────────────────
# Export Functions
# ─────────────────────────────────────────────────────────────


def export_predictions_to_csv(predictions_list, filename="predictions.csv"):
    """Export list of predictions to CSV."""
    df = pd.DataFrame(predictions_list)
    csv = df.to_csv(index=False)
    return csv


# ─────────────────────────────────────────────────────────────
# Validation Functions
# ─────────────────────────────────────────────────────────────


def validate_inputs(budget, cast_pop, vote_avg):
    """Validate user inputs."""
    errors = []

    if budget < 1_000 or budget > 1_000_000_000:
        errors.append("Budget must be between $1K and $1B")

    if cast_pop < 0 or cast_pop > 100:
        errors.append("Cast popularity must be between 0 and 100")

    if vote_avg < 0 or vote_avg > 10:
        errors.append("Vote average must be between 0 and 10")

    return errors


def get_success_threshold(roi):
    """Categorize success based on ROI."""
    if roi >= 2.0:
        return "Blockbuster", "#2ca02c"
    elif roi >= 1.0:
        return "Profitable", "#1f77b4"
    elif roi >= 0:
        return "Break-even", "#ff7f0e"
    else:
        return "Unprofitable", "#d62728"
