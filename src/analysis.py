"""
analysis.py
-----------
Statistical analysis helpers for the Star Power Index.

Week 4 will build on this module for:
  - Multiple Linear Regression (OLS)
  - One-way ANOVA (genre groups)
  - Tukey HSD post-hoc comparisons
  - Assumption checks: normality, homoscedasticity, VIF

Usage
-----
    from src.analysis import run_ols, run_anova
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _add_constant(df: pd.DataFrame) -> pd.DataFrame:
    """Add an intercept column (constant=1) to DataFrame."""
    return pd.DataFrame({"const": 1.0, **df})


def run_ols(
    df: pd.DataFrame,
    formula: str = "log_revenue ~ log_budget + log_cast_pop + vote_average",
    verbose: bool = True,
):
    """
    Fit an OLS regression model using a Patsy formula string.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame with required columns
    formula : str
        Patsy formula (e.g., "log_revenue ~ log_budget")
    verbose : bool
        If True, print model summary

    Returns
    -------
    statsmodels OLS results object
    """
    model = ols(formula, data=df).fit()
    if verbose:
        print(model.summary())
    return model


def check_vif(
    df: pd.DataFrame, predictors: list[str], verbose: bool = True
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors for a list of predictor columns.
    VIF > 10 suggests problematic multicollinearity.
    Manual implementation to avoid statsmodels import issues.
    """
    X = df[predictors].dropna().copy()
    X = _add_constant(X)  # adds intercept column

    vif_data = pd.DataFrame({"Feature": X.columns, "VIF": np.nan})

    # Compute VIF for each predictor (excluding const)
    for i, col in enumerate(X.columns):
        if col == "const":
            vif_data.loc[i, "VIF"] = 1.0
        else:
            # Regress this predictor on all other predictors
            other_cols = [c for c in X.columns if c != col]
            if len(other_cols) == 0:
                vif_data.loc[i, "VIF"] = 1.0
            else:
                # OLS: col ~ other_cols
                model = ols(f"{col} ~ {' + '.join(other_cols)}", data=X).fit()
                vif = 1 / (1 - model.rsquared)
                vif_data.loc[i, "VIF"] = vif

    vif_data["VIF"] = vif_data["VIF"].round(3)

    if verbose:
        print("\n=== VIF (multicollinearity check) ===")
        print(vif_data.to_string(index=False))
    return vif_data


# ─────────────────────────────────────────────────────────────
# One-way ANOVA  (genre groups)
# ─────────────────────────────────────────────────────────────


def run_anova(
    df: pd.DataFrame,
    group_col: str = "primary_genre",
    value_col: str = "log_revenue",
    verbose: bool = True,
) -> dict:
    """
    One-way ANOVA testing whether mean log_revenue differs across genres.

    H0: μ_Action = μ_Comedy = μ_Drama = ... (all genre means equal)
    H1: At least one genre mean differs

    Returns
    -------
    dict with keys: 'F', 'p_value', 'groups', 'tukey_results'
    """
    groups = [
        grp[value_col].dropna().values
        for _, grp in df.groupby(group_col)
        if len(grp[value_col].dropna()) >= 10
    ]
    group_labels = [
        name
        for name, grp in df.groupby(group_col)
        if len(grp[value_col].dropna()) >= 10
    ]

    F, p = stats.f_oneway(*groups)

    if verbose:
        print(f"\n=== One-way ANOVA: {value_col} by {group_col} ===")
        print(f"F-statistic = {F:.4f}   p-value = {p:.6f}")
        if p < 0.05:
            print("→ Reject H0: genre means differ significantly (α = 0.05)")
        else:
            print("→ Fail to reject H0")

        # Tukey HSD post-hoc
        tukey = pairwise_tukeyhsd(
            endog=df.loc[df[group_col].isin(group_labels), value_col],
            groups=df.loc[df[group_col].isin(group_labels), group_col],
            alpha=0.05,
        )
        print("\n=== Tukey HSD Post-hoc ===")
        print(tukey.summary())
    else:
        tukey = pairwise_tukeyhsd(
            endog=df.loc[df[group_col].isin(group_labels), value_col],
            groups=df.loc[df[group_col].isin(group_labels), group_col],
            alpha=0.05,
        )

    return {"F": F, "p_value": p, "groups": group_labels, "tukey_results": tukey}


# ─────────────────────────────────────────────────────────────
# Assumption checks
# ─────────────────────────────────────────────────────────────


def check_normality_residuals(residuals: pd.Series, verbose: bool = True) -> None:
    """
    Shapiro-Wilk test on OLS residuals.
    Also prints skewness and kurtosis.
    """
    stat, p = stats.shapiro(residuals.dropna()[:5000])  # Shapiro limited to 5000
    if verbose:
        print(f"\n=== Normality of Residuals (Shapiro-Wilk, n≤5000) ===")
        print(f"W = {stat:.4f},  p = {p:.6f}")
        print(
            f"Skewness: {residuals.skew():.3f}   Kurtosis: {residuals.kurtosis():.3f}"
        )
        if p < 0.05:
            print("→ Residuals may deviate from normality; consider robust SE.")
        else:
            print("→ Residuals consistent with normality.")


def check_homoscedasticity(model, verbose: bool = True) -> tuple[float, float]:
    """
    Breusch-Pagan test for heteroscedasticity.
    Manual implementation using auxiliary regression to avoid import issues.

    Returns
    -------
    (bp_stat, p_value) : tuple
        Test statistic and p-value.
    """
    residuals = model.resid
    exog = model.model.exog

    # Auxiliary regression: squared residuals ~ exog
    from statsmodels.regression.linear_model import OLS

    squared_resid = residuals**2
    aux_model = OLS(squared_resid, exog).fit()

    n = len(residuals)
    bp_stat = n * aux_model.rsquared
    df = exog.shape[1]
    p_value = 1 - stats.chi2.cdf(bp_stat, df)

    if verbose:
        print(f"\n=== Breusch-Pagan Homoscedasticity Test ===")
        print(f"BP stat = {bp_stat:.4f},  p = {p_value:.6f}")
        if p_value < 0.05:
            print("→ Evidence of heteroscedasticity; consider robust SE.")
        else:
            print("→ No significant heteroscedasticity detected.")

    return bp_stat, p_value
