"""
main.py
-------
Star Power Index Greenlight Tool - Main Streamlit Application

Team Ghanti Tininini | King's College Nepal, Westcliff University
Course: DATA 200 Applied Statistical Analysis | Professor: Regmi
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Import streamlit first (required for error messages)
try:
    import streamlit as st
except ImportError:
    print("❌ Critical: streamlit not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


# ── Environment checks ──────────────────────────────────────────
def verify_environment():
    """Check that required dependencies are available."""
    missing = []
    optional_missing = []

    # Required
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import plotly
    except ImportError:
        missing.append("plotly")

    # Optional but needed for full functionality
    try:
        import statsmodels
    except ImportError:
        optional_missing.append("statsmodels (needed for predictions)")
    try:
        import scipy
    except ImportError:
        optional_missing.append("scipy (needed for statistical tests)")

    if missing:
        st.error("❌ Critical dependencies missing: " + ", ".join(missing))
        st.markdown("""
        ### Install Requirements

        ```bash
        pip install -r requirements.txt
        ```

        If using a virtual environment (recommended):
        ```bash
        python -m venv venv
        venv\\Scripts\\activate   # Windows
        # or source venv/bin/activate  # Mac/Linux
        pip install -r requirements.txt
        ```
        """)
        st.stop()

    if optional_missing:
        st.warning("⚠️ Some features limited. Missing: " + ", ".join(optional_missing))


verify_environment()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Star Power Index - Greenlight Tool",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────
def load_custom_css():
    st.markdown(
        """
    <style>
    :root {
        --primary: #1f77b4;
        --secondary: #ff7f0e;
        --accent: #2ca02c;
        --dark-bg: #0e1117;
        --card-bg: #1e2129;
        --text-main: #ffffff;
        --text-muted: #b0b3b8;
    }
    .stApp { background-color: var(--dark-bg); color: var(--text-main); }
    .main-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle { font-size: 1.1rem; color: var(--text-muted); margin-bottom: 2rem; }
    .team-banner {
        background: var(--card-bg); padding: 1rem 1.5rem; border-radius: 8px;
        border-left: 4px solid var(--primary); margin-bottom: 2rem;
    }
    [data-testid="stSidebar"] { background-color: var(--card-bg); }
    .sidebar-header { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; }
    .input-panel {
        background: var(--card-bg); padding: 1.2rem; border-radius: 10px;
        margin-bottom: 1rem;
    }
    .input-panel h3 { color: var(--primary); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.95rem; }
    .stButton button {
        background: linear-gradient(90deg, var(--primary), #3776ab); color: white;
        border: none; border-radius: 8px; padding: 0.6rem 1.5rem; font-weight: 600;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #3776ab, var(--primary));
        transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .disclaimer {
        font-size: 0.8rem; color: var(--text-muted); font-style: italic;
        padding: 1rem; background: var(--card-bg); border-radius: 5px;
        border-left: 3px solid var(--secondary); margin-top: 2rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ── Data Loading ───────────────────────────────────────────────
def load_clean_data():
    """Load cleaned dataset from outputs folder."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    clean_csv = os.path.join(output_dir, "tmdb_cleaned.csv")
    if os.path.exists(clean_csv):
        df = pd.read_csv(clean_csv)
        if "primary_genre" in df.columns:
            df["primary_genre"] = pd.Categorical(df["primary_genre"])
        return df
    return None


# ── Formatting ─────────────────────────────────────────────────
def format_currency(value):
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    else:
        return f"${value:,.0f}"


def format_percentage(value):
    return f"{value:.1%}"


# ── MoviePredictor ─────────────────────────────────────────────
class MoviePredictor:
    """Predict movie revenue using OLS regression."""

    def __init__(self, df):
        self.df = df
        self.model_full = None
        self.model_simple = None
        self._fit()

    def _fit(self):
        df_model = self.df.copy()
        if not hasattr(df_model["primary_genre"].dtype, "categories"):
            df_model["primary_genre"] = pd.Categorical(df_model["primary_genre"])

        # Lazy import to avoid issues if statsmodels partially broken
        from statsmodels.formula.api import ols

        self.model_full = ols(
            "log_revenue ~ log_budget + log_cast_pop + vote_average + C(primary_genre)",
            data=df_model,
        ).fit()
        self.model_simple = ols(
            "log_revenue ~ log_budget + log_cast_pop + vote_average", data=df_model
        ).fit()

    def predict(self, budget, cast_pop, vote_avg, genre):
        df_model = self.df.copy()
        if not hasattr(df_model["primary_genre"].dtype, "categories"):
            df_model["primary_genre"] = pd.Categorical(df_model["primary_genre"])

        input_data = pd.DataFrame(
            {
                "log_budget": [np.log1p(budget)],
                "log_cast_pop": [np.log1p(cast_pop)],
                "vote_average": [vote_avg],
                "primary_genre": [genre],
            }
        )
        input_data["primary_genre"] = pd.Categorical(
            input_data["primary_genre"],
            categories=df_model["primary_genre"].cat.categories,
        )

        try:
            pred_log = self.model_full.predict(input_data)[0]
            revenue = np.expm1(pred_log)
            pred_int = self.model_full.get_prediction(input_data).conf_int(alpha=0.05)
            ci_low = np.expm1(pred_int[0][0])
            ci_high = np.expm1(pred_int[0][1])
            roi = (revenue - budget) / budget if budget else 0
            return {
                "revenue": revenue,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "roi": roi,
                "model": "full",
            }
        except Exception as e:
            st.warning(f"Genre model failed: {e}. Using aggregate model.")
            input_simple = pd.DataFrame(
                {
                    "log_budget": [np.log1p(budget)],
                    "log_cast_pop": [np.log1p(cast_pop)],
                    "vote_average": [vote_avg],
                }
            )
            pred_log = self.model_simple.predict(input_simple)[0]
            revenue = np.expm1(pred_log)
            roi = (revenue - budget) / budget if budget else 0
            return {
                "revenue": revenue,
                "ci_lower": revenue * 0.7,
                "ci_upper": revenue * 1.3,
                "roi": roi,
                "model": "simple",
            }


# ── Header ─────────────────────────────────────────────────────
def render_header():
    st.markdown(
        '<div class="main-title">⭐ Star Power Index</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Greenlight Tool — Interactive Movie Success Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="team-banner">
    <strong>Team Ghanti Tininini</strong> | King's College Nepal, Westcliff University<br>
    Course: DATA 200 Applied Statistical Analysis | Professor: Regmi<br>
    <em>Research Question: Does cast popularity independently predict global box office revenue?</em>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Page 1: Movie Predictor ───────────────────────────────────
def show_prediction_page(df):
    st.header("🎬 Movie Success Predictor")
    predictor = MoviePredictor(df)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎛️ Movie Parameters")
        with st.container():
            budget = st.slider(
                "Production Budget (USD)",
                1_000_000,
                400_000_000,
                50_000_000,
                step=5_000_000,
                format="$%d",
            )
            st.markdown(f"**Budget:** {format_currency(budget)}")
            st.markdown("<br>", unsafe_allow_html=True)

            cast_pop = st.slider(
                "Top-3 Cast Average TMDB Popularity (0-100)",
                0.0,
                100.0,
                30.0,
                step=0.5,
                help="Average popularity of your lead actors",
            )
            st.markdown("<br>", unsafe_allow_html=True)

            vote_avg = st.slider(
                "Expected TMDB Audience Rating (0-10)", 0.0, 10.0, 6.5, step=0.1
            )
            st.markdown("<br>", unsafe_allow_html=True)

            genres = sorted(df["primary_genre"].unique())
            genre = st.selectbox(
                "Primary Genre",
                options=genres,
                index=genres.index("Action") if "Action" in genres else 0,
            )

        if st.button("🔮 Predict Revenue", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                result = predictor.predict(budget, cast_pop, vote_avg, genre)
            with col2:
                st.markdown("### 📊 Results")
                st.metric(
                    "Predicted Global Revenue",
                    format_currency(result["revenue"]),
                    delta=f"vs {format_currency(budget)} budget",
                )
                st.metric(
                    "Return on Investment (ROI)",
                    format_percentage(result["roi"]),
                    delta="Positive" if result["roi"] > 0 else "Negative",
                )
                st.markdown(
                    f"**95% CI:** {format_currency(result['ci_lower'])} – {format_currency(result['ci_upper'])}"
                )
                st.markdown("---")
                st.markdown("### 🎯 Verdict")
                if result["roi"] > 2.0:
                    st.success(
                        f"🚀 BLOCKBUSTER — {format_percentage(result['roi'])} ROI"
                    )
                elif result["roi"] > 1.0:
                    st.success(
                        f"✅ PROFITABLE — {format_percentage(result['roi'])} ROI"
                    )
                elif result["roi"] > 0:
                    st.warning(f"⚖️ BREAK-EVEN — {format_percentage(result['roi'])} ROI")
                else:
                    st.error(f"⚠️ HIGH RISK — {format_percentage(result['roi'])} ROI")
                if result.get("model") == "simple":
                    st.info("Note: Using aggregate model (genre effects not applied).")

    st.markdown("---")
    st.markdown("### 🎬 Comparable Movies")
    similar = df[df["primary_genre"] == genre].nlargest(8, "revenue")
    if len(similar) > 0:
        fig = px.bar(
            similar,
            x="revenue",
            y="title",
            color="cast_popularity_score",
            orientation="h",
            title=f"Top {genre} Films (Color = Cast Popularity)",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# ── Page 2: Data Explorer ─────────────────────────────────────
def show_exploration_page(df):
    st.header("📊 Data Exploration Dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Movies", f"{len(df):,}")
    c2.metric("Avg Budget", format_currency(df["budget"].mean()))
    c3.metric("Avg Revenue", format_currency(df["revenue"].mean()))
    c4.metric("Avg ROI", format_percentage(df["roi"].mean()))
    c5.metric("Genres", f"{df['primary_genre'].nunique()}")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "💰 Financial", "⭐ Star Power", "🎭 Genres"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.histogram(
                    df,
                    x="budget",
                    nbins=50,
                    title="Budget Distribution",
                    color_discrete_sequence=["#1f77b4"],
                ),
                True,
            )
        with col2:
            st.plotly_chart(
                px.histogram(
                    df,
                    x="revenue",
                    nbins=50,
                    title="Revenue Distribution",
                    color_discrete_sequence=["#ff7f0e"],
                ),
                True,
            )
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                px.histogram(
                    df,
                    x="cast_popularity_score",
                    nbins=50,
                    title="Cast Popularity",
                    color_discrete_sequence=["#2ca02c"],
                ),
                True,
            )
        with col4:
            st.plotly_chart(
                px.histogram(
                    df,
                    x="vote_average",
                    nbins=50,
                    title="Audience Ratings",
                    color_discrete_sequence=["#d62728"],
                ),
                True,
            )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                df,
                x="budget",
                y="revenue",
                color="primary_genre",
                size="cast_popularity_score",
                hover_data=["title", "vote_average"],
                title="Budget vs Revenue (Bubble = Popularity)",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, True)
        with col2:
            # Log-Log scatter with manual trendline (avoids statsmodels import issue)
            fig = px.scatter(
                df,
                x="log_budget",
                y="log_revenue",
                color="primary_genre",
                title="Log-Log: Budget vs Revenue",
            )
            try:
                z = np.polyfit(df["log_budget"], df["log_revenue"], 1)
                p = np.poly1d(z)
                fig.add_scatter(
                    x=df["log_budget"].sort_values(),
                    y=p(df["log_budget"].sort_values()),
                    mode="lines",
                    name="Trend",
                    line=dict(color="red", dash="dash"),
                )
            except Exception:
                pass
            fig.update_layout(height=600)
            st.plotly_chart(fig, True)
        st.subheader("🔗 Correlation Matrix")
        corr = df[
            ["budget", "revenue", "cast_popularity_score", "vote_average", "roi"]
        ].corr()
        st.plotly_chart(
            px.imshow(
                corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r"
            ),
            True,
        )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            # Star Power vs Revenue with manual trendline
            fig = px.scatter(
                df,
                x="cast_popularity_score",
                y="revenue",
                color="primary_genre",
                title="Star Power vs Revenue",
            )
            try:
                z = np.polyfit(df["cast_popularity_score"], df["revenue"], 1)
                p = np.poly1d(z)
                fig.add_scatter(
                    x=df["cast_popularity_score"].sort_values(),
                    y=p(df["cast_popularity_score"].sort_values()),
                    mode="lines",
                    name="Trend",
                    line=dict(color="red", dash="dash"),
                )
            except Exception:
                pass
            fig.update_layout(height=600)
            st.plotly_chart(fig, True)
        with col2:
            gp = (
                df.groupby("primary_genre")["cast_popularity_score"]
                .mean()
                .sort_values(ascending=True)
            )
            fig = px.bar(
                x=gp.values,
                y=gp.index,
                orientation="h",
                title="Avg Cast Popularity by Genre",
                color=gp.values,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(showlegend=False, height=600)
            st.plotly_chart(fig, True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            gs = df.groupby("primary_genre")[["budget", "revenue"]].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=gs["primary_genre"],
                    y=gs["budget"],
                    name="Avg Budget",
                    marker_color="#1f77b4",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=gs["primary_genre"],
                    y=gs["revenue"],
                    name="Avg Revenue",
                    marker_color="#ff7f0e",
                )
            )
            fig.update_layout(barmode="group", title="Budget vs Revenue by Genre")
            st.plotly_chart(fig, True)
        with col2:
            roi_g = (
                df.groupby("primary_genre")["roi"].mean().sort_values(ascending=True)
            )
            fig = px.bar(
                x=roi_g.values,
                y=roi_g.index,
                orientation="h",
                title="Avg ROI by Genre",
                color=roi_g.values,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, True)
        st.subheader("📋 Genre Distribution")
        gd = pd.DataFrame(
            {
                "Genre": df["primary_genre"].value_counts().index,
                "Count": df["primary_genre"].value_counts().values,
                "Percentage": (df["primary_genre"].value_counts(normalize=True) * 100)
                .round(1)
                .values,
            }
        )
        st.dataframe(gd, use_container_width=True, hide_index=True)


# ── Page 3: Model Insights ─────────────────────────────────────
def show_model_insights_page(df):
    st.header("📈 Model Insights")
    st.markdown(
        "Statistical analysis of whether cast popularity independently predicts box office revenue."
    )

    # Check if analysis modules are available
    try:
        from src.analysis import (
            run_ols,
            run_anova,
            check_vif,
            check_homoscedasticity,
            check_normality_residuals,
        )

        ANALYSIS = True
    except ImportError as e:
        ANALYSIS = False
        import_error = str(e)

    if not ANALYSIS:
        st.error("⚠️ Statistical analysis modules could not be loaded.")
        st.code(f"Error: {import_error}")
        st.markdown("""
        Ensure `src/analysis.py` is accessible and dependencies are installed:
        ```bash
        pip install statsmodels scipy
        ```
        """)
        return

    with st.spinner("Running models..."):
        from statsmodels.formula.api import ols

        m1 = ols("log_revenue ~ log_budget", data=df).fit()
        m2 = ols(
            "log_revenue ~ log_budget + log_cast_pop + vote_average + C(primary_genre)",
            data=df,
        ).fit()

    st.subheader("🔬 Model Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model 1: Budget Only**")
        st.code(
            f"log_revenue ~ log_budget\nR² = {m1.rsquared:.3f}\nAdj.R² = {m1.rsquared_adj:.3f}\nAIC = {m1.aic:.1f}"
        )
    with c2:
        st.markdown("**Model 2: Full (adds Cast + Ratings + Genre)**")
        st.code(
            f"+ log_cast_pop + vote_average + C(genre)\nR² = {m2.rsquared:.3f}\nAdj.R² = {m2.rsquared_adj:.3f}\nAIC = {m2.aic:.1f}"
        )

    delta = m2.rsquared - m1.rsquared
    st.success(
        f" Adding cast popularity & ratings explains an extra {delta:.1%} variance (ΔR² = {delta:.3f})"
    )

    st.subheader("📊 Coefficients (Full Model)")
    coefs = pd.DataFrame(
        {
            "Coefficient": m2.params,
            "Std Error": m2.bse,
            "t-value": m2.tvalues,
            "P-value": m2.pvalues,
            "Significant (p<0.05)": m2.pvalues < 0.05,
        }
    ).round(4)
    st.dataframe(coefs, use_container_width=True)

    st.subheader("🔍 Multicollinearity (VIF)")
    # Use our safe check_vif function instead of direct statsmodels import
    vif = check_vif(df, ["log_budget", "log_cast_pop", "vote_average"], verbose=False)
    st.dataframe(vif, use_container_width=True, hide_index=True)
    if vif["VIF"].iloc[1:].max() < 5:
        st.success("All VIF < 5 — no concerning multicollinearity")
    else:
        st.warning("Some VIF > 5 — possible collinearity")

    st.subheader("🎭 ANOVA: Genre Revenue Differences")
    from scipy import stats as scipy_stats

    groups = [
        g["log_revenue"].dropna().values
        for _, g in df.groupby("primary_genre")
        if len(g) >= 10
    ]
    labels = [n for n, g in df.groupby("primary_genre") if len(g) >= 10]
    F, p = scipy_stats.f_oneway(*groups)
    c1, c2, c3 = st.columns(3)
    c1.metric("F-statistic", f"{F:.2f}")
    c2.metric("P-value", f"{p:.2e}")
    c3.metric("Significant?", "✅ Yes" if p < 0.05 else "❌ No")
    if p < 0.05:
        st.success("Revenue differs significantly across genres (p < 0.05)")
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        tukey = pairwise_tukeyhsd(df["log_revenue"], df["primary_genre"])
        with st.expander("Tukey HSD Post-hoc Results"):
            st.text(tukey.summary())
    else:
        st.info("No significant genre differences")

    st.subheader("📊 Residual Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        from scipy import stats as scipy_stats2

        w, p_norm = scipy_stats2.shapiro(m2.resid[:5000])
        st.markdown(
            f"**Shapiro-Wilk**\n\nW = {w:.4f}, p = {p_norm:.6f}\n\n{'✓ Normal' if p_norm >= 0.05 else '✗ Non-normal'}"
        )
    with col2:
        # Use our safe homoscedasticity check (avoids problematic statsmodels import)
        bp_stat, bp_p = check_homoscedasticity(m2, verbose=False)
        st.markdown(
            f"**Breusch-Pagan**\n\nBP = {bp_stat:.4f}, p = {bp_p:.6f}\n\n{'✓ Homoscedastic' if bp_p >= 0.05 else '✗ Heteroscedastic'}"
        )

    import scipy.stats as stats_sp

    qq = stats_sp.probplot(m2.resid, dist="norm", plot=None)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qq[0][0],
            y=qq[0][1],
            mode="markers",
            name="Residuals",
            marker=dict(color="#1f77b4", size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qq[0][0],
            y=qq[1][0] + qq[1][1] * qq[0][0],
            mode="lines",
            name="Normal",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )
    fig.update_layout(title="Q-Q Plot", height=500)
    st.plotly_chart(fig, True)


# ── Page 4: About ─────────────────────────────────────────────
def show_about_page(df):
    st.header("📚 About the Project")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 Research Question
        Does cast popularity (TMDB score) independently predict global box office revenue, beyond production budget?

        ### 🛠️ Methodology
        - Data: TMDB 5000 movies → 2,641 cleaned films
        - Models: OLS regression, One-way ANOVA, Tukey HSD
        - Diagnostics: VIF, Shapiro-Wilk, Breusch-Pagan

        ### 🎯 Greenlight Tool Purpose
        - Estimate potential film revenue
        - Compare ROI expectations across genres
        - Quantify star power value in casting
        - Assess risk/reward for different budgets

        ---
        ### 👥 Team Ghanti Tininini
        | Name | Role |
        |---|---|
        | Anuprash Pokharel | Project Lead |
        | Kabit Khadka | Statistical Analyst |
        | Prashanna Dhami | App Developer |
        | Sarjyant Maharjan | Data Engineer |

        **Course:** DATA 200 Applied Statistical Analysis  
        **Institution:** King's College Nepal, Westcliff University  
        **Professor:** Regmi
        """)
    with col2:
        st.markdown("### 📊 Dataset Overview")
        desc = (
            df[["budget", "revenue", "cast_popularity_score", "vote_average", "roi"]]
            .describe()
            .loc[["mean", "std", "min", "max"]]
        )
        st.dataframe(desc.T, use_container_width=True)
        st.markdown("### 📅 Time Span")
        ymin, ymax = int(df["release_year"].min()), int(df["release_year"].max())
        st.metric("From", f"{ymin}")
        st.metric("To", f"{ymax}")
        st.markdown("### 🎭 Top Genres")
        for g, c in df["primary_genre"].value_counts().head(5).items():
            st.markdown(f"**{g}:** {c} films")
    st.markdown("---")
    st.warning(
        "⚠️ Predictions are estimates based on historical patterns. Actual results may vary due to marketing, competition, timing, and other factors."
    )


# ── Main Application ───────────────────────────────────────────
def main():
    load_custom_css()
    render_header()

    df = load_clean_data()
    if df is None or df.empty:
        st.error("❌ Clean dataset not found (`outputs/tmdb_cleaned.csv`)")
        st.markdown("""
        ### 🚀 Setup Steps

        1. **Download data** from Kaggle: `tmdb_5000_movies.csv` & `tmdb_5000_credits.csv`
           Place them in the `data/` folder

        2. **Install dependencies:**
           ```bash
           pip install -r requirements.txt
           ```

        3. **Run preprocessing:**
           ```bash
           python -m src.loader         # Fetch cast data from TMDB API
           python -m src.preprocessing   # Clean & feature engineer
           ```

        4. **Launch app:**
           ```bash
           streamlit run main.py
           ```
        """)
        return

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header">⭐ Navigation</div>', unsafe_allow_html=True
        )
        st.markdown("---")
        page = st.radio(
            "Go to",
            ["🎬 Movie Predictor", "📊 Data Explorer", "📈 Model Insights", "📚 About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("### 📊 Dataset")
        st.metric("Movies", f"{len(df):,}")
        st.metric("Genres", f"{df['primary_genre'].nunique()}")
        st.metric(
            "Years", f"{int(df['release_year'].min())}–{int(df['release_year'].max())}"
        )
        st.markdown("---")
        st.markdown(
            """<small style="color: var(--text-muted);">**Star Power Index** v1.0<br>Team Ghanti Tininini<br>DATA 200</small>""",
            unsafe_allow_html=True,
        )

    if page == "🎬 Movie Predictor":
        show_prediction_page(df)
    elif page == "📊 Data Explorer":
        show_exploration_page(df)
    elif page == "📈 Model Insights":
        show_model_insights_page(df)
    elif page == "📚 About":
        show_about_page(df)

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ Estimates based on historical patterns. Actual results may vary.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
