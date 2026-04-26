# ⭐ Star Power Index — DATA 200 Applied Statistical Analysis

**Team Ghanti Tininini** | King's College Nepal, Westcliff University  
**Course:** DATA 200 Applied Statistical Analysis | **Professor:** Regmi

> **🚨 Important:** This project requires **Python 3.10–3.11** for full compatibility. Python 3.12 may work with `statsmodels>=0.14.3`, but if you encounter import errors, consider using Python 3.11.

## 📌 Research Question
Does cast popularity (TMDB popularity score) independently predict global box office revenue, beyond what production budget alone explains?

## 🚀 Quick Start (After Setup)

1. **Prepare data** — Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in `data/`
2. **Preprocess** — `python -m src.loader && python -m src.preprocessing`
3. **Run app** — `streamlit run main.py`

## 👥 Team Members
| Name | Role |
|---|---|
| Anuprash Pokharel | Project Lead — timeline & final report |
| Kabit Khadka | Statistical Analyst — ANOVA & model validation |
| Prashanna Dhami | App Developer — Greenlight Tool |
| Sarjyant Maharjan | Data Engineer — TMDB cleaning & feature engineering |

---

## 🗂️ Project Structure
```
star_power_index/
├── data/                   # Raw datasets (place downloaded CSV files here)
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── src/                    # Core Python modules
│   ├── config.py           — paths, constants, TMDB API key
│   ├── loader.py           — load & merge datasets + TMDB API calls
│   ├── preprocessing.py    — cleaning, feature engineering pipeline
│   ├── eda.py              — descriptive stats & visualisations
│   ├── analysis.py         — regression, ANOVA, diagnostic tests
│   └── app_utils.py        — helper utilities (formatters, filters)
├── outputs/                # Generated cleaned CSV and plots
│   └── tmdb_cleaned.csv    # ← Required for the Streamlit app
├── notebooks/              # Jupyter notebooks
│   └── week3_eda.ipynb     — exploratory data analysis
├── tests/                  # Unit tests
│   └── test_preprocessing.py
├── main.py                 # 🚀 Streamlit Greenlight Tool
├── requirements.txt        # Python dependencies
├── .env.example            # environment template for TMDB API key
└── README.md
```

---

## ⚙️ Setup Instructions

### 📦 Environment Setup (Recommended)

This project uses a virtual environment. Using one is strongly recommended to avoid package conflicts.

**1. Clone & navigate**
```bash
cd star_power_index
```

**2. Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Python 3.12 Note:**  
> If you encounter `TypeError: deprecate_kwarg() missing 1 required positional argument: 'new_arg_name'`,  
> this is a known compatibility issue with `statsmodels==0.14.2` and Python 3.12.  
> The `requirements.txt` specifies `statsmodels>=0.14.3` which resolves this.  
> Ensure you are using a clean virtual environment with the pinned versions.

### 4. Add your TMDB API key (optional for live API calls)
```bash
cp .env.example .env
# Edit .env and set: TMDB_API_KEY=your_key_here
```
Note: The preprocessing uses cached data; API key is optional for full functionality.

### 5. Download datasets from Kaggle
Place both files inside the `data/` folder:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

### 6. Run the preprocessing pipeline
```bash
python -m src.loader         # Fetch cast popularity from TMDB (requires API key)
python -m src.preprocessing   # Clean, feature engineer, save outputs/tmdb_cleaned.csv
```

### 7. Launch the Greenlight Tool
```bash
streamlit run main.py
```

---

## 📊 Project Components

### Core Analysis Pipeline
- **Data Preprocessing** (`src/preprocessing.py`) - Cleaning and feature engineering
- **Exploratory Analysis** (`src/eda.py`) - Descriptive statistics and visualizations
- **Statistical Modeling** (`src/analysis.py`) - Regression, ANOVA, hypothesis testing
- **Data Loading** (`src/loader.py`) - TMDB dataset ingestion and API integration

### Interactive Greenlight Tool (Streamlit App)
**Launch:**
```bash
streamlit run main.py
```

The Greenlight Tool is an interactive web application that allows users to:

- **Predict movie revenue** based on budget, cast popularity, expected ratings, and genre
- **Explore the dataset** through interactive Plotly visualisations
- **View model insights** including regression coefficients, ANOVA results, and diagnostic tests
- **Compare** predicted movies against actual top performers

**Pages:**
1. **🎬 Movie Predictor** – Input parameters → get revenue prediction + ROI + confidence interval + success verdict
2. **📊 Data Explorer** – Distributions, financial relationships, star power analysis, genre deep dive
3. **📈 Model Insights** – Full regression output, VIF, ANOVA, residual diagnostics
4. **📚 About** – Project methodology, team info, dataset overview

**Prerequisite:** Run preprocessing first to generate `outputs/tmdb_cleaned.csv`.

---

## 📊 Key Variables
| Variable | Type | Description |
|---|---|---|
| `revenue` | Continuous | Global box office gross (USD) — dependent variable |
| `budget` | Continuous | Production budget (USD) |
| `cast_popularity_score` | Continuous | Mean TMDB popularity of top-3 billed cast |
| `vote_average` | Continuous | Audience rating 0–10 |
| `vote_count` | Discrete | Number of TMDB votes |
| `genres` | Categorical | Genre labels (exploded for analysis) |
| `roi` | Derived | (revenue − budget) / budget |
| `log_revenue` | Derived | log1p(revenue) |
| `log_budget` | Derived | log1p(budget) |
| `log_cast_pop` | Derived | log1p(cast_popularity_score) |

---

## 📅 Weekly Progress
- **Week 1** ✅ Group formation & topic finalization
- **Week 2** ✅ Literature review (4 peer-reviewed sources)
- **Week 3** ✅ EDA — cleaning, visualisations, correlations
- **Week 4** ⏳ Model selection & hypothesis development
- **Week 5** ⏳ Statistical analysis & validation
- **Week 6** ⏳ Statistical modelling continued
- **Week 7** ⏳ Application development (Greenlight Tool)
- **Week 8** ⏳ Final presentation & peer evaluation
