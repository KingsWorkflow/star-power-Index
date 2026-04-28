"""
main.py  —  Star Power Index · Greenlight Tool
Team Ghanti Tininini | King's College Nepal, Westcliff University
DATA 200 Applied Statistical Analysis | Professor: Regmi
"""
import sys, os, warnings
warnings.filterwarnings("ignore")

try:
    import streamlit as st
except ImportError:
    print("pip install streamlit"); sys.exit(1)

def verify_environment():
    missing = [p for p in ["pandas","numpy","plotly"] if not __import__("importlib").util.find_spec(p)]
    optional = [p for p in ["statsmodels","scipy"] if not __import__("importlib").util.find_spec(p)]
    if missing:
        st.error("❌ Missing: " + ", ".join(missing)); st.code("pip install -r requirements.txt"); st.stop()
    if optional:
        st.warning("⚠️ Limited functionality — missing: " + ", ".join(optional))
verify_environment()

import pandas as pd, plotly.express as px, plotly.graph_objects as go, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="Star Power Index", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")

# ── Statsmodels compat layer ──────────────────────────────────
def _import_statsmodels():
    try:
        from packaging.version import Version
        import statsmodels as _sm
        if Version(_sm.__version__) < Version("0.14.3"):
            raise RuntimeError(f"statsmodels {_sm.__version__} < 0.14.3 — run: pip install 'statsmodels>=0.14.3'")
        from statsmodels.formula.api import ols
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        from statsmodels.stats.diagnostic import het_breuschpagan as _hbp
        import statsmodels.api as sm; import scipy.stats as sp
        def _bp(resid, exog):
            try: r = _hbp(resid, exog); return float(r[0]), float(r[1])
            except Exception:
                resid2 = (resid**2) / (resid**2).mean()
                aux = sm.OLS(resid2, exog).fit(); lm = len(resid)*aux.rsquared
                from scipy.stats import chi2; return float(lm), float(chi2.sf(lm, exog.shape[1]-1))
        return ols, variance_inflation_factor, pairwise_tukeyhsd, _bp, sm, sp, True, ""
    except Exception as e:
        return None,None,None,None,None,None,False,str(e)

# ── Plotly theme ──────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit,system-ui", color="#94a3b8", size=11),
    title_font=dict(family="Playfair Display,Georgia,serif", color="#e8eaf0", size=14),
    legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="rgba(255,255,255,0.07)", borderwidth=1, font=dict(size=11,color="#94a3b8")),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748b"), title_font=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748b"), title_font=dict(color="#94a3b8")),
    margin=dict(t=50,b=40,l=40,r=20),
)
def sc(fig, h=420): fig.update_layout(height=h,**PL); st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
def fm(v): return f"${v/1e9:.2f}B" if v>=1e9 else f"${v/1e6:.0f}M" if v>=1e6 else f"${v:,.0f}"
def fp(v): return f"{v:.1%}"
st.markdown("""
<style>

/* container alignment */
.nav-center {
    display: flex;
    gap: 0.5rem;
}

/* base button style */
div[data-testid="stButton"] > button {
    all: unset;
    cursor: pointer;
    padding: 0.55rem 1rem;
    border-radius: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    color: #94a3b8;
    background: transparent;
    border: 1px solid rgba(148,163,184,0.15);
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    justify-content: center;
}

/* hover */
div[data-testid="stButton"] > button:hover {
    border: 1px solid rgba(212,175,55,0.4);
    color: #d4af37;
    background: rgba(212,175,55,0.08);
}

/* ACTIVE button (gold pill) */
button.nav-active {
    color: #d4af37 !important;
    border: 1px solid rgba(212,175,55,0.6) !important;
    background: linear-gradient(
        90deg,
        rgba(212,175,55,0.15),
        rgba(212,175,55,0.05)
    ) !important;
    box-shadow: 0 0 12px rgba(212,175,55,0.15);
}

/* optional dot */
.nav-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)
# ── CSS ───────────────────────────────────────────────────────
def load_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --obsidian:#080a0f; --void:#0d1117; --surface:#111827; --surface2:#1a2235; --surface3:#1f2d45;
    --border:rgba(255,255,255,0.07); --border-glow:rgba(212,175,55,0.3);
    --gold:#d4af37; --gold-bright:#f5d060; --gold-dim:#8b7328; --gold-glow:rgba(212,175,55,0.15);
    --crimson:#c0392b; --emerald:#10b981; --sky:#38bdf8; --muted:#64748b; --text:#e8eaf0; --text-dim:#94a3b8;
    --fd:'Playfair Display',Georgia,serif; --fm:'DM Mono','Courier New',monospace; --fb:'Outfit',system-ui,sans-serif;
}

html,body,[data-testid="stAppViewContainer"]{background:var(--obsidian)!important;color:var(--text)!important;}
.stApp{background:var(--obsidian)!important;}
[data-testid="stAppViewContainer"]>.main{padding:0;}
.block-container{padding:0 2.5rem 4rem!important;max-width:1440px;}

.stApp::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");opacity:.6;}

/* ── CINEMATIC TOP NAV ───────────────────────────────────── */
.spi-nav {
    position: sticky; top: 0; z-index: 9999;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2rem;
    height: 64px;
    background: rgba(8,10,15,0.92);
    backdrop-filter: blur(20px) saturate(1.8);
    -webkit-backdrop-filter: blur(20px) saturate(1.8);
    border-bottom: 1px solid rgba(212,175,55,0.12);
    box-shadow: 0 1px 0 rgba(212,175,55,0.06), 0 8px 32px rgba(0,0,0,0.5);
}

/* Animated gold underline */
.spi-nav::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent 0%,rgba(212,175,55,0.6) 30%,rgba(245,208,96,0.9) 50%,rgba(212,175,55,0.6) 70%,transparent 100%);
    background-size:200% 100%;
    animation: nav-shimmer 4s linear infinite;
}
@keyframes nav-shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* Brand */
.spi-brand {
    display:flex; align-items:center; gap:0.75rem;
    text-decoration:none; flex-shrink:0;
}
.spi-brand-icon {
    width:36px; height:36px;
    background:linear-gradient(135deg,#d4af37,#8b7328);
    border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-family:var(--fd); font-size:1.1rem; color:#080a0f; font-weight:900;
    box-shadow:0 0 16px rgba(212,175,55,0.4);
    flex-shrink:0;
}
.spi-brand-text { display:flex; flex-direction:column; line-height:1.1; }
.spi-brand-name { font-family:var(--fd); font-size:0.95rem; font-weight:700; color:var(--text); letter-spacing:-0.01em; }
.spi-brand-sub  { font-family:var(--fm); font-size:0.5rem; color:var(--gold); letter-spacing:0.22em; text-transform:uppercase; }

/* Nav items */
.spi-nav-links { display:flex; align-items:center; gap:0.25rem; }

.spi-nav-item {
    position:relative;
    display:flex; align-items:center; gap:0.45rem;
    padding:0.45rem 1rem;
    border-radius:8px;
    font-family:var(--fm); font-size:0.68rem; font-weight:400;
    letter-spacing:0.1em; text-transform:uppercase;
    color:var(--muted);
    pointer-events: auto !important; /* Ensures clicks are registered */
    user-select: none;               /* Prevents text selection on double-click */
    cursor: pointer !important;
    border:1px solid transparent;
    transition:all 0.2s cubic-bezier(0.4,0,0.2,1);
    white-space:nowrap;
    user-select:none;
}
.spi-nav-item .nav-icon { font-size:0.8rem; opacity:0.7; transition:all 0.2s ease; }
.spi-nav-item:hover {
    color:var(--text);
    background:rgba(255,255,255,0.04);
    border-color:rgba(255,255,255,0.08);
}
.spi-nav-item:hover .nav-icon { opacity:1; transform:scale(1.1); }

.spi-nav-item.active {
    color:var(--gold-bright);
    background:linear-gradient(135deg,rgba(212,175,55,0.12),rgba(212,175,55,0.05));
    border-color:rgba(212,175,55,0.25);
    font-weight:500;
}
.spi-nav-item.active::after {
    content:'';
    position:absolute; bottom:-1px; left:20%; right:20%; height:2px;
    background:linear-gradient(90deg,transparent,var(--gold),transparent);
    border-radius:2px;
}
.spi-nav-item.active .nav-icon { opacity:1; }

/* Active pill dot */
.spi-nav-item.active .nav-dot {
    display:inline-block; width:5px; height:5px;
    background:var(--gold); border-radius:50%;
    box-shadow:0 0 6px var(--gold); flex-shrink:0;
}
.spi-nav-item:not(.active) .nav-dot { display:none; }

/* Nav right — live stats */
.spi-nav-stats {
    display:flex; align-items:center; gap:1.25rem; flex-shrink:0;
}
.spi-stat {
    display:flex; flex-direction:column; align-items:flex-end; line-height:1.2;
}
.spi-stat-val { font-family:var(--fm); font-size:0.75rem; color:var(--gold); font-weight:500; }
.spi-stat-lbl { font-family:var(--fm); font-size:0.5rem; color:var(--muted); letter-spacing:0.15em; text-transform:uppercase; }
.spi-stat-sep { width:1px; height:28px; background:rgba(255,255,255,0.07); }

/* Version badge */
.spi-version {
    font-family:var(--fm); font-size:0.55rem; color:var(--gold-dim);
    border:1px solid rgba(212,175,55,0.2); border-radius:4px;
    padding:0.15rem 0.4rem; letter-spacing:0.1em;
    background:rgba(212,175,55,0.04);
}

/* Hover tooltip on nav items */
.spi-nav-item .nav-tooltip {
    position:absolute; top:calc(100% + 10px); left:50%; transform:translateX(-50%);
    background:var(--surface); border:1px solid rgba(212,175,55,0.2); border-radius:6px;
    padding:0.35rem 0.65rem; white-space:nowrap;
    font-family:var(--fb); font-size:0.72rem; color:var(--text-dim); letter-spacing:0;
    opacity:0; pointer-events:none;
    transition:opacity 0.15s ease, transform 0.15s ease;
    transform:translateX(-50%) translateY(-4px);
    box-shadow:0 8px 24px rgba(0,0,0,0.4);
    z-index:1000;
    text-transform:none;
}
.spi-nav-item:hover .nav-tooltip {
    opacity:1; transform:translateX(-50%) translateY(0);
}
.spi-nav-item .nav-tooltip::before {
    content:''; position:absolute; top:-5px; left:50%; transform:translateX(-50%);
    border:5px solid transparent; border-bottom-color:rgba(212,175,55,0.2);
    border-top:none;
}

/* Page content offset */
.spi-page-content { padding-top: 2rem; }

/* ── Sidebar (collapsed by default, toggled via nav) */
[data-testid="stSidebar"] { display:none!important; }

/* ── Page header ── */
.page-header{padding:2.5rem 0 2rem;border-bottom:1px solid var(--border);margin-bottom:2.5rem;position:relative;}
.page-header::after{content:'';position:absolute;bottom:-1px;left:0;width:120px;height:1px;background:linear-gradient(90deg,var(--gold),transparent);}
.header-eyebrow{font-family:var(--fm);font-size:.65rem;letter-spacing:.25em;text-transform:uppercase;color:var(--gold);margin-bottom:.75rem;}
.header-title{font-family:var(--fd);font-size:clamp(2rem,4vw,3.5rem);font-weight:700;color:var(--text);line-height:1.1;letter-spacing:-.02em;margin-bottom:.5rem;}
.header-title span{color:var(--gold);}
.header-desc{font-family:var(--fb);font-size:.95rem;color:var(--text-dim);max-width:600px;line-height:1.7;}
.header-badge{display:inline-flex;align-items:center;gap:.4rem;padding:.25rem .75rem;background:var(--gold-glow);border:1px solid var(--gold-dim);border-radius:100px;font-family:var(--fm);font-size:.65rem;color:var(--gold);letter-spacing:.1em;text-transform:uppercase;margin-top:1rem;}

/* ── Cards ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.5rem;position:relative;overflow:hidden;transition:border-color .3s,box-shadow .3s;}
.card:hover{border-color:var(--border-glow);box-shadow:0 0 30px rgba(212,175,55,.06);}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(212,175,55,.4),transparent);}
.card-title{font-family:var(--fm);font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;}

/* ── Metrics ── */
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1.25rem 1.5rem!important;}
[data-testid="stMetricLabel"]{font-family:var(--fm)!important;font-size:.62rem!important;letter-spacing:.18em!important;text-transform:uppercase!important;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:var(--fd)!important;font-size:1.75rem!important;font-weight:700!important;color:var(--text)!important;}
[data-testid="stMetricDelta"]{font-family:var(--fm)!important;font-size:.75rem!important;}

/* ── Inputs ── */
.stSlider>div{padding:.25rem 0;}
.stSlider [data-testid="stTickBar"]{display:none;}
div[data-baseweb="slider"] div[role="slider"]{background:var(--gold)!important;border:2px solid var(--gold-bright)!important;box-shadow:0 0 12px var(--gold-glow)!important;width:18px!important;height:18px!important;}
div[data-baseweb="slider"]>div>div:last-child>div{background:var(--gold)!important;}
div[data-baseweb="slider"]>div>div:first-child{background:var(--surface3)!important;}
.stSlider label,.stSelectbox label{font-family:var(--fm)!important;font-size:.65rem!important;letter-spacing:.15em!important;text-transform:uppercase!important;color:var(--muted)!important;}
.stSelectbox>div>div{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;}

/* ── Buttons ── */
.stButton button[kind="primary"]{background:linear-gradient(135deg,var(--gold),#b8960f)!important;color:var(--obsidian)!important;border:none!important;border-radius:8px!important;font-family:var(--fm)!important;font-size:.75rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;font-weight:500!important;padding:.75rem 2rem!important;box-shadow:0 4px 20px rgba(212,175,55,.3)!important;transition:all .25s!important;}
.stButton button[kind="primary"]:hover{transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(212,175,55,.45)!important;}
.stButton button[kind="secondary"]{background:transparent!important;color:var(--text-dim)!important;border:1px solid var(--border)!important;border-radius:8px!important;font-family:var(--fm)!important;font-size:.7rem!important;letter-spacing:.1em!important;}

/* ── Tabs ── */
[data-testid="stTabs"] button{font-family:var(--fm)!important;font-size:.7rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--muted)!important;border:none!important;padding:.6rem 1.25rem!important;border-radius:0!important;border-bottom:2px solid transparent!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--gold)!important;border-bottom-color:var(--gold)!important;background:transparent!important;}
[data-testid="stTabs"]>div:first-child{border-bottom:1px solid var(--border)!important;gap:0!important;}

/* ── Alerts ── */
[data-testid="stAlert"]{border-radius:8px!important;border:1px solid!important;font-family:var(--fb)!important;}
.stSuccess{background:rgba(16,185,129,.08)!important;border-color:rgba(16,185,129,.3)!important;}
.stError{background:rgba(192,57,43,.08)!important;border-color:rgba(192,57,43,.3)!important;}
.stWarning{background:rgba(212,175,55,.08)!important;border-color:rgba(212,175,55,.3)!important;}
.stInfo{background:rgba(56,189,248,.08)!important;border-color:rgba(56,189,248,.3)!important;}

hr{border:none!important;border-top:1px solid var(--border)!important;margin:2rem 0!important;}
[data-testid="stDataFrame"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;overflow:hidden!important;}

/* ── Verdict ── */
.verdict-box{padding:1.5rem;border-radius:12px;text-align:center;margin:1rem 0;}
.verdict-blockbuster{background:linear-gradient(135deg,rgba(212,175,55,.12),rgba(245,208,96,.06));border:1px solid rgba(212,175,55,.4);}
.verdict-profitable{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.3);}
.verdict-breakeven{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.3);}
.verdict-risk{background:rgba(192,57,43,.08);border:1px solid rgba(192,57,43,.3);}
.verdict-grade{font-family:var(--fd);font-size:3rem;font-weight:900;line-height:1;margin-bottom:.25rem;}
.verdict-label{font-family:var(--fm);font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;}

.ci-bar-wrap{background:var(--surface2);border-radius:6px;height:8px;position:relative;margin:.75rem 0;overflow:hidden;}
.ci-bar-fill{position:absolute;height:100%;background:linear-gradient(90deg,var(--gold-dim),var(--gold));border-radius:6px;}

.section-title{font-family:var(--fd);font-size:1.4rem;font-weight:600;color:var(--text);letter-spacing:-.01em;margin:2rem 0 1.25rem;display:flex;align-items:center;gap:.6rem;}
.section-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);margin-left:.5rem;}

.input-group{background:#131c2e;border:1px solid rgba(212,175,55,.18);border-radius:14px;padding:1.5rem 1.5rem 1.75rem;position:relative;box-shadow:inset 0 1px 0 rgba(212,175,55,.08),0 4px 24px rgba(0,0,0,.4);}
.input-group-title{font-family:var(--fm);font-size:.62rem;letter-spacing:.22em;text-transform:uppercase;color:var(--gold);margin-bottom:1.25rem;margin-top:.25rem;display:flex;align-items:center;gap:.6rem;opacity:.9;}
.input-group-title::before{content:'';width:18px;height:1px;background:linear-gradient(90deg,var(--gold),transparent);flex-shrink:0;}

.film-row{display:flex;justify-content:space-between;align-items:center;padding:.75rem 0;border-bottom:1px solid var(--border);}
.film-row:last-child{border-bottom:none;}
.film-title{font-family:var(--fb);font-size:.875rem;color:var(--text);}
.film-revenue{font-family:var(--fm);font-size:.8rem;color:var(--gold);}
.film-pop{font-family:var(--fm);font-size:.7rem;color:var(--text-dim);background:var(--surface2);padding:.15rem .5rem;border-radius:4px;}

[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;}
[data-testid="stExpander"] summary{font-family:var(--fm)!important;font-size:.75rem!important;color:var(--text-dim)!important;}

::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--void);}
::-webkit-scrollbar-thumb{background:var(--surface3);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--gold-dim);}

/* Streamlit chrome */
#MainMenu,footer{display:none!important;}
[data-testid="stToolbar"]{display:none!important;}
[data-testid="stHeader"]{background:transparent!important;border-bottom:none!important;height:0!important;min-height:0!important;}
[data-testid="stSidebarCollapsedControl"]{display:none!important;}

/* Diag cards */
.diag-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.25rem 1.5rem;}
.diag-card::before{content:'';display:block;width:100%;height:1px;background:linear-gradient(90deg,transparent,rgba(212,175,55,.3),transparent);margin-bottom:1rem;}
.diag-card-label{font-family:var(--fm);font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;}
.diag-card-value{font-family:var(--fm);font-size:1.1rem;color:var(--text);margin-bottom:.25rem;}
.diag-card-sub{font-family:var(--fm);font-size:.72rem;color:var(--muted);margin-bottom:.75rem;}
.diag-card-verdict{font-size:.82rem;}
.verdict-ok{color:#10b981;} .verdict-warn{color:#94a3b8;}

.version-banner{background:rgba(212,175,55,.06);border:1px solid rgba(212,175,55,.25);border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:2rem;}
.version-banner-title{font-family:var(--fm);font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--gold);margin-bottom:.5rem;}
.version-banner-body{font-family:var(--fb);font-size:.85rem;color:var(--text-dim);line-height:1.7;}
</style>""", unsafe_allow_html=True)


def load_clean_data():
    path = os.path.join(os.path.dirname(__file__), "outputs", "tmdb_cleaned.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "primary_genre" in df.columns:
            df["primary_genre"] = pd.Categorical(df["primary_genre"])
        return df
    return None


class MoviePredictor:
    def __init__(self, df):
        self.df = df; self.model_full = self.model_simple = None
        ols, *_, available, _ = _import_statsmodels()
        if available:
            dm = df.copy()
            if not hasattr(dm["primary_genre"].dtype, "categories"):
                dm["primary_genre"] = pd.Categorical(dm["primary_genre"])
            self.model_full   = ols("log_revenue ~ log_budget + log_cast_pop + vote_average + C(primary_genre)", data=dm).fit()
            self.model_simple = ols("log_revenue ~ log_budget + log_cast_pop + vote_average", data=dm).fit()

    def predict(self, budget, cast_pop, vote_avg, genre):
        if self.model_full is None:
            rev = self.df["revenue"].mean() * (budget/self.df["budget"].mean())**0.7
            rev *= max(0.5, 1+(cast_pop-30)*0.004)
            return dict(revenue=rev, ci_lower=rev*.55, ci_upper=rev*1.45, roi=(rev-budget)/budget, model="fallback")
        dm = self.df.copy()
        if not hasattr(dm["primary_genre"].dtype, "categories"):
            dm["primary_genre"] = pd.Categorical(dm["primary_genre"])
        inp = pd.DataFrame({"log_budget":[np.log1p(budget)],"log_cast_pop":[np.log1p(cast_pop)],"vote_average":[vote_avg],
                            "primary_genre":pd.Categorical([genre], categories=dm["primary_genre"].cat.categories)})
        try:
            pl = self.model_full.predict(inp)[0]; rev = np.expm1(pl)
            ci = self.model_full.get_prediction(inp).conf_int(alpha=0.05)
            return dict(revenue=rev, ci_lower=np.expm1(ci[0][0]), ci_upper=np.expm1(ci[0][1]), roi=(rev-budget)/budget, model="full")
        except Exception:
            pl = self.model_simple.predict(inp[["log_budget","log_cast_pop","vote_average"]])[0]; rev = np.expm1(pl)
            return dict(revenue=rev, ci_lower=rev*.7, ci_upper=rev*1.3, roi=(rev-budget)/budget, model="simple")


# ── TOP NAVIGATION ────────────────────────────────────────────
PAGES = [
    ("✦", "Greenlight Predictor", "Revenue forecasting via OLS regression"),
    ("◈", "Data Observatory",     "Distributions, correlations & genre dynamics"),
    ("⊞", "Model Laboratory",     "OLS · ANOVA · VIF · Residual diagnostics"),
    ("◉", "Project Brief",        "Team, hypotheses & methodology"),
]
def render_top_nav(df):
    # ── 1. Define pages ──
    page_names = [name for _, name, _ in PAGES]

    # ── 2. Session state ──
    if "page" not in st.session_state:
        st.session_state.page = page_names[0]

    # ── 3. Stats ──
    stats_html = ""
    if df is not None:
        stats = [
            (f"{len(df):,}", "Films"),
            (fm(df["revenue"].mean()), "Avg Rev"),
            (fp(df["roi"].mean()), "Avg ROI"),
            (f"{df['cast_popularity_score'].mean():.1f}", "Avg SPI"),
        ]
        stats_html = "".join([
            f'<div class="spi-stat"><span class="spi-stat-val">{v}</span><span class="spi-stat-lbl">{l}</span></div>'
            f'<div class="spi-stat-sep"></div>' for v, l in stats
        ]).rstrip('<div class="spi-stat-sep"></div>')

    # ── 4. Layout ──
    left, center, right = st.columns([1.5, 4, 2])

    # ── Brand ──
    with left:
        st.markdown("""
        <div class="spi-brand">
            <div class="spi-brand-icon">✦</div>
            <div class="spi-brand-text">
                <span class="spi-brand-name">Star Power Index</span><br>
                <span class="spi-brand-sub">DATA 200 · Greenlight Tool</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Navigation (styled buttons) ──
    with center:
        nav_cols = st.columns(len(PAGES))

        for i, (icon, name, tooltip) in enumerate(PAGES):
            is_active = st.session_state.page == name
            label = f"{icon} {name}"

            with nav_cols[i]:
                clicked = st.button(
                    label,
                    key=f"nav_{name}",
                    use_container_width=True
                )

                # ✅ Apply active styling via JS (safe use: styling only)
                if is_active:
                    st.markdown(f"""
                    <script>
                    const btns = window.parent.document.querySelectorAll('button');
                    btns.forEach(btn => {{
                        if (btn.innerText.trim() === "{label}") {{
                            btn.classList.add("nav-active");
                        }}
                    }});
                    </script>
                    """, unsafe_allow_html=True)

                # ✅ Routing logic (REAL FIX)
                if clicked:
                    st.session_state.page = name
                    st.rerun()

    # ── Stats ──
    with right:
        st.markdown(f"""
        <div class="spi-nav-stats">
            {stats_html}
            <span class="spi-version">v1.0</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Page wrapper ──
    st.markdown('<div class="spi-page-content">', unsafe_allow_html=True)

    return st.session_state.page
# ── PAGES ─────────────────────────────────────────────────────
def page_predictor(df):
    st.markdown("""
    <div class="page-header">
        <div class="header-eyebrow">✦ Greenlight Tool</div>
        <div class="header-title">Film Revenue <span>Predictor</span></div>
        <div class="header-desc">Input your film's parameters to receive a data-driven revenue forecast powered by OLS regression trained on 2,641 films.</div>
        <div class="header-badge">◈ OLS Model · R² ≈ 0.65 · Genre-stratified</div>
    </div>""", unsafe_allow_html=True)
    predictor = MoviePredictor(df)
    col_input, col_result = st.columns([1.1, 0.9], gap="large")
    with col_input:
        holes = "".join(['<span style="display:inline-block;width:13px;height:13px;border:1.5px solid rgba(212,175,55,0.45);border-radius:3px;background:rgba(212,175,55,0.05);"></span>' for _ in range(14)])
        st.markdown(f'<div style="display:flex;gap:5px;margin-bottom:1rem;align-items:center;">{holes}<span style="font-family:\'DM Mono\',monospace;font-size:0.52rem;letter-spacing:0.22em;color:rgba(212,175,55,0.35);margin-left:0.4rem;">35MM</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="input-group"><div class="input-group-title">Financial Parameters</div></div>', unsafe_allow_html=True)
        budget = st.slider("Production Budget (USD)", 1_000_000, 400_000_000, 50_000_000, 5_000_000)
        st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:.78rem;color:#d4af37;margin-top:-.25rem;margin-bottom:.75rem;padding:.35rem .75rem;background:rgba(212,175,55,.07);border-radius:6px;display:inline-block;">✦ {fm(budget)}</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="input-group"><div class="input-group-title">Cast & Audience Signals</div></div>', unsafe_allow_html=True)
        cast_pop = st.slider("Top-3 Cast Avg TMDB Popularity Score", 0.0, 200.0, 30.0, 0.5)
        if cast_pop>=80:   pt,pc="SUPERSTAR TIER","#d4af37"
        elif cast_pop>=40: pt,pc="MAJOR STAR TIER","#10b981"
        elif cast_pop>=15: pt,pc="ESTABLISHED TIER","#38bdf8"
        else:              pt,pc="EMERGING TIER","#94a3b8"
        st.markdown(f'<span style="font-family:\'DM Mono\',monospace;font-size:.62rem;letter-spacing:.15em;color:{pc};border:1px solid {pc}44;padding:.2rem .6rem;border-radius:4px;background:{pc}11;">{pt}</span>', unsafe_allow_html=True)
        st.markdown('<div style="height:.5rem;"></div>', unsafe_allow_html=True)
        vote_avg = st.slider("Expected TMDB Audience Rating (0–10)", 0.0, 10.0, 6.5, 0.1)
        genres = sorted(df["primary_genre"].unique())
        genre = st.selectbox("Primary Genre", genres, index=genres.index("Action") if "Action" in genres else 0)
        btn = st.button("✦  Run Greenlight Analysis", type="primary", use_container_width=True)
    with col_result:
        if btn:
            with st.spinner("Running OLS model..."):
                res = predictor.predict(budget, cast_pop, vote_avg, genre)
            roi = res["roi"]
            if roi>2.0:   vc,gr,vl,vcol="verdict-blockbuster","A","BLOCKBUSTER","#d4af37"
            elif roi>1.0: vc,gr,vl,vcol="verdict-profitable","B","PROFITABLE","#10b981"
            elif roi>0:   vc,gr,vl,vcol="verdict-breakeven","C","BREAK-EVEN","#38bdf8"
            else:         vc,gr,vl,vcol="verdict-risk","D","HIGH RISK","#c0392b"
            st.markdown(f'<div class="verdict-box {vc}"><div class="verdict-grade" style="color:{vcol};">{gr}</div><div class="verdict-label" style="color:{vcol};">{vl}</div></div>', unsafe_allow_html=True)
            m1c,m2c = st.columns(2)
            with m1c: st.metric("Predicted Revenue", fm(res["revenue"]), delta=f"{fp(roi)} ROI")
            with m2c:
                cir = res["ci_upper"]-res["ci_lower"]
                st.metric("CI Range (95%)", fm(cir), delta=f"±{fm(cir/2)}")
            clip = max(0,(res["ci_lower"]/max(res["ci_upper"],1))*100)
            st.markdown(f"""<div style="margin:1rem 0;">
                <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:#64748b;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem;">95% Prediction Interval</div>
                <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:.72rem;color:#94a3b8;margin-bottom:.3rem;">
                    <span>{fm(res['ci_lower'])}</span><span style="color:#d4af37;">{fm(res['revenue'])}</span><span>{fm(res['ci_upper'])}</span>
                </div>
                <div class="ci-bar-wrap"><div class="ci-bar-fill" style="left:{clip:.0f}%;width:{100-clip:.0f}%;"></div></div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:1rem;">Comparable Films</div>', unsafe_allow_html=True)
            for _, row in df[df["primary_genre"]==genre].nlargest(6,"revenue")[["title","revenue","cast_popularity_score"]].iterrows():
                t=row["title"]; t=t[:32]+"…" if len(t)>32 else t
                st.markdown(f'<div class="film-row"><span class="film-title">{t}</span><span class="film-pop">SPI {row["cast_popularity_score"]:.0f}</span><span class="film-revenue">{fm(row["revenue"])}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:var(--surface);border:1px dashed rgba(212,175,55,.2);border-radius:14px;padding:3rem 2rem;text-align:center;margin-top:2rem;">
                <div style="font-size:2.5rem;margin-bottom:1rem;opacity:.4;">✦</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#94a3b8;margin-bottom:.5rem;">Awaiting Parameters</div>
                <div style="font-family:'DM Mono',monospace;font-size:.7rem;color:#64748b;letter-spacing:.1em;text-transform:uppercase;">Configure your film on the left<br>then run the analysis</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Revenue Landscape by Genre</div>', unsafe_allow_html=True)
    gr_rev = df.groupby("primary_genre").agg(R=("revenue","mean")).reset_index().sort_values("R",ascending=True)
    fig = go.Figure(go.Bar(x=gr_rev["R"],y=gr_rev["primary_genre"],orientation="h",
        marker=dict(color=["#d4af37" if g==genre else "#1f2d45" for g in gr_rev["primary_genre"]],
                    line=dict(color=["#d4af37" if g==genre else "#2a3d5a" for g in gr_rev["primary_genre"]],width=1)),
        hovertemplate="<b>%{y}</b><br>Avg Revenue: %{x:$,.0f}<extra></extra>"))
    fig.update_layout(showlegend=False,title="Average Revenue by Genre",xaxis_title="Average Revenue (USD)",**PL,height=380)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})


def page_exploration(df):
    st.markdown("""<div class="page-header"><div class="header-eyebrow">◈ Data Observatory</div>
        <div class="header-title">Dataset <span>Explorer</span></div>
        <div class="header-desc">Visual analysis of 2,641 TMDB films — distributions, correlations, and genre dynamics.</div>
    </div>""", unsafe_allow_html=True)
    k1,k2,k3,k4,k5=st.columns(5)
    with k1: st.metric("Films Analysed",f"{len(df):,}")
    with k2: st.metric("Avg Budget",fm(df["budget"].mean()))
    with k3: st.metric("Avg Revenue",fm(df["revenue"].mean()))
    with k4: st.metric("Avg ROI",fp(df["roi"].mean()))
    with k5: st.metric("Genres",str(df["primary_genre"].nunique()))
    st.markdown("<br>",unsafe_allow_html=True)
    tab1,tab2,tab3,tab4=st.tabs(["  DISTRIBUTIONS  ","  FINANCIALS  ","  STAR POWER  ","  GENRE BREAKDOWN  "])
    with tab1:
        c1,c2=st.columns(2)
        with c1: fig=px.histogram(df,x="budget",nbins=50,title="Budget Distribution",color_discrete_sequence=["#3b82f6"]); fig.update_traces(marker_line_color="rgba(0,0,0,.3)",marker_line_width=.5); sc(fig)
        with c2: fig=px.histogram(df,x="revenue",nbins=50,title="Revenue Distribution",color_discrete_sequence=["#d4af37"]); fig.update_traces(marker_line_color="rgba(0,0,0,.3)",marker_line_width=.5); sc(fig)
        c3,c4=st.columns(2)
        with c3: fig=px.histogram(df,x="cast_popularity_score",nbins=50,title="Cast Popularity (SPI)",color_discrete_sequence=["#10b981"]); fig.update_traces(marker_line_color="rgba(0,0,0,.3)",marker_line_width=.5); sc(fig)
        with c4: fig=px.histogram(df,x="vote_average",nbins=40,title="Audience Rating",color_discrete_sequence=["#c0392b"]); fig.update_traces(marker_line_color="rgba(0,0,0,.3)",marker_line_width=.5); sc(fig)
        st.markdown('<div class="section-title">Log-Transformed (used in OLS models)</div>',unsafe_allow_html=True)
        c5,c6,c7=st.columns(3)
        for cn,cl,co,cx in [("log_revenue","log(Revenue)","#d4af37",c5),("log_budget","log(Budget)","#3b82f6",c6),("log_cast_pop","log(Cast Popularity)","#10b981",c7)]:
            with cx: fig=px.histogram(df,x=cn,nbins=40,title=cl,color_discrete_sequence=[co]); fig.update_traces(marker_line_color="rgba(0,0,0,.3)",marker_line_width=.5); sc(fig,300)
    with tab2:
        c1,c2=st.columns(2)
        with c1: fig=px.scatter(df,x="budget",y="revenue",color="primary_genre",size="cast_popularity_score",size_max=18,hover_data=["title","vote_average"],title="Budget vs Revenue (bubble=SPI)"); sc(fig,520)
        with c2:
            sub=df.dropna(subset=["log_budget","log_revenue"]); fig=px.scatter(sub,x="log_budget",y="log_revenue",color="primary_genre",opacity=.7,title="Log–Log: Budget vs Revenue")
            z=np.polyfit(sub["log_budget"],sub["log_revenue"],1); xr=np.linspace(sub["log_budget"].min(),sub["log_budget"].max(),100)
            fig.add_scatter(x=xr,y=np.poly1d(z)(xr),mode="lines",name="Trend (r=0.72)",line=dict(color="#d4af37",width=2,dash="dash")); sc(fig,520)
        st.markdown('<div class="section-title">Correlation Matrix</div>',unsafe_allow_html=True)
        corr=df[["budget","revenue","cast_popularity_score","vote_average","roi"]].corr()
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale=["#c0392b","#111827","#10b981"],title="Pearson Correlation Coefficients",zmin=-1,zmax=1); sc(fig,400)
    with tab3:
        c1,c2=st.columns(2)
        with c1:
            sub2=df.dropna(subset=["cast_popularity_score","revenue"]); fig=px.scatter(sub2,x="cast_popularity_score",y="revenue",color="primary_genre",size="budget",size_max=16,hover_data=["title"],title="⭐ Star Power vs Revenue")
            z2=np.polyfit(sub2["cast_popularity_score"],sub2["revenue"],1); xr2=np.linspace(sub2["cast_popularity_score"].min(),sub2["cast_popularity_score"].max(),100)
            fig.add_scatter(x=xr2,y=np.poly1d(z2)(xr2),mode="lines",name="Trend (r=0.48)",line=dict(color="#d4af37",width=2.5,dash="dot")); sc(fig,500)
        with c2:
            gp=df.groupby("primary_genre")["cast_popularity_score"].mean().sort_values()
            fig=go.Figure(go.Bar(x=gp.values,y=gp.index,orientation="h",marker=dict(color=gp.values,colorscale=[[0,"#1a2235"],[.5,"#8b7328"],[1,"#d4af37"]]),hovertemplate="<b>%{y}</b><br>Avg SPI: %{x:.1f}<extra></extra>"))
            fig.update_layout(title="Avg Cast Popularity by Genre",showlegend=False,**PL,height=500); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.markdown('<div class="section-title">Star Power vs ROI</div>',unsafe_allow_html=True)
        fig=px.scatter(df.dropna(subset=["cast_popularity_score","roi"]).query("roi<30"),x="cast_popularity_score",y="roi",color="primary_genre",opacity=.6,hover_data=["title"],title="SPI vs ROI (outliers clipped)")
        fig.add_hline(y=0,line_dash="dash",line_color="rgba(192,57,43,.5)",annotation_text="Break-even")
        fig.add_hline(y=1,line_dash="dot",line_color="rgba(16,185,129,.4)",annotation_text="100% ROI"); sc(fig,420)
    with tab4:
        c1,c2=st.columns(2)
        with c1:
            gs=df.groupby("primary_genre")[["budget","revenue"]].mean().reset_index().sort_values("revenue")
            fig=go.Figure(); fig.add_trace(go.Bar(x=gs["budget"],y=gs["primary_genre"],name="Avg Budget",orientation="h",marker_color="#3b82f6",opacity=.85))
            fig.add_trace(go.Bar(x=gs["revenue"],y=gs["primary_genre"],name="Avg Revenue",orientation="h",marker_color="#d4af37",opacity=.85))
            fig.update_layout(barmode="overlay",title="Budget vs Revenue by Genre",**PL,height=420); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with c2:
            rg=df.groupby("primary_genre")["roi"].mean().sort_values()
            fig=go.Figure(go.Bar(x=rg.values,y=rg.index,orientation="h",marker=dict(color=["#c0392b" if v<0 else "#d4af37" if v>2 else "#10b981" for v in rg.values]),hovertemplate="<b>%{y}</b><br>Avg ROI: %{x:.1%}<extra></extra>"))
            fig.update_layout(showlegend=False,title="Average ROI by Genre",xaxis_tickformat=".0%",**PL,height=420); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.markdown('<div class="section-title">Genre Breakdown Table</div>',unsafe_allow_html=True)
        gt=df.groupby("primary_genre").agg(Films=("revenue","count"),Avg_Budget=("budget","mean"),Avg_Revenue=("revenue","mean"),Avg_ROI=("roi","mean"),Avg_SPI=("cast_popularity_score","mean")).reset_index().sort_values("Avg_Revenue",ascending=False)
        gt["Avg_Budget"]=gt["Avg_Budget"].apply(fm); gt["Avg_Revenue"]=gt["Avg_Revenue"].apply(fm); gt["Avg_ROI"]=gt["Avg_ROI"].apply(fp); gt["Avg_SPI"]=gt["Avg_SPI"].round(1)
        gt.columns=["Genre","Films","Avg Budget","Avg Revenue","Avg ROI","Avg SPI"]
        st.dataframe(gt,hide_index=True,use_container_width=True)


def page_model_insights(df):
    st.markdown("""<div class="page-header"><div class="header-eyebrow">⊞ Model Laboratory</div>
        <div class="header-title">Statistical <span>Analysis</span></div>
        <div class="header-desc">OLS regression, ANOVA, VIF diagnostics, and residual analysis validating the Star Power Index hypothesis.</div>
    </div>""", unsafe_allow_html=True)
    ols,vif_fn,tukeyhsd,_bp,sm,sp,ok,err=_import_statsmodels()
    if not ok:
        import statsmodels as _sm
        st.markdown(f"""<div class="version-banner"><div class="version-banner-title">⚠ Statsmodels Version Incompatibility</div>
        <div class="version-banner-body">Installed: <code>statsmodels {_sm.__version__}</code> — requires ≥ 0.14.3.<br>
        Fix: <code>pip install "statsmodels>=0.14.3" --upgrade</code></div></div>""",unsafe_allow_html=True)
        corr=df[["budget","revenue","cast_popularity_score","vote_average","roi"]].corr()
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale=["#c0392b","#111827","#10b981"],zmin=-1,zmax=1,title="Pearson Correlation — Core Variables"); sc(fig,380)
        from scipy.stats import f_oneway
        groups=[g["log_revenue"].dropna().values for _,g in df.groupby("primary_genre") if len(g)>=10]
        F,p=f_oneway(*groups); a1,a2,a3=st.columns(3)
        a1.metric("ANOVA F-statistic",f"{F:.3f}"); a2.metric("p-value",f"{p:.2e}"); a3.metric("Result","Reject H₀ ✓" if p<0.05 else "Fail to Reject H₀")
        if p<0.05: st.success("✦ Revenue differs significantly across genres (α=0.05).")
        st.info("Full OLS, VIF & residual diagnostics require statsmodels ≥ 0.14.3."); return
    with st.spinner("Fitting models..."):
        m1=ols("log_revenue ~ log_budget",data=df).fit()
        m2=ols("log_revenue ~ log_budget + log_cast_pop + vote_average + C(primary_genre)",data=df).fit()
    st.markdown('<div class="section-title">Model Comparison</div>',unsafe_allow_html=True)
    mc1,mc2,mc3=st.columns(3)
    with mc1: st.metric("Model 1 R² (Budget Only)",f"{m1.rsquared:.3f}")
    with mc2: delta=m2.rsquared-m1.rsquared; st.metric("Model 2 R² (Full)",f"{m2.rsquared:.3f}",delta=f"+{delta:.3f} ΔR²")
    with mc3: st.metric("AIC Improvement",f"{m1.aic-m2.aic:,.0f}",delta="lower is better")
    st.success(f"✦ Adding cast popularity & ratings explains an additional **{delta:.1%}** of revenue variance (ΔR²={delta:.4f})")
    st.markdown('<div class="section-title">OLS Coefficients — Full Model</div>',unsafe_allow_html=True)
    cdf=pd.DataFrame({"Coefficient":m2.params,"Std Error":m2.bse,"t-statistic":m2.tvalues,"p-value":m2.pvalues,"Sig":m2.pvalues<0.05}).round(4)
    key=cdf.loc[["Intercept","log_budget","log_cast_pop","vote_average"]].copy()
    fig=go.Figure(go.Bar(x=key.index,y=key["Coefficient"],error_y=dict(type="data",array=key["Std Error"],visible=True,color="#64748b"),
        marker=dict(color=["#d4af37" if abs(v)==key["Coefficient"].abs().max() else "#3b82f6" for v in key["Coefficient"]]),
        hovertemplate="<b>%{x}</b><br>β=%{y:.4f}<extra></extra>"))
    fig.update_layout(title="Key Coefficient Estimates (±1 SE)",**PL,height=320); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with st.expander("Full coefficient table"): st.dataframe(cdf,use_container_width=True)
    st.markdown('<div class="section-title">Multicollinearity (VIF)</div>',unsafe_allow_html=True)
    X=sm.add_constant(df[["log_budget","log_cast_pop","vote_average"]].dropna())
    vdf=pd.DataFrame({"Feature":X.columns,"VIF":[vif_fn(X.values,i) for i in range(X.shape[1])]})
    vc1,vc2=st.columns([1,2])
    with vc1: st.dataframe(vdf.round(3),hide_index=True)
    with vc2:
        mv=vdf.iloc[1:]["VIF"].max()
        if mv<5: st.success(f"✓ All VIF<5 (max={mv:.2f}) — no problematic multicollinearity.")
        else: st.warning(f"⚠ VIF>5 detected (max={mv:.2f}). Consider regularisation.")
    st.markdown('<div class="section-title">One-Way ANOVA — Genre Revenue</div>',unsafe_allow_html=True)
    groups=[g["log_revenue"].dropna().values for _,g in df.groupby("primary_genre") if len(g)>=10]
    F,p=sp.f_oneway(*groups); av1,av2,av3=st.columns(3)
    av1.metric("F-statistic",f"{F:.3f}"); av2.metric("p-value",f"{p:.2e}"); av3.metric("Result","Reject H₀ ✓" if p<0.05 else "Fail to Reject H₀")
    if p<0.05:
        st.success("✦ Revenue differs significantly across genres (α=0.05).")
        tk=tukeyhsd(df["log_revenue"],df["primary_genre"],alpha=0.05)
        with st.expander("Tukey HSD Post-hoc"): st.text(str(tk.summary()))
    st.markdown('<div class="section-title">Residual Diagnostics</div>',unsafe_allow_html=True)
    rd1,rd2=st.columns(2)
    with rd1:
        w,pn=sp.shapiro(m2.resid[:5000]); ok2=pn>=0.05
        st.markdown(f"""<div class="diag-card"><div class="diag-card-label">Shapiro-Wilk Normality Test</div>
            <div class="diag-card-value">W = {w:.4f}</div><div class="diag-card-sub">p = {pn:.6f}</div>
            <div class="diag-card-verdict {'verdict-ok' if ok2 else 'verdict-warn'}">{'✓ Consistent with normality' if ok2 else '⚠ Non-normal — use robust SE'}</div></div>""",unsafe_allow_html=True)
    with rd2:
        bs,bp=_bp(m2.resid,m2.model.exog); ok3=bp>=0.05
        bsd=f"{bs:.4f}" if not np.isnan(bs) else "n/a"; bpd=f"{bp:.6f}" if not np.isnan(bp) else "n/a"
        st.markdown(f"""<div class="diag-card"><div class="diag-card-label">Breusch-Pagan Homoscedasticity</div>
            <div class="diag-card-value">BP = {bsd}</div><div class="diag-card-sub">p = {bpd}</div>
            <div class="diag-card-verdict {'verdict-ok' if ok3 else 'verdict-warn'}">{'✓ Homoscedastic residuals' if ok3 else '⚠ Heteroscedasticity — use HC3 SE'}</div></div>""",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    qq=sp.probplot(m2.resid,dist="norm",plot=None); fig=go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0],y=qq[0][1],mode="markers",name="Residuals",marker=dict(color="#d4af37",size=4,opacity=.6)))
    fig.add_trace(go.Scatter(x=qq[0][0],y=qq[1][0]+qq[1][1]*qq[0][0],mode="lines",name="Normal Reference",line=dict(color="#c0392b",width=2,dash="dash")))
    fig.update_layout(title="Q-Q Plot of OLS Residuals",xaxis_title="Theoretical Quantiles",yaxis_title="Sample Quantiles",**PL,height=400); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    fig2=go.Figure(go.Scatter(x=m2.fittedvalues,y=m2.resid,mode="markers",marker=dict(color="#3b82f6",size=4,opacity=.5),hovertemplate="Fitted:%{x:.2f}<br>Residual:%{y:.2f}<extra></extra>",name="Residuals"))
    fig2.add_hline(y=0,line_dash="dash",line_color="rgba(212,175,55,.5)")
    fig2.update_layout(title="Fitted Values vs Residuals",xaxis_title="Fitted Values",yaxis_title="Residuals",**PL,height=380); st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})


def page_about(df):
    st.markdown("""<div class="page-header"><div class="header-eyebrow">◉ Project Brief</div>
        <div class="header-title">About the <span>Star Power Index</span></div>
        <div class="header-desc">DATA 200 Applied Statistical Analysis — Team Ghanti Tininini</div>
    </div>""", unsafe_allow_html=True)
    c1,c2=st.columns([1.5,1],gap="large")
    with c1:
        st.markdown("""
        <div class="card" style="margin-bottom:1.25rem;"><div class="card-title">Research Question</div>
            <div style="font-family:var(--fd);font-size:1.15rem;line-height:1.6;color:var(--text);">Does cast popularity — measured via TMDB API popularity scores — independently predict global box office revenue, beyond what production budget alone explains?</div></div>
        <div class="card" style="margin-bottom:1.25rem;"><div class="card-title">Hypotheses</div>
            <div style="font-size:.875rem;line-height:1.8;color:var(--text-dim);">
                <strong style="color:var(--text);">H₁ (OLS):</strong> A 10% increase in cast popularity is associated with a statistically significant increase in log_revenue.<br><br>
                <strong style="color:var(--text);">H₂ (ANOVA):</strong> Mean log_revenue differs significantly across genre groups (α=0.05).<br><br>
                <strong style="color:var(--text);">H₃ (ROI):</strong> Films with above-median SPI achieve higher ROI at the same budget tier.</div></div>
        <div class="card" style="margin-bottom:1.25rem;"><div class="card-title">Methodology</div>
            <div style="font-size:.875rem;line-height:1.9;color:var(--text-dim);">
                <strong style="color:var(--gold);">Data</strong> — TMDB 5000 Movies + Credits, augmented with TMDB API. 4,803 raw → 2,641 clean films.<br><br>
                <strong style="color:var(--gold);">Features</strong> — log_revenue, log_budget, log_cast_pop, vote_average, primary_genre, ROI.<br><br>
                <strong style="color:var(--gold);">Models</strong> — OLS Regression, One-way ANOVA, Tukey HSD post-hoc.<br><br>
                <strong style="color:var(--gold);">Diagnostics</strong> — VIF, Shapiro-Wilk, Breusch-Pagan.</div></div>
        <div class="card"><div class="card-title">Literature Foundation</div>
            <div style="font-size:.8rem;line-height:2;color:var(--text-dim);font-family:var(--fm);">
                Feng et al. (2024) · Information Systems Frontiers<br>Bogaert et al. (2021) · Decision Support Systems<br>
                Oh et al. (2017) · Information &amp; Management<br>Shahid &amp; Islam (2023) · PeerJ Computer Science</div></div>""",unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card" style="margin-bottom:1.25rem;"><div class="card-title">Team Ghanti Tininini</div>',unsafe_allow_html=True)
        for n,r,h in [("Anuprash Pokharel","Project Lead","AnuPrasHPoKhareL29"),("Kabit Khadka","Statistical Analyst","kabit-k"),("Prashanna Dhami","App Developer","prashannaLeo"),("Sarjyant Maharjan","Data Engineer","SarjyantM")]:
            st.markdown(f'<div style="padding:.75rem 0;border-bottom:1px solid var(--border);"><div style="font-family:var(--fb);font-size:.9rem;color:var(--text);font-weight:500;">{n}</div><div style="font-family:var(--fm);font-size:.65rem;color:var(--muted);">{r} · @{h}</div></div>',unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        if df is not None:
            st.markdown('<div class="card"><div class="card-title">Dataset at a Glance</div>',unsafe_allow_html=True)
            desc=df[["budget","revenue","cast_popularity_score","vote_average","roi"]].describe().loc[["mean","std","min","max"]]
            desc.columns=["Budget","Revenue","SPI","Rating","ROI"]; st.dataframe(desc.T.round(2),use_container_width=True)
            st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("""<div class="card" style="margin-top:1.25rem;"><div class="card-title">Course Info</div>
            <div style="font-size:.8rem;line-height:2.2;color:var(--text-dim);font-family:var(--fm);">
                DATA 200 Applied Statistical Analysis<br>King's College Nepal<br>Westcliff University<br>Professor Regmi · April 2026</div></div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.warning("⚠️ Predictions are statistical estimates. Actual outcomes depend on marketing, competition, timing, and many factors outside this model.")


def show_setup():
    st.markdown("""<div class="page-header"><div class="header-eyebrow">✦ Setup Required</div>
        <div class="header-title">Star Power <span>Index</span></div>
        <div class="header-desc">Dataset not found. Complete the steps below to launch.</div>
    </div>""", unsafe_allow_html=True)
    st.error("❌ `outputs/tmdb_cleaned.csv` not found.")
    st.markdown("""
**1.** Download from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) → place in `data/`
```powershell
pip install -r requirements.txt
python -m src.preprocessing
streamlit run main.py
```""")


def main():
    load_css()
    df = load_clean_data()
    
    # render_top_nav now returns just the name (e.g. "Greenlight Predictor")
    current_page = render_top_nav(df)

    # Simplified routing logic
    if df is None or df.empty:
        show_setup()
    elif current_page == "Greenlight Predictor": page_predictor(df)
    elif current_page == "Data Observatory":     page_exploration(df)
    elif current_page == "Model Laboratory":     page_model_insights(df)
    elif current_page == "Project Brief":        page_about(df)
    else:                                        page_predictor(df)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid rgba(255,255,255,.05);margin-top:4rem;padding:1.25rem 0;display:flex;justify-content:space-between;align-items:center;">
        <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:#64748b;letter-spacing:.1em;">✦ STAR POWER INDEX · v1.0 · DATA 200</div>
        <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:#64748b;">TEAM GHANTI TINININI · KING'S COLLEGE NEPAL · WESTCLIFF UNIVERSITY</div>
        <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:#64748b;">⚠ ESTIMATES ONLY — NOT FINANCIAL ADVICE</div>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()