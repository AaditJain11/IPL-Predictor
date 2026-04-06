import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Colour Palette ────────────────────────────────────────────
BG      = '#06090f'
CARD    = '#0d1321'
NAVY    = '#0f1a2e'
ACCENT  = '#2563eb'      # electric blue — primary
ACCENT2 = '#0ea5e9'      # sky blue — secondary
TEAL    = '#14b8a6'      # vibrant teal — accent
WHITE   = '#f8fafc'      # soft white text
MUTED   = '#475569'
LINE    = '#1e2d45'
RED     = '#f43f5e'
GREEN   = '#10b981'

# ─── Plotly theme defaults ─────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family='DM Sans, sans-serif', color='#94a3b8', size=11),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(
        gridcolor=LINE, linecolor=LINE, tickcolor='#334155',
        showgrid=False, zeroline=False
    ),
    yaxis=dict(
        gridcolor=LINE, linecolor='transparent', tickcolor='#334155',
        showgrid=True, zeroline=False
    ),
    hoverlabel=dict(
        bgcolor=NAVY, bordercolor=ACCENT,
        font=dict(color=WHITE, family='DM Sans, sans-serif', size=12)
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0)', bordercolor='transparent',
        font=dict(color='#94a3b8')
    ),
)

# ─── Premium CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #06090f;
    color: #cbd5e1;
}
.stApp { background: #06090f; }
.block-container { padding-top: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #06090f; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 4px; }

/* ═══════════════════════════════
   HERO — full-bleed dramatic header
════════════════════════════════ */
.hero {
    padding: 4rem 2rem 2.5rem;
    text-align: center;
    position: relative;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(37,99,235,0.18) 0%, transparent 70%);
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 5px;
    color: #2563eb;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.6rem, 5vw, 4.2rem);
    font-weight: 400;
    color: #f8fafc;
    line-height: 1.05;
    letter-spacing: -1px;
}
.hero-title span {
    color: transparent;
    background: linear-gradient(135deg, #2563eb, #0ea5e9, #14b8a6);
    -webkit-background-clip: text;
    background-clip: text;
}
.hero-sub {
    font-size: 0.88rem;
    color: #475569;
    margin-top: 0.9rem;
    letter-spacing: 0.5px;
}
.hero-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #2563eb, #14b8a6, transparent);
    margin: 1.8rem auto 0;
}

/* ═══════════════════════════════
   STAT STRIP
════════════════════════════════ */
.stats-strip {
    display: flex;
    gap: 1px;
    background: #1e2d45;
    border: 1px solid #1e2d45;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.stat-item {
    flex: 1;
    background: #0d1321;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    transition: background 0.2s;
}
.stat-item:hover { background: #111c30; }
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #f8fafc;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.stat-num.accent { color: #2563eb; }
.stat-num.teal   { color: #14b8a6; }
.stat-label {
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 500;
}

/* ═══════════════════════════════
   SECTION TITLES
════════════════════════════════ */
.sec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #f8fafc;
    margin: 2.2rem 0 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e2d45, transparent);
}

/* ═══════════════════════════════
   TABS
════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1321;
    border-radius: 14px;
    padding: 5px;
    gap: 2px;
    border: 1px solid #1e2d45;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.83rem;
    font-weight: 500;
    border-radius: 10px;
    padding: 9px 20px;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #0f1a2e !important;
    color: #2563eb !important;
    border-bottom: none !important;
    box-shadow: 0 0 0 1px #1e3a5f;
}

/* ═══════════════════════════════
   SELECTBOX / SLIDER
════════════════════════════════ */
.stSelectbox label, .stSlider label {
    color: #475569 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background: #0d1321 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
}

/* ═══════════════════════════════
   PREDICT TAB — HERO BANNER
════════════════════════════════ */
.predict-hero {
    background:
        radial-gradient(ellipse 70% 80% at 50% 0%, rgba(37,99,235,0.22) 0%, transparent 65%),
        linear-gradient(160deg, #0d1321 0%, #06090f 100%);
    border: 1px solid #1e3a5f;
    border-radius: 24px;
    padding: 3.5rem 2rem 3rem;
    text-align: center;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.predict-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #2563eb 30%, #0ea5e9 60%, #14b8a6, transparent);
}
.predict-hero::after {
    content: '🏏';
    position: absolute;
    font-size: 12rem;
    opacity: 0.03;
    top: 40%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-15deg);
    pointer-events: none;
}
.predict-hero-eyebrow {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 5px;
    color: #2563eb;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.predict-hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    color: #f8fafc;
    margin: 0.2rem 0;
    line-height: 1.1;
}
.predict-hero-title em {
    font-style: italic;
    color: transparent;
    background: linear-gradient(135deg, #2563eb, #14b8a6);
    -webkit-background-clip: text;
    background-clip: text;
}
.predict-hero-sub {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 0.7rem;
    letter-spacing: 0.3px;
}

/* ═══════════════════════════════
   PREDICT — INPUT PANEL
════════════════════════════════ */
.predict-panel {
    background: #0d1321;
    border: 1px solid #1e2d45;
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.panel-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: #334155;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ═══════════════════════════════
   PREDICT BUTTON
════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #0ea5e9 100%);
    color: #ffffff;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    border-radius: 12px;
    padding: 0.9rem 2.5rem;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 24px rgba(37,99,235,0.35), 0 0 0 0 rgba(37,99,235,0);
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 40px rgba(37,99,235,0.5), 0 0 60px rgba(14,165,233,0.15);
}
.stButton > button:active {
    transform: translateY(-1px);
}

/* ═══════════════════════════════
   PREDICT RESULT CARD
════════════════════════════════ */
.result-outer {
    background:
        radial-gradient(ellipse 80% 60% at 50% 0%, rgba(37,99,235,0.2) 0%, transparent 70%),
        linear-gradient(160deg, #0d1321, #06090f);
    border: 1px solid #1e3a5f;
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-outer::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #2563eb 30%, #14b8a6 70%, transparent);
}
.result-glow {
    position: absolute;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(37,99,235,0.15) 0%, transparent 70%);
    top: -100px;
    left: 50%;
    transform: translateX(-50%);
    pointer-events: none;
}
.result-eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: #334155;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    position: relative;
}
.result-winner {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 4vw, 3.4rem);
    color: #f8fafc;
    line-height: 1.1;
    position: relative;
    margin: 0.4rem 0;
    text-shadow: 0 0 60px rgba(37,99,235,0.5);
}
.result-conf {
    font-size: 0.8rem;
    color: #334155;
    margin-top: 0.4rem;
    position: relative;
}
.prob-section {
    margin-top: 2rem;
    background: rgba(6,9,15,0.6);
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.5rem;
    position: relative;
}
.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.45rem;
    font-size: 0.82rem;
}
.prob-name { font-weight: 600; color: #94a3b8; }
.prob-pct  { font-weight: 700; color: #f8fafc; font-variant-numeric: tabular-nums; }
.bar-track {
    width: 100%;
    height: 7px;
    background: #0f1a2e;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.bar-fill-blue { height: 100%; background: linear-gradient(90deg, #1d4ed8, #2563eb); border-radius: 4px; }
.bar-fill-teal { height: 100%; background: linear-gradient(90deg, #0891b2, #14b8a6); border-radius: 4px; }

/* ═══════════════════════════════
   H2H CARDS
════════════════════════════════ */
.h2h-card {
    background: #0d1321;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.h2h-card:hover { border-color: #2563eb; }
.h2h-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #2563eb;
    line-height: 1;
}
.h2h-lbl {
    font-size: 0.67rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.4rem;
    font-weight: 600;
}

/* ═══════════════════════════════
   DISCLAIMER
════════════════════════════════ */
.disc {
    background: #0d1321;
    border: 1px solid #1e2d45;
    border-left: 3px solid #2563eb;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.65;
    margin-top: 2rem;
}

/* ═══════════════════════════════
   MISC
════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA & MODEL
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    import zipfile

    with zipfile.ZipFile("matches.zip", "r") as z:
        z.extractall(".")
        matches_name = [n for n in z.namelist() if n.endswith('.csv')][0]

    with zipfile.ZipFile("deliveries.zip", "r") as z:
        z.extractall(".")
        deliveries_name = [n for n in z.namelist() if n.endswith('.csv')][0]

    matches    = pd.read_csv(matches_name)
    deliveries = pd.read_csv(deliveries_name)

    if 'date' not in matches.columns:
        matches, deliveries = deliveries, matches

    matches['date']   = pd.to_datetime(matches['date'])
    matches['season'] = matches['season'].apply(lambda x: int(str(x)[:4]))

    team_map = {
        'Delhi Daredevils':           'Delhi Capitals',
        'Kings XI Punjab':            'Punjab Kings',
        'Royal Challengers Bangalore':'Royal Challengers Bengaluru',
        'Rising Pune Supergiant':     'Rising Pune Supergiants',
    }
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        matches[col] = matches[col].replace(team_map)
    for col in ['batting_team', 'bowling_team']:
        deliveries[col] = deliveries[col].replace(team_map)

    matches['city']           = matches['city'].fillna(matches['venue'].str.split(',').str[-1].str.strip())
    matches['winner']         = matches['winner'].fillna('No Result')
    matches['player_of_match']= matches['player_of_match'].fillna('No Award')
    for c in ['extras_type', 'player_dismissed', 'dismissal_kind', 'fielder']:
        deliveries[c] = deliveries[c].fillna('None')

    df = deliveries.merge(matches, left_on='match_id', right_on='id', how='left')
    return matches, deliveries, df


@st.cache_resource(show_spinner=False)
def train_model(_matches):
    encoders = {}
    cols = ['city', 'team1', 'team2', 'toss_winner', 'toss_decision']
    raw  = _matches[['season', 'city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']].copy().dropna()
    raw['result'] = (raw['winner'] == raw['team1']).astype(int)
    raw  = raw.drop(columns=['winner'])
    for col in cols:
        le = LabelEncoder()
        raw[col] = le.fit_transform(raw[col])
        encoders[col] = le
    X, y = raw.drop(columns=['result']), raw['result']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(X_tr, y_tr)
    return m, encoders


# ─── Plotly chart factory ──────────────────────────────────────
def apply_theme(fig, title='', height=400):
    """Apply the navy/blue theme to any Plotly figure."""
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(color='#94a3b8', size=13, family='DM Sans'), x=0, pad=dict(l=4)),
        height=height,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════
with st.spinner('Loading IPL data…'):
    matches, deliveries, df = load_data()
    model, encoders = train_model(matches)

# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Indian Premier League &nbsp;·&nbsp; 2008 – 2024</div>
    <h1 class="hero-title">IPL <span>Intelligence</span></h1>
    <p class="hero-sub">17 seasons of data &nbsp;·&nbsp; interactive analytics &nbsp;·&nbsp; AI match predictor</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── Stat Strip ────────────────────────────────────────────────
total_runs  = int(deliveries['batsman_runs'].sum())
top_scorer  = deliveries.groupby('batter')['batsman_runs'].sum().idxmax()
total_sixes = int((deliveries['batsman_runs'] == 6).sum())
total_fours = int((deliveries['batsman_runs'] == 4).sum())

st.markdown(f"""
<div class="stats-strip">
    <div class="stat-item">
        <div class="stat-num">{len(matches):,}</div>
        <div class="stat-label">Matches</div>
    </div>
    <div class="stat-item">
        <div class="stat-num accent">{deliveries.shape[0]:,}</div>
        <div class="stat-label">Balls Bowled</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">{total_runs:,}</div>
        <div class="stat-label">Total Runs</div>
    </div>
    <div class="stat-item">
        <div class="stat-num teal">{total_sixes:,}</div>
        <div class="stat-label">Sixes Hit</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">{total_fours:,}</div>
        <div class="stat-label">Fours Hit</div>
    </div>
    <div class="stat-item">
        <div class="stat-num accent" style="font-size:1.1rem;padding-top:0.3rem">{top_scorer}</div>
        <div class="stat-label">Top Scorer (All-time)</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS — Predict first
# ═══════════════════════════════════════════════════════════════
tab_predict, tab_match, tab_bat, tab_bowl, tab_adv = st.tabs([
    "🎯  Predict", "📊  Match Analysis", "🏏  Batting", "🎳  Bowling", "🔍  Advanced"
])


# ══════════════════════════════════════════════════
# TAB — PREDICT
# ══════════════════════════════════════════════════
with tab_predict:
    st.markdown("""
    <div class="predict-hero">
        <div class="predict-hero-eyebrow">AI-Powered · Historical Data 2008 – 2024</div>
        <div class="predict-hero-title">Who Will<br><em>Win Tonight?</em></div>
        <div class="predict-hero-sub">Select teams & venue for an instant win-probability forecast</div>
    </div>
    """, unsafe_allow_html=True)

    teams_list  = sorted(encoders['team1'].classes_)
    cities_list = sorted(encoders['city'].classes_)

    st.markdown('<div class="predict-panel"><div class="panel-label">Match Setup</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        city_sel = st.selectbox('🏟️  Venue City', cities_list)
    with col2:
        t1_sel = st.selectbox('🔵  Team 1', teams_list)
    with col3:
        t2_sel = st.selectbox('🔴  Team 2', [t for t in teams_list if t != t1_sel])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button('⚡  Generate Prediction'):
        try:
            team_wins   = matches[matches['winner'] != 'No Result']['winner'].value_counts()
            team_played = pd.concat([matches['team1'], matches['team2']]).value_counts()
            win_pct     = (team_wins / team_played * 100).fillna(0)

            t1_win_pct = win_pct.get(t1_sel, 50)
            t2_win_pct = win_pct.get(t2_sel, 50)

            model_team1, model_team2 = (t1_sel, t2_sel) if t1_win_pct >= t2_win_pct else (t2_sel, t1_sel)

            def make_input(toss_winner):
                return pd.DataFrame([{
                    'season':        2024,
                    'city':          encoders['city'].transform([city_sel])[0],
                    'team1':         encoders['team1'].transform([model_team1])[0],
                    'team2':         encoders['team2'].transform([model_team2])[0],
                    'toss_winner':   encoders['toss_winner'].transform([toss_winner])[0],
                    'toss_decision': encoders['toss_decision'].transform(['field'])[0]
                }])

            p1 = model.predict_proba(make_input(model_team1))[0]
            p2 = model.predict_proba(make_input(model_team2))[0]

            avg_m1 = (p1[1] + p2[1]) / 2
            avg_m2 = (p1[0] + p2[0]) / 2

            avg1, avg2 = (avg_m1, avg_m2) if model_team1 == t1_sel else (avg_m2, avg_m1)

            winner = t1_sel if avg1 > avg2 else t2_sel
            conf   = max(avg1, avg2) * 100
            pct1   = round(avg1 * 100, 1)
            pct2   = round(avg2 * 100, 1)

            # --- Result card ---
            st.markdown(f"""
            <div class="result-outer">
                <div class="result-glow"></div>
                <div class="result-eyebrow">Predicted Winner</div>
                <div class="result-winner">🏆&nbsp; {winner}</div>
                <div class="result-conf">Model confidence &nbsp;·&nbsp; {conf:.1f}%</div>

                <div class="prob-section">
                    <div class="prob-row">
                        <span class="prob-name" style="color:#2563eb">{t1_sel}</span>
                        <span class="prob-pct">{pct1}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill-blue" style="width:{pct1}%"></div>
                    </div>
                    <div class="prob-row">
                        <span class="prob-name" style="color:#14b8a6">{t2_sel}</span>
                        <span class="prob-pct">{pct2}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill-teal" style="width:{pct2}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- Win probability gauge (Plotly) ---
            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct1,
                number=dict(suffix="%", font=dict(color='#f8fafc', size=32, family='DM Serif Display')),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor='#334155',
                              tickfont=dict(color='#475569', size=10)),
                    bar=dict(color='#2563eb', thickness=0.25),
                    bgcolor='#0f1a2e',
                    bordercolor='transparent',
                    steps=[
                        dict(range=[0, 50],  color='#0a1628'),
                        dict(range=[50, 100], color='#0d1f3c'),
                    ],
                    threshold=dict(
                        line=dict(color='#14b8a6', width=3),
                        thickness=0.8,
                        value=50
                    )
                ),
                title=dict(text=f"<b>{t1_sel}</b> Win Probability",
                           font=dict(color='#94a3b8', size=13, family='DM Sans'))
            ))
            fig_gauge.update_layout(
                paper_bgcolor=CARD, plot_bgcolor=CARD,
                height=260,
                margin=dict(l=30, r=30, t=40, b=10),
                font=dict(family='DM Sans')
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # --- H2H ---
            h2h = matches[
                ((matches['team1'] == t1_sel) & (matches['team2'] == t2_sel)) |
                ((matches['team1'] == t2_sel) & (matches['team2'] == t1_sel))
            ]
            h2h_total = len(h2h)
            h2h_w1    = len(h2h[h2h['winner'] == t1_sel])
            h2h_w2    = len(h2h[h2h['winner'] == t2_sel])

            if h2h_total > 0:
                st.markdown('<div class="sec-title">Head-to-Head History</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#f8fafc">{h2h_total}</div><div class="h2h-lbl">Total Matches</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#2563eb">{h2h_w1}</div><div class="h2h-lbl">{t1_sel}</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#14b8a6">{h2h_w2}</div><div class="h2h-lbl">{t2_sel}</div></div>', unsafe_allow_html=True)

                # Stacked bar split
                fig_h2h = go.Figure()
                p_t1 = h2h_w1 / h2h_total * 100
                p_t2 = h2h_w2 / h2h_total * 100
                p_nr = (h2h_total - h2h_w1 - h2h_w2) / h2h_total * 100
                for name, val, color in [
                    (t1_sel, p_t1, '#2563eb'),
                    (t2_sel, p_t2, '#14b8a6'),
                    ('No Result', p_nr, '#1e2d45')
                ]:
                    if val > 0:
                        fig_h2h.add_trace(go.Bar(
                            name=name, x=[val], y=['H2H Split'],
                            orientation='h',
                            marker=dict(color=color),
                            text=f'{name}<br>{val:.1f}%',
                            textposition='inside',
                            insidetextanchor='middle',
                            textfont=dict(color='#f8fafc', size=11, family='DM Sans'),
                            hovertemplate=f'<b>{name}</b>: {val:.1f}%<extra></extra>',
                        ))
                fig_h2h.update_layout(
                    barmode='stack',
                    paper_bgcolor=CARD, plot_bgcolor=CARD,
                    height=120,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    showlegend=False,
                    bargap=0,
                )
                st.plotly_chart(fig_h2h, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("""
    <div class="disc">
        ⚠️ Predictions are based on historical IPL data (2008–2024) and do not account for current player form,
        injuries, or squad changes. Model accuracy ~52%, reflecting the inherently unpredictable nature of T20 cricket.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════
# TAB — MATCH ANALYSIS
# ══════════════════════
with tab_match:
    all_seasons = sorted(matches['season'].unique())
    season_range = st.select_slider(
        'Filter by Season Range',
        options=all_seasons,
        value=(all_seasons[0], all_seasons[-1])
    )
    m_f = matches[(matches['season'] >= season_range[0]) & (matches['season'] <= season_range[1])]
    d_f = deliveries[deliveries['match_id'].isin(m_f['id'])]

    # ── Matches per Season (Plotly bar — season as category string to fix x-axis)
    st.markdown('<div class="sec-title">Matches Per Season</div>', unsafe_allow_html=True)
    mps = m_f.groupby('season')['id'].count().reset_index()
    mps.columns = ['season', 'count']
    mps['season_str'] = mps['season'].astype(str)   # string category fixes label gaps

    fig = go.Figure(go.Bar(
        x=mps['season_str'],
        y=mps['count'],
        marker=dict(
            color=mps['count'],
            colorscale=[[0, '#1e3a5f'], [1, '#2563eb']],
            showscale=False,
            line=dict(width=0)
        ),
        text=mps['count'],
        textposition='outside',
        textfont=dict(color='#64748b', size=10),
        hovertemplate='<b>Season %{x}</b><br>Matches: %{y}<extra></extra>',
    ))
    apply_theme(fig, 'Number of Matches Per Season', height=380)
    fig.update_layout(xaxis=dict(type='category', tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)

    # ── Toss analysis
    st.markdown('<div class="sec-title">Toss Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        tt = m_f.groupby(['season', 'toss_decision'])['id'].count().reset_index()
        tt.columns = ['season', 'decision', 'count']
        fig = go.Figure()
        color_map = {'bat': ACCENT, 'field': TEAL}
        dash_map  = {'bat': 'solid', 'field': 'dot'}
        for dec in ['bat', 'field']:
            d = tt[tt['decision'] == dec]
            fig.add_trace(go.Scatter(
                x=d['season'].astype(str), y=d['count'],
                mode='lines+markers',
                name=dec.capitalize(),
                line=dict(color=color_map[dec], width=2.5, dash=dash_map[dec]),
                marker=dict(size=6),
                hovertemplate=f'<b>{dec.capitalize()}</b><br>Season %{{x}}: %{{y}} matches<extra></extra>'
            ))
        apply_theme(fig, 'Toss Decision Over Seasons', height=340)
        fig.update_layout(xaxis=dict(type='category', tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        m_f2 = m_f.copy()
        m_f2['toss_won_match'] = m_f2['toss_winner'] == m_f2['winner']
        tdw = m_f2[m_f2['winner'] != 'No Result'].groupby('toss_decision')['toss_won_match'].mean() * 100
        fig = go.Figure(go.Bar(
            x=tdw.index.str.capitalize(),
            y=tdw.values.round(1),
            marker=dict(color=[ACCENT, TEAL], line=dict(width=0)),
            text=[f'{v:.1f}%' for v in tdw.values],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=12),
            width=0.4,
            hovertemplate='<b>%{x}</b><br>Win rate: %{y:.1f}%<extra></extra>',
        ))
        apply_theme(fig, 'Win Rate by Toss Decision', height=340)
        fig.update_layout(yaxis=dict(range=[0, 75], ticksuffix='%'))
        st.plotly_chart(fig, use_container_width=True)

    # ── Team Win %
    st.markdown('<div class="sec-title">Team Win Percentage</div>', unsafe_allow_html=True)
    tw = m_f[m_f['winner'] != 'No Result']['winner'].value_counts().reset_index()
    tw.columns = ['team', 'wins']
    tm = pd.concat([m_f['team1'], m_f['team2']]).value_counts().reset_index()
    tm.columns = ['team', 'played']
    ts = tm.merge(tw, on='team', how='left').fillna(0)
    ts['win_pct'] = (ts['wins'] / ts['played'] * 100).round(1)
    ts = ts.sort_values('win_pct')
    bar_colors = [ACCENT if v >= 55 else TEAL if v >= 45 else RED for v in ts['win_pct']]
    fig = go.Figure(go.Bar(
        x=ts['win_pct'],
        y=ts['team'],
        orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f'{v:.1f}%' for v in ts['win_pct']],
        textposition='outside',
        textfont=dict(color='#64748b', size=9),
        hovertemplate='<b>%{y}</b><br>Win %: %{x:.1f}%<extra></extra>',
    ))
    apply_theme(fig, 'All-time Win Percentage by Team', height=500)
    fig.update_layout(xaxis=dict(ticksuffix='%'))
    st.plotly_chart(fig, use_container_width=True)

    # ── Super Overs
    st.markdown('<div class="sec-title">Super Over Matches</div>', unsafe_allow_html=True)
    so = m_f[m_f['super_over'] == 'Y']
    so_teams = pd.concat([so['team1'], so['team2']]).value_counts().reset_index()
    so_teams.columns = ['team', 'count']
    fig = go.Figure(go.Bar(
        x=so_teams['team'], y=so_teams['count'],
        marker=dict(color=ACCENT2, line=dict(width=0)),
        text=so_teams['count'],
        textposition='outside',
        textfont=dict(color='#64748b', size=10),
        hovertemplate='<b>%{x}</b><br>Super Overs: %{y}<extra></extra>',
    ))
    apply_theme(fig, f'Super Over Involvement by Team  ({len(so)} total)', height=340)
    fig.update_layout(xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════
# TAB — BATTING
# ══════════════════════
with tab_bat:
    all_s2 = sorted(matches['season'].unique())
    s_range2 = st.select_slider(
        'Filter by Season Range', options=all_s2,
        value=(all_s2[0], all_s2[-1]), key='bat_slider'
    )
    m_bat = matches[(matches['season'] >= s_range2[0]) & (matches['season'] <= s_range2[1])]
    d_bat = deliveries[deliveries['match_id'].isin(m_bat['id'])]
    df_bat = d_bat.merge(m_bat[['id','season']], left_on='match_id', right_on='id', how='left')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-title">Top Run Scorers</div>', unsafe_allow_html=True)
        ts_bat = d_bat.groupby('batter')['batsman_runs'].sum().reset_index()
        ts_bat.columns = ['batter', 'runs']
        ts_bat = ts_bat.sort_values('runs').tail(15)
        fig = go.Figure(go.Bar(
            x=ts_bat['runs'], y=ts_bat['batter'],
            orientation='h',
            marker=dict(
                color=ts_bat['runs'],
                colorscale=[[0,'#1e3a5f'],[1,'#2563eb']],
                showscale=False, line=dict(width=0)
            ),
            text=ts_bat['runs'],
            textposition='outside',
            textfont=dict(color='#64748b', size=9),
            hovertemplate='<b>%{y}</b><br>Runs: %{x:,}<extra></extra>',
        ))
        apply_theme(fig, 'Top 15 Run Scorers', height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-title">Top Six Hitters</div>', unsafe_allow_html=True)
        sx = d_bat[d_bat['batsman_runs'] == 6].groupby('batter').size().reset_index()
        sx.columns = ['batter', 'sixes']
        sx = sx.sort_values('sixes').tail(12)
        fig = go.Figure(go.Bar(
            x=sx['sixes'], y=sx['batter'],
            orientation='h',
            marker=dict(
                color=sx['sixes'],
                colorscale=[[0,'#0a4a40'],[1,'#14b8a6']],
                showscale=False, line=dict(width=0)
            ),
            text=sx['sixes'],
            textposition='outside',
            textfont=dict(color='#64748b', size=9),
            hovertemplate='<b>%{y}</b><br>Sixes: %{x}<extra></extra>',
        ))
        apply_theme(fig, 'Top Six Hitters', height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Phase run rate
    st.markdown('<div class="sec-title">Phase-wise Run Rate</div>', unsafe_allow_html=True)
    def get_phase(o):
        return 'Powerplay (1–6)' if o <= 5 else ('Middle (7–15)' if o <= 14 else 'Death (16–20)')
    d_bat2 = d_bat.copy()
    d_bat2['phase'] = d_bat2['over'].apply(get_phase)
    ph_r  = d_bat2.groupby('phase')['total_runs'].sum()
    ph_b  = d_bat2.groupby('phase')['ball'].count()
    ph_rr = (ph_r / ph_b * 6).round(2).reset_index()
    ph_rr.columns = ['phase', 'run_rate']
    order = ['Powerplay (1–6)', 'Middle (7–15)', 'Death (16–20)']
    ph_rr['phase'] = pd.Categorical(ph_rr['phase'], categories=order, ordered=True)
    ph_rr = ph_rr.sort_values('phase')
    c_map = {'Powerplay (1–6)': ACCENT, 'Middle (7–15)': TEAL, 'Death (16–20)': RED}
    fig = go.Figure(go.Bar(
        x=ph_rr['phase'],
        y=ph_rr['run_rate'],
        marker=dict(color=[c_map[p] for p in ph_rr['phase']], line=dict(width=0)),
        text=[f'{v:.2f}' for v in ph_rr['run_rate']],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=14),
        width=0.4,
        hovertemplate='<b>%{x}</b><br>Run Rate: %{y:.2f}<extra></extra>',
    ))
    apply_theme(fig, 'Run Rate by Match Phase', height=340)
    fig.update_layout(yaxis=dict(range=[0, ph_rr['run_rate'].max() * 1.25]))
    st.plotly_chart(fig, use_container_width=True)

    # Scoring trends
    st.markdown('<div class="sec-title">Scoring Trends Per Season</div>', unsafe_allow_html=True)
    df_bat['is_four'] = df_bat['batsman_runs'] == 4
    df_bat['is_six']  = df_bat['batsman_runs'] == 6
    bps = df_bat.groupby('season')[['is_four','is_six']].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bps['season'].astype(str), y=bps['is_four'],
        mode='lines+markers', name='Fours',
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=6),
        fill='tozeroy', fillcolor='rgba(37,99,235,0.07)',
        hovertemplate='<b>Fours</b> — Season %{x}: %{y:,}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=bps['season'].astype(str), y=bps['is_six'],
        mode='lines+markers', name='Sixes',
        line=dict(color=TEAL, width=2.5, dash='dot'),
        marker=dict(size=6),
        fill='tozeroy', fillcolor='rgba(20,184,166,0.07)',
        hovertemplate='<b>Sixes</b> — Season %{x}: %{y:,}<extra></extra>'
    ))
    apply_theme(fig, 'Fours & Sixes Per Season', height=360)
    fig.update_layout(xaxis=dict(type='category', tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════
# TAB — BOWLING
# ══════════════════════
with tab_bowl:
    all_s3 = sorted(matches['season'].unique())
    s_range3 = st.select_slider(
        'Filter by Season Range', options=all_s3,
        value=(all_s3[0], all_s3[-1]), key='bowl_slider'
    )
    m_bowl = matches[(matches['season'] >= s_range3[0]) & (matches['season'] <= s_range3[1])]
    d_bowl = deliveries[deliveries['match_id'].isin(m_bowl['id'])]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-title">Top Wicket Takers</div>', unsafe_allow_html=True)
        wkts = d_bowl[(d_bowl['is_wicket'] == 1) & (d_bowl['dismissal_kind'] != 'run out')]
        tw_b = wkts.groupby('bowler')['is_wicket'].count().reset_index()
        tw_b.columns = ['bowler', 'wickets']
        tw_b = tw_b.sort_values('wickets').tail(15)
        fig = go.Figure(go.Bar(
            x=tw_b['wickets'], y=tw_b['bowler'],
            orientation='h',
            marker=dict(
                color=tw_b['wickets'],
                colorscale=[[0,'#3d0a14'],[1,'#f43f5e']],
                showscale=False, line=dict(width=0)
            ),
            text=tw_b['wickets'],
            textposition='outside',
            textfont=dict(color='#64748b', size=9),
            hovertemplate='<b>%{y}</b><br>Wickets: %{x}<extra></extra>',
        ))
        apply_theme(fig, 'Top 15 Wicket Takers', height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-title">Dismissal Types</div>', unsafe_allow_html=True)
        dis = d_bowl[d_bowl['is_wicket'] == 1]['dismissal_kind'].value_counts().reset_index()
        dis.columns = ['kind', 'count']
        dis = dis.sort_values('count')
        colors = [ACCENT, TEAL, RED, GREEN, '#8b5cf6', '#f97316', '#14b8a6', '#ec4899']
        fig = go.Figure(go.Bar(
            x=dis['count'], y=dis['kind'],
            orientation='h',
            marker=dict(color=colors[:len(dis)], line=dict(width=0)),
            text=dis['count'],
            textposition='outside',
            textfont=dict(color='#64748b', size=9),
            hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>',
        ))
        apply_theme(fig, 'Dismissal Breakdown', height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Economy
    st.markdown('<div class="sec-title">Most Economical Bowlers</div>', unsafe_allow_html=True)
    br   = d_bowl.groupby('bowler')['total_runs'].sum().reset_index()
    br.columns = ['bowler', 'runs']
    lb   = d_bowl[~d_bowl['extras_type'].isin(['wides','noballs'])].groupby('bowler')['ball'].count().reset_index()
    lb.columns = ['bowler', 'balls']
    be   = br.merge(lb, on='bowler')
    be['overs']   = be['balls'] / 6
    be['economy'] = (be['runs'] / be['overs']).round(2)
    be   = be[be['balls'] >= 300].sort_values('economy').head(12)
    fig  = go.Figure(go.Bar(
        x=be['economy'], y=be['bowler'],
        orientation='h',
        marker=dict(
            color=be['economy'],
            colorscale=[[0, GREEN], [1, TEAL]],
            reversescale=True,
            showscale=False,
            line=dict(width=0)
        ),
        text=[f'{v:.2f}' for v in be['economy']],
        textposition='outside',
        textfont=dict(color='#64748b', size=9),
        hovertemplate='<b>%{y}</b><br>Economy: %{x:.2f}<extra></extra>',
    ))
    apply_theme(fig, 'Most Economical Bowlers (min 300 balls)', height=440)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════
# TAB — ADVANCED
# ══════════════════════
with tab_adv:
    all_s4 = sorted(matches['season'].unique())
    s_range4 = st.select_slider(
        'Filter by Season Range', options=all_s4,
        value=(all_s4[0], all_s4[-1]), key='adv_slider'
    )
    m_adv = matches[(matches['season'] >= s_range4[0]) & (matches['season'] <= s_range4[1])]

    # Chasing win %
    st.markdown('<div class="sec-title">Chasing Win % Over Seasons</div>', unsafe_allow_html=True)
    valid = m_adv[m_adv['winner'] != 'No Result'].copy()
    def res_type(row):
        return 'defend' if (row['toss_decision'] == 'field' and row['toss_winner'] == row['winner']) else 'chase'
    valid['rtype'] = valid.apply(res_type, axis=1)
    cdp = valid.groupby(['season', 'rtype'])['id'].count().reset_index()
    cdp = cdp.pivot(index='season', columns='rtype', values='id').fillna(0)
    if 'chase' in cdp.columns and 'defend' in cdp.columns:
        cdp['pct'] = (cdp['chase'] / (cdp['chase'] + cdp['defend']) * 100).round(1)
        fig = go.Figure()
        fig.add_hline(y=50, line=dict(color='#334155', width=1, dash='dash'))
        fig.add_trace(go.Scatter(
            x=cdp.index.astype(str), y=cdp['pct'],
            mode='lines+markers',
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=7, color=[RED if v < 50 else ACCENT for v in cdp['pct']]),
            fill='tozeroy', fillcolor='rgba(37,99,235,0.07)',
            hovertemplate='<b>Season %{x}</b><br>Chase win %: %{y:.1f}%<extra></extra>'
        ))
        apply_theme(fig, 'Chasing Win % Per Season', height=360)
        fig.update_layout(
            xaxis=dict(type='category', tickangle=-45),
            yaxis=dict(ticksuffix='%', range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)

    # H2H
    st.markdown('<div class="sec-title">Head-to-Head</div>', unsafe_allow_html=True)
    all_teams = sorted(m_adv['team1'].unique())
    col1, col2 = st.columns(2)
    with col1:
        ht1 = st.selectbox('Team A', all_teams, key='ht1')
    with col2:
        ht2 = st.selectbox('Team B', [t for t in all_teams if t != ht1], key='ht2')

    h2h = m_adv[
        ((m_adv['team1'] == ht1) & (m_adv['team2'] == ht2)) |
        ((m_adv['team1'] == ht2) & (m_adv['team2'] == ht1))
    ]
    w1 = len(h2h[h2h['winner'] == ht1])
    w2 = len(h2h[h2h['winner'] == ht2])
    tot = len(h2h)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#f8fafc">{tot}</div><div class="h2h-lbl">Total Matches</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#2563eb">{w1}</div><div class="h2h-lbl">{ht1}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#14b8a6">{w2}</div><div class="h2h-lbl">{ht2}</div></div>', unsafe_allow_html=True)

    if tot > 0:
        pct1 = w1/tot*100; pct2 = w2/tot*100; pct_nr = (tot-w1-w2)/tot*100
        fig = go.Figure()
        for name, val, color in [(ht1, pct1, ACCENT), (ht2, pct2, TEAL), ('No Result', pct_nr, '#1e2d45')]:
            if val > 0:
                fig.add_trace(go.Bar(
                    name=name, x=[val], y=['Split'],
                    orientation='h',
                    marker=dict(color=color, line=dict(width=0)),
                    text=f'<b>{name}</b><br>{val:.1f}%',
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='#f8fafc', size=11),
                    hovertemplate=f'<b>{name}</b>: {val:.1f}%<extra></extra>',
                ))
        fig.update_layout(
            barmode='stack',
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            height=110,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # POM Leaders
    st.markdown('<div class="sec-title">Player of the Match Leaders</div>', unsafe_allow_html=True)
    pom = m_adv[m_adv['player_of_match'] != 'No Award']['player_of_match'].value_counts().head(15).reset_index()
    pom.columns = ['player', 'awards']
    pom = pom.sort_values('awards')
    fig = go.Figure(go.Bar(
        x=pom['awards'], y=pom['player'],
        orientation='h',
        marker=dict(
            color=pom['awards'],
            colorscale=[[0,'#1e3a5f'],[1,'#2563eb']],
            showscale=False, line=dict(width=0)
        ),
        text=pom['awards'],
        textposition='outside',
        textfont=dict(color='#64748b', size=9),
        hovertemplate='<b>%{y}</b><br>Awards: %{x}<extra></extra>',
    ))
    apply_theme(fig, 'Most Player of the Match Awards', height=480)
    st.plotly_chart(fig, use_container_width=True)


# ─── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#1e2d45;font-size:0.72rem;padding:3rem 0 1.5rem;letter-spacing:1.5px;text-transform:uppercase;">
    IPL Intelligence &nbsp;·&nbsp; Data: 2008 – 2024 &nbsp;·&nbsp; Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)
