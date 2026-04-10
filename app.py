import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="IPL Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
# PALETTE  — pitch green + twilight sky + ember orange
# ═══════════════════════════════════════════════════════════════
BG      = '#060d0d'   # deep pitch-black
CARD    = '#0b1614'   # dark turf
BORDER  = '#152a26'   # subtle green border
ACCENT  = '#00c896'   # neon pitch-green  (primary)
AMBER   = '#f5a623'   # sunset orange     (secondary)
BLUE    = '#3d9be9'   # twilight sky      (tertiary)
RED     = '#f04f5a'   # danger
PURPLE  = '#a78bfa'   # dusk violet
MUTED   = '#4a7a70'
LINE    = '#0f2320'

ACTIVE_TEAMS = {
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad',
}

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #060d0d;
    color: #d4ede8;
}
.stApp { background: #060d0d; }
.block-container { max-width: 1280px !important; padding: 0.5rem 1.5rem 4rem !important; }

/* ── NAV BAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 0.4rem;
    border-bottom: 1px solid #0f2320;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.topbar-brand {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.1rem, 3vw, 1.4rem);
    font-weight: 800;
    color: #f0f9f6;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.topbar-brand span { color: #00c896; }
.topbar-credit {
    font-size: 0.78rem;
    color: #4a7a70;
    letter-spacing: 0.5px;
}
.topbar-credit b { color: #00c896; font-weight: 600; }

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 1rem 0 2rem;
    position: relative;
}
.hero-eyebrow {
    font-size: clamp(0.7rem, 2vw, 0.78rem);
    font-weight: 500;
    letter-spacing: 5px;
    color: #00c896;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.6rem, 7vw, 4rem);
    font-weight: 800;
    color: #f0f9f6;
    line-height: 1.05;
    letter-spacing: -1px;
}
.hero-title span { color: #00c896; }
.hero-sub {
    font-size: clamp(0.92rem, 2vw, 1rem);
    color: #4a7a70;
    margin-top: 0.8rem;
    letter-spacing: 0.5px;
}

/* ── STAT CARDS ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin-bottom: 2.5rem;
}
.stat-card {
    background: #0b1614;
    border: 1px solid #152a26;
    border-radius: 16px;
    padding: 1.2rem 1.2rem 1rem;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00c896, transparent);
    opacity: 0.5;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.5rem, 3vw, 2rem);
    font-weight: 700;
    color: #f0f9f6;
    line-height: 1;
    margin-bottom: 0.35rem;
    word-break: break-word;
}
.stat-lbl {
    font-size: clamp(0.68rem, 1.8vw, 0.74rem);
    color: #4a7a70;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 500;
}

/* ── SECTION HEADER ── */
.sec-hd {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.15rem, 2.5vw, 1.35rem);
    font-weight: 700;
    color: #f0f9f6;
    margin: 2rem 0 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    letter-spacing: -0.3px;
}
.sec-hd::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #152a26, transparent);
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1614;
    border-radius: 14px;
    padding: 5px;
    gap: 3px;
    border: 1px solid #152a26;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a7a70;
    font-family: 'Inter', sans-serif;
    font-size: clamp(0.78rem, 2vw, 0.82rem);
    font-weight: 500;
    border-radius: 10px;
    padding: 7px 14px;
    white-space: nowrap;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #0f2320 !important;
    color: #00c896 !important;
    border-bottom: none !important;
}

/* ── SELECTBOX ── */
.stSelectbox label {
    color: #4a7a70 !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background: #0b1614 !important;
    border: 1px solid #152a26 !important;
    border-radius: 12px !important;
    color: #d4ede8 !important;
}

/* ── PREDICT HERO BLOCK ── */
.pred-hero {
    background: linear-gradient(135deg, #0b1614 0%, #071410 60%, #0b1614 100%);
    border: 1px solid #1a3d35;
    border-radius: 24px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.pred-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00c896, #f5a623, transparent);
}
.pred-hero-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 5px;
    color: #00c896;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    font-weight: 500;
}
.pred-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 2.8rem);
    font-weight: 800;
    color: #f0f9f6;
    letter-spacing: -0.5px;
}
.pred-hero-sub {
    font-size: clamp(0.88rem, 2vw, 0.95rem);
    color: #4a7a70;
    margin-top: 0.5rem;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #00a37a, #00c896);
    color: #060d0d;
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 1.5rem;
    width: 100%;
    transition: all 0.25s ease;
    box-shadow: 0 4px 24px rgba(0,200,150,0.25);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00c896, #00e6ac);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,200,150,0.4);
}

/* ── RESULT CARD ── */
.result-card {
    background: linear-gradient(135deg, #0b1614, #071410);
    border: 1px solid #1a3d35;
    border-radius: 24px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00c896, #f5a623, transparent);
}
.result-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: #4a7a70;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-winner {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 2.8rem);
    font-weight: 800;
    color: #00c896;
    letter-spacing: -0.5px;
    text-shadow: 0 0 40px rgba(0,200,150,0.35);
}
.result-conf {
    font-size: 0.85rem;
    color: #4a7a70;
    margin-top: 0.4rem;
}
.prob-wrap {
    background: #060d0d;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-top: 1.5rem;
}
.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.88rem;
    margin-bottom: 0.45rem;
    color: #94b8b0;
}
.prob-track {
    width: 100%; height: 8px;
    background: #152a26;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.prob-fill-green  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #00a37a, #00c896); }
.prob-fill-amber  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #c47d0e, #f5a623); }

/* ── DISCLAIMER ── */
.disc {
    background: #0b1614;
    border: 1px solid #152a26;
    border-left: 3px solid #00c896;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #4a7a70;
    line-height: 1.65;
    margin-top: 1.5rem;
}

/* ── H2H CARDS ── */
.h2h-card {
    background: #0b1614;
    border: 1px solid #152a26;
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.h2h-num {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.7rem, 3vw, 2.1rem);
    font-weight: 700;
    color: #00c896;
    line-height: 1;
}
.h2h-lbl {
    font-size: clamp(0.68rem, 1.8vw, 0.76rem);
    color: #4a7a70;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
    word-break: break-word;
}

/* ── PAR SCORE ── */
.par-hero {
    background: linear-gradient(135deg, #0b1614, #071410);
    border: 1px solid #1a3d35;
    border-radius: 24px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.par-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00c896, #f5a623, transparent);
}
.par-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.7rem;
    margin: 1.5rem 0;
}
.par-stat {
    background: #0b1614;
    border: 1px solid #152a26;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.par-stat::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}
.par-stat.green::before  { background: #00c896; }
.par-stat.amber::before  { background: #f5a623; }
.par-stat.blue::before   { background: #3d9be9; }
.par-stat.red::before    { background: #f04f5a; }
.par-stat-num {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.4rem, 3vw, 1.8rem);
    font-weight: 700;
    color: #f0f9f6;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.par-stat-lbl {
    font-size: clamp(0.68rem, 1.8vw, 0.74rem);
    color: #4a7a70;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}
.win-table {
    background: #0b1614;
    border: 1px solid #152a26;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.win-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid #0f2320;
    font-size: clamp(0.8rem, 2vw, 0.86rem);
    gap: 0.5rem;
}
.win-row:last-child { border-bottom: none; }
.win-score { color: #d4ede8; font-weight: 600; min-width: 90px; }
.win-pct   { font-weight: 700; font-size: clamp(0.88rem, 2vw, 0.96rem); min-width: 40px; text-align: right; }
.win-bar-track { flex: 1; height: 7px; background: #152a26; border-radius: 4px; overflow: hidden; }
.win-bar-fill  { height: 100%; border-radius: 4px; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #060d0d; }
::-webkit-scrollbar-thumb { background: #152a26; border-radius: 4px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA & MODEL
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    matches    = pd.read_csv(os.path.join(base, 'matches.csv'))
    deliveries = pd.read_csv(os.path.join(base, 'deliveries.csv'))

    matches['date']   = pd.to_datetime(matches['date'])
    matches['season'] = matches['season'].apply(lambda x: int(str(x)[:4]))

    team_map = {
        'Delhi Daredevils':            'Delhi Capitals',
        'Kings XI Punjab':             'Punjab Kings',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Rising Pune Supergiant':      'Rising Pune Supergiants',
    }
    for col in ['team1','team2','toss_winner','winner']:
        matches[col] = matches[col].replace(team_map)
    for col in ['batting_team','bowling_team']:
        deliveries[col] = deliveries[col].replace(team_map)

    matches['city']            = matches['city'].fillna(matches['venue'].str.split(',').str[-1].str.strip())
    matches['winner']          = matches['winner'].fillna('No Result')
    matches['player_of_match'] = matches['player_of_match'].fillna('No Award')
    for c in ['extras_type','player_dismissed','dismissal_kind','fielder']:
        deliveries[c] = deliveries[c].fillna('None')

    df = deliveries.merge(matches, left_on='match_id', right_on='id', how='left')
    return matches, deliveries, df


@st.cache_resource(show_spinner=False)
def train_model(_matches):
    encoders = {}
    cols = ['city','team1','team2','toss_winner','toss_decision']
    raw  = _matches[['season','city','team1','team2','toss_winner','toss_decision','winner']].copy().dropna()
    raw['result'] = (raw['winner'] == raw['team1']).astype(int)
    raw  = raw.drop(columns=['winner'])
    for col in cols:
        le = LabelEncoder()
        raw[col] = le.fit_transform(raw[col])
        encoders[col] = le
    X, y = raw.drop(columns=['result']), raw['result']
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(Xtr, ytr)
    return m, encoders


def base_fig(w=10, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors='#6a9e94', labelsize=11)
    ax.xaxis.label.set_color('#4a7a70')
    ax.yaxis.label.set_color('#4a7a70')
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    for sp in ax.spines.values():
        sp.set_edgecolor(LINE)
    ax.grid(axis='y', color=LINE, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


# ═══════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════
with st.spinner(''):
    matches, deliveries, df = load_data()
    model, encoders         = train_model(matches)


# ═══════════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">IPL <span>Intelligence</span></div>
    <div class="topbar-credit">Built by <b>Aadit Jain</b> &nbsp;·&nbsp; Data: 2008 – 2024</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Indian Premier League · 17 Seasons</div>
    <div class="hero-title">Cricket, <span>Decoded</span></div>
    <div class="hero-sub">Deep analytics · Match prediction · Par score calculator</div>
</div>
""", unsafe_allow_html=True)

total_runs  = int(deliveries['batsman_runs'].sum())
total_sixes = int((deliveries['batsman_runs'] == 6).sum())
top_scorer  = deliveries.groupby('batter')['batsman_runs'].sum().idxmax()

st.markdown(f"""
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-num">{len(matches):,}</div>
        <div class="stat-lbl">Matches</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{deliveries.shape[0]:,}</div>
        <div class="stat-lbl">Balls Bowled</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{total_runs:,}</div>
        <div class="stat-lbl">Total Runs</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{total_sixes:,}</div>
        <div class="stat-lbl">Sixes Hit</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="font-size:clamp(0.9rem,2.2vw,1.1rem);padding-top:0.3rem">{top_scorer}</div>
        <div class="stat-lbl">Top Run Scorer</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_pred, tab_match, tab_bat, tab_bowl, tab_adv, tab_par = st.tabs([
    "Predict Winner",
    "Match Analysis",
    "Batting",
    "Bowling",
    "Advanced",
    "Par Score",
])


# ══════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════
with tab_pred:
    st.markdown("""
    <div class="pred-hero">
        <div class="pred-hero-eyebrow">Logistic Regression Model · Historical Data 2008–2024</div>
        <div class="pred-hero-title">Match Winner Predictor</div>
        <div class="pred-hero-sub">Select two teams and a venue city to get an instant win probability forecast</div>
    </div>
    """, unsafe_allow_html=True)

    teams_list  = sorted([t for t in encoders['team1'].classes_ if t in ACTIVE_TEAMS])
    cities_list = sorted(encoders['city'].classes_)

    c1, c2, c3 = st.columns(3)
    with c1:
        city_sel = st.selectbox('Venue City', cities_list)
    with c2:
        t1_sel = st.selectbox('Team 1', teams_list)
    with c3:
        t2_sel = st.selectbox('Team 2', [t for t in teams_list if t != t1_sel])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button('Run Prediction'):
        try:
            team_wins   = matches[matches['winner'] != 'No Result']['winner'].value_counts()
            team_played = pd.concat([matches['team1'], matches['team2']]).value_counts()
            win_pct_map = (team_wins / team_played * 100).fillna(0)

            mt1, mt2 = (
                (t1_sel, t2_sel)
                if win_pct_map.get(t1_sel, 50) >= win_pct_map.get(t2_sel, 50)
                else (t2_sel, t1_sel)
            )

            def make_inp(tw):
                return pd.DataFrame([{
                    'season':        2024,
                    'city':          encoders['city'].transform([city_sel])[0],
                    'team1':         encoders['team1'].transform([mt1])[0],
                    'team2':         encoders['team2'].transform([mt2])[0],
                    'toss_winner':   encoders['toss_winner'].transform([tw])[0],
                    'toss_decision': encoders['toss_decision'].transform(['field'])[0],
                }])

            p1 = model.predict_proba(make_inp(mt1))[0]
            p2 = model.predict_proba(make_inp(mt2))[0]
            avg_mt1 = (p1[1] + p2[1]) / 2
            avg_mt2 = (p1[0] + p2[0]) / 2
            avg1, avg2 = (avg_mt1, avg_mt2) if mt1 == t1_sel else (avg_mt2, avg_mt1)

            winner = t1_sel if avg1 > avg2 else t2_sel
            conf   = max(avg1, avg2) * 100
            pct1   = round(avg1 * 100, 1)
            pct2   = round(avg2 * 100, 1)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-eyebrow">Predicted Winner</div>
                <div class="result-winner">&#127942;&nbsp; {winner}</div>
                <div class="result-conf">Confidence: {conf:.1f}%</div>
                <div class="prob-wrap">
                    <div class="prob-row">
                        <span style="color:#00c896;font-weight:600">{t1_sel}</span>
                        <span style="color:#f0f9f6;font-weight:600">{pct1}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill-green" style="width:{pct1}%"></div>
                    </div>
                    <div class="prob-row" style="margin-top:0.9rem">
                        <span style="color:#f5a623;font-weight:600">{t2_sel}</span>
                        <span style="color:#f0f9f6;font-weight:600">{pct2}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill-amber" style="width:{pct2}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("""
    <div class="disc">
        Predictions use historical IPL data (2008–2024) and do not account for current
        player availability, injuries, pitch conditions, or recent form. Model accuracy is
        approximately 52%, which fairly reflects the unpredictable nature of T20 cricket.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# MATCH ANALYSIS
# ══════════════════════════════════════════════════════
with tab_match:

    st.markdown('<div class="sec-hd">Matches Per Season</div>', unsafe_allow_html=True)
    mps = matches.groupby('season')['id'].count().reset_index()
    mps.columns = ['season', 'count']

    fig, ax = base_fig(10, 4.5)
    bars = ax.bar(range(len(mps)), mps['count'], color=ACCENT, alpha=0.88, width=0.65, zorder=3)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                str(int(bar.get_height())),
                ha='center', color='#d4ede8', fontsize=10, fontweight='600')
    ax.set_xticks(range(len(mps)))
    ax.set_xticklabels([str(s) for s in mps['season']], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Matches')
    ax.set_title('Number of Matches Per Season', color='#d4ede8', fontsize=13, pad=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="sec-hd">Toss Decisions Over Seasons</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        tt = matches.groupby(['season','toss_decision'])['id'].count().reset_index()
        tt.columns = ['season','decision','count']
        seas = sorted(tt['season'].unique())
        bat_d   = tt[tt['decision']=='bat'].set_index('season').reindex(seas, fill_value=0)['count']
        field_d = tt[tt['decision']=='field'].set_index('season').reindex(seas, fill_value=0)['count']

        fig, ax = base_fig(6, 4.8)
        ax.fill_between(seas, bat_d,   alpha=0.12, color=AMBER)
        ax.fill_between(seas, field_d, alpha=0.12, color=ACCENT)
        ax.plot(seas, bat_d,   marker='o', markersize=6, color=AMBER,  linewidth=2.2, label='Bat first', zorder=3)
        ax.plot(seas, field_d, marker='D', markersize=5, color=ACCENT, linewidth=2.2, label='Field first', zorder=3, linestyle='--')
        ax.set_title('Toss Decision Trend', color='#d4ede8', fontsize=13, pad=14)
        ax.set_xlabel('Season')
        ax.set_ylabel('Count')
        ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor='#d4ede8', fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        m2 = matches.copy()
        m2['toss_won_match'] = m2['toss_winner'] == m2['winner']
        tdw = (m2[m2['winner']!='No Result']
               .groupby('toss_decision')['toss_won_match'].mean()*100).reset_index()
        tdw.columns = ['decision','win_pct']
        tdw['label'] = tdw['decision'].map({'bat':'Bat First','field':'Field First'})
        tdw['color'] = tdw['decision'].map({'bat':AMBER,'field':ACCENT})

        fig, ax = base_fig(6, 4.8)
        bars = ax.bar(tdw['label'], tdw['win_pct'], color=tdw['color'].tolist(),
                      width=0.42, zorder=3, alpha=0.9)
        for bar, (_, row) in zip(bars, tdw.iterrows()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                    f'{row["win_pct"]:.1f}%',
                    ha='center', color='#f0f9f6', fontsize=15, fontweight='700')
        ax.set_title('Win Rate by Toss Decision', color='#d4ede8', fontsize=13, pad=14)
        ax.set_ylabel('Win %')
        ax.set_ylim(0, 80)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="sec-hd">All-time Win Percentage</div>', unsafe_allow_html=True)
    tw = matches[matches['winner']!='No Result']['winner'].value_counts().reset_index()
    tw.columns = ['team','wins']
    tm = pd.concat([matches['team1'],matches['team2']]).value_counts().reset_index()
    tm.columns = ['team','played']
    ts = tm.merge(tw,on='team',how='left').fillna(0)
    ts['win_pct'] = (ts['wins']/ts['played']*100).round(1)
    ts = ts[ts['played']>=10].sort_values('win_pct')

    bar_colors = [ACCENT if v>=55 else AMBER if v>=45 else RED for v in ts['win_pct']]
    fig, ax = base_fig(10, max(5.5, len(ts)*0.48))
    bars = ax.barh(ts['team'], ts['win_pct'], color=bar_colors, alpha=0.9, height=0.65, zorder=3)
    for bar in bars:
        ax.text(bar.get_width()+0.6, bar.get_y()+bar.get_height()/2,
                f'{bar.get_width():.1f}%', va='center', color='#d4ede8', fontsize=11, fontweight='600')
    ax.set_title('Win % by Team (min 10 matches)', color='#d4ede8', fontsize=13, pad=14)
    ax.set_xlabel('Win %')
    ax.set_xlim(0, 90)
    patches = [mpatches.Patch(color=ACCENT,label='>55%'),
               mpatches.Patch(color=AMBER, label='45–55%'),
               mpatches.Patch(color=RED,   label='<45%')]
    ax.legend(handles=patches, facecolor=CARD, edgecolor=LINE, labelcolor='#d4ede8', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    so = matches[matches['super_over']=='Y']
    if len(so) > 0:
        st.markdown('<div class="sec-hd">Super Over Involvement</div>', unsafe_allow_html=True)
        so_teams = pd.concat([so['team1'],so['team2']]).value_counts().reset_index()
        so_teams.columns = ['team','count']
        fig, ax = base_fig(10, 4)
        ax.bar(so_teams['team'], so_teams['count'], color=AMBER, alpha=0.9, width=0.6, zorder=3)
        for bar in ax.patches:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    str(int(bar.get_height())), ha='center', color='#d4ede8', fontsize=11, fontweight='600')
        ax.set_title(f'Super Over Involvement  ({len(so)} total matches)', color='#d4ede8', fontsize=13, pad=14)
        ax.set_ylabel('Times Involved')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════
# BATTING
# ══════════════════════════════════════════════════════
with tab_bat:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-hd">Top Run Scorers</div>', unsafe_allow_html=True)
        ts_bat = deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
        ts_bat.columns = ['batter','runs']
        ts_bat = ts_bat.sort_values('runs', ascending=True).tail(15)
        fig, ax = base_fig(6, 6.5)
        ax.barh(ts_bat['batter'], ts_bat['runs'], color=ACCENT, alpha=0.9, height=0.65, zorder=3)
        for bar in ax.patches:
            ax.text(bar.get_width()+30, bar.get_y()+bar.get_height()/2,
                    f'{int(bar.get_width()):,}', va='center', color='#d4ede8', fontsize=9, fontweight='600')
        ax.set_title('Top 15 Run Scorers', color='#d4ede8', fontsize=13, pad=14)
        ax.set_xlabel('Total Runs')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="sec-hd">Top Six Hitters</div>', unsafe_allow_html=True)
        sx = deliveries[deliveries['batsman_runs']==6].groupby('batter').size().reset_index()
        sx.columns = ['batter','sixes']
        sx = sx.sort_values('sixes', ascending=True).tail(12)
        fig, ax = base_fig(6, 6.5)
        ax.barh(sx['batter'], sx['sixes'], color=RED, alpha=0.9, height=0.65, zorder=3)
        for bar in ax.patches:
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    str(int(bar.get_width())), va='center', color='#d4ede8', fontsize=9, fontweight='600')
        ax.set_title('Top 12 Six Hitters', color='#d4ede8', fontsize=13, pad=14)
        ax.set_xlabel('Sixes')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="sec-hd">Run Rate by Phase</div>', unsafe_allow_html=True)

    def get_phase(o):
        if o<=5: return 'Powerplay\n(1–6)'
        elif o<=14: return 'Middle\n(7–15)'
        else: return 'Death\n(16–20)'

    dp = deliveries.copy()
    dp['phase'] = dp['over'].apply(get_phase)
    ph_r = dp.groupby('phase')['total_runs'].sum()
    ph_b = dp.groupby('phase')['ball'].count()
    ph_rr = (ph_r/ph_b*6).round(2).reset_index()
    ph_rr.columns = ['phase','run_rate']
    order_map = {'Powerplay\n(1–6)':0,'Middle\n(7–15)':1,'Death\n(16–20)':2}
    ph_rr['ord'] = ph_rr['phase'].map(order_map)
    ph_rr = ph_rr.sort_values('ord')
    cmap = {'Powerplay\n(1–6)':ACCENT,'Middle\n(7–15)':AMBER,'Death\n(16–20)':RED}

    fig, ax = base_fig(10, 5)
    for _, row in ph_rr.iterrows():
        ax.bar(row['phase'], row['run_rate'], color=cmap.get(row['phase'],ACCENT),
               alpha=0.9, width=0.42, zorder=3)
        ax.text(row['phase'], row['run_rate']+0.07, f"{row['run_rate']:.2f}",
                ha='center', color='#f0f9f6', fontsize=15, fontweight='700')
    ax.set_title('Run Rate by Match Phase', color='#d4ede8', fontsize=13, pad=14)
    ax.set_ylabel('Run Rate (per over)')
    ax.set_ylim(0, max(ph_rr['run_rate'])*1.3)
    patches = [mpatches.Patch(color=ACCENT,label='Powerplay'),
               mpatches.Patch(color=AMBER, label='Middle'),
               mpatches.Patch(color=RED,   label='Death')]
    ax.legend(handles=patches, facecolor=CARD, edgecolor=LINE, labelcolor='#d4ede8', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="sec-hd">Boundaries Per Season</div>', unsafe_allow_html=True)
    dt = deliveries.merge(matches[['id','season']], left_on='match_id', right_on='id', how='left')
    dt['is_four'] = dt['batsman_runs'] == 4
    dt['is_six']  = dt['batsman_runs'] == 6
    bps = dt.groupby('season')[['is_four','is_six']].sum().reset_index()

    fig, ax = base_fig(10, 4.5)
    ax.plot(bps['season'], bps['is_four'], marker='o', markersize=6, color=ACCENT, linewidth=2.5, label='Fours', zorder=3)
    ax.plot(bps['season'], bps['is_six'],  marker='s', markersize=6, color=RED,    linewidth=2.5, label='Sixes', linestyle='--', zorder=3)
    ax.fill_between(bps['season'], bps['is_four'], alpha=0.08, color=ACCENT)
    ax.fill_between(bps['season'], bps['is_six'],  alpha=0.08, color=RED)
    ax.set_title('Fours and Sixes Per Season', color='#d4ede8', fontsize=13, pad=14)
    ax.set_xlabel('Season')
    ax.set_ylabel('Count')
    ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor='#d4ede8', fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════
# BOWLING
# ══════════════════════════════════════════════════════
with tab_bowl:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-hd">Top Wicket Takers</div>', unsafe_allow_html=True)
        wkts = deliveries[(deliveries['is_wicket']==1) & (deliveries['dismissal_kind']!='run out')]
        tw_b = wkts.groupby('bowler')['is_wicket'].count().reset_index()
        tw_b.columns = ['bowler','wickets']
        tw_b = tw_b.sort_values('wickets', ascending=True).tail(15)
        fig, ax = base_fig(6, 6.5)
        ax.barh(tw_b['bowler'], tw_b['wickets'], color=PURPLE, alpha=0.9, height=0.65, zorder=3)
        for bar in ax.patches:
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    str(int(bar.get_width())), va='center', color='#d4ede8', fontsize=9, fontweight='600')
        ax.set_title('Top 15 Wicket Takers', color='#d4ede8', fontsize=13, pad=14)
        ax.set_xlabel('Wickets')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="sec-hd">Dismissal Types</div>', unsafe_allow_html=True)
        dis = deliveries[deliveries['is_wicket']==1]['dismissal_kind'].value_counts().reset_index()
        dis.columns = ['kind','count']
        dis = dis[dis['kind']!='None'].sort_values('count', ascending=True)
        pal = [ACCENT,RED,AMBER,PURPLE,BLUE,'#f97316','#06b6d4','#ec4899','#eab308']
        fig, ax = base_fig(6, 6.5)
        ax.barh(dis['kind'], dis['count'], color=pal[:len(dis)], alpha=0.9, height=0.65, zorder=3)
        for bar in ax.patches:
            ax.text(bar.get_width()+8, bar.get_y()+bar.get_height()/2,
                    f'{int(bar.get_width()):,}', va='center', color='#d4ede8', fontsize=9, fontweight='600')
        ax.set_title('Dismissal Breakdown', color='#d4ede8', fontsize=13, pad=14)
        ax.set_xlabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="sec-hd">Most Economical Bowlers</div>', unsafe_allow_html=True)
    br = deliveries.groupby('bowler')['total_runs'].sum().reset_index()
    br.columns = ['bowler','runs']
    lb = (deliveries[~deliveries['extras_type'].isin(['wides','noballs'])]
          .groupby('bowler')['ball'].count().reset_index())
    lb.columns = ['bowler','balls']
    be = br.merge(lb, on='bowler')
    be['economy'] = (be['runs'] / (be['balls']/6)).round(2)
    be = be[be['balls']>=300].sort_values('economy').head(12)

    fig, ax = base_fig(10, 5.5)
    ax.barh(be['bowler'][::-1], be['economy'][::-1], color=ACCENT, alpha=0.9, height=0.65, zorder=3)
    for bar in ax.patches:
        ax.text(bar.get_width()+0.06, bar.get_y()+bar.get_height()/2,
                f'{bar.get_width():.2f}', va='center', color='#d4ede8', fontsize=9, fontweight='600')
    ax.set_title('Most Economical Bowlers (min 300 legal balls)', color='#d4ede8', fontsize=13, pad=14)
    ax.set_xlabel('Economy Rate')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════
# ADVANCED
# ══════════════════════════════════════════════════════
with tab_adv:
    st.markdown('<div class="sec-hd">Chasing Win % Over Seasons</div>', unsafe_allow_html=True)
    valid = matches[matches['winner']!='No Result'].copy()

    def res_type(row):
        if row['toss_decision']=='field':
            return 'defend' if row['toss_winner']==row['winner'] else 'chase'
        else:
            return 'defend' if row['toss_winner']==row['winner'] else 'chase'

    valid['rtype'] = valid.apply(res_type, axis=1)
    cdp = valid.groupby(['season','rtype'])['id'].count().reset_index()
    cdp = cdp.pivot(index='season', columns='rtype', values='id').fillna(0)

    if 'chase' in cdp.columns and 'defend' in cdp.columns:
        cdp['pct'] = (cdp['chase']/(cdp['chase']+cdp['defend'])*100).round(1)
        fig, ax = base_fig(10, 4.5)
        ax.plot(cdp.index, cdp['pct'], marker='o', markersize=6, color=ACCENT, linewidth=2.5, zorder=3)
        ax.axhline(50, color=MUTED, linewidth=1.2, linestyle='--')
        ax.fill_between(cdp.index, cdp['pct'], 50, where=cdp['pct']>=50, alpha=0.12, color=ACCENT)
        ax.fill_between(cdp.index, cdp['pct'], 50, where=cdp['pct']<50,  alpha=0.12, color=RED)
        ax.set_title('Chasing Win % Per Season', color='#d4ede8', fontsize=13, pad=14)
        ax.set_xlabel('Season')
        ax.set_ylabel('Chase Win %')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="sec-hd">Head to Head</div>', unsafe_allow_html=True)
    active_list = sorted(ACTIVE_TEAMS)
    c1, c2 = st.columns(2)
    with c1:
        ht1 = st.selectbox('Team A', active_list, key='ht1')
    with c2:
        ht2 = st.selectbox('Team B', [t for t in active_list if t!=ht1], key='ht2')

    h2h = matches[
        ((matches['team1']==ht1)&(matches['team2']==ht2))|
        ((matches['team1']==ht2)&(matches['team2']==ht1))
    ]
    w1 = len(h2h[h2h['winner']==ht1])
    w2 = len(h2h[h2h['winner']==ht2])
    tot = len(h2h)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num">{tot}</div><div class="h2h-lbl">Total Matches</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#00c896">{w1}</div><div class="h2h-lbl">{ht1}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#f5a623">{w2}</div><div class="h2h-lbl">{ht2}</div></div>', unsafe_allow_html=True)

    if tot > 0:
        p1 = w1/tot*100; p2 = w2/tot*100
        fig, ax = base_fig(10, 2.8)
        ax.set_xlim(0,100); ax.set_ylim(0,1)
        ax.barh(0.5, p1, height=0.45, color=ACCENT, alpha=0.9, left=0)
        ax.barh(0.5, p2, height=0.45, color=AMBER,  alpha=0.9, left=p1)
        if p1>10:
            ax.text(p1/2, 0.5, f'{ht1}\n{p1:.1f}%', ha='center', va='center', color='#060d0d', fontsize=10, fontweight='700')
        if p2>10:
            ax.text(p1+p2/2, 0.5, f'{ht2}\n{p2:.1f}%', ha='center', va='center', color='#060d0d', fontsize=10, fontweight='700')
        ax.axis('off')
        ax.set_facecolor(CARD)
        fig.patch.set_facecolor(BG)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="sec-hd">Player of the Match Leaders</div>', unsafe_allow_html=True)
    pom = matches[matches['player_of_match']!='No Award']['player_of_match'].value_counts().head(15).reset_index()
    pom.columns = ['player','awards']
    pom = pom.sort_values('awards', ascending=True)
    fig, ax = base_fig(10, 6)
    bars = ax.barh(pom['player'], pom['awards'], color=AMBER, alpha=0.9, height=0.65, zorder=3)
    for bar in bars:
        ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                str(int(bar.get_width())), va='center', color='#d4ede8', fontsize=11, fontweight='600')
    ax.set_title('Most Player of the Match Awards', color='#d4ede8', fontsize=13, pad=14)
    ax.set_xlabel('Awards')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════
# PAR SCORE
# ══════════════════════════════════════════════════════
with tab_par:
    st.markdown("""
    <div class="par-hero">
        <div class="hero-eyebrow" style="margin-bottom:0.7rem">Ground-specific · IPL 2008–2024</div>
        <div class="pred-hero-title">Par Score Calculator</div>
        <div class="pred-hero-sub">Pick a ground to see what first innings score is typically enough to win</div>
    </div>
    """, unsafe_allow_html=True)

    inn_runs = deliveries.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
    inn_runs.columns = ['match_id','inning','innings_runs']
    first_inn = inn_runs[inn_runs['inning']==1].copy()
    first_inn = first_inn.merge(
        matches[['id','venue','winner','team1','team2','season']],
        left_on='match_id', right_on='id', how='left'
    )
    first_inn = first_inn[first_inn['winner']!='No Result'].dropna(subset=['venue'])
    first_inn['bat_first_won'] = first_inn['winner'] == first_inn['team1']

    venue_map = {
        'M Chinnaswamy Stadium':'M Chinnaswamy Stadium, Bengaluru',
        'M. Chinnaswamy Stadium':'M Chinnaswamy Stadium, Bengaluru',
        'Wankhede Stadium':'Wankhede Stadium, Mumbai',
        'Eden Gardens':'Eden Gardens, Kolkata',
        'Feroz Shah Kotla':'Arun Jaitley Stadium, Delhi',
        'Arun Jaitley Stadium':'Arun Jaitley Stadium, Delhi',
        'MA Chidambaram Stadium':'MA Chidambaram Stadium, Chennai',
        'MA Chidambaram Stadium, Chepauk':'MA Chidambaram Stadium, Chennai',
        'Rajiv Gandhi International Stadium':'Rajiv Gandhi Stadium, Hyderabad',
        'Rajiv Gandhi International Stadium, Uppal':'Rajiv Gandhi Stadium, Hyderabad',
        'Punjab Cricket Association Stadium':'PCA Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali':'PCA Stadium, Mohali',
        'Sawai Mansingh Stadium':'Sawai Mansingh Stadium, Jaipur',
        'Dr DY Patil Sports Academy':'DY Patil Stadium, Mumbai',
        'DY Patil Stadium':'DY Patil Stadium, Mumbai',
        'Narendra Modi Stadium':'Narendra Modi Stadium, Ahmedabad',
        'Sardar Patel Stadium, Motera':'Narendra Modi Stadium, Ahmedabad',
        'Holkar Cricket Stadium':'Holkar Stadium, Indore',
        'Maharashtra Cricket Association Stadium':'MCA Stadium, Pune',
        'Brabourne Stadium':'Brabourne Stadium, Mumbai',
        'Himachal Pradesh Cricket Association Stadium':'HPCA Stadium, Dharamsala',
        'Himachal Pradesh Cricket Association Stadium, Dharamsala':'HPCA Stadium, Dharamsala',
        'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium':'Ekana Stadium, Lucknow',
        'Ekana Cricket Stadium':'Ekana Stadium, Lucknow',
        'Vidarbha Cricket Association Stadium, Jamtha':'VCA Stadium, Nagpur',
    }
    first_inn['venue'] = first_inn['venue'].replace(venue_map)
    vc = first_inn['venue'].value_counts()
    valid_venues = sorted(vc[vc>=10].index.tolist())

    sel_venue = st.selectbox('Select Ground', valid_venues)
    vd = first_inn[first_inn['venue']==sel_venue].copy()

    if len(vd) < 5:
        st.warning('Not enough matches at this ground yet.')
    else:
        avg_sc   = vd['innings_runs'].mean()
        med_sc   = vd['innings_runs'].median()
        high_sc  = vd['innings_runs'].max()
        bf_wins  = vd['bat_first_won'].mean()*100
        n_matches = len(vd)

        par_score = None
        for lo in range(100, 280, 20):
            band = vd[(vd['innings_runs']>=lo)&(vd['innings_runs']<lo+20)]
            if len(band)>=2 and band['bat_first_won'].mean()>=0.50:
                par_score = lo; break
        if par_score is None:
            for lo in range(100, 280, 5):
                band = vd[vd['innings_runs']>=lo]
                if len(band)>=3 and band['bat_first_won'].mean()>=0.50:
                    par_score = lo; break
        if par_score is None:
            par_score = int(avg_sc)

        st.markdown(f"""
        <div class="par-stat-grid">
            <div class="par-stat blue">
                <div class="par-stat-num">{avg_sc:.0f}</div>
                <div class="par-stat-lbl">Avg Score</div>
            </div>
            <div class="par-stat amber">
                <div class="par-stat-num">{med_sc:.0f}</div>
                <div class="par-stat-lbl">Median Score</div>
            </div>
            <div class="par-stat red">
                <div class="par-stat-num">{high_sc}</div>
                <div class="par-stat-lbl">Highest Score</div>
            </div>
            <div class="par-stat green">
                <div class="par-stat-num">{bf_wins:.0f}%</div>
                <div class="par-stat-lbl">Bat First Wins</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        scores_arr = vd['innings_runs'].values
        min_sc = max(80, int(scores_arr.min()//20)*20)
        max_sc = int(scores_arr.max()//20)*20+20
        band_starts = list(range(min_sc, max_sc, 20))

        st.markdown('<div class="sec-hd">Score vs Win Rate Breakdown</div>', unsafe_allow_html=True)
        st.caption('Win rate of team batting first from each scoring range. Green = usually wins, Amber = 50/50, Red = usually loses.')

        rows_html = ''
        for i, lo in enumerate(band_starts):
            hi = lo+20
            if i == len(band_starts)-1:
                band = vd[vd['innings_runs']>=lo]; lbl=f'{lo}+'
            else:
                band = vd[(vd['innings_runs']>=lo)&(vd['innings_runs']<hi)]; lbl=f'{lo}–{hi-1}'
            if len(band)<2: continue
            wr = band['bat_first_won'].mean()*100
            colour = ACCENT if wr>=60 else AMBER if wr>=45 else RED
            verdict = 'Usually wins' if wr>=60 else 'Roughly 50/50' if wr>=45 else 'Usually loses'
            bw = min(int(wr),100)
            rows_html += f"""
            <div class="win-row">
                <span class="win-score">{lbl} runs</span>
                <div class="win-bar-track"><div class="win-bar-fill" style="width:{bw}%;background:{colour}"></div></div>
                <span class="win-pct" style="color:{colour}">{wr:.0f}%</span>
                <span style="color:{colour};font-size:0.78rem;font-weight:600;min-width:110px;text-align:right">{verdict}</span>
                <span style="color:#4a7a70;font-size:0.74rem;min-width:60px;text-align:right">{len(band)} matches</span>
            </div>"""

        st.markdown(f'<div class="win-table">{rows_html}</div>', unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#152a26;font-size:0.76rem;padding:3rem 0 1.5rem;letter-spacing:1px;">
    IPL Intelligence &nbsp;·&nbsp; Built by Aadit Jain &nbsp;·&nbsp; Data: 2008–2024
</div>
""", unsafe_allow_html=True)
