import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Colour Palette ────────────────────────────────────────────
BG      = '#080c14'
CARD    = '#0e1420'
ACCENT  = '#3b82f6'
ACCENT2 = '#06b6d4'
CREAM   = '#f0e6d3'
RED     = '#ef4444'
GREEN   = '#10b981'
MUTED   = '#64748b'
LINE    = '#1e293b'

# ─── Premium CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #080c14;
    color: #e2e8f0;
}
.stApp { background: #080c14; }

/* ── Responsive containers ── */
.block-container {
    max-width: 1200px !important;
    padding: 1rem 1rem 3rem !important;
}
@media (max-width: 768px) {
    .block-container { padding: 0.5rem 0.5rem 2rem !important; }
}

/* ── Hero ── */
.hero {
    padding: 2.5rem 1rem 1.5rem;
    text-align: center;
    position: relative;
    margin-bottom: 1rem;
}
.hero::after {
    content: '';
    display: block;
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    margin: 1.2rem auto 0;
}
.hero-tag {
    font-size: clamp(0.82rem, 2.2vw, 0.72rem);
    font-weight: 500;
    letter-spacing: 4px;
    color: #3b82f6;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 6vw, 3.2rem);
    font-weight: 400;
    color: #f0e6d3;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: clamp(0.92rem, 2vw, 0.9rem);
    color: #64748b;
    margin-top: 0.8rem;
    letter-spacing: 1px;
}

/* ── Stat Cards ── */
.stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 2rem;
}
.stat-card {
    flex: 1 1 150px;
    background: #0e1420;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    min-width: 120px;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #3b82f6, transparent);
    border-radius: 14px 0 0 14px;
}
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.6rem, 3vw, 2rem);
    color: #f0e6d3;
    line-height: 1;
    margin-bottom: 0.3rem;
    word-break: break-word;
}
.stat-label {
    font-size: clamp(0.75rem, 1.8vw, 0.82rem);
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}

/* ── Section Titles ── */
.sec-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.3rem, 3vw, 1.5rem);
    color: #f0e6d3;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e293b, transparent);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0e1420;
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
    border: 1px solid #1e293b;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(0.82rem, 2vw, 0.85rem);
    font-weight: 500;
    border-radius: 8px;
    padding: 7px 14px;
    letter-spacing: 0.3px;
    transition: all 0.2s;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    background: #111827 !important;
    color: #3b82f6 !important;
    border-bottom: none !important;
}

/* ── Selectbox ── */
.stSelectbox label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background: #0e1420 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Predict Hero Banner ── */
.predict-hero {
    background: linear-gradient(135deg, #0e1420 0%, #0f172a 50%, #0e1420 100%);
    border: 1px solid #1e40af;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.predict-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #3b82f6, #06b6d4, transparent);
}
.predict-hero::after {
    content: '🏏';
    position: absolute;
    font-size: 8rem;
    opacity: 0.04;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
}
.predict-hero-tag {
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: #3b82f6;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.predict-hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.9rem, 4vw, 2.2rem);
    color: #f0e6d3;
    margin: 0.3rem 0;
}
.predict-hero-sub {
    font-size: clamp(0.9rem, 2vw, 0.85rem);
    color: #64748b;
    margin-top: 0.4rem;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: #ffffff;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1.5rem;
    width: 100%;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.45);
}

/* ── Prediction Result ── */
.result-wrap {
    background: linear-gradient(135deg, #0e1420, #0f172a);
    border: 1px solid #1e40af;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #3b82f6, #06b6d4, transparent);
}
.result-label {
    font-size: 0.72rem;
    letter-spacing: 3px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.result-team {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 4vw, 2.6rem);
    color: #3b82f6;
    margin: 0.3rem 0;
    text-shadow: 0 0 40px rgba(59,130,246,0.4);
}
.result-conf {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.4rem;
}
.prob-bar-wrap {
    margin-top: 1.5rem;
    background: #080c14;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    margin-bottom: 0.5rem;
    color: #94a3b8;
}
.prob-bar-outer {
    width: 100%;
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.prob-bar-inner {
    height: 100%;
    background: linear-gradient(90deg, #1d4ed8, #3b82f6);
    border-radius: 4px;
}

/* ── Disclaimer ── */
.disc {
    background: #0e1420;
    border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* ── H2H Card ── */
.h2h-card {
    background: #0e1420;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.h2h-num {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.7rem, 3vw, 2rem);
    color: #3b82f6;
    line-height: 1;
}
.h2h-lbl {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
    word-break: break-word;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }

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

    matches = pd.read_csv(matches_name)
    deliveries = pd.read_csv(deliveries_name)

    if 'date' not in matches.columns:
        matches, deliveries = deliveries, matches

    matches['date'] = pd.to_datetime(matches['date'])
    matches['season'] = matches['season'].apply(lambda x: int(str(x)[:4]))

    team_map = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
    }
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        matches[col] = matches[col].replace(team_map)
    for col in ['batting_team', 'bowling_team']:
        deliveries[col] = deliveries[col].replace(team_map)

    matches['city'] = matches['city'].fillna(
        matches['venue'].str.split(',').str[-1].str.strip()
    )
    matches['winner'] = matches['winner'].fillna('No Result')
    matches['player_of_match'] = matches['player_of_match'].fillna('No Award')
    for c in ['extras_type', 'player_dismissed', 'dismissal_kind', 'fielder']:
        deliveries[c] = deliveries[c].fillna('None')

    df = deliveries.merge(matches, left_on='match_id', right_on='id', how='left')
    return matches, deliveries, df


@st.cache_resource(show_spinner=False)
def train_model(_matches):
    encoders = {}
    cols = ['city', 'team1', 'team2', 'toss_winner', 'toss_decision']
    raw = _matches[
        ['season', 'city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']
    ].copy().dropna()
    raw['result'] = (raw['winner'] == raw['team1']).astype(int)
    raw = raw.drop(columns=['winner'])
    for col in cols:
        le = LabelEncoder()
        raw[col] = le.fit_transform(raw[col])
        encoders[col] = le
    X, y = raw.drop(columns=['result']), raw['result']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(X_tr, y_tr)
    return m, encoders


# ─── Plot helpers ──────────────────────────────────────────────
def base_fig(w=10, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    for sp in ax.spines.values():
        sp.set_edgecolor(LINE)
    ax.grid(axis='y', color=LINE, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


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
    <div class="hero-tag">Indian Premier League · 2008 – 2024</div>
    <h1 class="hero-title">IPL Intelligence</h1>
    <p class="hero-sub">17 seasons of data · deep analytics · match predictor</p>
</div>
""", unsafe_allow_html=True)

# ─── Top stats ─────────────────────────────────────────────────
total_runs  = int(deliveries['batsman_runs'].sum())
top_scorer  = deliveries.groupby('batter')['batsman_runs'].sum().idxmax()
total_sixes = int((deliveries['batsman_runs'] == 6).sum())

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-num">{len(matches):,}</div>
        <div class="stat-label">Matches Played</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{deliveries.shape[0]:,}</div>
        <div class="stat-label">Balls Bowled</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{total_runs:,}</div>
        <div class="stat-label">Total Runs</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{total_sixes:,}</div>
        <div class="stat-label">Total Sixes</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="font-size:clamp(0.9rem,2vw,1.2rem);padding-top:0.4rem">{top_scorer}</div>
        <div class="stat-label">All-time Top Scorer</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab5, tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict", "📊 Match Analysis", "🏏 Batting", "🎳 Bowling", "🔍 Advanced"
])


# ══════════════════════════════════════════
# TAB — PREDICT
# ══════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="predict-hero">
        <div class="predict-hero-tag">AI-Powered · Historical Data 2008–2024</div>
        <div class="predict-hero-title">Match Winner Predictor</div>
        <div class="predict-hero-sub">Select teams and venue to get an instant win probability forecast</div>
    </div>
    """, unsafe_allow_html=True)

    teams_list  = sorted(encoders['team1'].classes_)
    cities_list = sorted(encoders['city'].classes_)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        city_sel = st.selectbox('🏟️  Venue City', cities_list)
    with col2:
        t1_sel = st.selectbox('🔵  Team 1', teams_list)
    with col3:
        t2_options = [t for t in teams_list if t != t1_sel]
        t2_sel = st.selectbox('🔴  Team 2', t2_options)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button('⚡  Predict Match Winner'):
        if t1_sel == t2_sel:
            st.error("Please select two different teams.")
        else:
            try:
                team_wins   = matches[matches['winner'] != 'No Result']['winner'].value_counts()
                team_played = pd.concat([matches['team1'], matches['team2']]).value_counts()
                win_pct     = (team_wins / team_played * 100).fillna(0)

                t1_win_pct = win_pct.get(t1_sel, 50)
                t2_win_pct = win_pct.get(t2_sel, 50)

                if t1_win_pct >= t2_win_pct:
                    model_team1, model_team2 = t1_sel, t2_sel
                else:
                    model_team1, model_team2 = t2_sel, t1_sel

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

                avg_model_team1 = (p1[1] + p2[1]) / 2
                avg_model_team2 = (p1[0] + p2[0]) / 2

                if model_team1 == t1_sel:
                    avg1, avg2 = avg_model_team1, avg_model_team2
                else:
                    avg1, avg2 = avg_model_team2, avg_model_team1

                winner = t1_sel if avg1 > avg2 else t2_sel
                conf   = max(avg1, avg2) * 100
                pct1   = round(avg1 * 100, 1)
                pct2   = round(avg2 * 100, 1)

                h2h = matches[
                    ((matches['team1'] == t1_sel) & (matches['team2'] == t2_sel)) |
                    ((matches['team1'] == t2_sel) & (matches['team2'] == t1_sel))
                ]
                h2h_total = len(h2h)
                h2h_w1    = len(h2h[h2h['winner'] == t1_sel])
                h2h_w2    = len(h2h[h2h['winner'] == t2_sel])

                st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-label">Predicted Winner</div>
                    <div class="result-team">🏆 &nbsp;{winner}</div>
                    <div class="result-conf">Model confidence: {conf:.1f}%</div>
                    <div class="prob-bar-wrap">
                        <div class="prob-row">
                            <span style="color:#3b82f6;font-weight:600">{t1_sel}</span>
                            <span style="color:#f0e6d3;font-weight:600">{pct1}%</span>
                        </div>
                        <div class="prob-bar-outer">
                            <div class="prob-bar-inner" style="width:{pct1}%"></div>
                        </div>
                        <div class="prob-row" style="margin-top:0.8rem">
                            <span style="color:#06b6d4;font-weight:600">{t2_sel}</span>
                            <span style="color:#f0e6d3;font-weight:600">{pct2}%</span>
                        </div>
                        <div class="prob-bar-outer">
                            <div class="prob-bar-inner" style="width:{pct2}%; background: linear-gradient(90deg,#0891b2,#06b6d4)"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if h2h_total > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="sec-title">Head-to-Head History</div>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="h2h-card"><div class="h2h-num">{h2h_total}</div><div class="h2h-lbl">Total Matches</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#3b82f6">{h2h_w1}</div><div class="h2h-lbl">{t1_sel}</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#06b6d4">{h2h_w2}</div><div class="h2h-lbl">{t2_sel}</div></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    st.markdown("""
    <div class="disc" style="margin-top:2rem">
        ⚠️ Predictions are based on historical IPL data (2008–2024) and do not account for
        current player form, injuries, or squad changes. Model accuracy ~52%, reflecting
        the unpredictable nature of T20 cricket.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════
# TAB 1 — MATCH
# ══════════════════════
with tab1:

    # ── Matches Per Season ──
    st.markdown('<div class="sec-title">Matches Per Season</div>', unsafe_allow_html=True)
    mps = matches.groupby('season')['id'].count().reset_index()
    mps.columns = ['season', 'count']
    mps['season'] = mps['season'].astype(int)

    fig, ax = base_fig(10, 4)
    bars = ax.bar(mps['season'], mps['count'], color=ACCENT, alpha=0.85, width=0.6, zorder=3)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(bar.get_height())),
            ha='center', color='#94a3b8', fontsize=8
        )
    ax.set_xlabel('Season')
    ax.set_ylabel('Matches')
    ax.set_title('Number of Matches Per Season', color='#c0c0d0', fontsize=11, pad=12)
    ax.set_xticks(mps['season'])
    ax.set_xticklabels(mps['season'].astype(int), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Toss Analysis ──
    st.markdown('<div class="sec-title">Toss Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # ── FIX: Toss Decision Over Seasons — clearly differentiated bat vs field ──
        tt = matches.groupby(['season', 'toss_decision'])['id'].count().reset_index()
        tt.columns = ['season', 'decision', 'count']

        seasons_sorted = sorted(tt['season'].unique())
        bat_data   = tt[tt['decision'] == 'bat'].set_index('season').reindex(seasons_sorted, fill_value=0)
        field_data = tt[tt['decision'] == 'field'].set_index('season').reindex(seasons_sorted, fill_value=0)

        TOSS_BAT   = '#f59e0b'   # warm amber  — Bat First
        TOSS_FIELD = '#a855f7'   # vivid violet — Field (Bowl)

        fig, ax = base_fig(6, 4.5)
        ax.fill_between(seasons_sorted, bat_data['count'],   alpha=0.18, color=TOSS_BAT)
        ax.fill_between(seasons_sorted, field_data['count'], alpha=0.18, color=TOSS_FIELD)
        ax.plot(seasons_sorted, bat_data['count'],
                marker='o', markersize=6, color=TOSS_BAT,   linewidth=2.5, label='Bat First',    zorder=3)
        ax.plot(seasons_sorted, field_data['count'],
                marker='D', markersize=5, color=TOSS_FIELD, linewidth=2.5, label='Field (Bowl)', zorder=3,
                linestyle='--')

        # Annotate last data point for clarity
        if len(bat_data) and len(field_data):
            last = seasons_sorted[-1]
            ax.annotate('Bat', xy=(last, bat_data.loc[last, 'count']),
                        xytext=(4, 2), textcoords='offset points',
                        color=TOSS_BAT, fontsize=9, fontweight='700')
            ax.annotate('Field', xy=(last, field_data.loc[last, 'count']),
                        xytext=(4, -12), textcoords='offset points',
                        color=TOSS_FIELD, fontsize=9, fontweight='700')

        ax.set_title('Toss Decision Over Seasons', color='#c0c0d0', fontsize=11, pad=12)
        ax.set_xlabel('Season')
        ax.set_ylabel('Number of Teams')
        ax.legend(
            facecolor=CARD, edgecolor=LINE, labelcolor='#c0c0d0',
            fontsize=9, loc='upper left',
            handles=[
                mpatches.Patch(color=TOSS_BAT,   label='Bat First'),
                mpatches.Patch(color=TOSS_FIELD,  label='Field (Bowl)'),
            ]
        )
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # ── Win Rate by Toss Decision ──
        m2 = matches.copy()
        m2['toss_won_match'] = m2['toss_winner'] == m2['winner']
        tdw = (
            m2[m2['winner'] != 'No Result']
            .groupby('toss_decision')['toss_won_match']
            .mean() * 100
        ).reset_index()
        tdw.columns = ['decision', 'win_pct']
        tdw['label'] = tdw['decision'].map({'bat': 'Bat First', 'field': 'Field (Bowl)'})
        tdw['color'] = tdw['decision'].map({'bat': '#f59e0b', 'field': '#a855f7'})

        fig, ax = base_fig(6, 4.5)
        bars = ax.bar(tdw['label'], tdw['win_pct'],
                      color=tdw['color'].tolist(), width=0.4, zorder=3, alpha=0.9)
        for bar, (_, row) in zip(bars, tdw.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f'{row["win_pct"]:.1f}%',
                ha='center', color='#e2e8f0', fontsize=12, fontweight='600'
            )
        ax.set_title('Win Rate When Winning Toss', color='#c0c0d0', fontsize=11, pad=12)
        ax.set_ylabel('Win %')
        ax.set_ylim(0, 75)
        ax.tick_params(axis='x', colors='#94a3b8', labelsize=10)
        patches = [
            mpatches.Patch(color='#f59e0b', label='Bat First'),
            mpatches.Patch(color='#a855f7', label='Field (Bowl)'),
        ]
        ax.legend(handles=patches, facecolor=CARD, edgecolor=LINE,
                  labelcolor='#c0c0d0', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Team Win Percentage ──
    st.markdown('<div class="sec-title">Team Win Percentage</div>', unsafe_allow_html=True)
    tw = matches[matches['winner'] != 'No Result']['winner'].value_counts().reset_index()
    tw.columns = ['team', 'wins']
    tm = pd.concat([matches['team1'], matches['team2']]).value_counts().reset_index()
    tm.columns = ['team', 'played']
    ts = tm.merge(tw, on='team', how='left').fillna(0)
    ts['win_pct'] = (ts['wins'] / ts['played'] * 100).round(1)
    ts = ts[ts['played'] >= 10].sort_values('win_pct')

    fig, ax = base_fig(10, max(5, len(ts) * 0.45))
    bar_colors = [
        ACCENT if v >= 55 else ACCENT2 if v >= 45 else RED
        for v in ts['win_pct']
    ]
    bars = ax.barh(ts['team'], ts['win_pct'],
                   color=bar_colors, alpha=0.85, height=0.6, zorder=3)
    for bar in bars:
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.1f}%',
            va='center', color='#8a8a9a', fontsize=8
        )
    ax.set_title('All-time Win Percentage by Team (min 10 matches)',
                 color='#c0c0d0', fontsize=11, pad=12)
    ax.set_xlabel('Win %')
    ax.set_xlim(0, 80)
    patches = [
        mpatches.Patch(color=ACCENT,  label='> 55%'),
        mpatches.Patch(color=ACCENT2, label='45–55%'),
        mpatches.Patch(color=RED,     label='< 45%'),
    ]
    ax.legend(handles=patches, facecolor=CARD, edgecolor=LINE,
              labelcolor='#c0c0d0', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Super Overs ──
    st.markdown('<div class="sec-title">Super Over Matches</div>', unsafe_allow_html=True)
    so = matches[matches['super_over'] == 'Y']
    if len(so) > 0:
        so_teams = pd.concat([so['team1'], so['team2']]).value_counts().reset_index()
        so_teams.columns = ['team', 'count']
        fig, ax = base_fig(10, 3.5)
        ax.bar(so_teams['team'], so_teams['count'],
               color=ACCENT, alpha=0.85, width=0.6, zorder=3)
        ax.set_title(
            f'Super Over Involvement by Team  ({len(so)} total)',
            color='#c0c0d0', fontsize=11, pad=12
        )
        ax.set_ylabel('Times Involved')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No super over data available.")


# ══════════════════════
# TAB 2 — BATTING
# ══════════════════════
with tab2:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-title">Top Run Scorers</div>', unsafe_allow_html=True)
        ts_bat = deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
        ts_bat.columns = ['batter', 'runs']
        ts_bat = ts_bat.sort_values('runs', ascending=True).tail(15)
        fig, ax = base_fig(6, 6)
        ax.barh(ts_bat['batter'], ts_bat['runs'],
                color=ACCENT, alpha=0.85, height=0.6, zorder=3)
        for bar in ax.patches:
            ax.text(
                bar.get_width() + 20,
                bar.get_y() + bar.get_height() / 2,
                f'{int(bar.get_width()):,}',
                va='center', color='#8a8a9a', fontsize=7
            )
        ax.set_title('Top 15 Run Scorers', color='#c0c0d0', fontsize=11, pad=12)
        ax.set_xlabel('Total Runs')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="sec-title">Top Six Hitters</div>', unsafe_allow_html=True)
        sx = deliveries[deliveries['batsman_runs'] == 6].groupby('batter').size().reset_index()
        sx.columns = ['batter', 'sixes']
        sx = sx.sort_values('sixes', ascending=True).tail(12)
        fig, ax = base_fig(6, 6)
        ax.barh(sx['batter'], sx['sixes'],
                color=ACCENT2, alpha=0.85, height=0.6, zorder=3)
        for bar in ax.patches:
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())),
                va='center', color='#8a8a9a', fontsize=7
            )
        ax.set_title('Top 12 Six Hitters', color='#c0c0d0', fontsize=11, pad=12)
        ax.set_xlabel('Sixes')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Phase-wise Run Rate ──
    st.markdown('<div class="sec-title">Phase-wise Run Rate</div>', unsafe_allow_html=True)

    def get_phase(o):
        if o <= 5:
            return 'Powerplay\n(Overs 1–6)'
        elif o <= 14:
            return 'Middle\n(Overs 7–15)'
        else:
            return 'Death\n(Overs 16–20)'

    d_phase = deliveries.copy()
    d_phase['phase'] = d_phase['over'].apply(get_phase)
    ph_r  = d_phase.groupby('phase')['total_runs'].sum()
    ph_b  = d_phase.groupby('phase')['ball'].count()
    ph_rr = (ph_r / ph_b * 6).round(2).reset_index()
    ph_rr.columns = ['phase', 'run_rate']

    order_map = {
        'Powerplay\n(Overs 1–6)': 0,
        'Middle\n(Overs 7–15)': 1,
        'Death\n(Overs 16–20)': 2
    }
    ph_rr['order'] = ph_rr['phase'].map(order_map)
    ph_rr = ph_rr.sort_values('order')

    c_map = {
        'Powerplay\n(Overs 1–6)': ACCENT,
        'Middle\n(Overs 7–15)': ACCENT2,
        'Death\n(Overs 16–20)': RED
    }

    fig, ax = base_fig(10, 4.5)
    bar_list = []
    for _, row in ph_rr.iterrows():
        b = ax.bar(row['phase'], row['run_rate'],
                   color=c_map.get(row['phase'], ACCENT),
                   alpha=0.85, width=0.4, zorder=3)
        bar_list.append(b)
        ax.text(
            row['phase'], row['run_rate'] + 0.06,
            f"{row['run_rate']:.2f}",
            ha='center', color='#e2e8f0', fontsize=13, fontweight='600'
        )

    ax.set_title('Run Rate by Match Phase', color='#c0c0d0', fontsize=11, pad=12)
    ax.set_ylabel('Run Rate (per over)')
    ax.set_ylim(0, max(ph_rr['run_rate']) * 1.25)
    ax.tick_params(axis='x', labelsize=9)
    patches = [
        mpatches.Patch(color=ACCENT,  label='Powerplay (1–6)'),
        mpatches.Patch(color=ACCENT2, label='Middle (7–15)'),
        mpatches.Patch(color=RED,     label='Death (16–20)'),
    ]
    ax.legend(handles=patches, facecolor=CARD, edgecolor=LINE,
              labelcolor='#c0c0d0', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Scoring Trends Per Season ──
    st.markdown('<div class="sec-title">Scoring Trends Per Season</div>', unsafe_allow_html=True)
    df_trend = deliveries.merge(matches[['id', 'season']], left_on='match_id', right_on='id', how='left')
    df_trend['is_four'] = df_trend['batsman_runs'] == 4
    df_trend['is_six']  = df_trend['batsman_runs'] == 6
    bps = df_trend.groupby('season')[['is_four', 'is_six']].sum().reset_index()

    fig, ax = base_fig(10, 4)
    ax.plot(bps['season'], bps['is_four'],
            marker='o', markersize=4, color=ACCENT,  linewidth=2, label='Fours', zorder=3)
    ax.plot(bps['season'], bps['is_six'],
            marker='s', markersize=4, color=ACCENT2, linewidth=2, label='Sixes',
            linestyle='--', zorder=3)
    ax.fill_between(bps['season'], bps['is_four'], alpha=0.06, color=ACCENT)
    ax.fill_between(bps['season'], bps['is_six'],  alpha=0.06, color=ACCENT2)
    ax.set_title('Fours and Sixes Per Season', color='#c0c0d0', fontsize=11, pad=12)
    ax.set_xlabel('Season')
    ax.set_ylabel('Count')
    ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor='#c0c0d0', fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════
# TAB 3 — BOWLING
# ══════════════════════
with tab3:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-title">Top Wicket Takers</div>', unsafe_allow_html=True)
        wkts = deliveries[
            (deliveries['is_wicket'] == 1) &
            (deliveries['dismissal_kind'] != 'run out')
        ]
        tw_bowl = wkts.groupby('bowler')['is_wicket'].count().reset_index()
        tw_bowl.columns = ['bowler', 'wickets']
        tw_bowl = tw_bowl.sort_values('wickets', ascending=True).tail(15)

        fig, ax = base_fig(6, 6)
        ax.barh(tw_bowl['bowler'], tw_bowl['wickets'],
                color=RED, alpha=0.85, height=0.6, zorder=3)
        for bar in ax.patches:
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())),
                va='center', color='#8a8a9a', fontsize=7
            )
        ax.set_title('Top 15 Wicket Takers', color='#c0c0d0', fontsize=11, pad=12)
        ax.set_xlabel('Wickets')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="sec-title">Dismissal Types</div>', unsafe_allow_html=True)
        dis = deliveries[deliveries['is_wicket'] == 1]['dismissal_kind'].value_counts().reset_index()
        dis.columns = ['kind', 'count']
        dis = dis[dis['kind'] != 'None'].sort_values('count', ascending=True)

        pal = [ACCENT, ACCENT2, RED, GREEN, '#8b5cf6', '#f97316', '#14b8a6', '#ec4899', '#eab308']
        colors_used = pal[:len(dis)]

        fig, ax = base_fig(6, 6)
        ax.barh(dis['kind'], dis['count'],
                color=colors_used, alpha=0.85, height=0.6, zorder=3)
        for bar in ax.patches:
            ax.text(
                bar.get_width() + 5,
                bar.get_y() + bar.get_height() / 2,
                f'{int(bar.get_width()):,}',
                va='center', color='#8a8a9a', fontsize=7
            )
        ax.set_title('Dismissal Breakdown', color='#c0c0d0', fontsize=11, pad=12)
        ax.set_xlabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Most Economical Bowlers ──
    st.markdown('<div class="sec-title">Most Economical Bowlers</div>', unsafe_allow_html=True)
    br = deliveries.groupby('bowler')['total_runs'].sum().reset_index()
    br.columns = ['bowler', 'runs']
    lb = (
        deliveries[~deliveries['extras_type'].isin(['wides', 'noballs'])]
        .groupby('bowler')['ball'].count().reset_index()
    )
    lb.columns = ['bowler', 'balls']
    be = br.merge(lb, on='bowler')
    be['overs']   = be['balls'] / 6
    be['economy'] = (be['runs'] / be['overs']).round(2)
    be = be[be['balls'] >= 300].sort_values('economy').head(12)

    fig, ax = base_fig(10, 5)
    ax.barh(be['bowler'][::-1], be['economy'][::-1],
            color=GREEN, alpha=0.85, height=0.6, zorder=3)
    for bar in ax.patches:
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.2f}',
            va='center', color='#8a8a9a', fontsize=8
        )
    ax.set_title('Most Economical Bowlers (min 300 legal balls)',
                 color='#c0c0d0', fontsize=11, pad=12)
    ax.set_xlabel('Economy Rate')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════
# TAB 4 — ADVANCED
# ══════════════════════
with tab4:

    # ── Chasing Win % ──
    st.markdown('<div class="sec-title">Chasing Win % Over Seasons</div>', unsafe_allow_html=True)
    valid = matches[matches['winner'] != 'No Result'].copy()

    def res_type(row):
        if row['toss_decision'] == 'field' and row['toss_winner'] == row['winner']:
            return 'defend'
        elif row['toss_decision'] == 'bat' and row['toss_winner'] != row['winner']:
            return 'chase'
        elif row['toss_decision'] == 'field' and row['toss_winner'] != row['winner']:
            return 'chase'
        else:
            return 'defend'

    valid['rtype'] = valid.apply(res_type, axis=1)
    cdp = valid.groupby(['season', 'rtype'])['id'].count().reset_index()
    cdp = cdp.pivot(index='season', columns='rtype', values='id').fillna(0)

    if 'chase' in cdp.columns and 'defend' in cdp.columns:
        cdp['pct'] = (cdp['chase'] / (cdp['chase'] + cdp['defend']) * 100).round(1)
        fig, ax = base_fig(10, 4)
        ax.plot(cdp.index, cdp['pct'],
                marker='o', markersize=5, color=ACCENT, linewidth=2.2, zorder=3)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle='--', label='50% line')
        ax.fill_between(cdp.index, cdp['pct'], 50,
                        where=cdp['pct'] >= 50, alpha=0.12, color=ACCENT,  label='Chasing favoured')
        ax.fill_between(cdp.index, cdp['pct'], 50,
                        where=cdp['pct'] < 50,  alpha=0.12, color=RED,    label='Defending favoured')
        ax.set_title('Chasing Win % Per Season', color='#c0c0d0', fontsize=11, pad=12)
        ax.set_xlabel('Season')
        ax.set_ylabel('Chase Win %')
        ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor='#c0c0d0', fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Head to Head ──
    st.markdown('<div class="sec-title">Head to Head</div>', unsafe_allow_html=True)
    all_teams = sorted(matches['team1'].unique())
    col1, col2 = st.columns(2)
    with col1:
        ht1 = st.selectbox('Team A', all_teams, key='ht1')
    with col2:
        ht2 = st.selectbox('Team B', [t for t in all_teams if t != ht1], key='ht2')

    h2h = matches[
        ((matches['team1'] == ht1) & (matches['team2'] == ht2)) |
        ((matches['team1'] == ht2) & (matches['team2'] == ht1))
    ]
    w1  = len(h2h[h2h['winner'] == ht1])
    w2  = len(h2h[h2h['winner'] == ht2])
    tot = len(h2h)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num">{tot}</div><div class="h2h-lbl">Total Matches</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#3b82f6">{w1}</div><div class="h2h-lbl">{ht1}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="h2h-card"><div class="h2h-num" style="color:#06b6d4">{w2}</div><div class="h2h-lbl">{ht2}</div></div>', unsafe_allow_html=True)

    if tot > 0:
        pct1 = w1 / tot * 100
        pct2 = w2 / tot * 100
        fig, ax = base_fig(10, 2.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.barh(0.5, pct1, height=0.4, color=ACCENT,  alpha=0.9, left=0)
        ax.barh(0.5, pct2, height=0.4, color=ACCENT2, alpha=0.9, left=pct1)
        if pct1 > 10:
            ax.text(pct1 / 2,       0.5, f'{ht1}\n{pct1:.1f}%',
                    ha='center', va='center', color='#fff', fontsize=8, fontweight='600')
        if pct2 > 10:
            ax.text(pct1 + pct2/2,  0.5, f'{ht2}\n{pct2:.1f}%',
                    ha='center', va='center', color='#fff', fontsize=8, fontweight='600')
        ax.axis('off')
        ax.set_facecolor(CARD)
        fig.patch.set_facecolor(BG)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Player of the Match Leaders ──
    st.markdown('<div class="sec-title">Player of the Match Leaders</div>', unsafe_allow_html=True)
    pom = (
        matches[matches['player_of_match'] != 'No Award']['player_of_match']
        .value_counts().head(15).reset_index()
    )
    pom.columns = ['player', 'awards']
    pom = pom.sort_values('awards', ascending=True)

    fig, ax = base_fig(10, 5.5)
    bars = ax.barh(pom['player'], pom['awards'],
                   color=ACCENT, alpha=0.85, height=0.6, zorder=3)
    for bar in bars:
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())),
            va='center', color='#8a8a9a', fontsize=8
        )
    ax.set_title('Most Player of the Match Awards', color='#c0c0d0', fontsize=11, pad=12)
    ax.set_xlabel('Awards')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#1e293b;font-size:0.75rem;padding:3rem 0 1.5rem;letter-spacing:1px;">
    IPL Intelligence &nbsp;·&nbsp; Data: 2008–2024 &nbsp;·&nbsp; Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)
