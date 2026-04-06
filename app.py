import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Intelligence",
    page_icon="🏏",
    layout="wide"
)

# ─────────────────────────────────────────────
# RESPONSIVE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
body { background:#080c14; color:white; }

/* Responsive grid */
.stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}

/* Cards */
.stat-card {
    background:#0e1420;
    padding:1rem;
    border-radius:12px;
    text-align:center;
}

/* Responsive text */
.hero-title {
    font-size: clamp(1.8rem, 5vw, 3rem);
    text-align:center;
}

/* Mobile */
@media (max-width:768px){
    .hero-title { font-size:1.8rem; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING (OPTIMIZED)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists("matches.csv"):
        with zipfile.ZipFile("matches.zip", "r") as z:
            z.extractall()
    if not os.path.exists("deliveries.csv"):
        with zipfile.ZipFile("deliveries.zip", "r") as z:
            z.extractall()

    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    matches['date'] = pd.to_datetime(matches['date'])
    matches['season'] = matches['date'].dt.year

    matches['city'] = matches['city'].fillna("Unknown")
    matches['winner'] = matches['winner'].fillna("No Result")

    return matches, deliveries

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(matches):
    df = matches[['season','city','team1','team2','toss_winner','toss_decision','winner']].dropna()

    df['result'] = (df['winner'] == df['team1']).astype(int)
    df = df.drop(columns=['winner'])

    encoders = {}
    for col in ['city','team1','team2','toss_winner','toss_decision']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=['result'])
    y = df['result']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, encoders

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
matches, deliveries = load_data()
model, encoders = train_model(matches)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🏏 IPL Intelligence Dashboard</h1>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
total_runs = int(deliveries['total_runs'].sum())
total_matches = len(matches)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <h2>{total_matches}</h2>
        <p>Matches</p>
    </div>
    <div class="stat-card">
        <h2>{total_runs}</h2>
        <p>Total Runs</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Analysis", "🎯 Predictor"])

# ─────────────────────────────────────────────
# ANALYSIS TAB
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Matches Per Season")

    mps = matches.groupby('season')['id'].count()

    fig, ax = plt.subplots()
    ax.plot(mps.index, mps.values, marker='o')
    st.pyplot(fig)

    st.subheader("Top Scorers")
    top = deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots()
    top.plot(kind='bar', ax=ax)
    st.pyplot(fig)

# ─────────────────────────────────────────────
# PREDICTOR TAB
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Match Winner Predictor")

    teams = sorted(encoders['team1'].classes_)
    cities = sorted(encoders['city'].classes_)

    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        city = st.selectbox("City", cities)
    with col2:
        team1 = st.selectbox("Team 1", teams)
    with col3:
        team2 = st.selectbox("Team 2", [t for t in teams if t != team1])

    if st.button("Predict"):
        try:
            def encode(col, val):
                return encoders[col].transform([val])[0]

            input_df = pd.DataFrame([{
                'season': 2024,
                'city': encode('city', city),
                'team1': encode('team1', team1),
                'team2': encode('team2', team2),
                'toss_winner': encode('toss_winner', team1),
                'toss_decision': encode('toss_decision', 'field')
            }])

            prob = model.predict_proba(input_df)[0]

            win1 = prob[1] * 100
            win2 = prob[0] * 100

            winner = team1 if win1 > win2 else team2

            st.success(f"🏆 Predicted Winner: {winner}")
            st.write(f"{team1}: {win1:.2f}%")
            st.write(f"{team2}: {win2:.2f}%")

        except Exception as e:
            st.error(str(e))

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("### ⚠️ Model accuracy ~50–55% (T20 is unpredictable!)")
