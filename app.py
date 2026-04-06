import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Intelligence",
    page_icon="🏏",
    layout="wide"
)

# ─── Colours ───────────────────────────────────────────────────
BG      = '#06090f'
CARD    = '#0d1321'
NAVY    = '#0f1a2e'
ACCENT  = '#2563eb'
ACCENT2 = '#0ea5e9'
TEAL    = '#14b8a6'
WHITE   = '#f8fafc'
MUTED   = '#475569'
LINE    = '#1e2d45'
RED     = '#f43f5e'
GREEN   = '#10b981'

# ─── FIXED Plotly Theme ────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family='DM Sans', color='#94a3b8', size=11),
    margin=dict(l=20, r=20, t=50, b=20),

    xaxis=dict(
        gridcolor=LINE,
        linecolor=LINE,
        tickcolor='#334155',
        showgrid=False,
        zeroline=False
    ),

    yaxis=dict(
        gridcolor=LINE,
        linecolor='transparent',
        tickcolor='#334155',
        showgrid=True,
        zeroline=False
    ),

    hoverlabel=dict(
        bgcolor=NAVY,
        bordercolor=ACCENT,
        font=dict(color=WHITE, family='DM Sans', size=12)
    ),

    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='transparent',
        font=dict(color='#94a3b8')
    ),
)

# ─── Apply Theme ───────────────────────────────────────────────
def apply_theme(fig, title='', height=400):
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=title,
            font=dict(color='#94a3b8', size=13, family='DM Sans'),
            x=0
        ),
        height=height
    )
    return fig

# ─── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    matches['date'] = pd.to_datetime(matches['date'])
    matches['season'] = matches['season'].astype(str).str[:4].astype(int)

    df = deliveries.merge(matches, left_on='match_id', right_on='id')
    return matches, deliveries, df

# ─── Train Model ───────────────────────────────────────────────
@st.cache_resource
def train_model(matches):
    encoders = {}
    cols = ['city', 'team1', 'team2', 'toss_winner', 'toss_decision']

    raw = matches[cols + ['season', 'winner']].dropna().copy()
    raw['result'] = (raw['winner'] == raw['team1']).astype(int)
    raw = raw.drop(columns=['winner'])

    for col in cols:
        le = LabelEncoder()
        raw[col] = le.fit_transform(raw[col])
        encoders[col] = le

    X = raw.drop(columns=['result'])
    y = raw['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, encoders

# ─── Load ─────────────────────────────────────────────────────
matches, deliveries, df = load_data()
model, encoders = train_model(matches)

# ─── UI ───────────────────────────────────────────────────────
st.title("🏏 IPL Intelligence")

teams = sorted(encoders['team1'].classes_)
cities = sorted(encoders['city'].classes_)

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("City", cities)

with col2:
    team1 = st.selectbox("Team 1", teams)

with col3:
    team2 = st.selectbox("Team 2", [t for t in teams if t != team1])

# ─── Prediction ────────────────────────────────────────────────
if st.button("Predict Winner"):
    try:
        def make_input(toss_winner):
            return pd.DataFrame([{
                'season': 2024,
                'city': encoders['city'].transform([city])[0],
                'team1': encoders['team1'].transform([team1])[0],
                'team2': encoders['team2'].transform([team2])[0],
                'toss_winner': encoders['toss_winner'].transform([toss_winner])[0],
                'toss_decision': encoders['toss_decision'].transform(['field'])[0]
            }])

        p1 = model.predict_proba(make_input(team1))[0]
        p2 = model.predict_proba(make_input(team2))[0]

        prob1 = (p1[1] + p2[1]) / 2
        prob2 = (p1[0] + p2[0]) / 2

        winner = team1 if prob1 > prob2 else team2

        st.success(f"🏆 Predicted Winner: {winner}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob1 * 100,
            number={'suffix': "%"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(str(e))

# ─── Example Chart ─────────────────────────────────────────────
st.subheader("Matches Per Season")

mps = matches.groupby('season')['id'].count().reset_index()

fig = go.Figure(go.Bar(
    x=mps['season'].astype(str),
    y=mps['id']
))

apply_theme(fig, "Matches per Season")

st.plotly_chart(fig, use_container_width=True)
