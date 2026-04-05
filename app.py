import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide"
)

# ─── IPL Theme CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0a0a1a;
        color: #f0f0f0;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #12122a 50%, #0a0a1a 100%);
    }

    .ipl-header {
        background: linear-gradient(90deg, #1a1a3e 0%, #c8a800 50%, #1a1a3e 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 0 40px rgba(200, 168, 0, 0.3);
    }

    .ipl-header h1 {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #0a0a1a;
        margin: 0;
        letter-spacing: 4px;
        text-transform: uppercase;
    }

    .ipl-header p {
        color: #1a1a3e;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
        letter-spacing: 2px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a3e, #12122a);
        border: 1px solid #c8a800;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(200, 168, 0, 0.15);
    }

    .metric-card h2 {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.5rem;
        color: #c8a800;
        margin: 0;
    }

    .metric-card p {
        color: #a0a0c0;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .section-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #c8a800;
        border-left: 4px solid #c8a800;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #1a1a3e, #0a0a1a);
        border: 2px solid #c8a800;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(200, 168, 0, 0.2);
    }

    .prediction-winner {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #c8a800;
        margin: 0.5rem 0;
    }

    .prediction-confidence {
        font-size: 1.1rem;
        color: #a0a0c0;
    }

    .disclaimer {
        background: rgba(200, 168, 0, 0.1);
        border: 1px solid rgba(200, 168, 0, 0.3);
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #a0a0c0;
        margin-top: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #12122a;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a0a0c0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 1px;
        border-radius: 6px;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background: #c8a800 !important;
        color: #0a0a1a !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #c8a800, #f0cc00);
        color: #0a0a1a;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 2px;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #f0cc00, #c8a800);
        box-shadow: 0 4px 20px rgba(200, 168, 0, 0.4);
        transform: translateY(-2px);
    }

    hr {
        border-color: #c8a800;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    matches = pd.read_csv(r'D:\IPL EDA\matches.csv')
    deliveries = pd.read_csv(r'D:\IPL EDA\deliveries.csv')

    matches['date'] = pd.to_datetime(matches['date'])
    matches['season'] = matches['season'].apply(lambda x: int(str(x)[:4]))

    team_name_map = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
    }

    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        matches[col] = matches[col].replace(team_name_map)
    for col in ['batting_team', 'bowling_team']:
        deliveries[col] = deliveries[col].replace(team_name_map)

    matches['city'] = matches['city'].fillna(matches['venue'].str.split(',').str[-1].str.strip())
    matches['winner'] = matches['winner'].fillna('No Result')
    matches['player_of_match'] = matches['player_of_match'].fillna('No Award')

    deliveries['extras_type'] = deliveries['extras_type'].fillna('None')
    deliveries['player_dismissed'] = deliveries['player_dismissed'].fillna('None')
    deliveries['dismissal_kind'] = deliveries['dismissal_kind'].fillna('None')
    deliveries['fielder'] = deliveries['fielder'].fillna('None')

    df = deliveries.merge(matches, left_on='match_id', right_on='id', how='left')
    return matches, deliveries, df


@st.cache_resource
def train_model(matches):
    encoders = {}
    categorical_cols = ['city', 'team1', 'team2', 'toss_winner', 'toss_decision']

    ml_data = matches[['season', 'city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']].copy()
    ml_data = ml_data.dropna()
    ml_data['result'] = (ml_data['winner'] == ml_data['team1']).astype(int)
    ml_data = ml_data.drop(columns=['winner'])

    for col in categorical_cols:
        le = LabelEncoder()
        ml_data[col] = le.fit_transform(ml_data[col])
        encoders[col] = le

    X = ml_data.drop(columns=['result'])
    y = ml_data['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    lr_pred = model.predict(X_test)

    return model, encoders, y_test, lr_pred


# ─── Plot Helper ───────────────────────────────────────────────
def ipl_plot(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#12122a')
    ax.set_facecolor('#12122a')
    ax.tick_params(colors='#a0a0c0')
    ax.xaxis.label.set_color('#a0a0c0')
    ax.yaxis.label.set_color('#a0a0c0')
    ax.title.set_color('#c8a800')
    for spine in ax.spines.values():
        spine.set_edgecolor('#c8a800')
        spine.set_alpha(0.3)
    return fig, ax


# ─── Load ──────────────────────────────────────────────────────
matches, deliveries, df = load_data()
model, encoders, y_test, lr_pred = train_model(matches)

# ─── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="ipl-header">
    <h1>🏏 IPL Analytics Dashboard</h1>
    <p>Historical Analysis · 2008 – 2024 · 17 Seasons · 1095 Matches</p>
</div>
""", unsafe_allow_html=True)

# ─── Top Metrics ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><h2>{len(matches)}</h2><p>Total Matches</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><h2>{deliveries.shape[0]:,}</h2><p>Total Balls</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><h2>{matches["season"].nunique()}</h2><p>Seasons</p></div>', unsafe_allow_html=True)
with c4:
    total_runs = deliveries['batsman_runs'].sum()
    st.markdown(f'<div class="metric-card"><h2>{total_runs:,}</h2><p>Total Runs</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Match Analysis",
    "🏏 Batting",
    "🎯 Bowling",
    "📈 Advanced",
    "🤖 Predict Winner"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — MATCH ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Matches Per Season</div>', unsafe_allow_html=True)
    matches_per_season = matches.groupby('season')['id'].count().reset_index()
    matches_per_season.columns = ['season', 'match_count']

    fig, ax = ipl_plot()
    sns.barplot(data=matches_per_season, x='season', y='match_count', ax=ax, color='#c8a800')
    ax.set_title('Number of Matches Per Season', fontsize=14)
    ax.set_xlabel('Season')
    ax.set_ylabel('Matches')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Toss Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        toss_trend = matches.groupby(['season', 'toss_decision'])['id'].count().reset_index()
        toss_trend.columns = ['season', 'toss_decision', 'count']
        fig, ax = ipl_plot()
        for decision, color in zip(['bat', 'field'], ['#c8a800', '#4a90d9']):
            data = toss_trend[toss_trend['toss_decision'] == decision]
            ax.plot(data['season'], data['count'], marker='o', label=decision, color=color, linewidth=2)
        ax.set_title('Toss Decision Trend Over Seasons', fontsize=13)
        ax.set_xlabel('Season')
        ax.set_ylabel('Count')
        ax.legend(facecolor='#1a1a3e', edgecolor='#c8a800', labelcolor='#f0f0f0')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        matches['toss_winner_won'] = matches['toss_winner'] == matches['winner']
        toss_decision_win = matches[matches['winner'] != 'No Result'].groupby('toss_decision')['toss_winner_won'].mean() * 100
        fig, ax = ipl_plot()
        bars = ax.bar(toss_decision_win.index, toss_decision_win.values, color=['#c8a800', '#4a90d9'], width=0.4)
        ax.set_title('Win Rate by Toss Decision (%)', fontsize=13)
        ax.set_ylabel('Win %')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%', ha='center', color='#f0f0f0', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Team Win Percentage</div>', unsafe_allow_html=True)
    team_wins = matches[matches['winner'] != 'No Result']['winner'].value_counts().reset_index()
    team_wins.columns = ['team', 'wins']
    team_matches = pd.concat([matches['team1'], matches['team2']]).value_counts().reset_index()
    team_matches.columns = ['team', 'matches_played']
    team_stats = team_matches.merge(team_wins, on='team', how='left')
    team_stats['wins'] = team_stats['wins'].fillna(0)
    team_stats['win_pct'] = (team_stats['wins'] / team_stats['matches_played'] * 100).round(2)
    team_stats = team_stats.sort_values('win_pct', ascending=True)

    fig, ax = ipl_plot(figsize=(10, 7))
    bars = ax.barh(team_stats['team'], team_stats['win_pct'], color='#c8a800')
    ax.set_title('All Time Team Win Percentage', fontsize=14)
    ax.set_xlabel('Win %')
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.1f}%', va='center', color='#f0f0f0', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 2 — BATTING
# ══════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Top Run Scorers</div>', unsafe_allow_html=True)
        top_scorers = deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
        top_scorers.columns = ['batter', 'total_runs']
        top_scorers = top_scorers.sort_values('total_runs', ascending=True).tail(15)

        fig, ax = ipl_plot(figsize=(8, 7))
        ax.barh(top_scorers['batter'], top_scorers['total_runs'], color='#c8a800')
        ax.set_title('Top 15 Run Scorers in IPL', fontsize=13)
        ax.set_xlabel('Total Runs')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Top Six Hitters</div>', unsafe_allow_html=True)
        sixes = deliveries[deliveries['batsman_runs'] == 6].groupby('batter')['batsman_runs'].count().reset_index()
        sixes.columns = ['batter', 'sixes']
        sixes = sixes.sort_values('sixes', ascending=True).tail(10)

        fig, ax = ipl_plot(figsize=(8, 7))
        ax.barh(sixes['batter'], sixes['sixes'], color='#4a90d9')
        ax.set_title('Top 10 Six Hitters in IPL', fontsize=13)
        ax.set_xlabel('Sixes')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Phase-wise Run Rate</div>', unsafe_allow_html=True)

    def get_phase(over):
        if over <= 5:
            return 'Powerplay'
        elif over <= 14:
            return 'Middle'
        else:
            return 'Death'

    deliveries['phase'] = deliveries['over'].apply(get_phase)
    phase_runs = deliveries.groupby('phase')['total_runs'].sum().reset_index()
    phase_balls = deliveries.groupby('phase')['ball'].count().reset_index()
    phase_stats = phase_runs.merge(phase_balls, on='phase')
    phase_stats.columns = ['phase', 'total_runs', 'total_balls']
    phase_stats['run_rate'] = (phase_stats['total_runs'] / phase_stats['total_balls'] * 6).round(2)
    order = ['Powerplay', 'Middle', 'Death']
    phase_stats['phase'] = pd.Categorical(phase_stats['phase'], categories=order, ordered=True)
    phase_stats = phase_stats.sort_values('phase')

    fig, ax = ipl_plot()
    colors = ['#c8a800', '#4a90d9', '#e84040']
    bars = ax.bar(phase_stats['phase'], phase_stats['run_rate'], color=colors, width=0.4)
    ax.set_title('Run Rate by Match Phase', fontsize=13)
    ax.set_ylabel('Run Rate')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{bar.get_height():.2f}', ha='center', color='#f0f0f0', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Boundaries Per Season</div>', unsafe_allow_html=True)
    df['is_four'] = df['batsman_runs'] == 4
    df['is_six'] = df['batsman_runs'] == 6
    boundaries_per_season = df.groupby('season')[['is_four', 'is_six']].sum().reset_index()

    fig, ax = ipl_plot()
    ax.plot(boundaries_per_season['season'], boundaries_per_season['is_four'],
            marker='o', label='Fours', color='#c8a800', linewidth=2)
    ax.plot(boundaries_per_season['season'], boundaries_per_season['is_six'],
            marker='o', label='Sixes', color='#4a90d9', linewidth=2)
    ax.set_title('Fours and Sixes Per Season', fontsize=13)
    ax.set_xlabel('Season')
    ax.set_ylabel('Count')
    ax.legend(facecolor='#1a1a3e', edgecolor='#c8a800', labelcolor='#f0f0f0')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 3 — BOWLING
# ══════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Top Wicket Takers</div>', unsafe_allow_html=True)
        wickets = deliveries[
            (deliveries['is_wicket'] == 1) &
            (deliveries['dismissal_kind'] != 'run out')
        ]
        top_wickets = wickets.groupby('bowler')['is_wicket'].count().reset_index()
        top_wickets.columns = ['bowler', 'wickets']
        top_wickets = top_wickets.sort_values('wickets', ascending=True).tail(15)

        fig, ax = ipl_plot(figsize=(8, 7))
        ax.barh(top_wickets['bowler'], top_wickets['wickets'], color='#e84040')
        ax.set_title('Top 15 Wicket Takers in IPL', fontsize=13)
        ax.set_xlabel('Wickets')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Dismissal Types</div>', unsafe_allow_html=True)
        dismissals = deliveries[deliveries['is_wicket'] == 1]
        dismissal_types = dismissals['dismissal_kind'].value_counts().reset_index()
        dismissal_types.columns = ['dismissal_kind', 'count']

        fig, ax = ipl_plot(figsize=(8, 7))
        colors_d = ['#c8a800', '#4a90d9', '#e84040', '#50c878', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c']
        ax.barh(dismissal_types['dismissal_kind'], dismissal_types['count'],
                color=colors_d[:len(dismissal_types)])
        ax.set_title('Dismissal Type Breakdown', fontsize=13)
        ax.set_xlabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Most Economical Bowlers</div>', unsafe_allow_html=True)
    bowler_runs = deliveries.groupby('bowler')['total_runs'].sum().reset_index()
    bowler_runs.columns = ['bowler', 'runs_conceded']
    legal_balls = deliveries[~deliveries['extras_type'].isin(['wides', 'noballs'])]
    bowler_balls = legal_balls.groupby('bowler')['ball'].count().reset_index()
    bowler_balls.columns = ['bowler', 'balls_bowled']
    bowler_economy = bowler_runs.merge(bowler_balls, on='bowler')
    bowler_economy['overs_bowled'] = (bowler_economy['balls_bowled'] / 6).round(2)
    bowler_economy['economy'] = (bowler_economy['runs_conceded'] / bowler_economy['overs_bowled']).round(2)
    bowler_economy = bowler_economy[bowler_economy['balls_bowled'] >= 300]
    bowler_economy = bowler_economy.sort_values('economy').head(10)

    fig, ax = ipl_plot()
    ax.barh(bowler_economy['bowler'][::-1], bowler_economy['economy'][::-1], color='#50c878')
    ax.set_title('Most Economical Bowlers (Min 300 balls)', fontsize=13)
    ax.set_xlabel('Economy Rate')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 4 — ADVANCED
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Chasing vs Defending Win % Per Season</div>', unsafe_allow_html=True)

    valid = matches[matches['winner'] != 'No Result'].copy()

    def get_result(row):
        return 'defend' if row['toss_winner'] == row['winner'] and row['toss_decision'] == 'field' else 'chase'

    valid['match_result_type'] = valid.apply(get_result, axis=1)
    chase_defend = valid.groupby(['season', 'match_result_type'])['id'].count().reset_index()
    chase_defend.columns = ['season', 'result_type', 'count']
    chase_defend_pivot = chase_defend.pivot(index='season', columns='result_type', values='count').fillna(0)

    if 'chase' in chase_defend_pivot.columns and 'defend' in chase_defend_pivot.columns:
        chase_defend_pivot['chase_win_pct'] = (
            chase_defend_pivot['chase'] / (chase_defend_pivot['chase'] + chase_defend_pivot['defend']) * 100
        ).round(2)

        fig, ax = ipl_plot()
        ax.plot(chase_defend_pivot.index, chase_defend_pivot['chase_win_pct'],
                marker='o', color='#c8a800', linewidth=2.5, label='Chase Win %')
        ax.axhline(50, linestyle='--', color='#4a90d9', alpha=0.6, label='50% line')
        ax.fill_between(chase_defend_pivot.index, chase_defend_pivot['chase_win_pct'], 50,
                        where=chase_defend_pivot['chase_win_pct'] >= 50,
                        alpha=0.15, color='#c8a800')
        ax.set_title('Chasing Win % Per Season', fontsize=13)
        ax.set_xlabel('Season')
        ax.set_ylabel('Chase Win %')
        ax.legend(facecolor='#1a1a3e', edgecolor='#c8a800', labelcolor='#f0f0f0')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Head to Head</div>', unsafe_allow_html=True)
    all_teams = sorted(matches['team1'].unique())
    col1, col2 = st.columns(2)
    with col1:
        h2h_team1 = st.selectbox('Select Team 1', all_teams, key='h2h1')
    with col2:
        h2h_team2 = st.selectbox('Select Team 2', [t for t in all_teams if t != h2h_team1], key='h2h2')

    h2h = matches[
        ((matches['team1'] == h2h_team1) & (matches['team2'] == h2h_team2)) |
        ((matches['team1'] == h2h_team2) & (matches['team2'] == h2h_team1))
    ]
    t1_wins = len(h2h[h2h['winner'] == h2h_team1])
    t2_wins = len(h2h[h2h['winner'] == h2h_team2])
    total = len(h2h)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{total}</h2><p>Total Matches</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2>{t1_wins}</h2><p>{h2h_team1} Wins</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h2>{t2_wins}</h2><p>{h2h_team2} Wins</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Player of the Match Leaders</div>', unsafe_allow_html=True)
    pom = matches[matches['player_of_match'] != 'No Award']['player_of_match'].value_counts().head(15).reset_index()
    pom.columns = ['player', 'awards']
    pom = pom.sort_values('awards', ascending=True)

    fig, ax = ipl_plot(figsize=(10, 6))
    ax.barh(pom['player'], pom['awards'], color='#c8a800')
    ax.set_title('Most Player of the Match Awards', fontsize=13)
    ax.set_xlabel('Awards')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 5 — PREDICT WINNER
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Match Winner Predictor</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> Predictions are based on historical IPL data from 2008 to 2024.
        Results may not reflect current team compositions, player form or 2025/2026 season performance.
        Model accuracy is ~52% which reflects the inherently unpredictable nature of T20 cricket.
        <br><br>
        📌 <b>Note:</b> Enter the <b>Home Team</b> or <b>Stronger Team</b> as Team 1 and the
        <b>Away Team</b> or <b>Visiting Team</b> as Team 2 for best results.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    teams = sorted(encoders['team1'].classes_)
    cities = sorted(encoders['city'].classes_)

    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox('🏟️ City', cities)
    with col2:
        team1 = st.selectbox('🔵 Team 1 (Home / Stronger)', teams)
    with col3:
        team2 = st.selectbox('🔴 Team 2 (Away / Visiting)', [t for t in teams if t != team1])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button('🏏 PREDICT WINNER'):
        try:
            input1 = pd.DataFrame([{
                'season': 2024,
                'city': encoders['city'].transform([city])[0],
                'team1': encoders['team1'].transform([team1])[0],
                'team2': encoders['team2'].transform([team2])[0],
                'toss_winner': encoders['toss_winner'].transform([team1])[0],
                'toss_decision': encoders['toss_decision'].transform(['field'])[0]
            }])
            input2 = pd.DataFrame([{
                'season': 2024,
                'city': encoders['city'].transform([city])[0],
                'team1': encoders['team1'].transform([team1])[0],
                'team2': encoders['team2'].transform([team2])[0],
                'toss_winner': encoders['toss_winner'].transform([team2])[0],
                'toss_decision': encoders['toss_decision'].transform(['field'])[0]
            }])

            prob1 = model.predict_proba(input1)[0]
            prob2 = model.predict_proba(input2)[0]

            avg_prob_team1 = (prob1[1] + prob2[1]) / 2
            avg_prob_team2 = (prob1[0] + prob2[0]) / 2

            winner = team1 if avg_prob_team1 > avg_prob_team2 else team2
            confidence = max(avg_prob_team1, avg_prob_team2) * 100

            st.markdown(f"""
            <div class="prediction-card">
                <p style="color:#a0a0c0; font-size:1rem; text-transform:uppercase; letter-spacing:2px;">Predicted Winner</p>
                <div class="prediction-winner">🏆 {winner}</div>
                <div class="prediction-confidence">Confidence: {confidence:.1f}%</div>
                <br>
                <div style="background:#1a1a3e; border-radius:8px; padding:0.5rem; margin-top:0.5rem;">
                    <span style="color:#c8a800; font-weight:600;">{team1}</span>
                    <span style="color:#a0a0c0;"> {avg_prob_team1*100:.1f}% vs {avg_prob_team2*100:.1f}% </span>
                    <span style="color:#4a90d9; font-weight:600;">{team2}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    acc = accuracy_score(y_test, lr_pred) * 100
    f1 = f1_score(y_test, lr_pred)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{acc:.1f}%</h2><p>Model Accuracy</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2>{f1:.2f}</h2><p>F1 Score</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cm = confusion_matrix(y_test, lr_pred)
    fig, ax = ipl_plot(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', ax=ax,
                xticklabels=['Team2 Wins', 'Team1 Wins'],
                yticklabels=['Team2 Wins', 'Team1 Wins'])
    ax.set_title('Confusion Matrix', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─── Footer ────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center; color:#a0a0c0; font-size:0.8rem; padding:1rem;">
    Built with ❤️ using Python · Pandas · Scikit-learn · Streamlit &nbsp;|&nbsp; Data: IPL 2008–2024
</div>
""", unsafe_allow_html=True)