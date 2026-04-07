# IPL Intelligence

A data analytics and match prediction dashboard built with Python and Streamlit, covering 17 seasons of Indian Premier League cricket from 2008 to 2024.

---

## What This Project Does

IPL Intelligence lets you explore ball-by-ball and match-level IPL data through interactive charts, and uses a machine learning model to predict the winner of any match you configure. The entire app runs in the browser via Streamlit with no backend needed beyond the CSV data files.

There are five sections:

**Predict** is the main feature. You pick two teams and a venue city, and the model gives you a win probability for each side along with historical head-to-head context between the two teams.

**Match Analysis** covers how many matches were played each season, how toss decisions have shifted over time, which teams have the best all-time win rates, and how many super overs have happened.

**Batting** shows the top run scorers, the biggest six hitters, how run rates differ across the powerplay, middle overs, and death overs, and how the number of fours and sixes has changed season by season.

**Bowling** breaks down the top wicket takers, what types of dismissals are most common, and which bowlers have been the most economical over their careers.

**Advanced** looks at whether chasing or defending has been more successful in each season, lets you compare head-to-head records between any two current IPL franchises, and shows who has won the most Player of the Match awards.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** for the web interface
- **Pandas and NumPy** for data processing
- **Matplotlib** for all charts
- **Scikit-learn** for the prediction model

---

## Getting Started

Clone the repository and install the dependencies:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

Place your IPL dataset files in the root directory. The app expects two zip files named `matches.zip` and `deliveries.zip`, each containing a single CSV inside. The matches CSV needs a `date` column and the deliveries CSV needs `match_id`, `batter`, `bowler`, `batsman_runs`, `total_runs`, and `is_wicket` among others.

Then run:

```bash
streamlit run app.py
```

---

## The Prediction Model

The predictor uses **Logistic Regression** trained on every IPL match from 2008 to 2024. The features it learns from are the season year, venue city, the two teams playing, who won the toss, and what decision they made.

**The model currently sits at around 52% accuracy, which is only marginally better than a coin flip.** This is not a bug or a mistake — it reflects something genuinely true about T20 cricket. A single match at this format is extraordinarily hard to predict because the margin between a win and a loss is so small, luck plays a real role, and the outcome can shift in the space of one over. Logistic Regression with historical aggregate features simply cannot capture that volatility.

The deeper reason the accuracy is low is that the features being used are structural: which teams are playing, where, and who won the toss. What they do not include is the information that actually swings individual matches — current player form, injury news, pitch conditions on the day, recent head-to-head performance, squad changes, and player matchup statistics. Without those dynamic inputs, no model trained purely on historical data will predict T20 results with high confidence.

**This project is a work in progress.** As I continue learning machine learning, the plan is to gradually upgrade the approach. That means experimenting with ensemble methods like Random Forest and Gradient Boosting, incorporating rolling performance features for players and teams, and eventually exploring deep learning approaches. The goal is not just better accuracy but a more honest representation of where the uncertainty in any prediction actually comes from.

---

## Data

The dataset used is the publicly available Kaggle IPL dataset covering the 2008 to 2024 seasons. It includes match-level data such as toss results, venue, teams, and outcomes, as well as ball-by-ball delivery data covering every run, wicket, and extra.

Team names have been standardised across seasons so that, for example, Delhi Daredevils and Delhi Capitals are treated as the same franchise throughout the historical record.

The Head-to-Head comparison in the Advanced tab only includes the **10 current active franchises**. Defunct teams like Deccan Chargers, Pune Warriors, Kochi Tuskers Kerala, Gujarat Lions, and Rising Pune Supergiants have been excluded from that section because there is no meaningful ongoing rivalry to analyse for teams that no longer exist.

---

## What Is Coming Next

- Replace Logistic Regression with Random Forest and XGBoost
- Add rolling team form features (last 5 match win rate, recent run rate)
- Player-level aggregates as model inputs
- Toss impact analysis broken down by venue
- Season-by-season model performance comparison

---

## A Note on the Predictions

Please treat the predictions as an interesting data-driven perspective, not a reliable forecast. The disclaimer in the app says it best: T20 cricket is unpredictable by nature, and a model trained on historical patterns will never fully capture what happens on the day. The numbers are there to inform and provoke thought, not to be taken at face value.

---

Built by a cricket fan learning machine learning, one season at a time.
