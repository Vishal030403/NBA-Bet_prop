
import streamlit as st
import joblib
import numpy as np
from scipy.stats import poisson
import os
from datetime import date

st.set_page_config(page_title="NBA Predictor")

st.title("🏀 NBA Points Predictor")

# ---------- PLAYER & TEAM LISTS ----------
players = [
    "LeBron James", "Stephen Curry", "Kevin Durant",
    "Luka Dončić", "Giannis Antetokounmpo",
    "Jayson Tatum", "Joel Embiid", "Nikola Jokić",
    "Devin Booker", "Damian Lillard"
]

teams = [
    "Boston Celtics", "Los Angeles Lakers", "Golden State Warriors",
    "Milwaukee Bucks", "Denver Nuggets", "Dallas Mavericks",
    "Phoenix Suns", "Philadelphia 76ers", "Miami Heat",
    "New York Knicks"
]

# ---------- MODEL LOAD ----------
MODEL_PATH = "nba_prop_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ---------- GAME INFO ----------
st.subheader("Game Info")

game_date = st.date_input("Game Date", value=date.today())

player = st.selectbox(
    "Player Name",
    players,
    index=0
)

opponent = st.selectbox(
    "Opponent Team",
    teams,
    index=0
)

is_b2b = st.radio(
    "Back-to-Back Game?",
    ["No", "Yes"]
)

is_b2b_val = 1 if is_b2b == "Yes" else 0

# ---------- PERFORMANCE METRICS ----------
st.subheader("Performance Metrics")

pts5 = st.number_input("PTS Roll5", value=20.0)
pts10 = st.number_input("PTS Roll10", value=20.0)
mp5 = st.number_input("MP Roll5", value=30.0)
fg = st.number_input("FG%", value=0.45)

fga5 = st.number_input("FGA Roll5", value=15.0)
usage = st.number_input("Usage Proxy", value=20.0)
oppdef = st.number_input("Opponent Defense", value=110.0)
rest = st.number_input("Rest Days", value=2.0)
mpstd = st.number_input("Minutes Volatility", value=3.0)

# ---------- PREDICTION ----------
if st.button("Predict"):

    features = np.array([[ 
        pts5, pts10, mp5, fg,
        fga5, usage, oppdef,
        rest, is_b2b_val, mpstd
    ]])

    lam = float(model.predict(features)[0])
    line = np.floor(lam) + 0.5
    prob_over = 1 - poisson.cdf(np.floor(line), lam)

    st.success(f"Player: {player}")
    st.write(f"📅 Date: {game_date}")
    st.write(f"🏀 Opponent: {opponent}")

    st.metric("Predicted Points", f"{lam:.2f}")
    st.metric("Suggested Line", f"{line:.1f}")
    st.metric("Probability Over", f"{prob_over*100:.1f}%")

    if prob_over > 0.5:
        st.info("📈 Lean: OVER")
    else:
        st.info("📉 Lean: UNDER")

