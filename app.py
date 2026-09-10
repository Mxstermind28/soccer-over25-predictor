
import os
import math
from io import StringIO

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Soccer Over 2.5 Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Soccer Over 2.5 Goals Predictor")
st.caption("A statistical model for estimating the probability that a match finishes with 3+ total goals.")

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def over25_probability(xg):
    # P(total goals >= 3) for a Poisson distribution
    under_or_equal_2 = sum(poisson_pmf(k, xg) for k in range(3))
    return max(0.0, min(1.0, 1.0 - under_or_equal_2))

def prepare(df):
    df = df.copy()
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return df.sort_values("Date")

def model_prediction(df, home, away, n_recent=10):
    if home not in set(df.HomeTeam) | set(df.AwayTeam):
        raise ValueError(f"{home} was not found in the dataset.")
    if away not in set(df.HomeTeam) | set(df.AwayTeam):
        raise ValueError(f"{away} was not found in the dataset.")

    league_home_goals = df["FTHG"].mean()
    league_away_goals = df["FTAG"].mean()

    # Team attack/defense strengths, calculated separately for home and away.
    h_home = df[df.HomeTeam == home].tail(n_recent)
    h_away = df[df.AwayTeam == home].tail(n_recent)
    a_home = df[df.HomeTeam == away].tail(n_recent)
    a_away = df[df.AwayTeam == away].tail(n_recent)

    # Fallback to all matches if a team has too few home/away observations.
    if len(h_home) < 3: h_home = df[df.HomeTeam == home].tail(max(n_recent, 20))
    if len(h_away) < 3: h_away = df[df.AwayTeam == home].tail(max(n_recent, 20))
    if len(a_home) < 3: a_home = df[df.HomeTeam == away].tail(max(n_recent, 20))
    if len(a_away) < 3: a_away = df[df.AwayTeam == away].tail(max(n_recent, 20))

    home_attack = (h_home.FTHG.mean() / league_home_goals) if len(h_home) else 1
    home_defense = (h_away.FTAG.mean() / league_away_goals) if len(h_away) else 1
    away_attack = (a_away.FTAG.mean() / league_away_goals) if len(a_away) else 1
    away_defense = (a_home.FTHG.mean() / league_home_goals) if len(a_home) else 1

    # Expected goals. Small shrink toward league average reduces overfitting.
    raw_home_xg = league_home_goals * home_attack * away_defense
    raw_away_xg = league_away_goals * away_attack * home_defense

    total_matches = len(df)
    shrink = min(0.75, max(0.25, total_matches / 500))
    home_xg = shrink * raw_home_xg + (1 - shrink) * league_home_goals
    away_xg = shrink * raw_away_xg + (1 - shrink) * league_away_goals

    total_xg = home_xg + away_xg
    probability = over25_probability(total_xg)

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "total_xg": total_xg,
        "probability": probability,
        "sample_size": total_matches,
    }

@st.cache_data(ttl=3600)
def load_football_data(competition, seasons):
    token = st.session_state.get("api_token") or os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise ValueError("No football-data.org API token supplied.")
    rows = []
    headers = {"X-Auth-Token": token}
    for season in seasons:
        url = f"https://api.football-data.org/v4/competitions/{competition}/matches"
        r = requests.get(url, headers=headers, params={"season": season, "status": "FINISHED"}, timeout=30)
        r.raise_for_status()
        for m in r.json().get("matches", []):
            rows.append({
                "Date": m["utcDate"][:10],
                "HomeTeam": m["homeTeam"]["name"],
                "AwayTeam": m["awayTeam"]["name"],
                "FTHG": m["score"]["fullTime"]["home"],
                "FTAG": m["score"]["fullTime"]["away"],
            })
    return prepare(pd.DataFrame(rows))

st.sidebar.header("Data")
data_mode = st.sidebar.radio("Data source", ["Upload CSV", "football-data.org API"])

if data_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Historical results CSV", type=["csv"])
    st.sidebar.markdown("Required columns: `Date, HomeTeam, AwayTeam, FTHG, FTAG`")
    if uploaded:
        try:
            df = prepare(pd.read_csv(uploaded))
        except Exception as e:
            st.error(str(e))
            st.stop()
    else:
        st.info("Upload a historical results CSV to begin.")
        st.stop()
else:
    token = st.sidebar.text_input("football-data.org API token", type="password")
    st.session_state["api_token"] = token
    competition = st.sidebar.selectbox(
        "Competition",
        ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "ELC", "CL"],
        index=0
    )
    seasons = st.sidebar.multiselect("Seasons", [2026, 2025, 2024, 2023, 2022], default=[2025])
    if not token:
        st.warning("Enter your API token in the sidebar, or switch to Upload CSV.")
        st.stop()
    try:
        df = load_football_data(competition, seasons)
    except Exception as e:
        st.error(f"Could not load API data: {e}")
        st.stop()

teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("Home team", teams)
with c2:
    away_options = [t for t in teams if t != home]
    away = st.selectbox("Away team", away_options)

recent = st.slider("Recent matches used per team", 5, 20, 10)

if st.button("Calculate Over 2.5 probability", type="primary"):
    try:
        result = model_prediction(df, home, away, recent)
        p = result["probability"]
        st.divider()
        st.subheader(f"{home} vs {away}")

        a, b, c, d = st.columns(4)
        a.metric("Over 2.5", f"{p*100:.1f}%")
        b.metric("Under 2.5", f"{(1-p)*100:.1f}%")
        c.metric("Expected goals", f"{result['total_xg']:.2f}")
        d.metric("Model sample", f"{result['sample_size']} matches")

        if p >= 0.70:
            label = "HIGH"
        elif p >= 0.55:
            label = "MEDIUM-HIGH"
        elif p >= 0.45:
            label = "MEDIUM"
        else:
            label = "LOW"

        st.progress(p, text=f"Model confidence band: {label}")
        st.write(f"**Expected goals:** {home} {result['home_xg']:.2f} — {away} {result['away_xg']:.2f}")
        st.warning("This is a statistical estimate, not a guarantee or betting advice.")

        with st.expander("How the model works"):
            st.write(
                "The model estimates each team's scoring and conceding strength relative to the "
                "league averages, then converts the resulting expected total goals into a Poisson "
                "probability for 3 or more goals. A shrinkage step reduces overfitting when samples are small."
            )
    except Exception as e:
        st.error(str(e))

with st.expander("CSV format example"):
    st.code(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
        "2025-08-15,Team A,Team B,2,1\n"
        "2025-08-16,Team C,Team D,0,0\n"
    )
