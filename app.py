import os
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="GoalEdge | Over 2.5 Football Forecasts",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

JAMAICA_TZ = ZoneInfo("America/Jamaica")

CUSTOM_CSS = """
<style>
:root {
  --bg:#07110d;
  --panel:#0d1b15;
  --panel2:#11251c;
  --line:#1f3a2d;
  --text:#f3f7f4;
  --muted:#9fb3a7;
  --green:#35e582;
  --green2:#18b866;
  --amber:#f6bd4b;
  --red:#ff6b6b;
}
.stApp {
  background:
    radial-gradient(circle at 15% 0%, rgba(53,229,130,.10), transparent 28%),
    radial-gradient(circle at 95% 10%, rgba(24,184,102,.08), transparent 25%),
    var(--bg);
  color:var(--text);
}
.block-container {padding-top:1.6rem; max-width:1450px;}
[data-testid="stSidebar"] {background:#08140f; border-right:1px solid var(--line);}
.hero {
  padding:26px 28px;
  border:1px solid var(--line);
  border-radius:22px;
  background:linear-gradient(135deg, rgba(17,37,28,.96), rgba(8,20,15,.96));
  box-shadow:0 18px 50px rgba(0,0,0,.24);
  margin-bottom:18px;
}
.hero-kicker {color:var(--green); font-size:.78rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase;}
.hero h1 {font-size:2.4rem; line-height:1.05; margin:.35rem 0 .5rem;}
.hero p {color:var(--muted); max-width:850px; font-size:1.02rem; margin:0;}
.badge {
  display:inline-block; padding:6px 10px; border:1px solid #28563e; border-radius:999px;
  background:#0d2117; color:#9ff5bf; font-size:.78rem; font-weight:700; margin-right:6px; margin-top:10px;
}
.section-title {font-size:1.08rem; font-weight:800; margin:8px 0 12px;}
.metric-card {
  border:1px solid var(--line);
  background:linear-gradient(180deg,#10231a,#0b1812);
  border-radius:18px;
  padding:16px 18px;
  min-height:122px;
}
.metric-label {color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700;}
.metric-value {font-size:2rem; font-weight:850; margin-top:5px;}
.metric-sub {color:var(--muted); font-size:.82rem; margin-top:5px;}
.signal-high {color:var(--green);}
.signal-mid {color:var(--amber);}
.signal-low {color:var(--red);}
.match-card {
  border:1px solid var(--line);
  background:#0d1b15;
  border-radius:20px;
  padding:20px;
}
.small-muted {color:var(--muted); font-size:.84rem;}
hr {border-color:var(--line)!important;}
.stButton>button {
  border-radius:12px; font-weight:800; min-height:44px;
}
div[data-testid="stProgress"] > div > div > div {background-color:var(--green);}
[data-testid="stMetric"] {
  background:#0d1b15;
  border:1px solid var(--line);
  padding:14px;
  border-radius:16px;
}
.footer {
  text-align:center; color:#70877a; font-size:.78rem; padding:28px 0 10px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def secret_or_env(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def over_probability(total_xg, line):
    threshold = math.floor(line) + 1
    under = sum(poisson_pmf(k, total_xg) for k in range(threshold))
    return max(0.0, min(1.0, 1.0 - under))


def btts_probability(home_xg, away_xg):
    p_home_zero = math.exp(-home_xg)
    p_away_zero = math.exp(-away_xg)
    return max(0.0, min(1.0, 1 - p_home_zero - p_away_zero + math.exp(-(home_xg + away_xg))))


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
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["Over25"] = (df["TotalGoals"] >= 3).astype(int)
    return df.sort_values("Date").reset_index(drop=True)


def team_recent(df, team, n=10):
    m = df[(df.HomeTeam == team) | (df.AwayTeam == team)].tail(n).copy()
    if m.empty:
        return m

    def goals_for(r):
        return r.FTHG if r.HomeTeam == team else r.FTAG

    def goals_against(r):
        return r.FTAG if r.HomeTeam == team else r.FTHG

    m["GF"] = m.apply(goals_for, axis=1)
    m["GA"] = m.apply(goals_against, axis=1)
    m["TG"] = m["GF"] + m["GA"]
    m["O25"] = (m["TG"] >= 3).astype(int)
    return m


def team_form_summary(df, team, n=10):
    m = team_recent(df, team, n)
    if m.empty:
        return {"matches": 0, "gf": 0, "ga": 0, "avg_total": 0, "over25": 0}
    return {
        "matches": len(m),
        "gf": m.GF.mean(),
        "ga": m.GA.mean(),
        "avg_total": m.TG.mean(),
        "over25": m.O25.mean(),
    }


def model_prediction(df, home, away, n_recent=10):
    all_teams = set(df.HomeTeam) | set(df.AwayTeam)
    if home not in all_teams:
        raise ValueError(f"{home} was not found in the dataset.")
    if away not in all_teams:
        raise ValueError(f"{away} was not found in the dataset.")

    league_home_goals = max(df["FTHG"].mean(), 0.15)
    league_away_goals = max(df["FTAG"].mean(), 0.15)

    h_home = df[df.HomeTeam == home].tail(n_recent)
    h_away = df[df.AwayTeam == home].tail(n_recent)
    a_home = df[df.HomeTeam == away].tail(n_recent)
    a_away = df[df.AwayTeam == away].tail(n_recent)

    if len(h_home) < 3:
        h_home = df[df.HomeTeam == home].tail(max(n_recent, 20))
    if len(h_away) < 3:
        h_away = df[df.AwayTeam == home].tail(max(n_recent, 20))
    if len(a_home) < 3:
        a_home = df[df.HomeTeam == away].tail(max(n_recent, 20))
    if len(a_away) < 3:
        a_away = df[df.AwayTeam == away].tail(max(n_recent, 20))

    home_attack = (h_home.FTHG.mean() / league_home_goals) if len(h_home) else 1
    home_defense = (h_away.FTAG.mean() / league_away_goals) if len(h_away) else 1
    away_attack = (a_away.FTAG.mean() / league_away_goals) if len(a_away) else 1
    away_defense = (a_home.FTHG.mean() / league_home_goals) if len(a_home) else 1

    raw_home_xg = league_home_goals * home_attack * away_defense
    raw_away_xg = league_away_goals * away_attack * home_defense

    sample_factor = min(1.0, len(df) / 650)
    shrink = 0.35 + (0.40 * sample_factor)

    home_xg = shrink * raw_home_xg + (1 - shrink) * league_home_goals
    away_xg = shrink * raw_away_xg + (1 - shrink) * league_away_goals
    total_xg = home_xg + away_xg

    home_form = team_form_summary(df, home, n_recent)
    away_form = team_form_summary(df, away, n_recent)

    poisson_o25 = over_probability(total_xg, 2.5)
    empirical = (home_form["over25"] + away_form["over25"]) / 2
    recent_goal_signal = over_probability((home_form["avg_total"] + away_form["avg_total"]) / 2, 2.5)

    probability = 0.58 * poisson_o25 + 0.27 * empirical + 0.15 * recent_goal_signal
    probability = max(0.01, min(0.99, probability))

    agreement = 1 - min(1.0, abs(poisson_o25 - empirical))
    sample_quality = min(1.0, (home_form["matches"] + away_form["matches"]) / max(2 * n_recent, 1))
    confidence = 0.55 * agreement + 0.45 * sample_quality
    confidence = max(0.0, min(1.0, confidence))

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "total_xg": total_xg,
        "probability": probability,
        "poisson_probability": poisson_o25,
        "empirical_probability": empirical,
        "recent_probability": recent_goal_signal,
        "over15": over_probability(total_xg, 1.5),
        "over35": over_probability(total_xg, 3.5),
        "btts": btts_probability(home_xg, away_xg),
        "fair_odds": (1 / probability) if probability > 0 else None,
        "confidence": confidence,
        "sample_size": len(df),
        "home_form": home_form,
        "away_form": away_form,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_football_data(competition, seasons, token):
    if not token:
        raise ValueError("No football-data.org API token supplied.")
    rows = []
    headers = {"X-Auth-Token": token}
    for season in seasons:
        url = f"https://api.football-data.org/v4/competitions/{competition}/matches"
        r = requests.get(url, headers=headers, params={"season": season, "status": "FINISHED"}, timeout=30)
        r.raise_for_status()
        for m in r.json().get("matches", []):
            score = m.get("score", {}).get("fullTime", {})
            if score.get("home") is None or score.get("away") is None:
                continue
            rows.append({
                "Date": m["utcDate"][:10],
                "HomeTeam": m["homeTeam"]["name"],
                "AwayTeam": m["awayTeam"]["name"],
                "FTHG": score["home"],
                "FTAG": score["away"],
            })
    if not rows:
        raise ValueError("The API returned no finished matches for that selection.")
    return prepare(pd.DataFrame(rows))


def signal_label(p):
    if p >= 0.72:
        return "ELITE", "signal-high"
    if p >= 0.64:
        return "HIGH", "signal-high"
    if p >= 0.56:
        return "MODERATE", "signal-mid"
    return "LOW", "signal-low"


def html_metric(label, value, sub="", css_class=""):
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {css_class}">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>
    """


now = datetime.now(JAMAICA_TZ)
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">GoalEdge Football Intelligence</div>
      <h1>Over 2.5 Goals Forecasting Dashboard</h1>
      <p>Estimate goal-market probabilities using league scoring rates, team attack/defence strength,
      recent goal trends and a Poisson-based ensemble. Built for research, comparison and transparent model analysis.</p>
      <span class="badge">⚽ Over 2.5</span>
      <span class="badge">📈 Fair odds</span>
      <span class="badge">🧠 Ensemble model</span>
      <span class="badge">🕒 Jamaica {now.strftime('%I:%M %p')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    data_mode = st.radio("Data source", ["football-data.org API", "Upload CSV"], horizontal=False)

    df = None
    if data_mode == "football-data.org API":
        default_token = secret_or_env("FOOTBALL_DATA_TOKEN") or ""
        token = st.text_input("football-data.org API token", value=default_token, type="password")
        competition_map = {
            "Premier League": "PL",
            "La Liga": "PD",
            "Serie A": "SA",
            "Bundesliga": "BL1",
            "Ligue 1": "FL1",
            "Eredivisie": "DED",
            "Primeira Liga": "PPL",
            "Championship": "ELC",
            "Champions League": "CL",
        }
        competition_name = st.selectbox("Competition", list(competition_map.keys()))
        seasons = st.multiselect("Seasons", [2026, 2025, 2024, 2023, 2022, 2021], default=[2025, 2024])
        if token and seasons:
            try:
                with st.spinner("Loading historical matches..."):
                    df = load_football_data(competition_map[competition_name], seasons, token)
            except Exception as e:
                st.error(f"API data error: {e}")
        else:
            st.info("Add your API token and choose at least one season.")
    else:
        uploaded = st.file_uploader("Historical results CSV", type=["csv"])
        st.caption("Required: Date, HomeTeam, AwayTeam, FTHG, FTAG")
        if uploaded is not None:
            try:
                df = prepare(pd.read_csv(uploaded))
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    recent = st.slider("Recent matches per team", 5, 20, 10)
    st.caption("Higher values are steadier; lower values react faster to recent form.")


if df is None or df.empty:
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        st.markdown("### Start by connecting data")
        st.write("Use **football-data.org** from the sidebar or upload a historical results CSV.")
        st.code(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "2025-08-15,Team A,Team B,2,1\n"
            "2025-08-16,Team C,Team D,0,0"
        )
        st.info("For deployment, store your API token as a Streamlit secret named FOOTBALL_DATA_TOKEN.")
    st.stop()


teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
if len(teams) < 2:
    st.error("The dataset needs at least two teams.")
    st.stop()

overview1, overview2, overview3, overview4 = st.columns(4)
overview1.metric("Historical matches", f"{len(df):,}")
overview2.metric("Teams", len(teams))
overview3.metric("League avg goals", f"{df.TotalGoals.mean():.2f}")
overview4.metric("Historical O2.5 rate", f"{df.Over25.mean()*100:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Match Lab", "📊 Team Form", "🧪 Model Breakdown", "📚 Data"])

with tab1:
    st.markdown('<div class="section-title">Build a match forecast</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("Home team", teams, key="home_team")
    with c2:
        away_options = [t for t in teams if t != home]
        away = st.selectbox("Away team", away_options, key="away_team")

    run = st.button("Run GoalEdge forecast", type="primary", use_container_width=True)

    if run or "last_prediction" in st.session_state:
        if run:
            st.session_state["last_prediction"] = model_prediction(df, home, away, recent)
            st.session_state["last_match"] = (home, away)

        result = st.session_state["last_prediction"]
        display_home, display_away = st.session_state.get("last_match", (home, away))
        p = result["probability"]
        label, label_css = signal_label(p)

        st.markdown(f"### {display_home} vs {display_away}")
        st.caption(f"Model signal: **{label}** · Research confidence: {result['confidence']*100:.0f}%")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(html_metric("Over 2.5", f"{p*100:.1f}%", f"Fair odds {result['fair_odds']:.2f}", label_css), unsafe_allow_html=True)
        with m2:
            st.markdown(html_metric("Expected goals", f"{result['total_xg']:.2f}", f"{display_home} {result['home_xg']:.2f} · {display_away} {result['away_xg']:.2f}"), unsafe_allow_html=True)
        with m3:
            st.markdown(html_metric("BTTS Yes", f"{result['btts']*100:.1f}%", "Both teams to score"), unsafe_allow_html=True)
        with m4:
            st.markdown(html_metric("Model confidence", f"{result['confidence']*100:.0f}%", "Agreement + sample quality"), unsafe_allow_html=True)

        st.progress(float(p), text=f"Over 2.5 probability · {label}")

        st.markdown("#### Goal-market probability ladder")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Over 1.5", f"{result['over15']*100:.1f}%")
        p2.metric("Over 2.5", f"{result['probability']*100:.1f}%")
        p3.metric("Over 3.5", f"{result['over35']*100:.1f}%")
        p4.metric("Under 2.5", f"{(1-result['probability'])*100:.1f}%")

        if p >= 0.64:
            st.success("The model sees a comparatively strong Over 2.5 profile for this matchup.")
        elif p >= 0.56:
            st.warning("The model sees a moderate Over 2.5 profile. Review form and market price before drawing conclusions.")
        else:
            st.info("The current data does not produce a strong Over 2.5 signal.")

        st.caption("Probability estimates are model outputs, not guarantees or betting advice.")

with tab2:
    home_for_tab = st.session_state.get("last_match", (teams[0], teams[1]))[0]
    away_for_tab = st.session_state.get("last_match", (teams[0], teams[1]))[1]
    h = team_form_summary(df, home_for_tab, recent)
    a = team_form_summary(df, away_for_tab, recent)

    left, right = st.columns(2)
    with left:
        st.markdown(f"### {home_for_tab}")
        x1, x2, x3 = st.columns(3)
        x1.metric("Avg goals for", f"{h['gf']:.2f}")
        x2.metric("Avg goals against", f"{h['ga']:.2f}")
        x3.metric("O2.5 recent", f"{h['over25']*100:.0f}%")
        recent_home = team_recent(df, home_for_tab, recent)
        if not recent_home.empty:
            view = recent_home[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "TotalGoals"]].sort_values("Date", ascending=False)
            st.dataframe(view, use_container_width=True, hide_index=True)

    with right:
        st.markdown(f"### {away_for_tab}")
        y1, y2, y3 = st.columns(3)
        y1.metric("Avg goals for", f"{a['gf']:.2f}")
        y2.metric("Avg goals against", f"{a['ga']:.2f}")
        y3.metric("O2.5 recent", f"{a['over25']*100:.0f}%")
        recent_away = team_recent(df, away_for_tab, recent)
        if not recent_away.empty:
            view = recent_away[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "TotalGoals"]].sort_values("Date", ascending=False)
            st.dataframe(view, use_container_width=True, hide_index=True)

with tab3:
    if "last_prediction" not in st.session_state:
        st.info("Run a match forecast first to see the ensemble breakdown.")
    else:
        r = st.session_state["last_prediction"]
        st.markdown("### Ensemble components")
        comp = pd.DataFrame({
            "Component": ["Poisson scoring model", "Recent O2.5 frequency", "Recent goal environment"],
            "Probability": [r["poisson_probability"], r["empirical_probability"], r["recent_probability"]],
            "Weight": [0.58, 0.27, 0.15],
        })
        comp["Probability %"] = (comp["Probability"] * 100).round(1)
        comp["Weight %"] = (comp["Weight"] * 100).round(0)
        st.dataframe(comp[["Component", "Probability %", "Weight %"]], use_container_width=True, hide_index=True)

        st.markdown("### Interpretation")
        st.write(
            "The final Over 2.5 forecast combines a Poisson estimate from expected goals with the two teams' "
            "recent Over 2.5 frequency and recent total-goal environment. Confidence is based on model agreement "
            "and whether enough recent matches are available."
        )
        st.write(f"Current historical sample: **{r['sample_size']:,} matches**.")

with tab4:
    st.markdown("### Historical dataset")
    st.dataframe(df.sort_values("Date", ascending=False).head(250), use_container_width=True, hide_index=True)
    st.download_button(
        "Download loaded data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="goaledge_loaded_history.csv",
        mime="text/csv",
    )
    with st.expander("CSV format"):
        st.code(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "2025-08-15,Team A,Team B,2,1\n"
            "2025-08-16,Team C,Team D,0,0"
        )

st.markdown(
    '<div class="footer">GoalEdge · Football forecasting research dashboard · Probabilities are estimates, not guarantees.</div>',
    unsafe_allow_html=True,
)
