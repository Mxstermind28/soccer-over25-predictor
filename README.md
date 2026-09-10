# Soccer Over 2.5 Goals Predictor

A Streamlit app that estimates the probability of **Over 2.5 total goals** in a soccer match.

## What it does

- Select a home and away team.
- Uses historical results to estimate expected goals.
- Calculates `P(total goals >= 3)` with a Poisson model.
- Shows Over 2.5 %, Under 2.5 %, expected goals, and sample size.
- Works from a CSV or from the football-data.org API.
- Includes a shrinkage step to reduce overfitting on small samples.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## Deploy from GitHub

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. On Streamlit Community Cloud, create a new app from that repository.
4. Set the main file to `app.py`.
5. Deploy.

## API option

The app supports football-data.org. Create an API token through their website and enter it in the app.

For public deployment, prefer storing the token as a Streamlit secret instead of hard-coding it.

Example `.streamlit/secrets.toml`:

```toml
FOOTBALL_DATA_TOKEN = "YOUR_TOKEN"
```

You can then adapt `app.py` to read `st.secrets["FOOTBALL_DATA_TOKEN"]`.

## CSV option

Required columns:

- `Date`
- `HomeTeam`
- `AwayTeam`
- `FTHG`
- `FTAG`

## Important limitation

This is a baseline statistical model, not a guaranteed predictor. A production-grade model should be backtested on historical seasons and can be improved with xG, shots, injuries, lineups, rest days, home advantage, market odds, and league-specific parameters.

Never interpret the probability as certainty.
