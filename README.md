# GoalEdge — Soccer Over 2.5 Predictor

GoalEdge is a Streamlit football forecasting website focused on **Over 2.5 goals** research.

## Website features

- Professional responsive dashboard
- Over 1.5, Over 2.5 and Over 3.5 probabilities
- BTTS probability
- Expected goals estimate
- Fair decimal odds for Over 2.5
- Team recent-form analysis
- Ensemble model breakdown
- Research confidence indicator
- football-data.org API support
- CSV upload support
- Streamlit Secrets support

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Worldwide league coverage

GoalEdge now supports dynamic competition discovery through API-Football instead of hard-coding a small league list. API-Football currently advertises 1,245 leagues and cups across countries and international competitions. Coverage and available seasons vary by competition.

For Streamlit Community Cloud, add:

```toml
API_FOOTBALL_KEY = "YOUR_API_FOOTBALL_KEY"
```

The original football-data.org source remains available as a secondary provider.

## football-data.org API

Create a football-data.org API token and either enter it in the website sidebar or store it securely.

For Streamlit Community Cloud, add this in **App settings → Secrets**:

```toml
FOOTBALL_DATA_TOKEN = "YOUR_TOKEN"
```

Do not commit your real API key to GitHub.

## CSV format

Required columns:

- `Date`
- `HomeTeam`
- `AwayTeam`
- `FTHG`
- `FTAG`

Example:

```csv
Date,HomeTeam,AwayTeam,FTHG,FTAG
2025-08-15,Team A,Team B,2,1
2025-08-16,Team C,Team D,0,0
```

## Current model

The website combines:

1. League scoring averages
2. Home/away attack and defence strengths
3. Expected-goal estimates
4. Poisson goal probabilities
5. Recent Over 2.5 rates
6. Recent total-goal environment
7. Sample-quality and model-agreement confidence

The current ensemble weights for Over 2.5 are:

- 58% Poisson scoring model
- 27% recent O2.5 frequency
- 15% recent goal environment

## Deploy with Streamlit Community Cloud

1. Connect your GitHub account to Streamlit Community Cloud.
2. Select repository `Mxstermind28/soccer-over25-predictor`.
3. Select branch `main`.
4. Set the main file to `app.py`.
5. Add `FOOTBALL_DATA_TOKEN` under app secrets.
6. Deploy.

## Important

GoalEdge is a football forecasting **research tool**. Model probabilities are estimates and should not be treated as guaranteed outcomes or guaranteed returns.


## Worldwide mode

Choose **API-Football — worldwide** in the sidebar. GoalEdge downloads the provider's current competition catalog, then lets you filter by country/region, search leagues and cups, choose an available season, and run the same GoalEdge model on that competition.

"Every league" means every competition available from the connected data provider. No football data API literally covers every organized league on Earth, and detailed data availability varies by season and competition.
