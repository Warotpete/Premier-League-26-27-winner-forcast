# Football Forecast Lab

[Open the live app](https://football-forecast.streamlit.app/)

Free-first Premier League prediction and model showcase using Python and Streamlit.
UCL is planned; no UCL probabilities are implemented yet.

## Local setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python fetch_data.py --start 2016 --end 2026
.venv\Scripts\python fetch_fixtures.py
.venv\Scripts\python -m streamlit run app.py
```

The downloader requests Premier League result CSVs from Football-Data.co.uk.
Unavailable seasons are reported and never replaced with fabricated data.
Season filenames use `2627.csv` for 2026–27. Match columns: Date, HomeTeam,
AwayTeam, FTHG, FTAG. Check the source's terms before redistributing data.

## Model

Regularized attack/defence Poisson goal model, learned home advantage,
one-year recency half-life. Only results through the selected cutoff enter
training. Independent scores produce consistent results and goal differences.
Actual results are fixed, and remaining home/away pairings are simulated.
The current-season gameweek view uses the official free FPL schedule. Completed
FPL scores override corresponding raw CSV results. Refresh it with
`python fetch_fixtures.py`; no API key is needed.
No injuries, transfers, or event-based xG are used.
Rare ties after points, GD and GF are randomly resolved. Unknown promoted
clubs receive average strength. Model parameters remain fixed during simulations.

## Current-season gameweek points

The main dashboard is fixed to 2026–27; historical season selection remains in
backtesting. Points by gameweek shows exact expected cumulative points (3 × win
probability + draw probability), actual totals, and simulated 10th–90th percentile
intervals for all clubs. Compare clubs, inspect any gameweek, and export all rows.
Gameweek 0 is the zero-points starting state. Future scores are excluded at the
selected cutoff. Missing past results are projected and explicitly flagged.
Postponed matches follow FPL gameweek assignment, including blank/double weeks;
this is not a calendar-date standings chart. Unassigned fixtures block the full
curve until refreshed. Each curve is conditional on one current forecast cutoff,
not a record of forecasts issued in earlier weeks.

Monthly chronological backtests report log loss, multiclass Brier score,
accuracy and home-win calibration against historical outcome frequencies.
These checks evaluate matches, not calibration of championship probabilities.
This is a baseline to improve and benchmark, not an established accuracy claim.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Export reproducible results

```powershell
python run_forecast.py --season 2627 --cutoff 2026-09-20 --backtest 2526
```

Outputs go to `reports/`, with cutoff, source freshness and validation metrics
recorded in `run_metadata.json`. The initially downloaded 2026–27 source stops
on September 20, 2026; these results are a snapshot, not an October 11 forecast.
