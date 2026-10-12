from datetime import date, datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
import streamlit as st
from engine import load_matches, GoalModel, simulate, backtest

st.set_page_config(page_title='Football Forecast Lab', page_icon='⚽', layout='wide')
st.title('⚽ Football Forecast Lab')
st.caption('Premier League forecasts • transparent models • historical validation')

@st.cache_data
def read_data(signature):
    return load_matches()

files = list(Path('data/raw').glob('*.csv'))
matches = read_data(tuple((str(p),p.stat().st_mtime) for p in files))
if matches.empty:
    st.info('Download historical results first: python fetch_data.py')
    st.stop()
today = datetime.now(ZoneInfo('America/Los_Angeles')).date()
matches = matches[matches.date <= pd.Timestamp(today)]
seasons = sorted(matches.season.unique(), reverse=True)
season = st.sidebar.selectbox('Season', seasons, format_func=lambda s:f'20{s[:2]}–20{s[2:]}')
start_year = 2000+int(season[:2])
cutoff = st.sidebar.date_input('Include results through', value=min(today,date(start_year+1,6,30)),
    min_value=date(start_year,7,1),max_value=date(start_year+1,6,30))
simulations = st.sidebar.select_slider('Season simulations',options=[1000,5000,10000,25000],value=10000)
season_matches = matches[matches.season==season]
teams = sorted(set(season_matches.home)|set(season_matches.away))
played = season_matches[season_matches.date<=pd.Timestamp(cutoff)]
if season != '2627':
    st.warning('Historical season selected. This is a retrospective forecast, not a current 2026–27 prediction. Current-season data must be available to forecast that season.')
st.caption(f'Data available through {matches.date.max():%d %b %Y}. Forecast uses results through {cutoff:%d %b %Y}.')
if not played.empty and (pd.Timestamp(cutoff)-played.date.max()).days > 7 and len(played)<380:
    st.warning(f'Season results stop at {played.date.max():%d %b %Y}. Later played matches may be missing and will be treated as unplayed. Refresh the source before using this as a current forecast.')
forecast_tab, match_tab, method_tab, validation_tab = st.tabs(['Season forecast','Match explorer','Model & assumptions','Backtesting'])

@st.cache_data
def forecast(signature,season,cutoff,simulations):
    data=load_matches()
    cutoff_end=pd.Timestamp(cutoff)+pd.Timedelta(days=1)
    model=GoalModel().fit(data,cutoff_end)
    season_data=data[data.season==season]
    teams=sorted(set(season_data.home)|set(season_data.away))
    return simulate(model,season_data[season_data.date<cutoff_end],teams,simulations)

with forecast_tab:
    try:
        with st.spinner('Fitting goal model and simulating the remaining season…'):
            result,distribution,remaining = forecast(tuple((str(p),p.stat().st_mtime) for p in files),season,cutoff,simulations)
        c1,c2,c3=st.columns(3)
        c1.metric('Title favourite',result.iloc[0].team)
        c2.metric('Favourite’s title chance',f'{result.iloc[0].title_probability:.1%}')
        c3.metric('Remaining matches',remaining)
        st.bar_chart(result.set_index('team').title_probability, horizontal=True)
        st.dataframe(result,hide_index=True,column_config={
            'title_probability':st.column_config.NumberColumn('Title chance',format='percent'),
            'top_four':st.column_config.NumberColumn('Top four',format='percent'),
            'relegation':st.column_config.NumberColumn('Relegation',format='percent'),
            'expected_points':st.column_config.NumberColumn('Expected points',format='%.1f'),
            'expected_position':st.column_config.NumberColumn('Expected finish',format='%.1f')},use_container_width=True)
        selected=st.selectbox('Explore finishing positions',teams)
        st.bar_chart(distribution[[selected]])
        st.caption('Points low/high show the 10th and 90th percentiles. Top four is a finishing-position probability, not a guarantee of Champions League qualification.')
        st.download_button('Download forecast CSV',result.to_csv(index=False),'forecast.csv','text/csv')
    except (ValueError,RuntimeError) as exc:
        st.error(str(exc))

with match_tab:
    left,right=st.columns(2)
    home=left.selectbox('Home club',teams)
    away=right.selectbox('Away club',[t for t in teams if t!=home])
    try:
        model=GoalModel().fit(matches,pd.Timestamp(cutoff)+pd.Timedelta(days=1))
        p=model.probabilities(home,away)
        lh,la=model.rates(home,away)
        cols=st.columns(3)
        for col,label,value in zip(cols,['Home win','Draw','Away win'],p):
            col.metric(label,f'{value:.1%}')
        st.caption(f'Model expected goals: {home} {lh:.2f} · {away} {la:.2f}. These are statistical goal expectations, not event-based xG data.')
    except (ValueError,RuntimeError) as exc:
        st.error(str(exc))

with method_tab:
    st.subheader('A reproducible baseline')
    st.write('The model estimates each club’s attack and defence strength using historical scores. Home advantage is learned from the data. Matches have a one-year weighting half-life; regularization keeps sparsely observed clubs near average.')
    st.latex(r'\lambda_{home}=\exp(\mu+h+a_{home}+d_{away})')
    st.latex(r'\lambda_{away}=\exp(\mu+a_{away}+d_{home})')
    st.write('Independent Poisson draws produce scorelines. Completed results stay fixed; every remaining home/away pairing is simulated. Finishing positions are ranked by points, goal difference, then goals scored.')
    st.warning('Baseline limitations: no injuries, lineups, transfer values, or event-based xG; team strength stays fixed during each simulation. Exact remaining fixture dates are not ingested yet. Ties after goals scored are resolved randomly rather than implementing head-to-head/play-off rules. Promoted clubs with no training history use average strength.')
    st.write('Historical season membership is inferred from that season’s observed clubs. Historical forecasts therefore assume the participant list was known. Future results never enter model fitting or completed standings.')
    st.info('UCL is the next milestone. Its forecasts are not implemented or represented by placeholder probabilities.')
    st.markdown('[Results source: Football-Data.co.uk](https://www.football-data.co.uk/)')

with validation_tab:
    st.write('Out-of-time evaluation: refit at each month’s start using only earlier results. Compare match log loss against historical home/draw/away frequencies. Lower log loss and Brier score are better.')
    valid=[s for s in seasons if len(matches[matches.date < matches[matches.season==s].date.min()])>=100]
    if valid:
        target=st.selectbox('Backtest season',valid)
        if st.button('Run historical backtest'):
            with st.spinner('Running monthly historical fits…'):
                try:
                    scores=backtest(matches,target)
                    cols=st.columns(4)
                    for col,label,value in zip(cols,['Model log loss','Baseline log loss','Multiclass Brier','Accuracy'],
                        [scores.log_loss.mean(),scores.baseline_log_loss.mean(),scores.brier.mean(),scores.correct.mean()]):
                        col.metric(label,f'{value:.3f}')
                    st.subheader('Home-win calibration')
                    scores['bin']=pd.cut(scores.home_probability,bins=[0,.2,.4,.6,.8,1],include_lowest=True)
                    calibration=scores.groupby('bin',observed=True).agg(predicted=('home_probability','mean'),observed=('actual',lambda x:(x==0).mean()),matches=('actual','size'))
                    calibration.index=calibration.index.astype(str)
                    st.line_chart(calibration[['predicted','observed']])
                    st.dataframe(calibration)
                    st.caption('Small bins are noisy. This evaluates match forecasts; it does not yet validate season-level title probabilities.')
                    st.download_button('Download backtest',scores.to_csv(index=False),'backtest.csv','text/csv')
                except (ValueError,RuntimeError) as exc:
                    st.error(str(exc))
    else:
        st.info('Download more historical seasons to enable backtesting.')
