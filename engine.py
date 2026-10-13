"""Time-weighted Poisson baseline and Premier League season simulation."""
from itertools import permutations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

def load_matches(root='data/raw'):
    frames = []
    for file in sorted(Path(root).glob('*.csv')):
        frame = pd.read_csv(file)
        frame = frame.rename(columns={'Date':'date', 'HomeTeam':'home', 'AwayTeam':'away', 'FTHG':'hg', 'FTAG':'ag'})
        if not {'date','home','away','hg','ag'}.issubset(frame.columns):
            continue
        frame['date'] = pd.to_datetime(frame['date'], format='mixed', dayfirst=True, errors='coerce')
        frame['season'] = file.stem
        frames.append(frame[['date','home','away','hg','ag','season']].dropna())
    if not frames:
        return pd.DataFrame(columns=['date','home','away','hg','ag','season'])
    result = pd.concat(frames).sort_values('date').drop_duplicates(['date','home','away'])
    official_path = Path('data/fixtures/2627.csv')
    if root == 'data/raw' and official_path.exists():
        official = load_fixtures(official_path)
        done = official[official.finished & official.hg.notna() & official.ag.notna()].copy()
        done['date'] = done.kickoff.dt.tz_localize(None).dt.normalize()
        done['season'] = '2627'
        done = done[['date','home','away','hg','ag','season']].dropna()
        result = pd.concat([result,done]).drop_duplicates(['season','home','away'],keep='last').sort_values('date')
    result[['hg','ag']] = result[['hg','ag']].astype(int)
    return result.reset_index(drop=True)

def load_fixtures(path='data/fixtures/2627.csv'):
    fixtures = pd.read_csv(path)
    fixtures['kickoff'] = pd.to_datetime(fixtures.kickoff,utc=True,errors='coerce')
    fixtures['gameweek'] = pd.to_numeric(fixtures.gameweek,errors='coerce').astype('Int64')
    fixtures['finished'] = fixtures.finished.astype(str).str.lower().eq('true')
    if len(fixtures)!=380 or fixtures.duplicated(['home','away']).any():
        raise ValueError('Expected a unique 380-match season schedule.')
    teams=set(fixtures.home)|set(fixtures.away)
    if len(teams)!=20 or (fixtures.home==fixtures.away).any():
        raise ValueError('Invalid season teams.')
    assigned=fixtures.gameweek.dropna()
    if not assigned.between(1,38).all():
        raise ValueError('Gameweeks must be between 1 and 38.')
    return fixtures

def gameweek_points(model, fixtures, cutoff, simulations=10000, seed=42):
    """Cumulative points by assigned FPL gameweek, conditional on known results.

    Unassigned postponed fixtures cannot be placed on a gameweek curve.
    Actual totals are shown only while every contributing team fixture is known.
    """
    if fixtures.gameweek.isna().any():
        raise ValueError('Some fixtures have no assigned gameweek. Refresh the schedule after reassignment; a complete gameweek forecast cannot yet be produced.')
    if simulations<1:
        raise ValueError('At least one simulation is required.')
    teams=sorted(set(fixtures.home)|set(fixtures.away))
    index={t:i for i,t in enumerate(teams)}
    n=len(teams)
    cutoff_end=pd.Timestamp(cutoff,tz='UTC')+pd.Timedelta(days=1)
    known=fixtures.finished & fixtures.hg.notna() & fixtures.ag.notna() & (fixtures.kickoff<cutoff_end)
    fixtures=fixtures.copy()
    fixtures['known']=known
    samples=np.zeros((simulations,n),dtype=int)
    expected=np.zeros(n)
    actual=np.zeros(n,dtype=int)
    actual_complete=np.ones(n,dtype=bool)
    rng=np.random.default_rng(seed)
    rows=[]
    for week in range(0,39):
        for f in fixtures[fixtures.gameweek==week].sort_values('id').itertuples():
            i,j=index[f.home],index[f.away]
            if f.known:
                hp=3 if f.hg>f.ag else 1 if f.hg==f.ag else 0
                ap=3 if f.ag>f.hg else 1 if f.hg==f.ag else 0
                samples[:,i]+=hp
                samples[:,j]+=ap
                expected[i]+=hp
                expected[j]+=ap
                actual[i]+=hp
                actual[j]+=ap
            else:
                lh,la=model.rates(f.home,f.away)
                hg,ag=rng.poisson(lh,simulations),rng.poisson(la,simulations)
                samples[:,i]+=np.where(hg>ag,3,np.where(hg==ag,1,0))
                samples[:,j]+=np.where(ag>hg,3,np.where(hg==ag,1,0))
                p=model.probabilities(f.home,f.away)
                expected[i]+=3*p[0]+p[1]
                expected[j]+=3*p[2]+p[1]
                actual_complete[[i,j]]=False
        low,high=np.quantile(samples,[.1,.9],axis=0)
        for i,team in enumerate(teams):
            rows.append({'gameweek':week,'team':team,'expected_points':expected[i],
                'points_low':low[i],'points_high':high[i],
                'actual_points':actual[i] if actual_complete[i] else np.nan})
    return pd.DataFrame(rows)

class GoalModel:
    def fit(self, matches, cutoff):
        matches = matches[matches.date < pd.Timestamp(cutoff)].copy()
        if len(matches) < 100:
            raise ValueError('At least 100 earlier matches are required.')
        self.teams = sorted(set(matches.home) | set(matches.away))
        self.index = {team:i for i,team in enumerate(self.teams)}
        n = len(self.teams)
        h = matches.home.map(self.index).to_numpy()
        a = matches.away.map(self.index).to_numpy()
        hg, ag = matches.hg.to_numpy(), matches.ag.to_numpy()
        age = (pd.Timestamp(cutoff) - matches.date).dt.days.to_numpy()
        weights = np.exp(-np.log(2)*age/365)
        def loss(x):
            attack, defense = x[2:2+n], x[2+n:]
            lh = x[0]+x[1]+attack[h]+defense[a]
            la = x[0]+attack[a]+defense[h]
            return np.sum(weights*(np.exp(lh)-hg*lh+np.exp(la)-ag*la)) + 3*np.sum(x[2:]**2)
        result = minimize(loss, np.zeros(2+2*n), method='L-BFGS-B', bounds=[(-2,2)]*(2+2*n))
        if not result.success:
            raise RuntimeError(f'Model did not converge: {result.message}')
        self.params = result.x
        return self

    def rates(self, home, away):
        n = len(self.teams)
        def strength(team):
            i = self.index.get(team)
            return (0.,0.) if i is None else (self.params[2+i], self.params[2+n+i])
        ha, hd = strength(home)
        aa, ad = strength(away)
        return np.exp(self.params[0]+self.params[1]+ha+ad), np.exp(self.params[0]+aa+hd)

    def probabilities(self, home, away):
        lh, la = self.rates(home, away)
        cap = max(15, int(poisson.ppf(1-1e-10, max(lh,la))))
        matrix = np.outer(poisson.pmf(np.arange(cap+1),lh), poisson.pmf(np.arange(cap+1),la))
        p = np.array([np.tril(matrix,-1).sum(), np.trace(matrix), np.triu(matrix,1).sum()])
        return p/p.sum()

def simulate(model, played, teams, simulations=10000, seed=42):
    teams = sorted(teams)
    idx = {t:i for i,t in enumerate(teams)}
    if len(teams) != 20:
        raise ValueError(f'Expected 20 season teams, found {len(teams)}.')
    if len(played) != len(played.drop_duplicates(['home','away'])):
        raise ValueError('Duplicate season fixtures.')
    points = np.zeros((simulations,20),dtype=int)
    gd = points.copy()
    gf = points.copy()
    def add(home, away, hg, ag):
        i,j=idx[home],idx[away]
        points[:,i] += np.where(hg>ag,3,np.where(hg==ag,1,0))
        points[:,j] += np.where(ag>hg,3,np.where(hg==ag,1,0))
        gd[:,i] += hg-ag
        gd[:,j] += ag-hg
        gf[:,i] += hg
        gf[:,j] += ag
    for row in played.itertuples():
        add(row.home,row.away,row.hg,row.ag)
    completed = set(zip(played.home,played.away))
    remaining = [(h,a) for h,a in permutations(teams,2) if (h,a) not in completed]
    rng = np.random.default_rng(seed)
    for h,a in remaining:
        lh,la = model.rates(h,a)
        add(h,a,rng.poisson(lh,simulations),rng.poisson(la,simulations))
    # Random resolution only when points, goal difference and goals scored all tie.
    order = np.lexsort((rng.random(points.shape),-gf,-gd,-points),axis=1)
    positions = np.empty_like(order)
    positions[np.arange(simulations)[:,None],order] = np.arange(1,21)
    result = pd.DataFrame({'team':teams,'title_probability':(positions==1).mean(axis=0),
        'expected_points':points.mean(axis=0),'points_low':np.quantile(points,.1,axis=0),
        'points_high':np.quantile(points,.9,axis=0),'expected_position':positions.mean(axis=0),
        'top_four':(positions<=4).mean(axis=0),'relegation':(positions>=18).mean(axis=0)})
    distributions = pd.DataFrame({t:np.bincount(positions[:,i],minlength=21)[1:]/simulations for i,t in enumerate(teams)},index=range(1,21))
    return result.sort_values('title_probability',ascending=False), distributions, len(remaining)

def backtest(matches, season):
    test = matches[matches.season==season].sort_values('date')
    if test.empty:
        raise ValueError('No matches in selected season.')
    # Refit once per calendar month, always on data before the month begins.
    rows=[]
    for month, batch in test.groupby(test.date.dt.to_period('M')):
        model = GoalModel().fit(matches,month.start_time)
        prior = matches[matches.date<month.start_time]
        counts = np.bincount(np.where(prior.hg>prior.ag,0,np.where(prior.hg==prior.ag,1,2)),minlength=3)+1
        baseline = counts/counts.sum()
        for match in batch.itertuples():
            p=model.probabilities(match.home,match.away)
            actual=0 if match.hg>match.ag else 1 if match.hg==match.ag else 2
            rows.append({'date':match.date,'home':match.home,'away':match.away,'actual':actual,
                'home_probability':p[0],'log_loss':-np.log(max(p[actual],1e-12)),
                'baseline_log_loss':-np.log(baseline[actual]),'correct':int(p.argmax()==actual),
                'brier':np.sum((p-np.eye(3)[actual])**2)})
    return pd.DataFrame(rows)
