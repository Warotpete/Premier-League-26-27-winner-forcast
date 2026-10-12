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
    result[['hg','ag']] = result[['hg','ag']].astype(int)
    return result.reset_index(drop=True)

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
