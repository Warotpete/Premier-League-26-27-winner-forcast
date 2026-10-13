"""Fetch official FPL fixtures and gameweek deadlines for 2026–27."""
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import pandas as pd

BASE='https://fantasy.premierleague.com/api/'
ALIASES={'Man Utd':'Man United','Spurs':'Tottenham',"Nott'm Forest": "Nott'm Forest",'Coventry City':'Coventry','Hull City':'Hull','Ipswich Town':'Ipswich'}

def get(path):
    request=urllib.request.Request(BASE+path,headers={'User-Agent':'FootballForecastLab/1.0'})
    return json.load(urllib.request.urlopen(request,timeout=30))

def fetch():
    bootstrap=get('bootstrap-static/')
    fixtures=get('fixtures/')
    events=bootstrap['events']
    if not events or not str(events[0]['deadline_time']).startswith('2026-'):
        raise ValueError('API season is not 2026–27; refusing to overwrite this season.')
    names={t['id']:ALIASES.get(t['name'],t['name']) for t in bootstrap['teams']}
    rows=[]
    for f in fixtures:
        rows.append({'id':f['id'],'gameweek':f['event'],'kickoff':f['kickoff_time'],
            'home':names[f['team_h']],'away':names[f['team_a']],
            'hg':f['team_h_score'],'ag':f['team_a_score'],'finished':f['finished']})
    frame=pd.DataFrame(rows)
    if len(frame)!=380 or frame.duplicated(['home','away']).any() or len(names)!=20:
        raise ValueError('Unexpected league schedule; refusing to save.')
    root=Path('data/fixtures')
    root.mkdir(parents=True,exist_ok=True)
    frame.to_csv(root/'2627.csv',index=False)
    metadata={'season':'2627','source':BASE+'fixtures/','fetched_at':datetime.now(timezone.utc).isoformat(),
        'events':[{'gameweek':e['id'],'deadline':e['deadline_time'],'finished':e['finished']} for e in events]}
    (root/'2627.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Saved {len(frame)} fixtures; {int(frame.finished.sum())} finished; {int(frame.gameweek.isna().sum())} without gameweek assignment.')

if __name__=='__main__':
    fetch()
