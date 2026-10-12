"""Export a reproducible forecast and monthly backtest without the dashboard."""
import argparse
import json
from pathlib import Path
import pandas as pd
from engine import load_matches, GoalModel, simulate, backtest

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--season',default='2627')
    parser.add_argument('--cutoff',required=True,help='Last included result date, YYYY-MM-DD')
    parser.add_argument('--simulations',type=int,default=10000)
    parser.add_argument('--backtest',default=None,help='Optional historical season, e.g. 2526')
    args=parser.parse_args()
    data=load_matches()
    end=pd.Timestamp(args.cutoff)+pd.Timedelta(days=1)
    season=data[data.season==args.season]
    played=season[season.date<end]
    model=GoalModel().fit(data,end)
    result,positions,remaining=simulate(model,played,set(season.home)|set(season.away),args.simulations)
    root=Path('reports')
    root.mkdir(exist_ok=True)
    result.to_csv(root/'forecast.csv',index=False)
    positions.to_csv(root/'positions.csv',index_label='position')
    metadata={'season':args.season,'cutoff':args.cutoff,'last_result':str(played.date.max().date()),
        'simulations':args.simulations,'seed':42,'remaining_matches':remaining,
        'model':'regularized time-weighted independent Poisson',
        'caveat':'Missing results treated as unplayed; fixed strengths; simplified final tiebreak.'}
    if args.backtest:
        scores=backtest(data,args.backtest)
        scores.to_csv(root/'backtest.csv',index=False)
        metadata['backtest']={'season':args.backtest,'matches':len(scores),
            **scores[['log_loss','baseline_log_loss','brier','correct']].mean().to_dict()}
    (root/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(result.head().to_string(index=False))
    print(f'Saved outputs to {root.resolve()}')
