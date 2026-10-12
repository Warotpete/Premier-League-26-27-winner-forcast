"""Download freely available league results; never invent unavailable seasons."""
from pathlib import Path
import argparse
import urllib.request
import pandas as pd

def fetch(start, end):
    root = Path('data/raw')
    root.mkdir(parents=True, exist_ok=True)
    for year in range(start, end + 1):
        code = f'{year % 100:02d}{(year + 1) % 100:02d}'
        url = f'https://www.football-data.co.uk/mmz4281/{code}/E0.csv'
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'FootballModelShowcase/1.0'})
            payload = urllib.request.urlopen(request, timeout=30).read()
            from io import BytesIO
            frame = pd.read_csv(BytesIO(payload))
            required = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'}
            if not required.issubset(frame.columns):
                raise ValueError('Response does not contain match results')
            (root / f'{code}.csv').write_bytes(payload)
            print(f'{code}: downloaded {len(frame)} rows')
        except Exception as exc:
            print(f'{code}: unavailable ({exc})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=2016)
    parser.add_argument('--end', type=int, default=2026)
    args = parser.parse_args()
    fetch(args.start, args.end)
