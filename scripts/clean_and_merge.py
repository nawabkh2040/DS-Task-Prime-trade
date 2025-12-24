"""Clean and merge Hyperliquid historical trades with Fear & Greed index.
Produces:
 - data/clean_historical.csv
 - data/clean_sentiment.csv
 - data/merged_by_date.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--historical", default="historical_data.csv")
    p.add_argument("--sentiment", default="fear_greed_index.csv")
    p.add_argument("--out_dir", default="data")
    return p.parse_args()


def load_sentiment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # prefer 'date' column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    else:
        raise ValueError('No date/timestamp column in sentiment file')
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def load_historical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # parse IST timestamp e.g. '02-12-2024 22:50'
    if 'Timestamp IST' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    elif 'Timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT
    # normalize numeric columns
    for col in ['Execution Price','Size Tokens','Size USD','Closed PnL','Fee']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = df['timestamp'].dt.floor('D')
    # trade-level win
    df['win'] = (df['Closed PnL'] > 0).astype(int)
    agg = df.groupby(['date']).agg(
        trades=('Account','count'),
        total_pnl=('Closed PnL','sum'),
        avg_pnl=('Closed PnL','mean'),
        win_rate=('win','mean'),
        notional_usd=('Size USD','sum')
    ).reset_index()
    return agg


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = load_historical(Path(args.historical))
    sent = load_sentiment(Path(args.sentiment))

    hist.to_csv(out_dir / 'clean_historical.csv', index=False)
    sent.to_csv(out_dir / 'clean_sentiment.csv', index=False)

    daily = aggregate_daily(hist)

    # merge on date
    sent_daily = sent.copy()
    sent_daily['date'] = sent_daily['date'].dt.floor('D')
    merged = daily.merge(sent_daily, on='date', how='left')
    merged.to_csv(out_dir / 'merged_by_date.csv', index=False)
    print('Wrote:', out_dir / 'clean_historical.csv', out_dir / 'clean_sentiment.csv', out_dir / 'merged_by_date.csv')

if __name__ == '__main__':
    main()
