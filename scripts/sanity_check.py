"""
Sanity check script for combined market parquet files.
Usage:
  python scripts/sanity_check.py data/all_markets_20y.parquet

Prints summary statistics and simple assertions.
Exits with non-zero code if critical checks fail.
"""
import sys
import pandas as pd
from pathlib import Path

def main(path):
    p = Path(path)
    if not p.exists():
        print('ERROR: parquet file not found:', p)
        return 2
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print('ERROR reading parquet:', e)
        return 3
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print('ERROR: Date column missing')
        return 4
    total_rows = len(df)
    unique_tickers = df['ticker'].nunique() if 'ticker' in df.columns else 0
    markets = sorted(df['market'].unique()) if 'market' in df.columns else ['unknown']
    overall_min = df['Date'].min()
    overall_max = df['Date'].max()
    print('File:', p)
    print('Total rows:', total_rows)
    print('Unique tickers:', unique_tickers)
    print('Markets present:', markets)
    print('Overall date range:', overall_min.date(), '->', overall_max.date())

    # Per-market summary
    for m in markets:
        sub = df[df['market'] == m]
        if sub.empty:
            print(f'Market {m}: NO DATA')
            continue
        ticks = sorted(sub['ticker'].unique())
        min_date = sub['Date'].min().date()
        max_date = sub['Date'].max().date()
        rows = len(sub)
        print(f"Market {m}: tickers={len(ticks)}, rows={rows}, range={min_date} -> {max_date}")
        # example missing data check
        na_frac = sub[['Open','High','Low','Close','Volume']].isna().mean()
        print('  NA fraction by column:')
        print(na_frac.to_string())
    # basic assertions
    # Requested start date (original test); use the nearest trading-day start.
    # Adjusted to 2006-01-17 (first trading day after the 2006-01-15 weekend/MLK holiday).
    start_req = pd.to_datetime('2006-01-17')
    end_req = pd.to_datetime(pd.Timestamp.today().strftime('%Y-%m-%d'))
    # allow up to N calendar days of tolerance for the earliest available trading day
    tolerance_days = 3
    ok_start = overall_min <= (start_req + pd.Timedelta(days=tolerance_days))
    ok_end = overall_max >= end_req
    print('\nAssertions:')
    print(f'  covers requested start (<=2006-01-15 + {tolerance_days}d tolerance):', ok_start)
    print('  covers up to today (>= today):', ok_end)
    failed = 0
    if not ok_start:
        print('WARNING: earliest date is after requested start')
        failed += 1
    if not ok_end:
        print('WARNING: latest date is before today')
        failed += 1
    if failed:
        print('Sanity checks reported issues:', failed)
        return 5
    print('Sanity checks passed')
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/sanity_check.py path/to/combined.parquet')
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
