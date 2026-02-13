#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import time
from models.WaveAnalyzer import WaveAnalyzer
from models.helpers import convert_yf_data, plot_pattern


def get_daily(symbol: str, start_days: int = 720) -> pd.DataFrame:
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(days=start_days)
    raw = yf.download(symbol, start=start_date, end=end_date)
    df = convert_yf_data(raw)
    return df


def run_sliding(symbol: str,
                days: int = 720,
                slide_weeks: int = 1,
                min_weeks: int = 4,
                max_weeks: int = 12,
                up_to: int = 10,
                grow_weeks: int = 1,
                top_n: int = 1):
    df = get_daily(symbol, start_days=days)
    wa = WaveAnalyzer(df=df, verbose=False)

    try:
        results = wa.sliding_adaptive_impulses(
            df=df,
            symbol=symbol,
            timeframe="1D",
            slide_weeks=slide_weeks,
            min_weeks=min_weeks,
            max_weeks=max_weeks,
            up_to=up_to,
            grow_weeks=grow_weeks,
            top_n=top_n,
        )
    except Exception as e:
        # Fall back to a resilient manual loop: try each start_row and continue on errors
        print(f"sliding_adaptive_impulses raised an exception: {e}. Falling back to manual per-window scanning.")
        results = []
        bars_per_week = wa._bars_per_week("1D")
        slide_step = slide_weeks * bars_per_week
        min_len = min_weeks * bars_per_week
        start_row = 0
        while start_row <= len(df) - min_len:
            try:
                res = wa.find_best_impulse_adaptive_window(
                    base_df=df,
                    start_row=start_row,
                    timeframe="1D",
                    min_weeks=min_weeks,
                    max_weeks=max_weeks,
                    up_to=up_to,
                    grow_weeks=grow_weeks,
                    top_n=top_n,
                )
            except Exception as e2:
                print(f"Error scanning window at start_row={start_row}: {e2}")
                res = None

            if res:
                results.append(res)

            start_row += slide_step

    print(f"Found {len(results)} detection windows")

    for i, r in enumerate(results):
        best = r["best"]
        window_df = r["window_df"]
        title = f"{symbol} 1D {r['date_start']} to {r['date_end']} (window={int(r['window_weeks'])}w, cfg={best.wave_config}, score={best.score:.3f})"

        print(f"Saving detection {i+1}/{len(results)}: {title}")
        plot_pattern(
            df=window_df,
            wave_pattern=best.pattern,
            title=title,
            symbol=symbol,
            timeframe="1D",
            rule_name=best.rule_name,
            score=best.score,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sliding adaptive scan and export all detections')
    parser.add_argument('symbol', help='Ticker symbol (e.g. AAPL)')
    parser.add_argument('--days', type=int, default=720)
    parser.add_argument('--slide_weeks', type=int, default=1)
    parser.add_argument('--min_weeks', type=int, default=4)
    parser.add_argument('--max_weeks', type=int, default=12)
    parser.add_argument('--up_to', type=int, default=10)
    parser.add_argument('--grow_weeks', type=int, default=1)
    parser.add_argument('--top_n', type=int, default=1)

    args = parser.parse_args()
    run_sliding(
        symbol=args.symbol,
        days=args.days,
        slide_weeks=args.slide_weeks,
        min_weeks=args.min_weeks,
        max_weeks=args.max_weeks,
        up_to=args.up_to,
        grow_weeks=args.grow_weeks,
        top_n=args.top_n,
    )
