from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from models.WaveAnalyzer import WaveAnalyzer
from models.helpers import plot_pattern, convert_yf_data


def get_daily(symbol: str, start_days: int = 720) -> pd.DataFrame:
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(days=start_days)
    raw = yf.download(symbol, start=start_date, end=end_date)
    df = convert_yf_data(raw)
    return df


def main():
    symbol = "AAPL"  # change to "BTC-USD" if you want
    timeframe = "1D"

    df = get_daily(symbol, start_days=900)

    wa = WaveAnalyzer(df=df, verbose=False)

    # Sliding adaptive windows:
    # start 4 weeks; if no wave found, grow by 1 week up to 12 weeks
    results = wa.sliding_adaptive_impulses(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        slide_weeks=1,
        min_weeks=4,
        max_weeks=12,
        grow_weeks=1,
        up_to=10,
        top_n=1,
    )

    print(f"Found {len(results)} windows with at least one impulse candidate")

    for r in results:
        best = r["best"]
        window_df = r["window_df"]

        title = f"{symbol} {timeframe} {r['date_start']} to {r['date_end']}  (window={int(r['window_weeks'])}w, cfg={best.wave_config}, score={best.score:.3f})"

        plot_pattern(
            df=window_df,
            wave_pattern=best.pattern,
            title=title,
            symbol=symbol,
            timeframe=timeframe,
            rule_name=best.rule_name,
            score=best.score,
        )


if __name__ == "__main__":
    main()
