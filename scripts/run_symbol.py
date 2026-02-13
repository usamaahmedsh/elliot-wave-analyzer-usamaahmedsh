from __future__ import annotations
import argparse
import time
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern, convert_yf_data
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Iterable


def get_symbol_daily(symbol: str, start_days: int = 720) -> pd.DataFrame:
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(days=start_days)
    raw = yf.download(symbol, start=start_date, end=end_date)
    df = convert_yf_data(raw)
    return df


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("Date")
    o = df["Open"].resample("W").first()
    h = df["High"].resample("W").max()
    l = df["Low"].resample("W").min()
    c = df["Close"].resample("W").last()
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
    out = out.dropna()
    out = out.reset_index()
    return out


def scan_impulse(df: pd.DataFrame,
                 symbol: str,
                 timeframe: str) -> None:
    # defensive: ensure Low column exists and has values
    if "Low" not in df.columns or df["Low"].dropna().empty:
        print(f"[{timeframe}] No 'Low' data for {symbol}, skipping")
        return

    idx_start = int(np.argmin(np.array(list(df["Low"]))))

    wa = WaveAnalyzer(df=df, verbose=False)
    wave_options_impulse = WaveOptionsGenerator5(up_to=10)  # keep smaller for demo

    impulse = Impulse("impulse")
    leading_diagonal = LeadingDiagonal("leading_diagonal")
    rules_to_check = [impulse, leading_diagonal]

    print(f"[{timeframe}] Start at idx: {idx_start}")
    print(f"[{timeframe}] will run up to {wave_options_impulse.number / 1e6}M combinations.")

    wavepatterns_up = set()

    for new_option_impulse in wave_options_impulse.options_sorted:
        waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)

        if not waves_up:
            continue

        wavepattern_up = WavePattern(waves_up, verbose=True)

        for rule in rules_to_check:
            if wavepattern_up.check_rule(rule):
                if wavepattern_up in wavepatterns_up:
                    continue

                wavepatterns_up.add(wavepattern_up)
                score = wavepattern_up.score_rule(rule)

                print(f"[{timeframe}] {rule.name} found: {new_option_impulse.values}, score={score:.3f}")
                plot_pattern(
                    df=df,
                    wave_pattern=wavepattern_up,
                    title=f"{symbol} {timeframe} {new_option_impulse}",
                    symbol=symbol,
                    timeframe=timeframe,
                    rule_name=rule.name,
                    score=score,
                )

    if not wavepatterns_up:
        print(f"[{timeframe}] No valid impulse pattern found.")


def _symbols_from_file(path: str) -> list[str]:
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(s)
    except Exception as e:
        print(f"Failed to read symbols file '{path}': {e}")
    return out


def run_for_symbols(symbols: Iterable[str], start_days: int = 720, delay: float = 1.0) -> None:
    for symbol in symbols:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        print("\n==============================")
        print(f"Processing: {symbol}")
        print("==============================\n")
        try:
            df_daily = get_symbol_daily(symbol, start_days=start_days)
            # If sliding mode requested, run the adaptive sliding scan; otherwise run the anchored single-start scan
            if globals().get('FLAGS', None) and getattr(FLAGS, 'sliding', False):
                wa = WaveAnalyzer(df=df_daily, verbose=False)
                results = wa.sliding_adaptive_impulses(
                    df=df_daily,
                    symbol=symbol,
                    timeframe="1D",
                    slide_weeks=FLAGS.slide_weeks,
                    min_weeks=FLAGS.min_weeks,
                    max_weeks=FLAGS.max_weeks,
                    up_to=FLAGS.up_to,
                    grow_weeks=FLAGS.grow_weeks,
                    top_n=FLAGS.top_n,
                )
                print(f"[1D sliding] Found {len(results)} windows")
                for r in results:
                    best = r["best"]
                    window_df = r["window_df"]
                    title = f"{symbol} 1D {r['date_start']} to {r['date_end']} (window={int(r['window_weeks'])}w, cfg={best.wave_config}, score={best.score:.3f})"
                    plot_pattern(
                        df=window_df,
                        wave_pattern=best.pattern,
                        title=title,
                        symbol=symbol,
                        timeframe="1D",
                        rule_name=best.rule_name,
                        score=best.score,
                    )
            else:
                scan_impulse(df_daily, symbol=symbol, timeframe="1D")

            df_weekly = resample_to_weekly(df_daily)
            if globals().get('FLAGS', None) and getattr(FLAGS, 'sliding', False):
                wa2 = WaveAnalyzer(df=df_weekly, verbose=False)
                results2 = wa2.sliding_adaptive_impulses(
                    df=df_weekly,
                    symbol=symbol,
                    timeframe="1W",
                    slide_weeks=FLAGS.slide_weeks,
                    min_weeks=FLAGS.min_weeks,
                    max_weeks=FLAGS.max_weeks,
                    up_to=FLAGS.up_to,
                    grow_weeks=FLAGS.grow_weeks,
                    top_n=FLAGS.top_n,
                )
                print(f"[1W sliding] Found {len(results2)} windows")
                for r in results2:
                    best = r["best"]
                    window_df = r["window_df"]
                    title = f"{symbol} 1W {r['date_start']} to {r['date_end']} (window={int(r['window_weeks'])}w, cfg={best.wave_config}, score={best.score:.3f})"
                    plot_pattern(
                        df=window_df,
                        wave_pattern=best.pattern,
                        title=title,
                        symbol=symbol,
                        timeframe="1W",
                        rule_name=best.rule_name,
                        score=best.score,
                    )
            else:
                scan_impulse(df_weekly, symbol=symbol, timeframe="1W")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        # polite delay to avoid hitting rate limits when running many symbols
        time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run impulsive wave scan for one or more symbols")
    parser.add_argument("symbols", nargs="*", help="One or more ticker symbols (e.g. AAPL MSFT)")
    parser.add_argument("-f", "--file", help="Path to a file containing symbols (one per line)")
    parser.add_argument("-d", "--days", type=int, default=720, help="Lookback days for daily data")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay (seconds) between symbol requests")
    # Sliding/adaptive options
    parser.add_argument("--sliding", action="store_true", help="Run sliding adaptive-window scan instead of a single anchored scan")
    parser.add_argument("--slide-weeks", type=int, default=1, help="Weeks to slide between windows when using --sliding")
    parser.add_argument("--min-weeks", type=int, default=4, help="Minimum window size in weeks for adaptive growth")
    parser.add_argument("--max-weeks", type=int, default=12, help="Maximum window size in weeks for adaptive growth")
    parser.add_argument("--up-to", type=int, default=10, help="WaveOptionsGenerator up_to parameter (combinatorial depth)")
    parser.add_argument("--grow-weeks", type=int, default=1, help="Grow step in weeks for adaptive window")
    parser.add_argument("--top-n", type=int, default=1, help="Top-N candidates to keep per window when using sliding mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # expose parsed args to module so run_for_symbols can check sliding flags
    global FLAGS
    FLAGS = args
    symbols: list[str] = []
    if args.file:
        symbols.extend(_symbols_from_file(args.file))
    if args.symbols:
        symbols.extend(args.symbols)
    if not symbols:
        # default fallback
        symbols = ["AAPL"]

    # normalize symbols and deduplicate while preserving order
    seen = set()
    normalized: list[str] = []
    for s in symbols:
        s_up = s.strip().upper()
        if s_up and s_up not in seen:
            normalized.append(s_up)
            seen.add(s_up)

    run_for_symbols(normalized, start_days=args.days, delay=args.delay)
