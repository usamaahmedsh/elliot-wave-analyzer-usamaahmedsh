#!/usr/bin/env python3
"""Build a small starter validation set (CSV) by running a short pipeline on GOOG.

This script is conservative: it picks a few windows where the scanner found a
pattern (labels them 'true') and a few windows where no pattern was found
(labels 'false'). Labels are suggestions and should be reviewed by a human.
"""
import asyncio
import csv
from pathlib import Path
import time

from pipeline.config import PipelineConfig
from pipeline.fetcher import fetch_symbols
from scripts.pipeline_run import build_windows_for_df
from pipeline.numba_warm import prewarm_numba


OUT = Path('tests') / 'validation_set.csv'
OUT.parent.mkdir(exist_ok=True)


def write_rows(rows):
    with OUT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['symbol', 'start_row', 'window_len', 'label', 'notes'])
        w.writeheader()
        w.writerows(rows)


async def main():
    cfg = PipelineConfig.load_from_file('configs.yaml')
    cfg.days = 365
    cfg.max_windows = 50
    cfg.processes = 1
    cfg.profile = False

    print('fetching GOOG...')
    fetched = await fetch_symbols(['GOOG'], start_days=cfg.days, concurrency=1, source='yfinance')
    df = fetched.get('GOOG')
    if df is None or df.empty:
        print('no data for GOOG')
        return

    print('building windows...')
    windows, shm = build_windows_for_df(df, cfg)
    print('got', len(windows), 'windows')

    # pre-warm numba to avoid first-call overhead
    try:
        prewarm_numba()
    except Exception:
        pass

    # Synchronous scan per-window (avoid multiprocessing) to pick positives/negatives
    positives = []
    negs = []
    picked = set()

    for idx, w in enumerate(windows):
        if len(positives) >= 6 and len(negs) >= 6:
            break
        sr, wl, ctx = w
        # obtain arrays
        if ctx.get('lows') is not None:
            lows = ctx['lows'][sr: sr + wl]
            highs = ctx['highs'][sr: sr + wl]
            dates = ctx['dates'][sr: sr + wl]
        else:
            wnd = ctx['base_df'].iloc[sr: sr + wl]
            lows = wnd['Low'].to_numpy()
            highs = wnd['High'].to_numpy()
            dates = wnd['Date'].to_numpy()

        from models.WaveAnalyzer import WaveAnalyzer
        import numpy as _np
        try:
            local_idx_start = int(_np.argmin(lows))
        except Exception:
            continue

        wa = WaveAnalyzer(df=None, lows=lows, highs=highs, dates=dates, verbose=False)
        try:
            cand = wa.scan_impulses(idx_start=local_idx_start, up_to=min(10, cfg.up_to), top_n=1, scan_cfg={'cpu_batch_size': cfg.cpu_batch_size, 'cpu_top_k': cfg.cpu_top_k})
        except Exception:
            cand = []

        if cand and len(positives) < 6:
            positives.append({'symbol': 'GOOG', 'start_row': int(sr), 'window_len': int(wl), 'label': 'true', 'notes': 'auto-picked: scanner returned candidate'})
            picked.add(int(sr))
        elif not cand and len(negs) < 6:
            negs.append({'symbol': 'GOOG', 'start_row': int(sr), 'window_len': int(wl), 'label': 'false', 'notes': 'auto-picked: no candidate'})

    rows = positives + negs
    write_rows(rows)
    print('wrote', OUT, 'positives=', len(positives), 'negatives=', len(negs))


if __name__ == '__main__':
    asyncio.run(main())
