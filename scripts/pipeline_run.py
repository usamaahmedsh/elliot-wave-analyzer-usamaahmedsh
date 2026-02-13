#!/usr/bin/env python3
"""Orchestrator for the new pipeline: async fetch -> parallel window scans -> optional GPU scoring -> export results
"""
import argparse
import asyncio
import time
import os
from pathlib import Path
import zipfile
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.fetcher import fetch_symbols
from pipeline.executor import parallel_scan_windows
from pipeline.gpu_accel import GPUAccelerator
import json
from pathlib import Path


def build_windows_for_df(df, cfg: PipelineConfig):
    # compute bars per week heuristic (same as WaveAnalyzer)
    bars_per_week = 1 if False else 7
    slide_step = cfg.slide_weeks * bars_per_week
    min_len = cfg.min_weeks * bars_per_week

    # prepare base arrays once to avoid repeated conversions in workers
    try:
        lows_arr = df['Low'].to_numpy()
        highs_arr = df['High'].to_numpy()
        dates_arr = df['Date'].to_numpy()
    except Exception:
        # fallback: convert via list
        lows_arr = np.array(list(df['Low']))
        highs_arr = np.array(list(df['High']))
        dates_arr = np.array(list(df['Date']))

    windows = []
    start_row = 0
    while start_row <= len(df) - min_len:
        for window_len in range(min_len, cfg.max_weeks * bars_per_week + 1, cfg.grow_weeks * bars_per_week):
            if start_row + window_len > len(df):
                break
            # include numpy arrays in the context so workers can avoid pandas slicing/copying
            windows.append((start_row, window_len, {'base_df': df, 'lows': lows_arr, 'highs': highs_arr, 'dates': dates_arr}))
        start_row += slide_step
    return windows


async def run_pipeline(symbols, cfg: PipelineConfig):
    # async fetch (respect config concurrency)
    phase_times = {}
    t0 = time.time()
    fetch_start = time.time()
    fetched = await fetch_symbols(symbols, start_days=cfg.days, concurrency=cfg.concurrency)
    phase_times['fetch'] = time.time() - fetch_start

    all_results = {}
    all_payloads = []
    for s, df in fetched.items():
        if df.empty:
            print(f"No data for {s}, skipping")
            continue

        bw_start = time.time()
        windows = build_windows_for_df(df, cfg)
        print(f"{s}: prepared {len(windows)} windows to scan (cfg up_to={cfg.up_to})")

        # cheap volatility pre-filter: drop windows whose close-return std is below min_volatility
        if getattr(cfg, 'min_volatility', 0.0) and len(windows) > 0:
            filtered = []
            for w in windows:
                start_row, window_len, context = w
                wnd = context['base_df'].iloc[start_row : start_row + window_len]
                # prefer 'Close' or 'Adj Close'
                if 'Close' in wnd.columns:
                    series = wnd['Close'].astype(float)
                elif 'Adj Close' in wnd.columns:
                    series = wnd['Adj Close'].astype(float)
                else:
                    # fallback to first numeric column
                    series = wnd.select_dtypes(include=['number']).iloc[:, 0]
                if series.size < 2:
                    vol = 0.0
                else:
                    ret = series.pct_change().dropna()
                    vol = float(ret.std()) if not ret.empty else 0.0
                if vol >= float(cfg.min_volatility):
                    filtered.append(w)
            print(f"{s}: filtered windows by volatility {len(windows)} -> {len(filtered)} (min_vol={cfg.min_volatility})")
            windows = filtered

        # optional skip flat windows: if enabled, require simple price range > small epsilon
        if getattr(cfg, 'skip_flat_windows', False) and len(windows) > 0:
            filtered2 = []
            for w in windows:
                start_row, window_len, context = w
                wnd = context['base_df'].iloc[start_row : start_row + window_len]
                if 'Low' in wnd.columns and 'High' in wnd.columns:
                    lo = float(wnd['Low'].min())
                    hi = float(wnd['High'].max())
                else:
                    nums = wnd.select_dtypes(include=['number'])
                    lo = float(nums.min().min())
                    hi = float(nums.max().max())
                if (hi - lo) / (lo + 1e-9) > 1e-4:
                    filtered2.append(w)
            print(f"{s}: skipped flat windows {len(windows)} -> {len(filtered2)}")
            windows = filtered2

        phase_times.setdefault(s, {})
        phase_times[s]['build_windows'] = time.time() - bw_start

        # limit number of windows to keep runtime reasonable (use config cap)
        max_windows = max(1, min(len(windows), cfg.max_windows))
        windows = windows[:max_windows]

        # --- vectorized pre-score: compute cheap features per-window and prune ---
        pre_top_k = int(getattr(cfg, 'pre_score_top_k', 0))
        pre_thresh = float(getattr(cfg, 'pre_score_threshold', 0.0))
        pre_weights = getattr(cfg, 'pre_score_weights', (0.4, 0.3, 0.2, 0.1))
        if (pre_top_k > 0 or pre_thresh > 0.0) and len(windows) > 0:
            feats = []
            for w in windows:
                sr, wl, ctx = w
                lows_arr = ctx.get('lows')
                highs_arr = ctx.get('highs')
                if lows_arr is None:
                    wnd = ctx['base_df'].iloc[sr: sr + wl]
                    lows_win = wnd['Low'].to_numpy()
                    highs_win = wnd['High'].to_numpy()
                else:
                    lows_win = lows_arr[sr: sr + wl]
                    highs_win = highs_arr[sr: sr + wl]
                # volatility: std of pct changes of lows (or closes)
                if lows_win.size < 2:
                    vol = 0.0
                else:
                    rets = (lows_win[1:] - lows_win[:-1]) / (lows_win[:-1] + 1e-9)
                    vol = float(rets.std())
                lo = float(lows_win.min())
                hi = float(highs_win.max())
                ran = (hi - lo) / (lo + 1e-9)
                # slope: simple (last-first)/n
                slope = abs(float((lows_win[-1] - lows_win[0]) / (len(lows_win) + 1e-9)))
                # extrema count via fast numba helper if available
                try:
                    from models.functions import count_extrema
                    ext = int(count_extrema(lows_win))
                except Exception:
                    ext = 0
                feats.append([vol, ran, float(ext), slope])

            # score features (use GPU scorer if available)
            gpu = GPUAccelerator() if cfg.gpu_enabled else None
            if gpu is not None:
                scores = gpu.score_features(feats)
            else:
                # linear combination
                w0, w1, w2, w3 = pre_weights
                scores = [w0 * f[0] + w1 * f[1] + w2 * min(f[2] / 10.0, 1.0) + w3 * f[3] for f in feats]

            # attach scores and filter
            scored = list(zip(windows, scores))
            # apply threshold then optionally top_k
            if pre_thresh > 0.0:
                scored = [ws for ws in scored if ws[1] >= pre_thresh]
            if pre_top_k > 0 and scored:
                scored = sorted(scored, key=lambda x: x[1], reverse=True)[:pre_top_k]
            windows = [ws[0] for ws in scored]

        cfg_dict = {
            'up_to': cfg.up_to,
            'top_n': cfg.top_n,
            'max_combinations': cfg.max_combinations,
            'chunk_size': cfg.chunk_size,
        }

        scan_start = time.time()
        results = parallel_scan_windows(windows, cfg=cfg_dict, processes=cfg.processes)
        phase_times[s]['scan'] = time.time() - scan_start

        # optional GPU scoring (batched)
        gpu = GPUAccelerator() if cfg.gpu_enabled else None
        if gpu is not None and results:
            bests = [r['best'] for r in results if r and 'best' in r]
            if bests:
                try:
                    new_scores = gpu.score_candidates(bests)
                    for r, sc in zip([r for r in results if r and 'best' in r], new_scores):
                        try:
                            r['best'].score = sc
                        except Exception:
                            pass
                except Exception:
                    # if GPU scoring fails, continue with original scores
                    pass

        all_results[s] = results

        # export detections: save images per-detection and collect payloads into a single JSON later
        # the images directory is configured in the script's main block via set_images_dir(...)
        from models.helpers import plot_pattern
        # determine top-N results to plot to limit I/O
        to_plot_idx = set()
        if cfg.save_images and results:
            try:
                sorted_idx = sorted(range(len(results)), key=lambda i: getattr(results[i]['best'], 'score', 0.0), reverse=True)
                top_n = max(1, int(getattr(cfg, 'save_images_top_n', 1)))
                to_plot_idx = set(sorted_idx[:top_n])
            except Exception:
                to_plot_idx = set()

        for i, r in enumerate(results):
            best = r['best']
            # reconstruct window DataFrame only when needed (avoids pickling DataFrames across processes)
            window_df = df.iloc[r['start_row'] : r['start_row'] + r['window_len']].reset_index(drop=True)
            title = f"{s} {r['date_start']} to {r['date_end']} (window={int(r['window_len']/7)}w, cfg={best.wave_config}, score={best.score:.3f})"
            if cfg.save_images and i in to_plot_idx:
                payload = plot_pattern(
                    df=window_df,
                    wave_pattern=best.pattern,
                    title=title,
                    symbol=s,
                    timeframe='1D',
                    rule_name=best.rule_name,
                    score=best.score,
                )
            else:
                # skip plotting but collect a minimal payload
                payload = {
                    'symbol': s,
                    'date_start': r.get('date_start'),
                    'date_end': r.get('date_end'),
                    'window_len': int(r.get('window_len', 0)),
                    'wave_config': getattr(best, 'wave_config', None),
                    'rule_name': getattr(best, 'rule_name', None),
                    'score': getattr(best, 'score', None),
                }
            if payload is not None:
                # attach extra metadata about detection window
                payload.update({
                    "date_start": r.get("date_start"),
                    "date_end": r.get("date_end"),
                    "window_len": int(r.get("window_len", 0)),
                    "wave_config": getattr(best, "wave_config", None),
                })
                all_payloads.append(payload)

    # write a single consolidated JSON result file into data/
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_data_dir = Path('data')
    out_data_dir.mkdir(exist_ok=True)
    results_path = out_data_dir / f'results_run_{ts}.json'
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(all_payloads, f, indent=2, default=str)

    print(f"Wrote {results_path} (images remain under ./images)")
    phase_times['total'] = time.time() - t0
    # print profiling summary if requested
    if getattr(cfg, 'profile', False):
        print('\nProfiling summary (seconds):')
        for k, v in phase_times.items():
            if isinstance(v, dict):
                print(f"Symbol: {k}")
                for kk, vv in v.items():
                    print(f"  {kk}: {vv:.2f}")
            else:
                print(f"{k}: {v:.2f}")
    return all_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('symbols', nargs='+')
    p.add_argument('--config', type=str, default='configs.yaml', help='path to YAML config file')
    p.add_argument('--source', type=str, default='yfinance', choices=['yfinance', 'hf'], help='data source')
    p.add_argument('--processes', type=int, default=None)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--out-dir', type=str, default='output', help='output directory (contains images/ and latest_results.json)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = PipelineConfig.load_from_file(args.config)
    # allow CLI overrides for a couple of knobs
    if args.processes is not None:
        cfg.processes = args.processes
    if args.gpu:
        cfg.gpu_enabled = True

    # prepare output dirs and configure helpers
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images = out_dir / 'images'
    out_images.mkdir(parents=True, exist_ok=True)

    # set images dir used by models.helpers
    from models.helpers import set_images_dir
    set_images_dir(str(out_images))

    # delete previous runs (timestamped files) before starting a fresh run
    import glob
    for f in glob.glob('data/results_run_*.json'):
        try:
            Path(f).unlink()
        except Exception:
            pass
    try:
        (out_dir / 'latest_results.json').unlink()
    except Exception:
        pass
    # clear images dir
    for p in out_images.glob('*'):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass

    # pass source through to fetcher by capturing it in the coroutine
    async def _run():
        # monkeypatch fetch_symbols import location by partialing source argument
        from functools import partial
        import pipeline.fetcher as fetcher_mod
        fetcher_mod.fetch_symbols = partial(fetcher_mod.fetch_symbols, source=(args.source))

        # ensure this run uses last 365 days as requested
        cfg.days = 365

        # run pipeline
        results = await run_pipeline(args.symbols, cfg)

        # after run, move consolidated results (timestamped) to out_dir/latest_results.json if present
        # run_pipeline writes a timestamped results file in data/; attempt to find the most recent one
        import glob
        from pathlib import Path as _P
        matches = sorted(glob.glob('data/results_run_*.json'))
        if matches:
            latest = matches[-1]
            _P(latest).rename(out_dir / 'latest_results.json')
        return results

    asyncio.run(_run())
