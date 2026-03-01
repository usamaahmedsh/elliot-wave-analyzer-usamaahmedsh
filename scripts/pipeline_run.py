#!/usr/bin/env python3
"""
Elliott Wave Pipeline Runner
- Loads data from local parquet files (data/hf_dataset_complete/)
- Ticker-level parallelism via parallel_scan_tickers (ProcessPoolExecutor)
- Checkpoint/resume support
- Multi-interval support (1d, 4h, 1h, 1wk)
- HPC/SLURM compatible
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.config import PipelineConfig
from pipeline.executor import parallel_scan_tickers
from pipeline.numba_warm import prewarm_numba


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.checkpoint_dir / "processed_symbols.json"
        self.results_file   = self.checkpoint_dir / "partial_results.json"
        self.failed_file    = self.checkpoint_dir / "failed_symbols.json"
        self._processed_cache: Optional[Set[str]] = None
        self._pending_processed: List[str] = []
        self._pending_failed: Dict[str, str] = {}
        self.BATCH = 50

    def load_processed_symbols(self) -> Set[str]:
        if self._processed_cache is not None:
            return self._processed_cache
        try:
            if self.processed_file.exists():
                with open(self.processed_file) as f:
                    self._processed_cache = set(json.load(f))
                    return self._processed_cache
        except Exception:
            pass
        self._processed_cache = set()
        return self._processed_cache

    def load_partial_results(self) -> List[Dict]:
        try:
            if self.results_file.exists():
                with open(self.results_file) as f:
                    return json.load(f)
        except Exception:
            backup = self.checkpoint_dir / "partial_results.backup.json"
            if backup.exists():
                try:
                    with open(backup) as f:
                        return json.load(f)
                except Exception:
                    pass
        return []

    def flush(self, all_results: List[Dict]):
        """Batch-write processed + failed symbols and partial results."""
        try:
            if self._pending_processed:
                processed = self.load_processed_symbols()
                processed.update(self._pending_processed)
                self._processed_cache = processed
                with open(self.processed_file, 'w') as f:
                    json.dump(sorted(processed), f)
                self._pending_processed.clear()

            if self._pending_failed:
                existing = {}
                if self.failed_file.exists():
                    try:
                        with open(self.failed_file) as f:
                            existing = json.load(f)
                    except Exception:
                        pass
                existing.update(self._pending_failed)
                with open(self.failed_file, 'w') as f:
                    json.dump(existing, f)
                self._pending_failed.clear()

            # Backup + write results
            if self.results_file.exists():
                import shutil
                shutil.copy2(self.results_file,
                             self.checkpoint_dir / "partial_results.backup.json")
            with open(self.results_file, 'w') as f:
                json.dump(all_results, f, default=str)
        except Exception as e:
            print(f"  [checkpoint] flush error: {e}")

    def on_ticker_done(self, symbol: str, results: List[Dict],
                       error: Optional[str], all_results: List[Dict]):
        """Called from main process after each ticker completes."""
        self._pending_processed.append(symbol)
        if error:
            self._pending_failed[symbol] = error[:200]
        if len(self._pending_processed) >= self.BATCH:
            self.flush(all_results)


# =============================================================================
# DATA LOADER  (local parquet)
# =============================================================================

def load_parquet_data(data_dir: Path, interval: str,
                      verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load market data from a local parquet file.
    Expected path: <data_dir>/market_data_<interval>.parquet
    Required columns: Date, Open, High, Low, Close, Volume, ticker
    """
    parquet_path = data_dir / f"market_data_{interval}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    if verbose:
        print(f"\n\U0001f4c2 Loading {parquet_path}...")

    df_all = pd.read_parquet(parquet_path)

    if verbose:
        print(f"   Loaded {len(df_all):,} rows")

    # Normalise column names
    df_all.columns = [c.strip() for c in df_all.columns]

    # Detect ticker column
    ticker_col = next((c for c in df_all.columns
                       if c.lower() in ('ticker', 'symbol')), None)
    if ticker_col is None:
        raise ValueError("Parquet has no 'ticker' or 'symbol' column")
    if ticker_col != 'ticker':
        df_all = df_all.rename(columns={ticker_col: 'ticker'})

    # Detect date column
    date_col = next((c for c in df_all.columns
                     if c.lower() in ('date', 'datetime', 'timestamp')), None)
    if date_col is None:
        raise ValueError("Parquet has no date column")
    if date_col != 'Date':
        df_all = df_all.rename(columns={date_col: 'Date'})

    df_all['Date'] = pd.to_datetime(df_all['Date'])

    tickers = sorted(df_all['ticker'].unique().tolist())
    if verbose:
        print(f"   Grouped into {len(tickers)} tickers")

    ticker_data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        sub = df_all[df_all['ticker'] == t].sort_values('Date').reset_index(drop=True)
        ticker_data[t] = sub

    return ticker_data


# =============================================================================
# PIPELINE
# =============================================================================

def run_interval(
    interval:       str,
    ticker_data:    Dict[str, pd.DataFrame],
    cfg:            PipelineConfig,
    output_dir:     Path,
    checkpoint_dir: Path,
    n_workers:      int,
    resume:         bool,
    verbose:        bool,
):
    """
    Run the full wave scan for one interval.
    Returns list of pattern dicts.
    """
    ckpt = CheckpointManager(checkpoint_dir / interval)

    # --- resume filtering ---
    symbols = list(ticker_data.keys())
    if resume:
        processed = ckpt.load_processed_symbols()
        symbols = [s for s in symbols if s not in processed]
        if verbose and processed:
            print(f"  [resume] {len(processed)} already done, {len(symbols)} remaining")

    if not symbols:
        print(f"  [{interval}] All symbols already processed, loading checkpoint...")
        return ckpt.load_partial_results()

    # Only pass tickers that still need processing
    ticker_subset = {s: ticker_data[s] for s in symbols}

    # --- build cfg_dict for workers ---
    bars_per_week = {'1d': 7, '1wk': 1, '4h': 42, '1h': 168}.get(interval, 7)
    cfg_dict = {
        'up_to':                cfg.up_to,
        'top_n':                cfg.top_n,
        'cpu_batch_size':       cfg.cpu_batch_size,
        'cpu_top_k':            getattr(cfg, 'cpu_top_k', 64),
        'max_combinations':     cfg.max_combinations,
        'scan_pattern_types':   getattr(cfg, 'scan_pattern_types', 'all'),
        'enable_multi_start':   getattr(cfg, 'enable_multi_start', False),
        'max_start_points':     getattr(cfg, 'max_start_points', 5),
        'max_seconds_per_scan': getattr(cfg, 'max_seconds_per_scan', 10.0),
        'max_windows':          cfg.max_windows,
        'slide_bars':           cfg.slide_weeks * bars_per_week,
        'min_bars':             cfg.min_weeks  * bars_per_week,
        'max_bars':             cfg.max_weeks  * bars_per_week,
    }

    # --- load prior partial results if resuming ---
    all_results: List[Dict] = []
    if resume:
        prior = ckpt.load_partial_results()
        if prior:
            all_results.extend(prior)
            if verbose:
                print(f"  [{interval}] Loaded {len(prior)} patterns from checkpoint")

    # --- checkpoint callback ---
    def on_done(symbol, results, error=None):
        for r in results:
            r['ticker']   = symbol
            r['interval'] = interval
        all_results.extend(results)
        ckpt.on_ticker_done(symbol, results, error, all_results)

    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"\U0001f680  Interval : {interval.upper()}  |  {len(ticker_subset)} tickers")
    print(f"\u2699\ufe0f   Workers  : {n_workers}  (SLURM_NTASKS)")
    print(f"{'='*70}\n")

    new_results = parallel_scan_tickers(
        ticker_data=ticker_subset,
        cfg_dict=cfg_dict,
        n_workers=n_workers,
        checkpoint_fn=on_done,
    )

    # Final checkpoint flush
    ckpt.flush(all_results)

    elapsed = time.time() - t0
    tickers_done = len(ticker_subset)
    rate = tickers_done / (elapsed / 60) if elapsed > 0 else 0
    cpu_eff = (tickers_done * (elapsed / max(tickers_done, 1))) / \
              (elapsed * n_workers) * 100 if elapsed > 0 else 0

    print(f"\n{'='*70}")
    print(f"\u2705  {interval.upper()} complete  \u2014  {elapsed/3600:.2f} h")
    print(f"    Tickers    : {tickers_done:,}  ({rate:.1f} tickers/min)")
    print(f"    Patterns   : {len(all_results):,}  (avg {len(all_results)/max(tickers_done,1):.1f}/ticker)")
    print(f"    Avg/ticker : {elapsed/max(tickers_done,1):.2f}s")
    print(f"    CPU eff.   : ~{cpu_eff:.0f}%  across {n_workers} workers")
    print(f"{'='*70}\n")

    return all_results


def run_pipeline(
    data_dir:       str,
    config_path:    str,
    output_dir:     str,
    checkpoint_dir: str,
    intervals:      List[str],
    resume:         bool,
    verbose:        bool,
):
    cfg       = PipelineConfig.load_from_file(config_path)
    data_path = Path(data_dir)
    out_path  = Path(output_dir)
    ckpt_path = Path(checkpoint_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    n_workers = int(
        os.environ.get('SLURM_NTASKS',
        os.environ.get('SLURM_CPUS_ON_NODE',
        os.cpu_count() or 1))
    )

    print(f"\u2699\ufe0f   Workers  : {n_workers}  (from SLURM_NTASKS)")

    # Numba warmup once in main process
    print("\U0001f525 Pre-warming Numba JIT...")
    prewarm_numba()
    print("   \u2705 Numba ready\n")

    all_patterns: List[Dict] = []

    for interval in intervals:
        try:
            ticker_data = load_parquet_data(data_path, interval, verbose=verbose)
            print(f"\U0001f4cb Found {len(ticker_data)} unique tickers for {interval}")
        except FileNotFoundError as e:
            print(f"  \u26a0\ufe0f  Skipping {interval}: {e}")
            continue

        results = run_interval(
            interval=interval,
            ticker_data=ticker_data,
            cfg=cfg,
            output_dir=out_path,
            checkpoint_dir=ckpt_path,
            n_workers=n_workers,
            resume=resume,
            verbose=verbose,
        )
        all_patterns.extend(results)

        # Save per-interval results
        interval_out = out_path / f"results_{interval}.json"
        clean = [{k: v for k, v in r.items() if k != '_pattern_obj'}
                 for r in results]
        with open(interval_out, 'w') as f:
            json.dump(clean, f, default=str)
        print(f"  Saved {len(clean)} patterns -> {interval_out}")

    # Save combined results
    combined_out = out_path / "results.json"
    clean_all = [{k: v for k, v in r.items() if k != '_pattern_obj'}
                 for r in all_patterns]
    with open(combined_out, 'w') as f:
        json.dump(clean_all, f, default=str)

    print(f"\n\U0001f4be Combined results ({len(clean_all)} patterns) -> {combined_out}")
    return clean_all


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Elliott Wave Pipeline")
    parser.add_argument('--data-dir',       default='data/hf_dataset_complete',
                        help='Directory containing market_data_<interval>.parquet files')
    parser.add_argument('--config',         default='configs.yaml')
    parser.add_argument('--output',         default='output',
                        help='Output directory (results saved as results_<interval>.json)')
    parser.add_argument('--checkpoint-dir', default='output/checkpoints')
    parser.add_argument('--intervals',      default='1d',
                        help='Comma-separated intervals: 1d,4h,1h,1wk')
    parser.add_argument('--resume',         action='store_true')
    parser.add_argument('--verbose',        action='store_true')
    # Legacy HF flags (ignored when parquet files exist)
    parser.add_argument('--hf-dataset',          default=None)
    parser.add_argument('--use-all-hf-symbols',  action='store_true')
    parser.add_argument('--symbols',             default=None)

    args = parser.parse_args()

    intervals = [i.strip() for i in args.intervals.split(',') if i.strip()]

    run_pipeline(
        data_dir=args.data_dir,
        config_path=args.config,
        output_dir=args.output,
        checkpoint_dir=args.checkpoint_dir,
        intervals=intervals,
        resume=args.resume,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
