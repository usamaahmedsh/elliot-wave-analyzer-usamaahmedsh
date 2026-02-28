#!/usr/bin/env python3
"""
Pipeline Runner for Validated Dataset
- Loads data from local parquet files (hf_dataset_complete)
- Processes each ticker SEPARATELY per interval (no mixing)
- Uses checkpointing for resume capability
- Optimized for maximum pattern extraction (up_to=15)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.config import PipelineConfig
from pipeline.executor import _worker_scan_window
from pipeline.numba_warm import prewarm_numba


class PipelineCheckpointManager:
    """Manages checkpoints for resumable pipeline runs"""
    
    def __init__(self, checkpoint_dir: Path, interval: str):
        self.checkpoint_dir = Path(checkpoint_dir) / interval
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.checkpoint_dir / "processed_tickers.json"
        self.results_file = self.checkpoint_dir / "patterns.json"
        self.failed_file = self.checkpoint_dir / "failed_tickers.json"
        self.stats_file = self.checkpoint_dir / "stats.json"
        
    def load_processed(self) -> Set[str]:
        """Load set of already processed tickers"""
        try:
            if self.processed_file.exists():
                with open(self.processed_file, 'r') as f:
                    return set(json.load(f))
        except Exception as e:
            print(f"⚠️  Could not load processed tickers: {e}")
        return set()
    
    def load_failed(self) -> Dict[str, str]:
        """Load dict of failed tickers with error messages"""
        try:
            if self.failed_file.exists():
                with open(self.failed_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_processed_set(self, processed: Set[str]):
        """Flush the full in-memory processed set to disk (replaces save_processed)."""
        try:
            with open(self.processed_file, 'w') as f:
                json.dump(sorted(list(processed)), f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save processed tickers: {e}")

    def save_failed(self, ticker: str, error_msg: str):
        """Mark a ticker as failed with error message"""
        try:
            failed = self.load_failed()
            failed[ticker] = error_msg
            with open(self.failed_file, 'w') as f:
                json.dump(failed, f, indent=2)
        except Exception:
            pass
    
    def load_patterns(self) -> List[Dict]:
        """Load existing patterns"""
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def save_patterns(self, patterns: List[Dict]):
        """Save patterns with backup"""
        try:
            if self.results_file.exists():
                backup = self.checkpoint_dir / "patterns.backup.json"
                import shutil
                shutil.copy2(self.results_file, backup)
            
            with open(self.results_file, 'w') as f:
                json.dump(patterns, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Could not save patterns: {e}")
    
    def save_stats(self, stats: Dict):
        """Save runtime statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception:
            pass


def load_parquet_data(data_dir: Path, interval: str, hf_repo: str = None) -> pd.DataFrame:
    """
    Load data from a local parquet file for a specific interval.
    If the local file does not exist and hf_repo is set, download it from
    HuggingFace Hub first and cache it under data_dir.

    hf_repo format: 'owner/dataset-name'
                    e.g. 'usamaahmedsh/elliott-wave-market-data-complete'
    The expected file path inside the HF dataset repo is:
        data/market_data_{interval}.parquet
    """
    parquet_file = data_dir / f"market_data_{interval}.parquet"

    if not parquet_file.exists():
        if hf_repo:
            print(f"📥 Local file not found. Downloading from HuggingFace: {hf_repo} ...")
            try:
                from huggingface_hub import hf_hub_download
                # Load .env so HF_TOKEN is available if set there
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                except ImportError:
                    pass  # python-dotenv optional; token can also be set directly in env
                token = os.environ.get("HF_TOKEN") or True  # True = use cached login
                data_dir.mkdir(parents=True, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=f"data/market_data_{interval}.parquet",
                    repo_type="dataset",
                    local_dir=str(data_dir),
                    local_dir_use_symlinks=False,
                    token=token,
                )
                # hf_hub_download may nest into data/ — resolve to expected path
                downloaded_path = Path(downloaded)
                if downloaded_path != parquet_file and downloaded_path.exists():
                    import shutil
                    shutil.move(str(downloaded_path), str(parquet_file))
                print(f"   ✅ Downloaded to {parquet_file}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {interval} data from HuggingFace repo '{hf_repo}': {e}\n"
                    f"Make sure 'huggingface_hub' is installed (pip install huggingface_hub) "
                    f"and you are logged in (huggingface-cli login) if the repo is private."
                )
        else:
            raise FileNotFoundError(
                f"Parquet file not found: {parquet_file}\n"
                f"Pass --hf-repo usamaahmedsh/elliott-wave-market-data-complete to download automatically."
            )

    print(f"📂 Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"   Loaded {len(df):,} rows")
    return df


def get_bars_per_week(interval: str) -> int:
    """Get number of bars per week for each interval"""
    intervals = {
        '1h': 24 * 7,      # 168 bars/week
        '4h': 6 * 7,       # 42 bars/week
        '1d': 7,           # 7 bars/week (approximation)
        '1wk': 1           # 1 bar/week
    }
    return intervals.get(interval, 7)


def get_window_config(interval: str) -> Dict:
    """Get window configuration based on interval"""
    # Adjust window sizes based on interval
    configs = {
        '1h': {
            'min_bars': 168 * 4,     # 4 weeks minimum
            'max_bars': 168 * 26,    # 6 months max window
            'slide_bars': 168,       # Slide 1 week
        },
        '4h': {
            'min_bars': 42 * 4,      # 4 weeks minimum
            'max_bars': 42 * 52,     # 1 year max window
            'slide_bars': 42,        # Slide 1 week
        },
        '1d': {
            'min_bars': 30,          # ~1 month minimum
            'max_bars': 365 * 2,     # 2 years max window
            'slide_bars': 7,         # Slide 1 week
        },
        '1wk': {
            'min_bars': 8,           # 2 months minimum
            'max_bars': 104,         # 2 years max window
            'slide_bars': 1,         # Slide 1 week
        }
    }
    return configs.get(interval, configs['1d'])


def _process_ticker_worker(args: Tuple) -> Tuple[str, List[Dict], Optional[str]]:
    """
    Top-level worker executed in a subprocess via ProcessPoolExecutor.
    Must be defined at module level to be picklable.

    Returns (ticker, patterns, error_msg_or_None).
    """
    ticker, lows_arr, highs_arr, dates_arr, interval, up_to, max_windows, cfg_dict = args

    # Build a minimal DataFrame the scanner needs
    import numpy as np
    import pandas as pd
    ticker_df = pd.DataFrame({'low': lows_arr, 'high': highs_arr, 'datetime': dates_arr})

    try:
        patterns = scan_ticker_patterns(
            df=ticker_df,
            ticker=ticker,
            interval=interval,
            up_to=up_to,
            max_windows=max_windows,
            cfg_dict=cfg_dict,
            verbose=False,
        )
        return ticker, patterns, None
    except Exception as e:
        return ticker, [], str(e)


def scan_ticker_patterns(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
    up_to: int = 15,
    max_windows: int = 1000,
    cfg_dict: Dict = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Scan a single ticker's data for Elliott Wave patterns.
    Uses sequential scanning with skip-ahead to avoid duplicates.
    """
    window_cfg = get_window_config(interval)
    min_bars = window_cfg['min_bars']
    max_bars = window_cfg['max_bars']
    slide_bars = window_cfg['slide_bars']

    # Prepare arrays
    lows_arr = df['low'].to_numpy()
    highs_arr = df['high'].to_numpy()

    # Handle datetime column
    if 'datetime' in df.columns:
        dates_arr = df['datetime'].to_numpy()
    elif 'date' in df.columns:
        dates_arr = df['date'].to_numpy()
    else:
        dates_arr = np.arange(len(df))

    n_total = len(lows_arr)

    if n_total < min_bars:
        if verbose:
            print(f"   ⚠️  Insufficient data: {n_total} bars (need {min_bars})")
        return []

    # Use caller-supplied cfg_dict if provided, otherwise fall back to safe defaults
    if cfg_dict is None:
        cfg_dict = {
            'up_to': up_to,
            'top_n': 10,
            'cpu_batch_size': 1024,
            'cpu_top_k': 64,
            'scan_pattern_types': 'all',
            'enable_multi_start': True,
            'max_start_points': 5,
            'max_combinations': 500000,
        }
    else:
        # Always keep up_to in sync with the value passed explicitly
        cfg_dict = dict(cfg_dict)
        cfg_dict['up_to'] = up_to
    
    results = []
    current_idx = 0
    windows_scanned = 0
    
    while current_idx < n_total and windows_scanned < max_windows:
        # Calculate window bounds
        end_idx = min(current_idx + max_bars, n_total)
        window_len = end_idx - current_idx
        
        if window_len < min_bars:
            break
        
        # Prepare context
        context = {
            'symbol': ticker,
            'lows': lows_arr,
            'highs': highs_arr,
            'dates': dates_arr
        }
        
        # Scan this window
        window_tuple = (current_idx, window_len, context)
        
        try:
            result = _worker_scan_window(window_tuple, cfg_dict)
        except Exception as e:
            if verbose:
                print(f"      Window {windows_scanned} error: {e}")
            current_idx += slide_bars
            windows_scanned += 1
            continue
        
        windows_scanned += 1
        
        if result and result.get('best', {}).get('score', 0) > 0:
            # Add metadata
            result['ticker'] = ticker
            result['interval'] = interval
            result['window_start_idx'] = current_idx
            result['window_end_idx'] = end_idx
            
            # Convert dates to strings for JSON
            if 'date_start' in result:
                result['date_start'] = str(result['date_start'])
            if 'date_end' in result:
                result['date_end'] = str(result['date_end'])
            
            results.append(result)
            
            # Skip ahead past this pattern
            pattern_idx_end = result['best'].get('idx_end', 0)
            skip_to = current_idx + pattern_idx_end + slide_bars
            current_idx = skip_to
        else:
            current_idx += slide_bars
    
    if verbose:
        print(f"   📊 Scanned {windows_scanned} windows, found {len(results)} patterns")
    
    return results


def run_pipeline_for_interval(
    data_dir: Path,
    interval: str,
    output_dir: Path,
    checkpoint_dir: Path,
    cfg: 'PipelineConfig',
    up_to: int = 15,
    max_windows: int = 1000,
    resume: bool = True,
    verbose: bool = True,
    workers: int = None,
    hf_repo: str = None,
):
    """Run the pipeline for a single interval using parallel ticker processing."""
    print(f"\n{'='*70}")
    print(f"🚀 Processing interval: {interval.upper()}")
    print(f"{'='*70}")

    start_time = time.time()

    # Load data (downloads from HuggingFace if local file is absent)
    df = load_parquet_data(data_dir, interval, hf_repo=hf_repo)

    # Determine datetime column name once
    date_col = 'datetime' if 'datetime' in df.columns else 'date'

    # Get unique tickers
    all_tickers = df['ticker'].unique().tolist()
    print(f"📋 Found {len(all_tickers)} unique tickers")

    # Setup checkpoint manager
    checkpoint_mgr = PipelineCheckpointManager(checkpoint_dir, interval)

    # Filter out already processed tickers if resuming
    if resume:
        processed = checkpoint_mgr.load_processed()
        if processed:
            print(f"📂 Resuming: {len(processed)} tickers already processed")
        tickers = [t for t in all_tickers if t not in processed]
        if not tickers:
            print("✅ All tickers already processed for this interval!")
            return checkpoint_mgr.load_patterns()
        print(f"🔄 Remaining: {len(tickers)} tickers to process")
    else:
        processed = set()
        tickers = all_tickers

    # Load existing patterns
    all_patterns = checkpoint_mgr.load_patterns() if resume else []

    # ── Pre-group the DataFrame into per-ticker numpy arrays (done once) ──
    # Single groupby split is O(N) once vs O(N*T) for per-ticker boolean filters.
    print(f"⏳ Grouping {len(df):,} rows by ticker (one-time cost)...", flush=True)
    t0 = time.time()
    ticker_set = set(tickers)
    grouped = df[df['ticker'].isin(ticker_set)].sort_values(date_col).groupby('ticker')
    ticker_arrays: Dict[str, Tuple] = {}
    for ticker, grp in grouped:
        ticker_arrays[ticker] = (
            grp['low'].to_numpy(),
            grp['high'].to_numpy(),
            grp[date_col].to_numpy(),
        )
    print(f"✅ Grouping done in {time.time()-t0:.1f}s — {len(ticker_arrays)} tickers ready", flush=True)

    # ── Determine worker count ──
    # CLI --workers overrides cfg.ticker_workers; 0 in either means auto.
    explicit = workers if workers is not None else cfg.ticker_workers
    n_workers = explicit if explicit > 0 else max(1, (os.cpu_count() or 4) - 1)
    print(f"⚙️  Using {n_workers} parallel workers")

    # ── Build the scan config dict from PipelineConfig (passed to every worker) ──
    scan_cfg_dict = {
        'up_to': up_to,
        'top_n': cfg.top_n_per_window,
        'cpu_batch_size': cfg.cpu_batch_size,
        'cpu_top_k': cfg.cpu_top_k,
        'scan_pattern_types': cfg.scan_pattern_types,
        'enable_multi_start': cfg.enable_multi_start,
        'max_start_points': cfg.max_start_points,
        'max_combinations': cfg.max_combinations,
    }

    total_tickers = len(tickers)
    patterns_per_ticker: List[int] = []
    ticker_times: List[float] = []

    # ── Parallel processing with ProcessPoolExecutor ──
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        print(f"📤 Submitting {len(tickers)} tickers to {n_workers} workers...", flush=True)
        for ticker in tickers:
            lows_arr, highs_arr, dates_arr = ticker_arrays[ticker]
            fut = executor.submit(
                _process_ticker_worker,
                (ticker, lows_arr, highs_arr, dates_arr,
                 interval, up_to, max_windows, scan_cfg_dict),
            )
            futures[fut] = ticker
        print(f"✅ All {len(futures)} futures submitted — workers running...", flush=True)

        completed = 0
        for fut in as_completed(futures):
            ticker = futures[fut]
            completed += 1
            ticker_start = time.time()

            try:
                result_ticker, patterns, error_msg = fut.result()
            except Exception as e:
                error_msg = str(e)
                patterns = []
                result_ticker = ticker

            ticker_elapsed = time.time() - ticker_start
            ticker_times.append(ticker_elapsed)

            if error_msg:
                if verbose:
                    print(f"   ❌ [{completed}/{total_tickers}] {result_ticker}: {error_msg}")
                checkpoint_mgr.save_failed(result_ticker, error_msg)
            else:
                patterns_per_ticker.append(len(patterns))
                if patterns:
                    all_patterns.extend(patterns)

                # Track processed (in-memory; flushed every 50 tickers)
                processed.add(result_ticker)

                if verbose:
                    pct = completed / total_tickers * 100
                    # Rolling ETA
                    if ticker_times:
                        avg = sum(ticker_times) / len(ticker_times)
                        remaining_sec = (total_tickers - completed) * avg / n_workers
                        eta_str = (f"{remaining_sec/60:.1f} min" if remaining_sec < 3600
                                   else f"{remaining_sec/3600:.1f} h")
                    else:
                        eta_str = "?"
                    score_str = ""
                    if patterns:
                        scores = [p.get('ensemble_score', p.get('best', {}).get('score', 0)) for p in patterns]
                        score_str = f" best={max(scores):.3f}"
                    print(f"   ✅ [{completed}/{total_tickers}] {result_ticker} "
                          f"({pct:.1f}%) — {len(patterns)} patterns{score_str} | ETA {eta_str}")

            # Flush checkpoint every 50 completed tickers
            if completed % 50 == 0:
                checkpoint_mgr.save_processed_set(processed)
                checkpoint_mgr.save_patterns(all_patterns)
                if verbose:
                    print(f"   💾 Checkpoint saved ({len(all_patterns)} patterns so far)")

    # Final flush
    checkpoint_mgr.save_processed_set(processed)
    checkpoint_mgr.save_patterns(all_patterns)

    elapsed = time.time() - start_time

    # Save stats
    stats = {
        'interval': interval,
        'total_tickers': len(all_tickers),
        'processed_tickers': len(processed),
        'total_patterns': len(all_patterns),
        'avg_patterns_per_ticker': (sum(patterns_per_ticker) / len(patterns_per_ticker)
                                    if patterns_per_ticker else 0),
        'runtime_seconds': elapsed,
        'completed_at': datetime.now().isoformat(),
    }
    checkpoint_mgr.save_stats(stats)

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ {interval.upper()} Complete!")
        print(f"   Tickers: {len(processed)}/{len(all_tickers)}")
        print(f"   Patterns: {len(all_patterns):,}")
        if patterns_per_ticker:
            print(f"   Avg patterns/ticker: {sum(patterns_per_ticker)/len(patterns_per_ticker):.1f}")
        print(f"   Runtime: {elapsed/60:.1f} minutes")
        print(f"{'='*70}")

    return all_patterns


def main():
    parser = argparse.ArgumentParser(
        description="Elliott Wave Pattern Detection Pipeline for Validated Dataset"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/hf_dataset_complete',
        help='Directory containing validated parquet files'
    )
    parser.add_argument(
        '--intervals',
        type=str,
        default='1d,1wk,4h,1h',
        help='Comma-separated list of intervals to process (default: 1d,1wk,4h,1h)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/validated_patterns',
        help='Directory to save results'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='output/checkpoints',
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--up-to',
        type=int,
        default=None,
        help='Maximum pattern complexity — overrides configs.yaml up_to'
    )
    parser.add_argument(
        '--max-windows',
        type=int,
        default=None,
        help='Maximum windows per ticker — overrides configs.yaml max_windows'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint (default: True)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, do not resume'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel worker processes (default: cpu_count - 1)'
    )
    parser.add_argument(
        '--hf-repo',
        type=str,
        default=None,
        help=(
            'HuggingFace dataset repo to download parquet files from when they '
            'are not present locally. Format: owner/repo-name  '
            'e.g. usamaahmedsh/elliott-wave-market-data-complete'
        )
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Show verbose progress'
    )
    
    args = parser.parse_args()

    # ── Load PipelineConfig from configs.yaml ──────────────────────────────
    # CLI args override individual knobs where both are specified.
    cfg = PipelineConfig.load_from_file('configs.yaml')

    # CLI flags take priority over configs.yaml for the two most common knobs
    if args.up_to is not None:
        cfg.up_to = args.up_to
    if args.max_windows is not None:
        cfg.max_windows = args.max_windows

    # Convenience: resolve effective values used in the banner
    effective_up_to = cfg.up_to
    effective_max_windows = cfg.max_windows
    effective_workers = (
        args.workers if args.workers is not None
        else (cfg.ticker_workers if cfg.ticker_workers > 0
              else max(1, (os.cpu_count() or 4) - 1))
    )

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    # Parse intervals
    intervals = [i.strip() for i in args.intervals.split(',')]

    # Validate intervals
    valid_intervals = {'1h', '4h', '1d', '1wk'}
    for interval in intervals:
        if interval not in valid_intervals:
            print(f"❌ Invalid interval: {interval}")
            print(f"   Valid intervals: {', '.join(valid_intervals)}")
            sys.exit(1)

    # Resume setting
    resume = args.resume and not args.no_resume

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     ELLIOTT WAVE PATTERN DETECTION - VALIDATED DATASET PIPELINE      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Data Directory:    {str(data_dir):<47} ║
║  HF Repo:           {(args.hf_repo or 'local only'):<47} ║
║  Intervals:         {', '.join(intervals):<47} ║
║  up_to:             {effective_up_to:<47} ║
║  Max Windows:       {effective_max_windows:<47} ║
║  Ticker Workers:    {effective_workers:<47} ║
║  cpu_batch_size:    {cfg.cpu_batch_size:<47} ║
║  max_combinations:  {cfg.max_combinations:<47,} ║
║  scan_pattern_types:{cfg.scan_pattern_types:<47} ║
║  enable_multi_start:{str(cfg.enable_multi_start):<47} ║
║  Checkpointing:     {'ENABLED' if resume else 'DISABLED':<47} ║
║  Output:            {str(output_dir):<47} ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Pre-warm Numba
    print("🔥 Pre-warming Numba JIT...")
    prewarm_numba()

    total_start = time.time()
    all_results = {}

    # Process each interval separately
    for interval in intervals:
        patterns = run_pipeline_for_interval(
            data_dir=data_dir,
            interval=interval,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            cfg=cfg,
            up_to=effective_up_to,
            max_windows=effective_max_windows,
            resume=resume,
            verbose=args.verbose,
            workers=args.workers,
            hf_repo=args.hf_repo,
        )
        all_results[interval] = patterns

    # Save combined results
    total_patterns = sum(len(p) for p in all_results.values())

    combined_output = output_dir / "all_patterns.json"
    combined_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'intervals': intervals,
            'up_to': effective_up_to,
            'total_patterns': total_patterns,
            'runtime_seconds': time.time() - total_start
        },
        'patterns_by_interval': {
            interval: len(patterns) for interval, patterns in all_results.items()
        },
        'patterns': []
    }

    # Flatten all patterns
    for interval, patterns in all_results.items():
        combined_data['patterns'].extend(patterns)

    with open(combined_output, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)

    total_elapsed = time.time() - total_start

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         PIPELINE COMPLETE                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total Patterns: {total_patterns:<52,} ║
║  By Interval:                                                        ║""")
    
    for interval, patterns in all_results.items():
        print(f"║    {interval}: {len(patterns):<54,} ║")
    
    if total_elapsed < 3600:
        runtime_str = f"{total_elapsed/60:.1f} minutes"
    else:
        runtime_str = f"{total_elapsed/3600:.1f} hours"
    
    print(f"""║  Runtime: {runtime_str:<57} ║
║  Output:  {str(combined_output):<57} ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
