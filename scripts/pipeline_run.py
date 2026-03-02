#!/usr/bin/env python3
"""
Elliott Wave Pipeline Runner
- Parquet / HuggingFace data loading
- Ticker-level parallelism via ProcessPoolExecutor (64 workers on HPC)
- tqdm progress bars with live stats
- Checkpoint / resume support
- SLURM-aware worker count (reads SLURM_NTASKS automatically)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# .env loader
# =============================================================================

def load_env_file():
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key   = key.strip()
                    value = value.strip()
                    if key and value and value != 'your_huggingface_token_here':
                        os.environ[key] = value
                        print(f"✅ Loaded {key} from .env file")

load_env_file()

from pipeline.config import PipelineConfig
from pipeline.executor import parallel_scan_tickers, _worker_scan_window
from pipeline.numba_warm import prewarm_numba


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, interval: str = ''):
        base = Path(checkpoint_dir)
        self.checkpoint_dir = base / interval if interval else base
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.checkpoint_dir / "processed_symbols.json"
        self.results_file   = self.checkpoint_dir / "partial_results.json"
        self.failed_file    = self.checkpoint_dir / "failed_symbols.json"
        self.stats_file     = self.checkpoint_dir / "runtime_stats.json"

    def load_processed_symbols(self) -> Set[str]:
        try:
            if self.processed_file.exists():
                with open(self.processed_file, 'r') as f:
                    return set(json.load(f))
        except Exception as e:
            print(f"⚠️  Could not load processed symbols: {e}")
        return set()

    def load_failed_symbols(self) -> Dict[str, str]:
        try:
            if self.failed_file.exists():
                with open(self.failed_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def save_processed_symbol(self, symbol: str):
        try:
            processed = self.load_processed_symbols()
            processed.add(symbol)
            with open(self.processed_file, 'w') as f:
                json.dump(sorted(list(processed)), f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save checkpoint for {symbol}: {e}")

    def save_failed_symbol(self, symbol: str, error_msg: str):
        try:
            failed = self.load_failed_symbols()
            failed[symbol] = error_msg[:500]   # truncate long tracebacks
            with open(self.failed_file, 'w') as f:
                json.dump(failed, f, indent=2)
        except Exception:
            pass

    def save_partial_results(self, results: List[Dict]):
        try:
            if self.results_file.exists():
                import shutil
                shutil.copy2(self.results_file,
                             self.checkpoint_dir / "partial_results.backup.json")
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Could not save partial results: {e}")

    def save_runtime_stats(self, stats: Dict):
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception:
            pass

    def load_partial_results(self) -> List[Dict]:
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load partial results: {e}")
            backup = self.checkpoint_dir / "partial_results.backup.json"
            if backup.exists():
                try:
                    with open(backup, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        return []


# =============================================================================
# DATA LOADING
# =============================================================================

def get_window_config(interval: str) -> Dict:
    """Return min/max/slide bar counts for a given interval."""
    return {
        '1h':  {'min_bars': 168 * 4,  'max_bars': 168 * 26, 'slide_bars': 168},
        '4h':  {'min_bars': 42  * 4,  'max_bars': 42  * 52, 'slide_bars': 42 },
        '1d':  {'min_bars': 30,        'max_bars': 730,       'slide_bars': 7  },
        '1wk': {'min_bars': 8,         'max_bars': 104,       'slide_bars': 1  },
    }.get(interval.lower(), {'min_bars': 30, 'max_bars': 730, 'slide_bars': 7})


def load_parquet_ticker_data(
    data_dir: Path,
    interval: str,
    hf_repo:  str = None,
) -> Dict[str, 'pd.DataFrame']:
    """Load ticker data from a local parquet file, downloading from HF if missing."""
    import pandas as pd

    parquet_file = data_dir / f"market_data_{interval}.parquet"

    if not parquet_file.exists():
        if hf_repo:
            print(f"📥 Downloading {interval} parquet from HuggingFace: {hf_repo} ...")
            try:
                from huggingface_hub import hf_hub_download
                token = os.environ.get("HF_TOKEN") or True
                data_dir.mkdir(parents=True, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id              = hf_repo,
                    filename             = f"market_data_{interval}.parquet",
                    repo_type            = "dataset",
                    local_dir            = str(data_dir),
                    local_dir_use_symlinks = False,
                    token                = token,
                )
                dl = Path(downloaded)
                if dl != parquet_file and dl.exists():
                    import shutil
                    shutil.move(str(dl), str(parquet_file))
                print(f"   ✅ Saved to {parquet_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to download parquet: {e}")
        else:
            raise FileNotFoundError(
                f"Parquet not found: {parquet_file}\n"
                f"Pass --hf-repo to download automatically."
            )

    print(f"📂 Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"   Loaded {len(df):,} rows")

    # Normalise column names
    col_map     = {}
    cols_lower  = {c.lower(): c for c in df.columns}
    for want, candidates in [
        ('Date',   ['date', 'datetime', 'timestamp']),
        ('High',   ['high']),
        ('Low',    ['low']),
        ('Open',   ['open']),
        ('Close',  ['close']),
        ('Volume', ['volume']),
    ]:
        if want not in df.columns:
            for cand in candidates:
                if cand in cols_lower:
                    col_map[cols_lower[cand]] = want
                    break
    if col_map:
        df = df.rename(columns=col_map)

    ticker_data: Dict[str, pd.DataFrame] = {}
    for ticker, grp in df.sort_values('Date').groupby('ticker'):
        ticker_data[ticker] = grp.reset_index(drop=True)

    print(f"   Grouped into {len(ticker_data)} tickers")
    return ticker_data


def load_hf_dataset_api(
    dataset_name: str,
    symbols:      List[str] = None,
    verbose:      bool = True,
) -> Dict[str, 'pd.DataFrame']:
    """Load ticker data via HuggingFace Datasets API (streams full dataset)."""
    try:
        from datasets import load_dataset
        import pandas as pd

        if verbose:
            print(f"📦 Loading from HF Datasets API: {dataset_name}")
        hf_token = os.environ.get('HF_TOKEN')
        dataset  = load_dataset(dataset_name, split='train', token=hf_token)
        if verbose:
            print(f"✅ {len(dataset):,} rows loaded")

        df          = dataset.to_pandas()
        df_filtered = df[df['ticker'].isin(symbols)] if symbols else df
        all_syms    = symbols or df['ticker'].unique().tolist()

        ticker_data: Dict[str, pd.DataFrame] = {}
        for ticker in (tqdm(all_syms, desc="Grouping tickers") if verbose else all_syms):
            grp = df_filtered[df_filtered['ticker'] == ticker].copy()
            if len(grp) == 0:
                continue
            grp = grp.sort_values('Date')
            if not all(c in grp.columns for c in ['Date', 'High', 'Low']):
                continue
            ticker_data[ticker] = grp.reset_index(drop=True)

        if verbose:
            print(f"✅ {len(ticker_data)} symbols ready")
        return ticker_data

    except ImportError:
        print("❌ pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"❌ HF load failed: {e}")
        sys.exit(1)


# =============================================================================
# DEDUPLICATION
# =============================================================================

def deduplicate_patterns(patterns: List[Dict], verbose: bool = False) -> List[Dict]:
    """Keep highest-scoring pattern per (rule, w1_low, w3_high, w5_high) signature."""
    if not patterns:
        return patterns
    seen: Dict = {}
    for p in patterns:
        best  = p.get('best', {})
        waves = best.get('waves', {})
        sig   = (
            best.get('rule_name', 'unknown'),
            round(waves.get('wave1', {}).get('low',  0), 2),
            round(waves.get('wave3', {}).get('high', 0), 2),
            round(waves.get('wave5', {}).get('high', 0), 2),
        )
        score = p.get('ensemble_score', best.get('score', 0))
        if sig not in seen or score > seen[sig][1]:
            seen[sig] = (p, score)
    deduped = [p for p, _ in seen.values()]
    if verbose and len(patterns) != len(deduped):
        print(f"   🔄 Deduped: {len(patterns)} → {len(deduped)}")
    return deduped


# =============================================================================
# PER-INTERVAL RUNNER
# =============================================================================

def run_pipeline_for_interval(
    ticker_data:    Dict[str, 'pd.DataFrame'],
    interval:       str,
    cfg:            PipelineConfig,
    output_dir:     Path,
    checkpoint_dir: Path,
    resume:         bool = True,
    verbose:        bool = True,
    n_workers:      int  = None,
) -> List[Dict]:
    """
    Run the full scan for one interval across all tickers in parallel.
    Workers are determined in priority order:
        1. n_workers argument
        2. SLURM_NTASKS env var  (set automatically by salloc/sbatch)
        3. SLURM_CPUS_ON_NODE
        4. os.cpu_count()
    """
    # ── Worker count ─────────────────────────────────────────────────────────
    if n_workers is None:
        n_workers = int(
            os.environ.get('SLURM_NTASKS',
            os.environ.get('SLURM_CPUS_ON_NODE',
            os.cpu_count() or 1))
        )

    print(f"\n{'='*70}")
    print(f"🚀  Interval : {interval.upper()}  |  {len(ticker_data)} tickers")
    print(f"⚙️   Workers  : {n_workers}  "
          f"({'SLURM_NTASKS' if os.environ.get('SLURM_NTASKS') else 'cpu_count'})")
    print(f"{'='*70}")
    start_time = time.time()

    checkpoint_mgr = CheckpointManager(checkpoint_dir, interval)

    # ── Resume filtering ──────────────────────────────────────────────────────
    all_tickers = list(ticker_data.keys())
    if resume:
        processed = checkpoint_mgr.load_processed_symbols()
        tickers   = [t for t in all_tickers if t not in processed]
        if processed:
            print(f"📂 Resuming: {len(processed)} done, {len(tickers)} remaining")
        if not tickers:
            print("✅ Already complete!")
            return checkpoint_mgr.load_partial_results()
    else:
        processed = set()
        tickers   = all_tickers

    ticker_data_filtered = {t: ticker_data[t] for t in tickers}

    # ── Build cfg_dict sent to every worker ──────────────────────────────────
    wcfg       = get_window_config(interval)
    slide_bars = cfg.slide_weeks * 7  if interval == '1d' else wcfg['slide_bars']
    min_bars   = cfg.min_weeks   * 7  if interval == '1d' else wcfg['min_bars']
    max_bars   = cfg.max_weeks   * 7  if interval == '1d' else wcfg['max_bars']

    cfg_dict = {
        # scan knobs
        'up_to':                cfg.up_to,
        'top_n':                cfg.top_n,
        'cpu_batch_size':       cfg.cpu_batch_size,
        'cpu_top_k':            getattr(cfg, 'cpu_top_k', 64),
        'scan_pattern_types':   getattr(cfg, 'scan_pattern_types', 'all'),
        'enable_multi_start':   getattr(cfg, 'enable_multi_start', False),
        'max_start_points':     getattr(cfg, 'max_start_points', 5),
        'max_combinations':     cfg.max_combinations,
        'max_seconds_per_scan': getattr(cfg, 'max_seconds_per_scan', 1e9),
        # window knobs
        'slide_bars':  slide_bars,
        'min_bars':    min_bars,
        'max_bars':    max_bars,
        'max_windows': cfg.max_windows,
    }

    # ── Load partial results if resuming ─────────────────────────────────────
    all_patterns: List[Dict] = checkpoint_mgr.load_partial_results() if resume else []
    _save_counter = [0]   # mutable counter for closure

    # ── Checkpoint callback (runs in main process) ───────────────────────────
    def _on_ticker_done(symbol: str, results: List[Dict], error: str = None):
        if error:
            checkpoint_mgr.save_failed_symbol(symbol, error)
            if verbose:
                # tqdm.write keeps output above the progress bar
                tqdm.write(f"   ⚠️  {symbol} failed: {error[:120]}")
        else:
            all_patterns.extend(results)
            checkpoint_mgr.save_processed_symbol(symbol)
            _save_counter[0] += 1
            # Flush to disk every 25 tickers
            if _save_counter[0] % 25 == 0:
                checkpoint_mgr.save_partial_results(all_patterns)
                checkpoint_mgr.save_runtime_stats({
                    'interval':    interval,
                    'n_workers':   n_workers,
                    'done':        _save_counter[0],
                    'total':       len(tickers),
                    'patterns':    len(all_patterns),
                    'elapsed_s':   time.time() - start_time,
                    'updated':     datetime.now().isoformat(),
                })

    # ── Run parallel scan ─────────────────────────────────────────────────────
    parallel_scan_tickers(
        ticker_data   = ticker_data_filtered,
        cfg_dict      = cfg_dict,
        n_workers     = n_workers,
        checkpoint_fn = _on_ticker_done,
    )

    # Final checkpoint save
    checkpoint_mgr.save_partial_results(all_patterns)
    checkpoint_mgr.save_runtime_stats({
        'interval':    interval,
        'n_workers':   n_workers,
        'done':        len(tickers),
        'total':       len(tickers),
        'patterns':    len(all_patterns),
        'elapsed_s':   time.time() - start_time,
        'updated':     datetime.now().isoformat(),
    })

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    avg_t   = elapsed / max(len(tickers), 1)
    cpu_eff = min(100.0, (len(tickers) * avg_t) / (elapsed * n_workers) * 100)
    failed  = checkpoint_mgr.load_failed_symbols()

    runtime_str = (f"{elapsed/60:.1f} min"
                   if elapsed < 3600 else f"{elapsed/3600:.2f} h")

    print(f"\n{'='*70}")
    print(f"✅  {interval.upper()} complete  —  {runtime_str}")
    print(f"    Tickers    : {len(tickers):,}  "
          f"({len(tickers)/elapsed*60:.1f} tickers/min)")
    print(f"    Patterns   : {len(all_patterns):,}  "
          f"(avg {len(all_patterns)/max(len(tickers),1):.1f}/ticker)")
    print(f"    Avg/ticker : {avg_t:.2f}s")
    print(f"    CPU eff.   : ~{cpu_eff:.0f}%  across {n_workers} workers")
    if failed:
        print(f"    ⚠️  Failed  : {len(failed)} tickers  "
              f"(see {checkpoint_mgr.failed_file})")
    print(f"{'='*70}")

    return all_patterns


# =============================================================================
# TOP-LEVEL PIPELINE
# =============================================================================

def run_pipeline(
    symbols:        Optional[List[str]],
    config_path:    str,
    output_path:    str,
    checkpoint_dir: str           = None,
    hf_dataset:     str           = None,
    data_dir:       str           = None,
    hf_repo:        str           = None,
    intervals:      List[str]     = None,
    n_workers:      int           = None,
    verbose:        bool          = True,
    resume:         bool          = True,
):
    total_start = time.time()
    cfg         = PipelineConfig.load_from_file(config_path)

    if not intervals:
        intervals = [getattr(cfg, 'interval', '1d')]

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path('output/checkpoints')
    out_dir  = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Worker count — resolve once, print clearly
    if n_workers is None:
        n_workers = int(
            os.environ.get('SLURM_NTASKS',
            os.environ.get('SLURM_CPUS_ON_NODE',
            os.cpu_count() or 1))
        )
    src = ('SLURM_NTASKS'    if os.environ.get('SLURM_NTASKS') else
           'SLURM_CPUS_ON_NODE' if os.environ.get('SLURM_CPUS_ON_NODE') else
           'cpu_count')
    print(f"⚙️   Workers  : {n_workers}  (from {src})")

    print("🔥 Pre-warming Numba JIT...")
    prewarm_numba()
    print("   ✅ Numba ready\n")

    all_results_by_interval: Dict[str, List[Dict]] = {}

    for interval in intervals:
        # ── Load data ────────────────────────────────────────────────────────
        if data_dir:
            ticker_data = load_parquet_ticker_data(
                data_dir = Path(data_dir),
                interval = interval,
                hf_repo  = hf_repo,
            )
        elif hf_dataset:
            ticker_data = load_hf_dataset_api(hf_dataset, symbols, verbose)
        else:
            from pipeline.fetcher import fetch_symbols
            ticker_data = asyncio.run(fetch_symbols(
                symbols or [],
                start_days  = getattr(cfg, 'days', 720),
                concurrency = cfg.concurrency,
                interval    = interval,
            ))

        if symbols is not None:
            sym_set     = set(symbols)
            ticker_data = {t: df for t, df in ticker_data.items() if t in sym_set}

        print(f"📋 Found {len(ticker_data)} unique tickers for {interval}")

        patterns = run_pipeline_for_interval(
            ticker_data    = ticker_data,
            interval       = interval,
            cfg            = cfg,
            output_dir     = out_dir,
            checkpoint_dir = ckpt_dir,
            resume         = resume,
            verbose        = verbose,
            n_workers      = n_workers,
        )
        all_results_by_interval[interval] = patterns

    # ── Save output JSON ──────────────────────────────────────────────────────
    all_patterns = []
    for pats in all_results_by_interval.values():
        for r in pats:
            r.pop('_pattern_obj', None)   # not JSON-serialisable
        all_patterns.extend(pats)

    total_elapsed = time.time() - total_start

    output_data = {
        'metadata': {
            'generated_at':    datetime.now().isoformat(),
            'intervals':       intervals,
            'n_workers':       n_workers,
            'total_patterns':  len(all_patterns),
            'runtime_seconds': total_elapsed,
            'config':          config_path,
            'data_dir':        data_dir,
            'hf_repo':         hf_repo,
            'hf_dataset':      hf_dataset,
        },
        'patterns_by_interval': {
            iv: len(pats) for iv, pats in all_results_by_interval.items()
        },
        'patterns': all_patterns,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    runtime_str = (f"{total_elapsed/60:.1f} min"
                   if total_elapsed < 3600
                   else f"{total_elapsed/3600:.1f} h")

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                       PIPELINE COMPLETE                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total patterns : {len(all_patterns):<50,} ║
║  By interval:                                                        ║""")
    for iv, pats in all_results_by_interval.items():
        print(f"║    {iv}: {len(pats):<54,} ║")
    print(f"""║  Runtime  : {runtime_str:<56} ║
║  Workers  : {n_workers:<56} ║
║  Output   : {str(output_path):<56} ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Elliott Wave Pipeline")
    parser.add_argument('--symbols',             type=str, default=None)
    parser.add_argument('--symbols-file',        type=str, default=None)
    parser.add_argument('--config',              type=str, default='configs.yaml')
    parser.add_argument('--output',              type=str, default='output/results.json')
    parser.add_argument('--checkpoint-dir',      type=str, default='output/checkpoints')
    parser.add_argument('--data-dir',            type=str, default=None,
                        help='Directory containing market_data_{interval}.parquet files')
    parser.add_argument('--hf-repo',             type=str, default=None,
                        help='HuggingFace repo ID to download parquet from if missing locally')
    parser.add_argument('--hf-dataset',          type=str, default=None,
                        help='HuggingFace dataset name (Datasets API)')
    parser.add_argument('--intervals',           type=str, default=None,
                        help='Comma-separated intervals: 1d,1wk,4h,1h')
    parser.add_argument('--workers',             type=int, default=None,
                        help='Number of parallel worker processes '
                             '(default: SLURM_NTASKS or cpu_count)')
    parser.add_argument('--verbose',             action='store_true', default=True)
    parser.add_argument('--resume',              action='store_true', default=True)
    parser.add_argument('--no-resume',           action='store_true')
    parser.add_argument('--use-all-hf-symbols',  action='store_true')
    args = parser.parse_args()

    # ── Resolve symbols ───────────────────────────────────────────────────────
    symbols = None
    if args.data_dir or (args.hf_dataset and args.use_all_hf_symbols):
        symbols = None
        src = args.data_dir or args.hf_dataset
        print(f"📋 Will use ALL tickers from: {src}")
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"📋 {len(symbols)} symbols from --symbols")
    elif args.symbols_file:
        sp = Path(args.symbols_file)
        if not sp.exists():
            print(f"❌ Symbols file not found: {args.symbols_file}")
            sys.exit(1)
        symbols = [l.strip() for l in sp.read_text().splitlines() if l.strip()]
        print(f"📋 {len(symbols)} symbols from {args.symbols_file}")
    else:
        default = project_root / "data" / "sp500_tickers.txt"
        if default.exists():
            symbols = [l.strip() for l in default.read_text().splitlines() if l.strip()]
            print(f"📋 {len(symbols)} symbols from {default}")
        else:
            print("❌ No symbols. Use --symbols, --symbols-file, or --data-dir")
            sys.exit(1)

    # ── Resolve intervals ─────────────────────────────────────────────────────
    intervals = None
    if args.intervals:
        valid     = {'1h', '4h', '1d', '1wk'}
        intervals = [i.strip() for i in args.intervals.split(',') if i.strip()]
        for iv in intervals:
            if iv not in valid:
                print(f"❌ Invalid interval '{iv}'. Valid: {valid}")
                sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        symbols        = symbols,
        config_path    = args.config,
        output_path    = args.output,
        checkpoint_dir = args.checkpoint_dir,
        hf_dataset     = args.hf_dataset,
        data_dir       = args.data_dir,
        hf_repo        = args.hf_repo,
        intervals      = intervals,
        n_workers      = args.workers,
        verbose        = args.verbose,
        resume         = args.resume and not args.no_resume,
    )


if __name__ == '__main__':
    main()
