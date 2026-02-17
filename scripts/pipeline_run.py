#!/usr/bin/env python3
"""
Enhanced Pipeline Runner with:
- Hugging Face dataset integration
- Checkpoint/resume support
- Verbose progress tracking
- HPC compatibility
"""
import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.config import PipelineConfig
from pipeline.executor import parallel_scan_windows
from pipeline.numba_warm import prewarm_numba


class CheckpointManager:
    """Manages checkpoints for resumable pipeline runs with enhanced error tracking"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.checkpoint_dir / "processed_symbols.json"
        self.results_file = self.checkpoint_dir / "partial_results.json"
        self.failed_file = self.checkpoint_dir / "failed_symbols.json"
        self.stats_file = self.checkpoint_dir / "runtime_stats.json"
        
    def load_processed_symbols(self) -> Set[str]:
        """Load set of already processed symbols"""
        try:
            if self.processed_file.exists():
                with open(self.processed_file, 'r') as f:
                    return set(json.load(f))
        except Exception as e:
            print(f"⚠️  Warning: Could not load processed symbols: {e}")
        return set()
    
    def load_failed_symbols(self) -> Dict[str, str]:
        """Load dict of failed symbols with error messages"""
        try:
            if self.failed_file.exists():
                with open(self.failed_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load failed symbols: {e}")
        return {}
    
    def save_processed_symbol(self, symbol: str):
        """Mark a symbol as processed"""
        try:
            processed = self.load_processed_symbols()
            processed.add(symbol)
            with open(self.processed_file, 'w') as f:
                json.dump(sorted(list(processed)), f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save processed symbol {symbol}: {e}")
    
    def save_failed_symbol(self, symbol: str, error_msg: str):
        """Mark a symbol as failed with error message"""
        try:
            failed = self.load_failed_symbols()
            failed[symbol] = error_msg
            with open(self.failed_file, 'w') as f:
                json.dump(failed, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save failed symbol {symbol}: {e}")
    
    def save_partial_results(self, results: List[Dict]):
        """Save partial results with backup"""
        try:
            # Create backup of existing results
            if self.results_file.exists():
                backup_file = self.checkpoint_dir / "partial_results.backup.json"
                import shutil
                shutil.copy2(self.results_file, backup_file)
            
            # Save new results
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Warning: Could not save partial results: {e}")
    
    def save_runtime_stats(self, stats: Dict):
        """Save runtime statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Warning: Could not save runtime stats: {e}")
    
    def load_partial_results(self) -> List[Dict]:
        """Load partial results with fallback to backup"""
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load partial results: {e}")
            # Try backup
            backup_file = self.checkpoint_dir / "partial_results.backup.json"
            if backup_file.exists():
                print(f"   Attempting to load from backup...")
                try:
                    with open(backup_file, 'r') as f:
                        return json.load(f)
                except Exception as e2:
                    print(f"   Backup also failed: {e2}")
        return []


def load_hf_dataset(dataset_name: str, symbols: List[str] = None, verbose: bool = True):
    """
    Load data from Hugging Face dataset
    
    Args:
        dataset_name: HF dataset name (e.g., 'usamaahmedsh/financial-markets-dataset-15y-train')
        symbols: List of specific symbols to load (None = all)
        verbose: Show progress
    """
    try:
        from datasets import load_dataset
        import pandas as pd
        
        if verbose:
            print(f"📦 Loading dataset from Hugging Face: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split='train')
        
        if verbose:
            print(f"✅ Loaded {len(dataset)} rows")
        
        # Convert to pandas
        df = dataset.to_pandas()
        
        # Group by ticker
        ticker_data = {}
        
        if symbols:
            # Filter to requested symbols
            df_filtered = df[df['ticker'].isin(symbols)]
            if verbose:
                print(f"🔍 Filtered to {len(symbols)} symbols: {', '.join(symbols)}")
        else:
            df_filtered = df
            symbols = df['ticker'].unique().tolist()
            if verbose:
                print(f"📊 Found {len(symbols)} unique symbols")
        
        # Group by ticker
        for ticker in (tqdm(symbols, desc="Processing tickers") if verbose else symbols):
            ticker_df = df_filtered[df_filtered['ticker'] == ticker].copy()
            
            if len(ticker_df) == 0:
                if verbose:
                    print(f"⚠️  No data found for {ticker}")
                continue
            
            # Sort by date
            ticker_df = ticker_df.sort_values('Date')
            
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in ticker_df.columns for col in required_cols):
                if verbose:
                    print(f"⚠️  Missing columns for {ticker}, skipping")
                continue
            
            ticker_data[ticker] = ticker_df
        
        if verbose:
            print(f"✅ Loaded data for {len(ticker_data)} symbols")
            
        return ticker_data
        
    except ImportError:
        print("❌ ERROR: 'datasets' library not installed")
        print("Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        sys.exit(1)


def build_windows_for_df(df, cfg: PipelineConfig):
    """Build time windows from dataframe"""
    bars_per_week = 7  # Daily data
    
    overlap_ratio = getattr(cfg, 'window_overlap_ratio', 0.0)
    
    if overlap_ratio > 0:
        base_slide = cfg.slide_weeks * bars_per_week
        slide_step = max(1, int(base_slide * (1.0 - overlap_ratio)))
    else:
        slide_step = cfg.slide_weeks * bars_per_week
    
    min_len = cfg.min_weeks * bars_per_week

    # Prepare arrays
    lows_arr = df['Low'].to_numpy()
    highs_arr = df['High'].to_numpy()
    dates_arr = df['Date'].to_numpy()

    n_total = len(lows_arr)
    windows = []
    
    for start_idx in range(0, n_total, slide_step):
        end_idx = start_idx + cfg.max_weeks * bars_per_week
        if end_idx > n_total:
            end_idx = n_total
        
        if (end_idx - start_idx) < min_len:
            break
        
        windows.append((start_idx, end_idx))
        
        if len(windows) >= cfg.max_windows:
            break
    
    return windows, (lows_arr, highs_arr, dates_arr)


def run_pipeline(
    symbols: List[str],
    config_path: str,
    output_path: str,
    checkpoint_dir: str = None,
    hf_dataset: str = None,
    verbose: bool = True,
    resume: bool = False
):
    """
    Run the pattern detection pipeline
    
    Args:
        symbols: List of ticker symbols
        config_path: Path to config file
        output_path: Path to save results
        checkpoint_dir: Directory for checkpoints
        hf_dataset: Hugging Face dataset name
        verbose: Show progress
        resume: Resume from checkpoint
    """
    start_time = time.time()
    
    # Load config with auto device detection
    cfg = PipelineConfig.load_from_file(config_path, auto_detect=True)
    
    # Show device info
    if verbose:
        try:
            from pipeline.device import get_optimal_config, print_device_info
            device_cfg = get_optimal_config()
            print_device_info(device_cfg)
            print()
        except Exception:
            pass
    
    # Setup checkpoint manager
    checkpoint_mgr = None
    if checkpoint_dir:
        checkpoint_mgr = CheckpointManager(Path(checkpoint_dir))
        
        if resume:
            processed = checkpoint_mgr.load_processed_symbols()
            if verbose and processed:
                print(f"📂 Resuming: {len(processed)} symbols already processed")
                print(f"   Already done: {', '.join(sorted(processed))}")
            
            # Filter out processed symbols
            symbols = [s for s in symbols if s not in processed]
            if not symbols:
                print("✅ All symbols already processed!")
                return
            
            if verbose:
                print(f"🔄 Remaining: {len(symbols)} symbols to process")
    
    # Pre-warm Numba
    if verbose:
        print("🔥 Pre-warming Numba JIT...")
    prewarm_numba()
    
    # Load data
    if hf_dataset:
        ticker_data = load_hf_dataset(hf_dataset, symbols, verbose)
    else:
        # Fallback to yfinance
        if verbose:
            print("📥 Fetching data from yfinance...")
        from pipeline.fetcher import fetch_symbols
        # Use cfg.days for historical data (365 = 1 year, 5475 = 15 years)
        start_days = getattr(cfg, 'days', 720)
        ticker_data = asyncio.run(fetch_symbols(symbols, start_days=start_days, concurrency=cfg.concurrency))
    
    # Collect all results
    all_results = []
    
    # Load partial results if resuming
    if resume and checkpoint_mgr:
        partial = checkpoint_mgr.load_partial_results()
        if partial:
            all_results.extend(partial)
            if verbose:
                print(f"📂 Loaded {len(partial)} patterns from previous runs")
    
    # Process each symbol
    total_symbols = len(symbols)
    symbol_times = []  # Track time per symbol for estimation
    patterns_count = []  # Track patterns per symbol
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🚀 Starting pattern detection for {total_symbols} symbols")
        print(f"{'='*70}\n")
    
    for idx, symbol in enumerate(symbols, 1):
        symbol_start_time = time.time()
        
        if verbose:
            print(f"\n[{idx}/{total_symbols}] Processing {symbol}...")
            print(f"   Progress: {idx/total_symbols*100:.1f}% complete")
            print(f"   Remaining: {total_symbols - idx} symbols")
            
            # Time estimation based on completed symbols
            if symbol_times:
                avg_time = sum(symbol_times) / len(symbol_times)
                remaining_count = total_symbols - idx + 1  # +1 to include current
                estimated_remaining_sec = avg_time * remaining_count
                
                # Format time nicely
                if estimated_remaining_sec < 3600:
                    est_str = f"{estimated_remaining_sec/60:.1f} minutes"
                elif estimated_remaining_sec < 86400:
                    est_str = f"{estimated_remaining_sec/3600:.1f} hours"
                else:
                    est_str = f"{estimated_remaining_sec/86400:.1f} days ({estimated_remaining_sec/3600:.1f} hours)"
                
                print(f"   ⏱️  Estimated remaining: {est_str} (avg {avg_time:.1f}s per symbol)")
                
                # ETA calculation
                eta_timestamp = time.time() + estimated_remaining_sec
                eta_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta_timestamp))
                print(f"   🎯 Estimated completion: {eta_str}")
                
                # Show cumulative stats
                if patterns_count:
                    total_patterns_so_far = sum(patterns_count)
                    avg_patterns = total_patterns_so_far / len(patterns_count)
                    estimated_total_patterns = int(avg_patterns * total_symbols)
                    print(f"   📊 Patterns so far: {total_patterns_so_far} (avg {avg_patterns:.1f} per symbol)")
                    print(f"   📈 Estimated total patterns: ~{estimated_total_patterns:,}")

        # Wrap entire symbol processing in try-except
        try:
            df = ticker_data.get(symbol)
            if df is None or len(df) == 0:
                if verbose:
                    print(f"   ⚠️  No data available for {symbol}, skipping")
                if checkpoint_mgr:
                    checkpoint_mgr.save_failed_symbol(symbol, "No data available")
                continue
            
            if verbose:
                print(f"   📊 Data: {len(df)} bars from {df['Date'].min()} to {df['Date'].max()}")
            
            # Build windows
            try:
                windows, (lows, highs, dates) = build_windows_for_df(df, cfg)
                
                if verbose:
                    print(f"   🔍 Generated {len(windows)} windows")
                
                if not windows:
                    if verbose:
                        print(f"   ⚠️  No valid windows, skipping")
                    if checkpoint_mgr:
                        checkpoint_mgr.save_failed_symbol(symbol, "No valid windows generated")
                    continue
            except Exception as e:
                error_msg = f"Window generation failed: {str(e)}"
                if verbose:
                    print(f"   ❌ {error_msg}")
                if checkpoint_mgr:
                    checkpoint_mgr.save_failed_symbol(symbol, error_msg)
                symbol_elapsed = time.time() - symbol_start_time
                symbol_times.append(symbol_elapsed)
                patterns_count.append(0)
                continue
            
            # Run pattern detection
            if verbose:
                print(f"   ⚙️  Running wave analyzer...")
            
            # Prepare windows in the format expected by parallel_scan_windows
            # Convert (start_idx, end_idx) to (start_row, window_len, context)
            prepared_windows = []
            for start_idx, end_idx in windows:
                window_len = end_idx - start_idx
                context = {
                    'symbol': symbol,
                    'lows': lows,
                    'highs': highs,
                    'dates': dates
                }
                prepared_windows.append((start_idx, window_len, context))
            
            # Prepare config dict for executor
            cfg_dict = {
                'up_to': cfg.up_to,
                'top_n': cfg.top_n,
                'cpu_batch_size': cfg.cpu_batch_size,
                'scan_pattern_types': getattr(cfg, 'scan_pattern_types', 'all'),
                'enable_multi_start': getattr(cfg, 'enable_multi_start', False),
                'max_start_points': getattr(cfg, 'max_start_points', 5),
                'max_combinations': cfg.max_combinations
            }
            
            window_start = time.time()
            try:
                symbol_results = parallel_scan_windows(
                    windows=prepared_windows,
                    cfg=cfg_dict,
                    processes=cfg.processes
                )
            except Exception as e:
                error_msg = f"Pattern detection failed: {str(e)}"
                if verbose:
                    print(f"   ❌ {error_msg}")
                if checkpoint_mgr:
                    checkpoint_mgr.save_failed_symbol(symbol, error_msg)
                symbol_elapsed = time.time() - symbol_start_time
                symbol_times.append(symbol_elapsed)
                patterns_count.append(0)
                continue
            
            window_time = time.time() - window_start
            
            patterns_count.append(len(symbol_results))
            
            if verbose:
                print(f"   ✅ Found {len(symbol_results)} patterns in {window_time:.1f}s")
                if symbol_results:
                    top_score = max((r.get('ensemble_score', r.get('score', 0)) for r in symbol_results if r), default=0)
                    print(f"   📈 Top score: {top_score:.3f}")
            
            # Save images for top patterns if enabled
            if getattr(cfg, 'save_images', False) and symbol_results:
                try:
                    save_top_n = getattr(cfg, 'save_images_top_n', 5)
                    if verbose:
                        print(f"   🖼️  Saving top {min(save_top_n, len(symbol_results))} pattern images...")
                    
                    # Set images directory for this symbol
                    from models.helpers import set_images_dir
                    images_dir = Path(cfg.out_dir) / "images" / symbol
                    images_dir.mkdir(parents=True, exist_ok=True)
                    set_images_dir(str(images_dir))
                    
                    # Sort by ensemble_score (if available) or score
                    sorted_results = sorted(
                        [r for r in symbol_results if r],
                        key=lambda x: x.get('ensemble_score', x.get('score', 0)),
                        reverse=True
                    )[:save_top_n]
                    
                    saved_count = 0
                    for i, result in enumerate(sorted_results, 1):
                        try:
                            # Get the pattern object (temporary, will be removed before JSON export)
                            found_pattern = result.get('_pattern_obj')
                            if not found_pattern or not hasattr(found_pattern, 'pattern'):
                                continue
                            
                            # Extract pattern details
                            wave_pattern = found_pattern.pattern
                            best_dict = result.get('best', {})
                            rule_name = best_dict.get('rule_name', 'unknown')
                            ensemble_score = result.get('ensemble_score', 0)
                            
                            # Create a DataFrame slice for this pattern
                            start_row = result.get('start_row', 0)
                            window_len = result.get('window_len', len(df))
                            end_row = min(start_row + window_len, len(df))
                            df_slice = df.iloc[start_row:end_row].copy()
                            
                            # Import plot function
                            from models.helpers import plot_pattern
                            
                            # Generate plot with custom filename prefix
                            title = f"{symbol} - Pattern #{i} (Score: {ensemble_score:.3f})"
                            plot_pattern(
                                df=df_slice,
                                wave_pattern=wave_pattern,
                                title=title,
                                symbol=symbol,
                                timeframe="1D",
                                rule_name=rule_name,
                                score=ensemble_score
                            )
                            saved_count += 1
                            
                            if verbose and i <= 3:
                                print(f"      Saved pattern #{i}: {rule_name}, score={ensemble_score:.3f}")
                        except Exception as e:
                            if verbose and i <= 3:
                                print(f"      ⚠️  Failed to save image #{i}: {e}")
                            continue
                    
                    if verbose and saved_count > 0:
                        print(f"   ✅ Saved {saved_count}/{save_top_n} images")
                        
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Image saving failed: {e}")
                    # Continue even if image saving fails
                        continue
            
            all_results.extend(symbol_results)
            
            # Run pattern analysis if enabled
            if getattr(cfg, 'analyze_patterns', False) and symbol_results:
                from tools.analyze_patterns import analyze_score_distribution, analyze_pattern_types
                
                if verbose:
                    print(f"   📊 Pattern Analysis:")
                
                # Score distribution
                ensemble_stats = analyze_score_distribution(symbol_results, 'ensemble_score')
                if ensemble_stats:
                    print(f"      Ensemble: mean={ensemble_stats['mean']:.3f}, median={ensemble_stats['median']:.3f}, max={ensemble_stats['max']:.3f}")
                
                # Pattern types
                pattern_types = analyze_pattern_types(symbol_results)
                if pattern_types:
                    types_str = ", ".join([f"{k}:{v}" for k, v in pattern_types.items()])
                    print(f"      Types: {types_str}")
                
                # Quality tiers
                excellent = len([p for p in symbol_results if p.get('ensemble_score', 0) >= 0.85])
                good = len([p for p in symbol_results if 0.70 <= p.get('ensemble_score', 0) < 0.85])
                fair = len([p for p in symbol_results if 0.50 <= p.get('ensemble_score', 0) < 0.70])
                poor = len([p for p in symbol_results if p.get('ensemble_score', 0) < 0.50])
                print(f"      Quality: Excellent={excellent}, Good={good}, Fair={fair}, Poor={poor}")
            
            # Save checkpoint after each symbol
            if checkpoint_mgr:
                try:
                    checkpoint_mgr.save_processed_symbol(symbol)
                    checkpoint_mgr.save_partial_results(all_results)
                    
                    # Save runtime stats every 10 symbols
                    if idx % 10 == 0 or idx == total_symbols:
                        runtime_stats = {
                            'symbols_processed': idx,
                            'total_symbols': total_symbols,
                            'patterns_found': len(all_results),
                            'avg_time_per_symbol': sum(symbol_times) / len(symbol_times) if symbol_times else 0,
                            'avg_patterns_per_symbol': sum(patterns_count) / len(patterns_count) if patterns_count else 0,
                            'elapsed_time': time.time() - start_time,
                            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        checkpoint_mgr.save_runtime_stats(runtime_stats)
                    
                    if verbose:
                        print(f"   💾 Checkpoint saved ({idx}/{total_symbols} symbols)")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Checkpoint save failed: {e}")
            
            # Track symbol processing time for estimation
            symbol_elapsed = time.time() - symbol_start_time
            symbol_times.append(symbol_elapsed)
            
            if verbose:
                print(f"   ⏲️  Symbol completed in {symbol_elapsed:.1f}s")
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if verbose:
                print(f"   ❌ {error_msg}")
                import traceback
                traceback.print_exc()
            
            # Save to failed symbols
            if checkpoint_mgr:
                checkpoint_mgr.save_failed_symbol(symbol, error_msg)
            
            # Still track time even on error for better estimates
            symbol_elapsed = time.time() - symbol_start_time
            symbol_times.append(symbol_elapsed)
            continue
    
    # Save final results
    if verbose:
        print(f"\n{'='*70}")
        print(f"💾 Saving results to {output_path}")
    
    # Clean up pattern objects before JSON export (they can't be serialized)
    for result in all_results:
        if '_pattern_obj' in result:
            del result['_pattern_obj']
    
    output_data = {
        'metadata': {
            'total_symbols': total_symbols,
            'total_patterns': len(all_results),
            'config': config_path,
            'hf_dataset': hf_dataset,
            'runtime_seconds': time.time() - start_time
        },
        'patterns': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    
    # Load failed symbols for summary
    failed_symbols = {}
    if checkpoint_mgr:
        failed_symbols = checkpoint_mgr.load_failed_symbols()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Pipeline Complete!")
        print(f"{'='*70}")
        print(f"📊 Summary:")
        print(f"   Symbols attempted: {total_symbols}")
        
        successful = total_symbols - len(failed_symbols)
        if failed_symbols:
            print(f"   Successfully processed: {successful}")
            print(f"   Failed: {len(failed_symbols)}")
        else:
            print(f"   Successfully processed: {total_symbols} (100%)")
        
        print(f"   Total patterns detected: {len(all_results)}")
        
        # Format total runtime nicely
        if elapsed < 3600:
            runtime_str = f"{elapsed:.1f}s ({elapsed/60:.1f} min)"
        elif elapsed < 86400:
            runtime_str = f"{elapsed/3600:.1f} hours ({elapsed/60:.0f} min)"
        else:
            runtime_str = f"{elapsed/86400:.1f} days ({elapsed/3600:.1f} hours)"
        
        print(f"   Total runtime: {runtime_str}")
        
        if symbol_times:
            avg_time = sum(symbol_times) / len(symbol_times)
            min_time = min(symbol_times)
            max_time = max(symbol_times)
            
            print(f"   Time per symbol:")
            print(f"      Average: {avg_time:.1f}s")
            print(f"      Min: {min_time:.1f}s")
            print(f"      Max: {max_time:.1f}s")
            
            # Patterns per symbol stats
            if all_results:
                from collections import Counter
                patterns_per_symbol = Counter(r.get('symbol', 'unknown') for r in all_results)
                if patterns_per_symbol:
                    avg_patterns = len(all_results) / len(patterns_per_symbol)
                    min_patterns = min(patterns_per_symbol.values())
                    max_patterns = max(patterns_per_symbol.values())
                    print(f"   Patterns per symbol:")
                    print(f"      Average: {avg_patterns:.1f}")
                    print(f"      Min: {min_patterns}")
                    print(f"      Max: {max_patterns}")
        
        print(f"   Output: {output_path}")
        
        # Show failed symbols if any
        if failed_symbols:
            print(f"\n   ⚠️  Failed Symbols ({len(failed_symbols)}):")
            for sym, error in list(failed_symbols.items())[:10]:  # Show first 10
                print(f"      {sym}: {error[:60]}...")
            if len(failed_symbols) > 10:
                print(f"      ... and {len(failed_symbols) - 10} more")
                if checkpoint_mgr:
                    print(f"      See {checkpoint_mgr.failed_file} for full list")
        
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Elliott Wave Pattern Detection Pipeline"
    )
    parser.add_argument(
        '--symbols',
        type=str,
        required=False,
        default=None,
        help='Comma-separated list of ticker symbols (default: read from data/sp500_tickers.txt)'
    )
    parser.add_argument(
        '--symbols-file',
        type=str,
        default=None,
        help='Path to file with symbols (one per line). Default: data/sp500_tickers.txt'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/results.json',
        help='Path to save results'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--hf-dataset',
        type=str,
        help='Hugging Face dataset name'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose progress'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Parse symbols from various sources
    symbols = None
    
    if args.symbols:
        # Explicit comma-separated symbols
        symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"📋 Using {len(symbols)} symbols from --symbols argument")
    elif args.symbols_file:
        # Read from specified file
        symbols_path = Path(args.symbols_file)
        if not symbols_path.exists():
            print(f"❌ ERROR: Symbols file not found: {args.symbols_file}")
            sys.exit(1)
        symbols = [line.strip() for line in symbols_path.read_text().splitlines() if line.strip()]
        print(f"📋 Loaded {len(symbols)} symbols from {args.symbols_file}")
    else:
        # Default: read from data/sp500_tickers.txt
        default_symbols_file = project_root / "data" / "sp500_tickers.txt"
        if default_symbols_file.exists():
            symbols = [line.strip() for line in default_symbols_file.read_text().splitlines() if line.strip()]
            print(f"📋 Loaded {len(symbols)} symbols from {default_symbols_file}")
        else:
            print(f"❌ ERROR: No symbols provided and default file not found: {default_symbols_file}")
            print("Use --symbols 'AAPL,MSFT,...' or --symbols-file path/to/file.txt")
            sys.exit(1)
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    run_pipeline(
        symbols=symbols,
        config_path=args.config,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        hf_dataset=args.hf_dataset,
        verbose=args.verbose,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
