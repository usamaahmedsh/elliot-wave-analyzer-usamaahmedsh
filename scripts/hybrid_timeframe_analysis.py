#!/usr/bin/env python3
"""
Hybrid Timeframe Analysis Script

Analyzes Elliott Wave patterns across multiple timeframes (hourly, daily, weekly)
for a single stock to compare pattern detection rates.

Usage:
    python scripts/hybrid_timeframe_analysis.py --symbol AAPL
    python scripts/hybrid_timeframe_analysis.py --symbol MSFT --output output/hybrid_results.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.fetcher import fetch_symbols, INTERVAL_LIMITS
from pipeline.config import PipelineConfig
from pipeline.executor import _worker_scan_window


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    interval: str
    name: str
    max_days: int  # Max days of data available
    bars_per_day: float  # Approximate bars per calendar day
    min_bars: int  # Minimum pattern length in bars
    max_bars: int  # Maximum window size in bars
    slide_bars: int  # Slide step in bars


# Define timeframe-specific configurations
# For faster initial testing, use smaller windows. Scale up for production.
TIMEFRAME_CONFIGS = {
    '1h': TimeframeConfig(
        interval='1h',
        name='Hourly',
        max_days=365,  # 1 year (conservative to avoid yfinance issues)
        bars_per_day=6.5,  # Trading hours
        min_bars=40,  # ~1 week of hourly bars
        max_bars=200,  # ~1.5 months (smaller for faster testing)
        slide_bars=20,  # ~3 days (larger slide for faster testing)
    ),
    '1d': TimeframeConfig(
        interval='1d',
        name='Daily',
        max_days=2000,  # ~5.5 years (faster than full 15 years)
        bars_per_day=1,
        min_bars=40,  # ~2 months
        max_bars=200,  # ~10 months (smaller for faster testing)
        slide_bars=10,  # ~2 weeks (larger slide for faster testing)
    ),
    '1wk': TimeframeConfig(
        interval='1wk',
        name='Weekly',
        max_days=3650,  # 10 years
        bars_per_day=0.2,  # ~1 bar per week
        min_bars=20,  # ~5 months
        max_bars=104,  # ~2 years
        slide_bars=4,  # ~1 month
    ),
}


def sequential_scan_timeframe(df, tf_config: TimeframeConfig, symbol: str, cfg: PipelineConfig, verbose: bool = False) -> List[Dict]:
    """
    Sequential scanning with skip-ahead for a specific timeframe.
    
    Returns list of pattern results with timeframe metadata.
    """
    if df is None or len(df) == 0:
        return []
    
    lows_arr = df['Low'].to_numpy()
    highs_arr = df['High'].to_numpy()
    dates_arr = df['Date'].to_numpy()
    
    n_total = len(lows_arr)
    
    # Config for worker - use lower max_combinations for faster testing
    cfg_dict = {
        'up_to': cfg.up_to,
        'top_n': min(cfg.top_n, 3),  # Limit for speed
        'cpu_batch_size': cfg.cpu_batch_size,
        'scan_pattern_types': getattr(cfg, 'scan_pattern_types', 'all'),
        'enable_multi_start': getattr(cfg, 'enable_multi_start', False),
        'max_start_points': getattr(cfg, 'max_start_points', 5),
        'max_combinations': min(cfg.max_combinations, 10000)  # Cap for speed
    }
    
    results = []
    current_idx = 0
    windows_scanned = 0
    max_windows = min(cfg.max_windows, 500)  # Cap for testing
    
    while current_idx < n_total and windows_scanned < max_windows:
        # Calculate window bounds using timeframe-specific sizes
        end_idx = current_idx + tf_config.max_bars
        if end_idx > n_total:
            end_idx = n_total
        
        window_len = end_idx - current_idx
        if window_len < tf_config.min_bars:
            break
        
        # Prepare context
        context = {
            'symbol': symbol,
            'lows': lows_arr,
            'highs': highs_arr,
            'dates': dates_arr
        }
        
        # Scan this window
        window_tuple = (current_idx, window_len, context)
        result = _worker_scan_window(window_tuple, cfg_dict)
        windows_scanned += 1
        
        if result and result.get('best', {}).get('score', 0) > 0:
            # Valid pattern found - add timeframe metadata
            result['timeframe'] = tf_config.interval
            result['timeframe_name'] = tf_config.name
            results.append(result)
            
            # Skip to after the pattern ends
            pattern_idx_end = result['best'].get('idx_end', 0)
            pattern_abs_end = current_idx + pattern_idx_end
            skip_to = pattern_abs_end + tf_config.slide_bars
            
            if verbose:
                print(f"      [{tf_config.name}] Found pattern at idx {current_idx}-{pattern_abs_end}, skipping to {skip_to}")
            
            current_idx = skip_to
        else:
            current_idx += tf_config.slide_bars
    
    if verbose:
        print(f"   [{tf_config.name}] Scanned {windows_scanned} windows, found {len(results)} patterns")
    
    return results


def run_hybrid_analysis(symbol: str, timeframes: List[str] = None, output_path: str = None, verbose: bool = True):
    """
    Run Elliott Wave pattern detection across multiple timeframes.
    
    Args:
        symbol: Stock ticker symbol
        timeframes: List of timeframe intervals (default: ['1h', '1d', '1wk'])
        output_path: Path to save results JSON
        verbose: Print progress
    """
    if timeframes is None:
        timeframes = ['1h', '1d', '1wk']
    
    start_time = time.time()
    
    # Load base config
    cfg = PipelineConfig.load_from_file('configs.yaml', auto_detect=True)
    
    if verbose:
        print("=" * 70)
        print(f"HYBRID TIMEFRAME ANALYSIS: {symbol}")
        print("=" * 70)
        print(f"Timeframes: {', '.join(timeframes)}")
        print()
    
    all_results = {
        'symbol': symbol,
        'timeframes': {},
        'summary': {}
    }
    
    # Analyze each timeframe
    for tf_interval in timeframes:
        if tf_interval not in TIMEFRAME_CONFIGS:
            print(f"Unknown timeframe: {tf_interval}, skipping")
            continue
        
        tf_config = TIMEFRAME_CONFIGS[tf_interval]
        
        if verbose:
            print(f"\n[{tf_config.name.upper()}] Analyzing {tf_interval} data...")
            print(f"   Max history: {tf_config.max_days} days")
            print(f"   Window size: {tf_config.min_bars}-{tf_config.max_bars} bars")
        
        # Fetch data for this timeframe
        tf_start = time.time()
        data = asyncio.run(fetch_symbols(
            [symbol], 
            start_days=tf_config.max_days,
            interval=tf_interval
        ))
        
        df = data.get(symbol)
        if df is None or len(df) == 0:
            print(f"   No data available for {tf_interval}")
            all_results['timeframes'][tf_interval] = {
                'patterns': [],
                'count': 0,
                'bars': 0,
                'error': 'No data available'
            }
            continue
        
        if verbose:
            date_range = f"{df['Date'].min()} to {df['Date'].max()}"
            print(f"   Data: {len(df)} bars ({date_range})")
        
        # Run pattern detection
        patterns = sequential_scan_timeframe(df, tf_config, symbol, cfg, verbose=verbose)
        
        tf_time = time.time() - tf_start
        
        # Categorize patterns by type
        pattern_types = {}
        for p in patterns:
            rule = p['best'].get('rule_name', 'unknown')
            pattern_types[rule] = pattern_types.get(rule, 0) + 1
        
        # Store results
        all_results['timeframes'][tf_interval] = {
            'name': tf_config.name,
            'patterns': patterns,
            'count': len(patterns),
            'bars': len(df),
            'pattern_types': pattern_types,
            'runtime_seconds': tf_time
        }
        
        if verbose:
            print(f"   Found {len(patterns)} patterns in {tf_time:.1f}s")
            if pattern_types:
                types_str = ', '.join([f"{k}:{v}" for k, v in pattern_types.items()])
                print(f"   Types: {types_str}")
    
    # Generate summary
    total_patterns = sum(tf['count'] for tf in all_results['timeframes'].values())
    total_runtime = time.time() - start_time
    
    all_results['summary'] = {
        'total_patterns': total_patterns,
        'runtime_seconds': total_runtime,
        'patterns_by_timeframe': {
            tf: data['count'] for tf, data in all_results['timeframes'].items()
        },
        'all_pattern_types': {}
    }
    
    # Aggregate pattern types across timeframes
    for tf_data in all_results['timeframes'].values():
        for ptype, count in tf_data.get('pattern_types', {}).items():
            all_results['summary']['all_pattern_types'][ptype] = \
                all_results['summary']['all_pattern_types'].get(ptype, 0) + count
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Symbol: {symbol}")
        print(f"Total patterns found: {total_patterns}")
        print(f"Total runtime: {total_runtime:.1f}s")
        print()
        print("Patterns by timeframe:")
        for tf, data in all_results['timeframes'].items():
            print(f"   {data.get('name', tf)}: {data['count']} patterns ({data['bars']} bars)")
        print()
        print("Pattern types:")
        for ptype, count in all_results['summary']['all_pattern_types'].items():
            print(f"   {ptype}: {count}")
    
    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert patterns for JSON serialization
        def serialize_pattern(p):
            serializable = {}
            for k, v in p.items():
                if k == '_pattern_obj':
                    continue  # Skip non-serializable object
                if hasattr(v, 'isoformat'):
                    serializable[k] = v.isoformat()
                elif hasattr(v, 'tolist'):
                    serializable[k] = v.tolist()
                elif isinstance(v, dict):
                    serializable[k] = serialize_pattern(v)
                else:
                    serializable[k] = v
            return serializable
        
        output_data = {
            'symbol': all_results['symbol'],
            'summary': all_results['summary'],
            'timeframes': {}
        }
        
        for tf, data in all_results['timeframes'].items():
            output_data['timeframes'][tf] = {
                'name': data.get('name', tf),
                'count': data['count'],
                'bars': data['bars'],
                'pattern_types': data.get('pattern_types', {}),
                'runtime_seconds': data.get('runtime_seconds', 0),
                'patterns': [serialize_pattern(p) for p in data.get('patterns', [])]
            }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Hybrid Timeframe Elliott Wave Analysis')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--timeframes', type=str, default='1h,1d,1wk', 
                        help='Comma-separated timeframes (1h, 1d, 1wk)')
    parser.add_argument('--output', type=str, default='output/hybrid_analysis.json',
                        help='Output JSON file path')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    run_hybrid_analysis(
        symbol=args.symbol,
        timeframes=timeframes,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
