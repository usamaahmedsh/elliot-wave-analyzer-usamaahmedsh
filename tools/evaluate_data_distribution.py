#!/usr/bin/env python3
"""
Elliott Wave Pattern Dataset - Exploratory Data Analysis (EDA)

This script analyzes the distribution and characteristics of the generated
Elliott Wave pattern dataset.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(results_path: str) -> dict:
    """Load the results JSON file"""
    print(f"📂 Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded {data['metadata']['total_patterns']} patterns from {data['metadata']['total_symbols']} symbols")
    return data


def analyze_basic_stats(patterns: list) -> dict:
    """Basic statistics about the dataset"""
    stats = {
        'total_patterns': len(patterns),
        'unique_symbols': len(set(p['symbol'] for p in patterns)),
    }
    return stats


def analyze_pattern_types(patterns: list) -> dict:
    """Analyze distribution of pattern types"""
    # Count by rule_name (impulse vs corrective)
    rule_names = Counter(p['best']['rule_name'].lower() for p in patterns)
    
    # Count by scan_stats pattern_type
    scan_types = Counter(p['scan_stats']['pattern_type'].lower() for p in patterns)
    
    return {
        'by_rule_name': dict(rule_names),
        'by_scan_type': dict(scan_types)
    }


def analyze_scores(patterns: list) -> dict:
    """Analyze score distributions"""
    ensemble_scores = [p['ensemble_score'] for p in patterns]
    fib_scores = [p['fib_score'] for p in patterns]
    best_scores = [p['best']['score'] for p in patterns]
    
    def score_stats(scores, name):
        arr = np.array(scores)
        return {
            'name': name,
            'count': len(arr),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
        }
    
    # Quality buckets
    def quality_bucket(score):
        if score >= 0.85:
            return 'Excellent (≥0.85)'
        elif score >= 0.70:
            return 'Good (0.70-0.85)'
        elif score >= 0.50:
            return 'Fair (0.50-0.70)'
        else:
            return 'Poor (<0.50)'
    
    quality_dist = Counter(quality_bucket(s) for s in ensemble_scores)
    
    return {
        'ensemble': score_stats(ensemble_scores, 'ensemble_score'),
        'fibonacci': score_stats(fib_scores, 'fib_score'),
        'best_score': score_stats(best_scores, 'best_score'),
        'quality_distribution': dict(quality_dist)
    }


def analyze_symbols(patterns: list) -> dict:
    """Analyze patterns per symbol"""
    patterns_per_symbol = Counter(p['symbol'] for p in patterns)
    
    counts = list(patterns_per_symbol.values())
    
    # Top and bottom symbols
    top_10 = patterns_per_symbol.most_common(10)
    bottom_10 = patterns_per_symbol.most_common()[:-11:-1]
    
    return {
        'patterns_per_symbol': {
            'mean': float(np.mean(counts)),
            'median': float(np.median(counts)),
            'std': float(np.std(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts)),
        },
        'top_10_symbols': top_10,
        'bottom_10_symbols': bottom_10,
        'symbols_with_zero_patterns': sum(1 for c in counts if c == 0),
    }


def analyze_wave_configs(patterns: list) -> dict:
    """Analyze wave configuration distributions"""
    # Wave configs are like [5, 5, 7, 5, 4] representing degrees of each wave
    wave_configs = [tuple(p['best']['wave_config']) for p in patterns if p['best']['wave_config']]
    
    # Filter out configs with None values
    valid_configs = [wc for wc in wave_configs if None not in wc]
    
    config_counts = Counter(valid_configs)
    
    # Pattern lengths (idx_end - idx_start)
    pattern_lengths = [p['best']['idx_end'] - p['best']['idx_start'] for p in patterns]
    
    return {
        'total_valid_configs': len(valid_configs),
        'unique_configs': len(config_counts),
        'top_10_configs': config_counts.most_common(10),
        'pattern_length': {
            'mean': float(np.mean(pattern_lengths)),
            'median': float(np.median(pattern_lengths)),
            'std': float(np.std(pattern_lengths)),
            'min': int(np.min(pattern_lengths)),
            'max': int(np.max(pattern_lengths)),
        }
    }


def analyze_temporal(patterns: list) -> dict:
    """Analyze temporal distribution of patterns"""
    # Parse dates
    years = []
    durations = []
    
    for p in patterns:
        try:
            start = datetime.fromisoformat(p['date_start'].replace('.000000000', ''))
            end = datetime.fromisoformat(p['date_end'].replace('.000000000', ''))
            years.append(start.year)
            durations.append((end - start).days)
        except:
            continue
    
    year_dist = Counter(years)
    
    return {
        'year_distribution': dict(sorted(year_dist.items())),
        'duration_days': {
            'mean': float(np.mean(durations)),
            'median': float(np.median(durations)),
            'min': int(np.min(durations)),
            'max': int(np.max(durations)),
        }
    }


def analyze_by_pattern_type(patterns: list) -> dict:
    """Analyze scores broken down by pattern type"""
    impulse_patterns = [p for p in patterns if p['best']['rule_name'].lower() == 'impulse']
    corrective_patterns = [p for p in patterns if p['best']['rule_name'].lower() == 'corrective']
    
    def get_score_summary(pattern_list, name):
        if not pattern_list:
            return {'name': name, 'count': 0}
        scores = [p['ensemble_score'] for p in pattern_list]
        return {
            'name': name,
            'count': len(pattern_list),
            'mean_ensemble': float(np.mean(scores)),
            'median_ensemble': float(np.median(scores)),
        }
    
    return {
        'impulse': get_score_summary(impulse_patterns, 'impulse'),
        'corrective': get_score_summary(corrective_patterns, 'corrective'),
    }


def print_report(analysis: dict):
    """Print a formatted report"""
    print("\n" + "="*70)
    print("         ELLIOTT WAVE PATTERN DATASET - EDA REPORT")
    print("="*70)
    
    # Basic stats
    print("\n📊 BASIC STATISTICS")
    print("-"*50)
    print(f"  Total Patterns:     {analysis['basic']['total_patterns']:,}")
    print(f"  Unique Symbols:     {analysis['basic']['unique_symbols']}")
    
    # Pattern types
    print("\n📈 PATTERN TYPE DISTRIBUTION")
    print("-"*50)
    print("  By Rule Name:")
    for k, v in analysis['pattern_types']['by_rule_name'].items():
        pct = v / analysis['basic']['total_patterns'] * 100
        print(f"    {k.capitalize():15} {v:6,} ({pct:.1f}%)")
    
    # Scores
    print("\n⭐ SCORE DISTRIBUTIONS")
    print("-"*50)
    for score_type in ['ensemble', 'fibonacci', 'best_score']:
        s = analysis['scores'][score_type]
        print(f"  {s['name']}:")
        print(f"    Mean: {s['mean']:.3f}  Median: {s['median']:.3f}  Std: {s['std']:.3f}")
        print(f"    Range: [{s['min']:.3f}, {s['max']:.3f}]  IQR: [{s['q25']:.3f}, {s['q75']:.3f}]")
    
    print("\n  Quality Distribution (by ensemble score):")
    for quality, count in sorted(analysis['scores']['quality_distribution'].items()):
        pct = count / analysis['basic']['total_patterns'] * 100
        print(f"    {quality:20} {count:6,} ({pct:.1f}%)")
    
    # By pattern type
    print("\n📊 SCORES BY PATTERN TYPE")
    print("-"*50)
    for ptype in ['impulse', 'corrective']:
        p = analysis['by_pattern_type'][ptype]
        if p['count'] > 0:
            print(f"  {ptype.capitalize():12} n={p['count']:,}  mean={p['mean_ensemble']:.3f}  median={p['median_ensemble']:.3f}")
    
    # Symbols
    print("\n🏢 SYMBOL ANALYSIS")
    print("-"*50)
    s = analysis['symbols']['patterns_per_symbol']
    print(f"  Patterns per symbol: mean={s['mean']:.1f}, median={s['median']:.0f}, range=[{s['min']}, {s['max']}]")
    print("\n  Top 10 symbols (most patterns):")
    for sym, count in analysis['symbols']['top_10_symbols']:
        print(f"    {sym:10} {count:4} patterns")
    print("\n  Bottom 10 symbols (fewest patterns):")
    for sym, count in analysis['symbols']['bottom_10_symbols']:
        print(f"    {sym:10} {count:4} patterns")
    
    # Wave configs
    print("\n🌊 WAVE CONFIGURATION ANALYSIS")
    print("-"*50)
    wc = analysis['wave_configs']
    print(f"  Valid configurations: {wc['total_valid_configs']:,}")
    print(f"  Unique configurations: {wc['unique_configs']}")
    print(f"\n  Pattern length (bars):")
    pl = wc['pattern_length']
    print(f"    Mean: {pl['mean']:.1f}  Median: {pl['median']:.0f}  Range: [{pl['min']}, {pl['max']}]")
    print("\n  Top 10 wave configs:")
    for config, count in wc['top_10_configs']:
        print(f"    {str(config):20} {count:4} occurrences")
    
    # Temporal
    print("\n📅 TEMPORAL ANALYSIS")
    print("-"*50)
    print("  Patterns by start year:")
    for year, count in analysis['temporal']['year_distribution'].items():
        pct = count / analysis['basic']['total_patterns'] * 100
        bar = '█' * int(pct / 2)
        print(f"    {year}: {count:5} ({pct:5.1f}%) {bar}")
    
    td = analysis['temporal']['duration_days']
    print(f"\n  Pattern duration (days):")
    print(f"    Mean: {td['mean']:.0f}  Median: {td['median']:.0f}  Range: [{td['min']}, {td['max']}]")
    
    print("\n" + "="*70)
    print("                         END OF REPORT")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Elliott Wave Pattern Dataset EDA')
    parser.add_argument('--input', type=str, default='output/all_symbols_results.json',
                        help='Path to results JSON file')
    parser.add_argument('--output', type=str, default='output/eda_report.json',
                        help='Path to save JSON report')
    
    args = parser.parse_args()
    
    # Load data
    data = load_results(args.input)
    patterns = data['patterns']
    
    # Run analyses
    print("\n🔍 Running analyses...")
    
    analysis = {
        'basic': analyze_basic_stats(patterns),
        'pattern_types': analyze_pattern_types(patterns),
        'scores': analyze_scores(patterns),
        'symbols': analyze_symbols(patterns),
        'wave_configs': analyze_wave_configs(patterns),
        'temporal': analyze_temporal(patterns),
        'by_pattern_type': analyze_by_pattern_type(patterns),
    }
    
    # Print report
    print_report(analysis)
    
    # Save JSON report
    print(f"\n💾 Saving detailed report to: {args.output}")
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        elif isinstance(obj, tuple):
            return [convert_for_json(i) for i in obj]
        return obj
    
    analysis_json = convert_for_json(analysis)
    
    with open(args.output, 'w') as f:
        json.dump(analysis_json, f, indent=2)
    
    print("✅ Done!")


if __name__ == '__main__':
    main()
