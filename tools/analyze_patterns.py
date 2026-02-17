#!/usr/bin/env python3
"""
Pattern Quality Analysis Tool

Analyzes the output JSON from the Elliott Wave pipeline and provides:
- Statistical summaries of pattern scores
- Distribution analysis
- Quality filtering and reporting
- Pattern type breakdown
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics


def load_results(json_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_score_distribution(patterns: List[Dict[str, Any]], score_key: str = 'ensemble_score') -> Dict[str, float]:
    """Analyze the distribution of a specific score."""
    scores = [p.get(score_key, 0.0) for p in patterns if score_key in p]
    
    if not scores:
        return {}
    
    return {
        'count': len(scores),
        'min': min(scores),
        'max': max(scores),
        'mean': statistics.mean(scores),
        'median': statistics.median(scores),
        'stdev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
        'q25': statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else min(scores),
        'q75': statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else max(scores),
    }


def filter_by_score(patterns: List[Dict[str, Any]], min_score: float, score_key: str = 'ensemble_score') -> List[Dict[str, Any]]:
    """Filter patterns by minimum score threshold."""
    return [p for p in patterns if p.get(score_key, 0.0) >= min_score]


def analyze_pattern_types(patterns: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count patterns by type (impulse vs corrective)."""
    type_counts = {}
    
    for pattern in patterns:
        if 'best' in pattern:
            best = pattern['best']
            if isinstance(best, dict) and 'rule_name' in best:
                rule_name = best['rule_name']
                type_counts[rule_name] = type_counts.get(rule_name, 0) + 1
    
    return type_counts


def analyze_by_symbol(patterns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group patterns by symbol and analyze each."""
    by_symbol = {}
    
    for pattern in patterns:
        symbol = pattern.get('symbol', 'UNKNOWN')
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(pattern)
    
    results = {}
    for symbol, symbol_patterns in by_symbol.items():
        results[symbol] = {
            'count': len(symbol_patterns),
            'ensemble_score': analyze_score_distribution(symbol_patterns, 'ensemble_score'),
            'fib_score': analyze_score_distribution(symbol_patterns, 'fib_score'),
            'pattern_types': analyze_pattern_types(symbol_patterns),
        }
    
    return results


def print_score_stats(label: str, stats: Dict[str, float]):
    """Pretty print score statistics."""
    if not stats:
        print(f"  {label}: No data")
        return
    
    print(f"  {label}:")
    print(f"    Count:   {stats['count']}")
    print(f"    Min:     {stats['min']:.3f}")
    print(f"    Q25:     {stats['q25']:.3f}")
    print(f"    Median:  {stats['median']:.3f}")
    print(f"    Mean:    {stats['mean']:.3f}")
    print(f"    Q75:     {stats['q75']:.3f}")
    print(f"    Max:     {stats['max']:.3f}")
    print(f"    StdDev:  {stats['stdev']:.3f}")


def print_quality_tiers(patterns: List[Dict[str, Any]]):
    """Print pattern counts by quality tier."""
    tiers = [
        ('Excellent (0.85-1.0)', 0.85, 1.0),
        ('Good (0.70-0.85)', 0.70, 0.85),
        ('Fair (0.50-0.70)', 0.50, 0.70),
        ('Poor (<0.50)', 0.0, 0.50),
    ]
    
    print("\n" + "="*70)
    print("QUALITY TIERS (by Ensemble Score)")
    print("="*70)
    
    for tier_name, min_score, max_score in tiers:
        if max_score == 1.0:
            tier_patterns = [p for p in patterns if p.get('ensemble_score', 0.0) >= min_score]
        else:
            tier_patterns = [p for p in patterns if min_score <= p.get('ensemble_score', 0.0) < max_score]
        
        count = len(tier_patterns)
        percentage = (count / len(patterns) * 100) if patterns else 0
        print(f"  {tier_name:25s} {count:4d} patterns ({percentage:5.1f}%)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Elliott Wave pattern detection results')
    parser.add_argument('--input', '-i', default='output/results.json',
                       help='Path to results JSON file (default: output/results.json)')
    parser.add_argument('--min-score', type=float, default=0.0,
                       help='Minimum ensemble score to include (default: 0.0)')
    parser.add_argument('--export-filtered', '-e',
                       help='Export filtered patterns to this JSON file')
    parser.add_argument('--by-symbol', action='store_true',
                       help='Show per-symbol breakdown')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    try:
        results = load_results(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)
    
    patterns = results.get('patterns', [])
    metadata = results.get('metadata', {})
    
    print("\n" + "="*70)
    print("ELLIOTT WAVE PATTERN ANALYSIS")
    print("="*70)
    
    # Metadata
    print(f"\nTotal Symbols:  {metadata.get('total_symbols', 'N/A')}")
    print(f"Total Patterns: {metadata.get('total_patterns', len(patterns))}")
    print(f"Runtime:        {metadata.get('runtime_seconds', 'N/A')} seconds")
    
    if not patterns:
        print("\nNo patterns found in results.")
        return
    
    # Apply filter
    if args.min_score > 0:
        filtered_patterns = filter_by_score(patterns, args.min_score)
        print(f"\nFiltered to {len(filtered_patterns)} patterns with ensemble_score >= {args.min_score}")
        patterns = filtered_patterns
    
    if not patterns:
        print("\nNo patterns remaining after filtering.")
        return
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL SCORE DISTRIBUTION")
    print("="*70)
    
    ensemble_stats = analyze_score_distribution(patterns, 'ensemble_score')
    fib_stats = analyze_score_distribution(patterns, 'fib_score')
    
    # Extract rule scores from 'best' nested dict
    best_patterns = [p.get('best', {}) for p in patterns if 'best' in p and isinstance(p.get('best'), dict)]
    rule_stats = analyze_score_distribution(best_patterns, 'score')
    
    print_score_stats("Ensemble Score", ensemble_stats)
    print()
    print_score_stats("Fibonacci Score", fib_stats)
    print()
    print_score_stats("Rule Compliance Score", rule_stats)
    
    # Quality tiers
    print_quality_tiers(patterns)
    
    # Pattern types
    print("\n" + "="*70)
    print("PATTERN TYPE BREAKDOWN")
    print("="*70)
    
    pattern_types = analyze_pattern_types(patterns)
    for pattern_type, count in sorted(pattern_types.items(), key=lambda x: -x[1]):
        percentage = (count / len(patterns) * 100)
        print(f"  {pattern_type:20s} {count:4d} patterns ({percentage:5.1f}%)")
    
    # Per-symbol analysis
    if args.by_symbol:
        print("\n" + "="*70)
        print("PER-SYMBOL ANALYSIS")
        print("="*70)
        
        by_symbol = analyze_by_symbol(patterns)
        
        for symbol in sorted(by_symbol.keys()):
            symbol_data = by_symbol[symbol]
            print(f"\n{symbol}:")
            print(f"  Total patterns: {symbol_data['count']}")
            
            if symbol_data['ensemble_score']:
                print(f"  Ensemble score: mean={symbol_data['ensemble_score']['mean']:.3f}, "
                      f"median={symbol_data['ensemble_score']['median']:.3f}, "
                      f"max={symbol_data['ensemble_score']['max']:.3f}")
            
            if symbol_data['pattern_types']:
                types_str = ", ".join([f"{k}:{v}" for k, v in symbol_data['pattern_types'].items()])
                print(f"  Pattern types: {types_str}")
    
    # Export filtered results
    if args.export_filtered:
        filtered_results = {
            'metadata': {
                **metadata,
                'filtered_by': f'ensemble_score >= {args.min_score}',
                'original_count': metadata.get('total_patterns', len(results.get('patterns', []))),
                'filtered_count': len(patterns),
            },
            'patterns': patterns
        }
        
        with open(args.export_filtered, 'w') as f:
            json.dump(filtered_results, f, indent=2)
        
        print(f"\n✅ Exported {len(patterns)} filtered patterns to {args.export_filtered}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
