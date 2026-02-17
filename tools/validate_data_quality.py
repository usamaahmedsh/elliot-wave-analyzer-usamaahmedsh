#!/usr/bin/env python3
"""
Data Quality Validation for Elliott Wave Training Dataset

Validates pattern detection results before using them for ML training:
- Pattern score distributions
- Temporal coverage and gaps
- Label balance (for classification)
- Outlier detection
- Statistical sanity checks
"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter


class DataQualityValidator:
    def __init__(self, patterns_json: str):
        """Load and prepare pattern data for validation."""
        with open(patterns_json, 'r') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        self.patterns = data.get('patterns', [])
        self.df = pd.DataFrame(self.patterns)
        
        print(f"Loaded {len(self.patterns)} patterns from {patterns_json}")
        print(f"Total symbols: {self.metadata.get('total_symbols', 'N/A')}")
        print(f"Runtime: {self.metadata.get('runtime_seconds', 0):.1f}s\n")
    
    def check_completeness(self) -> Dict[str, Any]:
        """Check for missing or incomplete data."""
        print("=" * 70)
        print("1. COMPLETENESS CHECK")
        print("=" * 70)
        
        issues = []
        
        # Required fields
        required_fields = ['symbol', 'start_row', 'window_len', 'date_start', 
                          'date_end', 'ensemble_score', 'fib_score', 'best']
        
        for field in required_fields:
            missing = self.df[field].isna().sum()
            pct = (missing / len(self.df)) * 100
            print(f"  {field:20s} Missing: {missing:5d} ({pct:5.2f}%)")
            
            if missing > 0:
                issues.append(f"Missing {field}: {missing} records")
        
        # Check nested 'best' field
        if 'best' in self.df.columns:
            valid_best = sum(1 for p in self.patterns if isinstance(p.get('best'), dict))
            invalid_best = len(self.patterns) - valid_best
            print(f"  {'best (dict)':20s} Invalid: {invalid_best:5d} ({invalid_best/len(self.patterns)*100:5.2f}%)")
            
            if invalid_best > 0:
                issues.append(f"Invalid 'best' field: {invalid_best} records")
        
        print(f"\n  ✓ Total issues: {len(issues)}")
        return {'issues': issues, 'completeness_rate': 1 - (len(issues) / len(required_fields))}
    
    def check_score_distributions(self) -> Dict[str, Any]:
        """Validate score distributions and detect anomalies."""
        print("\n" + "=" * 70)
        print("2. SCORE DISTRIBUTION CHECK")
        print("=" * 70)
        
        issues = []
        
        # Ensemble scores
        ensemble_scores = self.df['ensemble_score'].dropna()
        print(f"\n  Ensemble Score Statistics:")
        print(f"    Count:  {len(ensemble_scores)}")
        print(f"    Mean:   {ensemble_scores.mean():.4f}")
        print(f"    Median: {ensemble_scores.median():.4f}")
        print(f"    Std:    {ensemble_scores.std():.4f}")
        print(f"    Min:    {ensemble_scores.min():.4f}")
        print(f"    Max:    {ensemble_scores.max():.4f}")
        
        # Check for suspicious patterns
        if ensemble_scores.std() < 0.01:
            issues.append("Ensemble scores have very low variance - possible duplication")
        
        if (ensemble_scores == ensemble_scores.iloc[0]).sum() > len(ensemble_scores) * 0.5:
            issues.append("Over 50% of ensemble scores are identical - suspicious!")
        
        # Quality tiers
        excellent = (ensemble_scores >= 0.85).sum()
        good = ((ensemble_scores >= 0.70) & (ensemble_scores < 0.85)).sum()
        fair = ((ensemble_scores >= 0.50) & (ensemble_scores < 0.70)).sum()
        poor = (ensemble_scores < 0.50).sum()
        
        print(f"\n  Quality Tier Distribution:")
        print(f"    Excellent (≥0.85): {excellent:5d} ({excellent/len(ensemble_scores)*100:5.1f}%)")
        print(f"    Good (0.70-0.85):  {good:5d} ({good/len(ensemble_scores)*100:5.1f}%)")
        print(f"    Fair (0.50-0.70):  {fair:5d} ({fair/len(ensemble_scores)*100:5.1f}%)")
        print(f"    Poor (<0.50):      {poor:5d} ({poor/len(ensemble_scores)*100:5.1f}%)")
        
        # Warn if distribution is too skewed
        if excellent > len(ensemble_scores) * 0.9:
            issues.append("Over 90% excellent scores - may indicate overfitting or validation issues")
        
        if poor > len(ensemble_scores) * 0.5:
            issues.append("Over 50% poor scores - detection may be unreliable")
        
        # Fibonacci scores
        fib_scores = self.df['fib_score'].dropna()
        print(f"\n  Fibonacci Score Statistics:")
        print(f"    Mean:   {fib_scores.mean():.4f}")
        print(f"    Median: {fib_scores.median():.4f}")
        
        print(f"\n  ✓ Total issues: {len(issues)}")
        return {'issues': issues, 'ensemble_mean': ensemble_scores.mean()}
    
    def check_temporal_coverage(self) -> Dict[str, Any]:
        """Check date ranges and temporal gaps."""
        print("\n" + "=" * 70)
        print("3. TEMPORAL COVERAGE CHECK")
        print("=" * 70)
        
        issues = []
        
        # Parse dates
        self.df['date_start_parsed'] = pd.to_datetime(self.df['date_start'], errors='coerce')
        self.df['date_end_parsed'] = pd.to_datetime(self.df['date_end'], errors='coerce')
        
        # Overall date range
        min_date = self.df['date_start_parsed'].min()
        max_date = self.df['date_end_parsed'].max()
        date_range_days = (max_date - min_date).days
        
        print(f"\n  Overall Date Range:")
        print(f"    Earliest: {min_date.strftime('%Y-%m-%d')}")
        print(f"    Latest:   {max_date.strftime('%Y-%m-%d')}")
        print(f"    Span:     {date_range_days} days ({date_range_days/365:.1f} years)")
        
        # Check for patterns in the future (data leakage!)
        today = pd.Timestamp.now()
        future_patterns = (self.df['date_end_parsed'] > today).sum()
        if future_patterns > 0:
            issues.append(f"WARNING: {future_patterns} patterns end in the future - DATA LEAKAGE!")
            print(f"\n  ⚠️  {future_patterns} patterns end after today - potential data leakage!")
        
        # Check pattern duration distribution
        self.df['pattern_duration'] = (self.df['date_end_parsed'] - self.df['date_start_parsed']).dt.days
        
        print(f"\n  Pattern Duration Statistics:")
        print(f"    Mean:   {self.df['pattern_duration'].mean():.1f} days")
        print(f"    Median: {self.df['pattern_duration'].median():.1f} days")
        print(f"    Min:    {self.df['pattern_duration'].min():.1f} days")
        print(f"    Max:    {self.df['pattern_duration'].max():.1f} days")
        
        # Warn about very short or very long patterns
        very_short = (self.df['pattern_duration'] < 7).sum()
        very_long = (self.df['pattern_duration'] > 730).sum()  # 2 years
        
        if very_short > 0:
            print(f"    ⚠️  {very_short} patterns shorter than 1 week")
            issues.append(f"{very_short} patterns shorter than 1 week")
        
        if very_long > 0:
            print(f"    ⚠️  {very_long} patterns longer than 2 years")
            issues.append(f"{very_long} patterns longer than 2 years")
        
        print(f"\n  ✓ Total issues: {len(issues)}")
        return {'issues': issues, 'date_range_years': date_range_days/365}
    
    def check_pattern_diversity(self) -> Dict[str, Any]:
        """Check for pattern type diversity and symbol coverage."""
        print("\n" + "=" * 70)
        print("4. PATTERN DIVERSITY CHECK")
        print("=" * 70)
        
        issues = []
        
        # Pattern type distribution
        pattern_types = []
        for p in self.patterns:
            if isinstance(p.get('best'), dict):
                pattern_types.append(p['best'].get('rule_name', 'unknown'))
        
        type_counts = Counter(pattern_types)
        
        print(f"\n  Pattern Type Distribution:")
        for ptype, count in type_counts.most_common():
            pct = (count / len(pattern_types)) * 100
            print(f"    {ptype:20s} {count:5d} ({pct:5.1f}%)")
        
        # Warn if extremely imbalanced
        if len(type_counts) > 1:
            most_common_pct = type_counts.most_common(1)[0][1] / len(pattern_types)
            if most_common_pct > 0.95:
                issues.append(f"Over 95% patterns are same type - very imbalanced for classification")
        
        # Symbol distribution
        symbol_counts = self.df['symbol'].value_counts()
        
        print(f"\n  Symbol Coverage:")
        print(f"    Unique symbols: {len(symbol_counts)}")
        print(f"    Patterns per symbol (avg): {symbol_counts.mean():.1f}")
        print(f"    Patterns per symbol (median): {symbol_counts.median():.1f}")
        print(f"    Min: {symbol_counts.min()}")
        print(f"    Max: {symbol_counts.max()}")
        
        # Check for over-representation
        overrepresented = symbol_counts[symbol_counts > symbol_counts.mean() + 2*symbol_counts.std()]
        if len(overrepresented) > 0:
            print(f"\n    Over-represented symbols (>2 std above mean):")
            for sym, count in overrepresented.head(5).items():
                print(f"      {sym}: {count} patterns")
            issues.append(f"{len(overrepresented)} symbols over-represented")
        
        print(f"\n  ✓ Total issues: {len(issues)}")
        return {'issues': issues, 'unique_symbols': len(symbol_counts)}
    
    def check_statistical_sanity(self) -> Dict[str, Any]:
        """Statistical sanity checks for ML readiness."""
        print("\n" + "=" * 70)
        print("5. STATISTICAL SANITY CHECKS")
        print("=" * 70)
        
        issues = []
        
        # Check for duplicates
        duplicates = self.df.duplicated(subset=['symbol', 'start_row', 'window_len']).sum()
        print(f"\n  Duplicate Patterns: {duplicates}")
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate patterns found")
        
        # Check for outliers in scores (using IQR method)
        Q1 = self.df['ensemble_score'].quantile(0.25)
        Q3 = self.df['ensemble_score'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((self.df['ensemble_score'] < (Q1 - 1.5 * IQR)) | 
                   (self.df['ensemble_score'] > (Q3 + 1.5 * IQR))).sum()
        
        print(f"  Score Outliers (IQR method): {outliers} ({outliers/len(self.df)*100:.1f}%)")
        
        # Check correlation between scores
        if 'ensemble_score' in self.df.columns and 'fib_score' in self.df.columns:
            correlation = self.df[['ensemble_score', 'fib_score']].corr().iloc[0, 1]
            print(f"  Ensemble-Fib correlation: {correlation:.3f}")
            
            if correlation > 0.99:
                issues.append("Ensemble and Fib scores are nearly identical - redundant features")
        
        # Sample size check
        print(f"\n  Sample Size Assessment:")
        if len(self.df) < 100:
            issues.append("Less than 100 samples - too small for ML training")
            print(f"    ⚠️  {len(self.df)} samples - VERY SMALL for ML")
        elif len(self.df) < 1000:
            issues.append("Less than 1000 samples - marginal for ML")
            print(f"    ⚠️  {len(self.df)} samples - small for robust ML")
        elif len(self.df) < 10000:
            print(f"    ✓ {len(self.df)} samples - adequate for ML")
        else:
            print(f"    ✓ {len(self.df)} samples - good size for ML")
        
        print(f"\n  ✓ Total issues: {len(issues)}")
        return {'issues': issues, 'sample_size': len(self.df)}
    
    def generate_summary_report(self, output_path: str = 'output/data_quality_report.txt'):
        """Generate comprehensive validation report."""
        print("\n" + "=" * 70)
        print("GENERATING SUMMARY REPORT")
        print("=" * 70)
        
        results = {
            'completeness': self.check_completeness(),
            'scores': self.check_score_distributions(),
            'temporal': self.check_temporal_coverage(),
            'diversity': self.check_pattern_diversity(),
            'statistical': self.check_statistical_sanity(),
        }
        
        # Count total issues
        total_issues = sum(len(r['issues']) for r in results.values())
        
        print(f"\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"\n  Total Patterns: {len(self.df)}")
        print(f"  Total Issues Found: {total_issues}")
        
        if total_issues == 0:
            print(f"\n  ✅ DATA QUALITY: EXCELLENT - Ready for ML training")
            quality_verdict = "EXCELLENT"
        elif total_issues <= 3:
            print(f"\n  ✓ DATA QUALITY: GOOD - Minor issues, proceed with caution")
            quality_verdict = "GOOD"
        elif total_issues <= 10:
            print(f"\n  ⚠️  DATA QUALITY: FAIR - Address issues before ML training")
            quality_verdict = "FAIR"
        else:
            print(f"\n  ❌ DATA QUALITY: POOR - Fix issues before proceeding")
            quality_verdict = "POOR"
        
        # Critical issues
        critical = []
        for category, result in results.items():
            for issue in result['issues']:
                if any(keyword in issue.lower() for keyword in ['leakage', 'duplicate', 'too small']):
                    critical.append(f"[{category.upper()}] {issue}")
        
        if critical:
            print(f"\n  🚨 CRITICAL ISSUES:")
            for issue in critical:
                print(f"    - {issue}")
        
        # Save detailed report
        with open(output_path, 'w') as f:
            f.write("ELLIOTT WAVE DATA QUALITY VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {len(self.df)} patterns\n")
            f.write(f"Quality Verdict: {quality_verdict}\n")
            f.write(f"Total Issues: {total_issues}\n\n")
            
            for category, result in results.items():
                f.write(f"\n{category.upper()}\n")
                f.write("-" * 70 + "\n")
                for issue in result['issues']:
                    f.write(f"  - {issue}\n")
        
        print(f"\n  📄 Detailed report saved to: {output_path}")
        
        return {
            'verdict': quality_verdict,
            'total_issues': total_issues,
            'sample_size': len(self.df),
            'results': results
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Elliott Wave pattern data quality for ML training')
    parser.add_argument('--input', '-i', default='output/results.json',
                       help='Path to patterns JSON file')
    parser.add_argument('--output', '-o', default='output/data_quality_report.txt',
                       help='Path to save detailed report')
    parser.add_argument('--strict', action='store_true',
                       help='Exit with error code if quality is not EXCELLENT')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run validation
    validator = DataQualityValidator(args.input)
    summary = validator.generate_summary_report(args.output)
    
    # Exit code based on quality
    if args.strict and summary['verdict'] != 'EXCELLENT':
        print(f"\n❌ Strict mode: Exiting with error due to quality verdict: {summary['verdict']}")
        sys.exit(1)
    
    if summary['verdict'] == 'POOR':
        sys.exit(2)
    
    print(f"\n✅ Validation complete!")


if __name__ == '__main__':
    main()
