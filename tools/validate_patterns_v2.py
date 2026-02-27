#!/usr/bin/env python3
"""
Elliott Wave Pattern Validation v2

This script validates detected patterns using the actual wave boundaries
stored in the JSON output file.

Key Difference from v1:
- v1 tried to guess wave boundaries from OHLCV data (unreliable)
- v2 uses the actual wave boundaries stored by the pattern detector
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def load_results(results_path: str) -> dict:
    """Load the results JSON file"""
    print(f"📂 Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded {data['metadata']['total_patterns']} patterns")
    return data


class WaveData:
    """Simple class to hold wave data for rule validation"""
    def __init__(self, wave_dict: dict):
        self.idx_start = wave_dict.get('idx_start', 0)
        self.idx_end = wave_dict.get('idx_end', 0)
        self.low = wave_dict.get('low', 0.0)
        self.high = wave_dict.get('high', 0.0)
        self.low_idx = wave_dict.get('low_idx', 0)
        self.high_idx = wave_dict.get('high_idx', 0)
        self.length = wave_dict.get('length', 0.0)
        self.duration = wave_dict.get('duration', 0)


def extract_waves(pattern: dict) -> Optional[Dict[str, WaveData]]:
    """
    Extract wave data from a pattern's stored wave boundaries.
    Returns None if wave boundaries are not available.
    """
    best = pattern.get('best', {})
    waves_dict = best.get('waves', {})
    
    if not waves_dict:
        return None
    
    waves = {}
    for wave_name, wave_info in waves_dict.items():
        if wave_info:
            waves[wave_name] = WaveData(wave_info)
    
    return waves if waves else None


# =============================================================================
# IMPULSE WAVE RULES
# =============================================================================

def check_impulse_w2_no_full_retrace(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 2 cannot retrace more than 100% of Wave 1 (W2 low > W1 low)"""
    w1, w2 = waves.get('wave1'), waves.get('wave2')
    if not w1 or not w2:
        return None, "Missing wave data"
    passes = w2.low > w1.low
    return passes, f"W2.low ({w2.low:.4f}) > W1.low ({w1.low:.4f})"


def check_impulse_w3_not_shortest(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 3 cannot be the shortest impulse wave"""
    w1, w3, w5 = waves.get('wave1'), waves.get('wave3'), waves.get('wave5')
    if not w1 or not w3 or not w5:
        return None, "Missing wave data"
    w3_is_shortest = (w3.length < w5.length) and (w3.length < w1.length)
    passes = not w3_is_shortest
    return passes, f"W1={w1.length:.4f}, W3={w3.length:.4f}, W5={w5.length:.4f}"


def check_impulse_w3_exceeds_w1(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 3 must exceed the end of Wave 1 (W3 high > W1 high)"""
    w1, w3 = waves.get('wave1'), waves.get('wave3')
    if not w1 or not w3:
        return None, "Missing wave data"
    passes = w3.high > w1.high
    return passes, f"W3.high ({w3.high:.4f}) > W1.high ({w1.high:.4f})"


def check_impulse_w4_no_overlap(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 4 cannot overlap Wave 1 price territory (W4 low > W1 high)"""
    w1, w4 = waves.get('wave1'), waves.get('wave4')
    if not w1 or not w4:
        return None, "Missing wave data"
    passes = w4.low > w1.high
    return passes, f"W4.low ({w4.low:.4f}) > W1.high ({w1.high:.4f})"


def check_impulse_w5_exceeds_w3(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 5 must exceed Wave 3 (W5 high > W3 high)"""
    w3, w5 = waves.get('wave3'), waves.get('wave5')
    if not w3 or not w5:
        return None, "Missing wave data"
    passes = w5.high > w3.high
    return passes, f"W5.high ({w5.high:.4f}) > W3.high ({w3.high:.4f})"


def check_impulse_w3_longer_than_w2(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 3 must be longer than Wave 2"""
    w2, w3 = waves.get('wave2'), waves.get('wave3')
    if not w2 or not w3:
        return None, "Missing wave data"
    passes = w3.length > w2.length
    return passes, f"W3.length ({w3.length:.4f}) > W2.length ({w2.length:.4f})"


# =============================================================================
# BEARISH IMPULSE RULES (mirror of bullish with highs/lows swapped)
# =============================================================================

def check_bearish_w2_no_full_retrace(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 2 cannot retrace more than 100% of Wave 1 (W2 high < W1 high for bearish)"""
    w1, w2 = waves.get('wave1'), waves.get('wave2')
    if not w1 or not w2:
        return None, "Missing wave data"
    passes = w2.high < w1.high
    return passes, f"W2.high ({w2.high:.4f}) < W1.high ({w1.high:.4f})"


def check_bearish_w3_not_shortest(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 3 cannot be the shortest impulse wave"""
    w1, w3, w5 = waves.get('wave1'), waves.get('wave3'), waves.get('wave5')
    if not w1 or not w3 or not w5:
        return None, "Missing wave data"
    w3_is_shortest = (w3.length < w5.length) and (w3.length < w1.length)
    passes = not w3_is_shortest
    return passes, f"W1={w1.length:.4f}, W3={w3.length:.4f}, W5={w5.length:.4f}"


def check_bearish_w3_exceeds_w1(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 3 must exceed the end of Wave 1 (W3 low < W1 low for bearish)"""
    w1, w3 = waves.get('wave1'), waves.get('wave3')
    if not w1 or not w3:
        return None, "Missing wave data"
    passes = w3.low < w1.low
    return passes, f"W3.low ({w3.low:.4f}) < W1.low ({w1.low:.4f})"


def check_bearish_w4_no_overlap(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 4 cannot overlap Wave 1 price territory (W4 high < W1 low for bearish)"""
    w1, w4 = waves.get('wave1'), waves.get('wave4')
    if not w1 or not w4:
        return None, "Missing wave data"
    passes = w4.high < w1.low
    return passes, f"W4.high ({w4.high:.4f}) < W1.low ({w1.low:.4f})"


def check_bearish_w5_exceeds_w3(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 5 must exceed Wave 3 (W5 low < W3 low for bearish)"""
    w3, w5 = waves.get('wave3'), waves.get('wave5')
    if not w3 or not w5:
        return None, "Missing wave data"
    passes = w5.low < w3.low
    return passes, f"W5.low ({w5.low:.4f}) < W3.low ({w3.low:.4f})"


def check_bearish_w3_longer_than_w2(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave 3 must be longer than Wave 2"""
    w2, w3 = waves.get('wave2'), waves.get('wave3')
    if not w2 or not w3:
        return None, "Missing wave data"
    passes = w3.length > w2.length
    return passes, f"W3.length ({w3.length:.4f}) > W2.length ({w2.length:.4f})"


# =============================================================================
# LEADING DIAGONAL RULES (relaxed W4 overlap)
# =============================================================================

def check_diagonal_w4_overlap(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Leading diagonal: Wave 4 CAN overlap Wave 1 (W4 low < W1 high is allowed)"""
    w1, w4 = waves.get('wave1'), waves.get('wave4')
    if not w1 or not w4:
        return None, "Missing wave data"
    # For leading diagonal, W4 can overlap W1, so this rule is always passed
    passes = w4.low < w1.high  # The opposite of impulse rule
    return passes, f"Diagonal: W4.low ({w4.low:.4f}) < W1.high ({w1.high:.4f})"


# =============================================================================
# CORRECTIVE WAVE RULES
# =============================================================================

def check_corrective_b_below_a(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave B peak is below Wave A peak"""
    wa, wb = waves.get('wave1'), waves.get('wave2')  # A=wave1, B=wave2
    if not wa or not wb:
        return None, "Missing wave data"
    passes = wa.high > wb.high
    return passes, f"WA.high ({wa.high:.4f}) > WB.high ({wb.high:.4f})"


def check_corrective_c_below_a(waves: Dict[str, WaveData]) -> Tuple[bool, str]:
    """Wave C low is below Wave A low"""
    wa, wc = waves.get('wave1'), waves.get('wave3')  # A=wave1, C=wave3
    if not wa or not wc:
        return None, "Missing wave data"
    passes = wa.low > wc.low
    return passes, f"WA.low ({wa.low:.4f}) > WC.low ({wc.low:.4f})"


# =============================================================================
# VALIDATION ENGINE
# =============================================================================

IMPULSE_RULES = [
    ("w2_no_retrace", check_impulse_w2_no_full_retrace),
    ("w3_not_shortest", check_impulse_w3_not_shortest),
    ("w3_exceeds_w1", check_impulse_w3_exceeds_w1),
    ("w4_no_overlap", check_impulse_w4_no_overlap),
    ("w5_exceeds_w3", check_impulse_w5_exceeds_w3),
    ("w3_longer_w2", check_impulse_w3_longer_than_w2),
]

BEARISH_IMPULSE_RULES = [
    ("w2_no_retrace", check_bearish_w2_no_full_retrace),
    ("w3_not_shortest", check_bearish_w3_not_shortest),
    ("w3_exceeds_w1", check_bearish_w3_exceeds_w1),
    ("w4_no_overlap", check_bearish_w4_no_overlap),
    ("w5_exceeds_w3", check_bearish_w5_exceeds_w3),
    ("w3_longer_w2", check_bearish_w3_longer_than_w2),
]

DIAGONAL_RULES = [
    ("w2_no_retrace", check_impulse_w2_no_full_retrace),
    ("w3_not_shortest", check_impulse_w3_not_shortest),
    ("w3_exceeds_w1", check_impulse_w3_exceeds_w1),
    ("w4_overlaps_w1", check_diagonal_w4_overlap),  # Diagonal allows overlap
    ("w5_exceeds_w3", check_impulse_w5_exceeds_w3),
    ("w3_longer_w2", check_impulse_w3_longer_than_w2),
]

CORRECTIVE_RULES = [
    ("b_below_a", check_corrective_b_below_a),
    ("c_below_a", check_corrective_c_below_a),
]


def validate_pattern(pattern: dict) -> Dict[str, Any]:
    """
    Validate a single pattern against Elliott Wave rules.
    Returns validation results dict.
    """
    waves = extract_waves(pattern)
    rule_name = pattern.get('best', {}).get('rule_name', 'unknown').lower()
    
    result = {
        'symbol': pattern.get('symbol'),
        'rule_name': rule_name,
        'has_wave_data': waves is not None,
        'violations': [],
        'passes': [],
        'unknown': [],
        'valid': None,
        'n_waves': len(waves) if waves else 0,
    }
    
    if not waves:
        result['valid'] = 'unknown'
        result['error'] = 'No wave boundary data available'
        return result
    
    # Select appropriate rules based on pattern type
    if rule_name == 'impulse':
        rules = IMPULSE_RULES
    elif rule_name in ['bearish_impulse', 'bearishimpulse']:
        rules = BEARISH_IMPULSE_RULES
    elif rule_name in ['leading_diagonal', 'leadingdiagonal']:
        rules = DIAGONAL_RULES
    elif rule_name == 'corrective':
        rules = CORRECTIVE_RULES
    else:
        result['valid'] = 'unknown'
        result['error'] = f'Unknown pattern type: {rule_name}'
        return result
    
    # Check each rule
    for rule_id, check_fn in rules:
        try:
            passes, detail = check_fn(waves)
            if passes is None:
                result['unknown'].append((rule_id, detail))
            elif passes:
                result['passes'].append((rule_id, detail))
            else:
                result['violations'].append((rule_id, detail))
        except Exception as e:
            result['unknown'].append((rule_id, str(e)))
    
    # Determine overall validity
    if result['violations']:
        result['valid'] = 'invalid'
    elif result['unknown'] and not result['passes']:
        result['valid'] = 'unknown'
    else:
        result['valid'] = 'valid'
    
    return result


def validate_all_patterns(results: dict) -> Dict[str, Any]:
    """Validate all patterns in the results file"""
    patterns = results.get('patterns', [])
    print(f"\n🔍 Validating {len(patterns)} patterns...")
    
    validation_results = []
    summary = {
        'total': len(patterns),
        'valid': 0,
        'invalid': 0,
        'unknown': 0,
        'no_wave_data': 0,
        'by_type': defaultdict(lambda: {'valid': 0, 'invalid': 0, 'unknown': 0}),
        'violations': Counter(),
        'sample_violations': defaultdict(list),
    }
    
    for i, pattern in enumerate(patterns):
        result = validate_pattern(pattern)
        validation_results.append(result)
        
        rule_name = result['rule_name']
        
        if not result['has_wave_data']:
            summary['no_wave_data'] += 1
            summary['unknown'] += 1
            summary['by_type'][rule_name]['unknown'] += 1
        elif result['valid'] == 'valid':
            summary['valid'] += 1
            summary['by_type'][rule_name]['valid'] += 1
        elif result['valid'] == 'invalid':
            summary['invalid'] += 1
            summary['by_type'][rule_name]['invalid'] += 1
            for violation, detail in result['violations']:
                summary['violations'][violation] += 1
                if len(summary['sample_violations'][violation]) < 3:
                    summary['sample_violations'][violation].append({
                        'symbol': result['symbol'],
                        'detail': detail
                    })
        else:
            summary['unknown'] += 1
            summary['by_type'][rule_name]['unknown'] += 1
    
    return {
        'summary': summary,
        'results': validation_results
    }


def print_validation_report(validation: Dict[str, Any]):
    """Print a formatted validation report"""
    summary = validation['summary']
    
    print("\n" + "="*70)
    print("ELLIOTT WAVE PATTERN VALIDATION REPORT (v2)")
    print("="*70)
    
    total = summary['total']
    
    # Overall stats
    print(f"\n📊 OVERALL VALIDATION RESULTS")
    print(f"   Total patterns: {total:,}")
    print(f"   ✅ Valid:        {summary['valid']:,} ({100*summary['valid']/total:.1f}%)")
    print(f"   ❌ Invalid:      {summary['invalid']:,} ({100*summary['invalid']/total:.1f}%)")
    print(f"   ❓ Unknown:      {summary['unknown']:,} ({100*summary['unknown']/total:.1f}%)")
    print(f"   📭 No wave data: {summary['no_wave_data']:,} ({100*summary['no_wave_data']/total:.1f}%)")
    
    # By pattern type
    print(f"\n📈 VALIDATION BY PATTERN TYPE")
    for ptype, counts in sorted(summary['by_type'].items()):
        type_total = counts['valid'] + counts['invalid'] + counts['unknown']
        if type_total > 0:
            valid_pct = 100 * counts['valid'] / type_total
            print(f"   {ptype:20s}: {type_total:5,} patterns | ✅ {valid_pct:5.1f}% valid")
    
    # Violations breakdown
    if summary['violations']:
        print(f"\n⚠️  RULE VIOLATIONS (Top 10)")
        for violation, count in summary['violations'].most_common(10):
            pct = 100 * count / total
            print(f"   {violation:20s}: {count:5,} ({pct:5.1f}%)")
            # Show sample violations
            for sample in summary['sample_violations'].get(violation, [])[:2]:
                print(f"      └─ {sample['symbol']}: {sample['detail']}")
    
    # Data quality assessment
    print(f"\n🎯 DATA QUALITY ASSESSMENT")
    if summary['no_wave_data'] == total:
        print("   ⚠️  NO WAVE BOUNDARY DATA IN OUTPUT FILE")
        print("   This means the pipeline was run with an older version that")
        print("   didn't save wave boundaries. Re-run the pipeline to get")
        print("   proper validation data.")
    elif summary['no_wave_data'] > 0:
        print(f"   ⚠️  {summary['no_wave_data']} patterns missing wave boundary data")
    
    if total > 0 and summary['no_wave_data'] < total:
        valid_rate = summary['valid'] / (total - summary['no_wave_data'])
        if valid_rate >= 0.95:
            print("   🏆 EXCELLENT: 95%+ patterns are valid")
        elif valid_rate >= 0.85:
            print("   ✅ GOOD: 85%+ patterns are valid")
        elif valid_rate >= 0.70:
            print("   ⚠️  FAIR: 70%+ patterns are valid - some rule relaxation may be needed")
        else:
            print("   ❌ POOR: <70% patterns valid - investigate rule implementation")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate Elliott Wave patterns')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to results JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output validation report')
    args = parser.parse_args()
    
    # Default paths
    if args.input:
        results_path = Path(args.input)
    else:
        results_path = Path(__file__).parent.parent / "output" / "results.json"
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "output" / "validation_report_v2.json"
    
    # Load results
    results = load_results(str(results_path))
    
    # Validate all patterns
    validation = validate_all_patterns(results)
    
    # Print report
    print_validation_report(validation)
    
    # Save results
    print(f"\n💾 Saving validation report to: {output_path}")
    
    # Convert to JSON-serializable format
    report_data = {
        'summary': {
            'total': validation['summary']['total'],
            'valid': validation['summary']['valid'],
            'invalid': validation['summary']['invalid'],
            'unknown': validation['summary']['unknown'],
            'no_wave_data': validation['summary']['no_wave_data'],
            'by_type': dict(validation['summary']['by_type']),
            'violations': dict(validation['summary']['violations']),
        },
        'sample_violations': dict(validation['summary']['sample_violations']),
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print("✅ Done!")


if __name__ == "__main__":
    main()
