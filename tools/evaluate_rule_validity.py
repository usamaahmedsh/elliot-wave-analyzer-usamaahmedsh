#!/usr/bin/env python3
"""
Elliott Wave Pattern Dataset - Rule Validation

This script validates detected patterns against Elliott Wave rules to assess
data quality and identify patterns that may not be valid.

Rules are based on:
- Impulse waves: R.N. Elliott's original rules
- Corrective waves: Standard ABC correction rules
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(results_path: str) -> dict:
    """Load the results JSON file"""
    print(f"📂 Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded {data['metadata']['total_patterns']} patterns")
    return data


def load_ohlcv_data(hf_dataset: str) -> Dict[str, Any]:
    """Load OHLCV data from HuggingFace dataset"""
    print(f"📦 Loading OHLCV data from: {hf_dataset}")
    try:
        from datasets import load_dataset
        import pandas as pd
        import os
        
        # Get HF token from environment
        hf_token = os.environ.get('HF_TOKEN')
        
        dataset = load_dataset(hf_dataset, split='train', token=hf_token)
        df = dataset.to_pandas()
        
        # Group by ticker
        ticker_data = {}
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].sort_values('Date').reset_index(drop=True)
            ticker_data[ticker] = ticker_df
        
        print(f"✅ Loaded OHLCV data for {len(ticker_data)} symbols")
        return ticker_data
    except Exception as e:
        print(f"⚠️  Could not load HuggingFace dataset: {e}")
        print("   Will validate patterns without price data (limited validation)")
        return {}


class WaveData:
    """Simple class to hold wave data for rule validation"""
    def __init__(self, idx_start: int, idx_end: int, low: float, high: float, 
                 low_idx: int, high_idx: int):
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.low = low
        self.high = high
        self.low_idx = low_idx
        self.high_idx = high_idx
        self.length = abs(high - low)
        self.duration = idx_end - idx_start


def extract_waves_from_pattern(pattern: dict, ohlcv_df) -> Optional[Dict[str, WaveData]]:
    """
    Extract wave data from a pattern using OHLCV data.
    
    The wave_config contains the degree/skip values for each wave.
    We need to reconstruct the actual wave positions from the OHLCV data.
    """
    try:
        wave_config = pattern['best']['wave_config']
        idx_start = pattern['best']['idx_start']
        idx_end = pattern['best']['idx_end']
        start_row = pattern['start_row']
        
        if wave_config is None or None in wave_config:
            return None
        
        if ohlcv_df is None or len(ohlcv_df) == 0:
            return None
        
        # Get the window of OHLCV data
        window_start = start_row
        window_end = start_row + pattern['window_len']
        
        if window_end > len(ohlcv_df):
            return None
        
        window_df = ohlcv_df.iloc[window_start:window_end].reset_index(drop=True)
        
        # Get the pattern slice
        pattern_df = window_df.iloc[idx_start:idx_end+1].reset_index(drop=True)
        
        if len(pattern_df) < 10:  # Need minimum data
            return None
        
        lows = pattern_df['Low'].values
        highs = pattern_df['High'].values
        
        rule_name = pattern['best']['rule_name'].lower()
        
        # Determine if this is a bullish (up) or bearish (down) pattern
        # For impulse: starts low, ends high (bullish) or starts high, ends low (bearish)
        is_bullish = lows[0] < highs[-1]
        
        waves = {}
        
        if rule_name == 'impulse':
            # 5 wave pattern: 1-2-3-4-5
            # For bullish: W1 up, W2 down, W3 up, W4 down, W5 up
            # We need to find the turning points
            
            # Simple approach: divide pattern into 5 segments based on wave_config
            # The wave_config values represent degrees/complexity
            total_config = sum(wave_config)
            if total_config == 0:
                return None
            
            # Calculate approximate positions for each wave
            n_bars = len(pattern_df)
            
            # Find key turning points using local extrema
            wave_boundaries = find_wave_boundaries_impulse(lows, highs, is_bullish)
            
            if wave_boundaries is None or len(wave_boundaries) < 6:
                return None
            
            # Create wave objects
            for i in range(5):
                w_start = wave_boundaries[i]
                w_end = wave_boundaries[i + 1]
                
                segment_lows = lows[w_start:w_end+1]
                segment_highs = highs[w_start:w_end+1]
                
                if len(segment_lows) == 0:
                    return None
                
                waves[f'wave{i+1}'] = WaveData(
                    idx_start=w_start,
                    idx_end=w_end,
                    low=float(np.min(segment_lows)),
                    high=float(np.max(segment_highs)),
                    low_idx=w_start + int(np.argmin(segment_lows)),
                    high_idx=w_start + int(np.argmax(segment_highs))
                )
        
        elif rule_name == 'corrective':
            # 3 wave pattern: A-B-C
            wave_boundaries = find_wave_boundaries_corrective(lows, highs, is_bullish)
            
            if wave_boundaries is None or len(wave_boundaries) < 4:
                return None
            
            wave_names = ['wave1', 'wave2', 'wave3']  # A, B, C mapped to wave1, wave2, wave3
            
            for i in range(3):
                w_start = wave_boundaries[i]
                w_end = wave_boundaries[i + 1]
                
                segment_lows = lows[w_start:w_end+1]
                segment_highs = highs[w_start:w_end+1]
                
                if len(segment_lows) == 0:
                    return None
                
                waves[wave_names[i]] = WaveData(
                    idx_start=w_start,
                    idx_end=w_end,
                    low=float(np.min(segment_lows)),
                    high=float(np.max(segment_highs)),
                    low_idx=w_start + int(np.argmin(segment_lows)),
                    high_idx=w_start + int(np.argmax(segment_highs))
                )
        
        return waves
    
    except Exception as e:
        return None


def find_wave_boundaries_impulse(lows: np.ndarray, highs: np.ndarray, is_bullish: bool) -> Optional[List[int]]:
    """
    Find approximate wave boundaries for an impulse pattern.
    Returns list of 6 indices: [W0_start, W1_end, W2_end, W3_end, W4_end, W5_end]
    """
    n = len(lows)
    if n < 20:
        return None
    
    # Simple segmentation: divide into 5 roughly equal parts
    # This is a simplification - real wave detection is more complex
    segment_size = n // 5
    
    boundaries = [0]
    for i in range(1, 6):
        boundaries.append(min(i * segment_size, n - 1))
    
    return boundaries


def find_wave_boundaries_corrective(lows: np.ndarray, highs: np.ndarray, is_bullish: bool) -> Optional[List[int]]:
    """
    Find approximate wave boundaries for a corrective pattern.
    Returns list of 4 indices: [A_start, A_end/B_start, B_end/C_start, C_end]
    """
    n = len(lows)
    if n < 10:
        return None
    
    # Simple segmentation: divide into 3 roughly equal parts
    segment_size = n // 3
    
    boundaries = [0]
    for i in range(1, 4):
        boundaries.append(min(i * segment_size, n - 1))
    
    return boundaries


# =============================================================================
# IMPULSE WAVE RULES
# =============================================================================

def check_impulse_rule_w2_no_full_retrace(waves: Dict[str, WaveData]) -> Tuple[bool, str, float]:
    """
    Rule: Wave 2 cannot retrace more than 100% of Wave 1
    Wave 2's low must be above Wave 1's starting low (Wave 0)
    """
    w1 = waves.get('wave1')
    w2 = waves.get('wave2')
    
    if not w1 or not w2:
        return None, "Missing wave data", 0.0
    
    # For bullish: W2 low > W1 low (start of wave 1)
    passes = w2.low > w1.low
    
    if passes:
        margin = (w2.low - w1.low) / w1.length * 100 if w1.length > 0 else 0
        return True, f"W2 low ({w2.low:.2f}) > W1 low ({w1.low:.2f})", margin
    else:
        margin = (w1.low - w2.low) / w1.length * 100 if w1.length > 0 else 0
        return False, f"W2 ({w2.low:.2f}) retraced below W1 start ({w1.low:.2f})", -margin


def check_impulse_rule_w3_not_shortest(waves: Dict[str, WaveData]) -> Tuple[bool, str, float]:
    """
    Rule: Wave 3 cannot be the shortest among waves 1, 3, and 5
    """
    w1 = waves.get('wave1')
    w3 = waves.get('wave3')
    w5 = waves.get('wave5')
    
    if not w1 or not w3 or not w5:
        return None, "Missing wave data", 0.0
    
    lengths = {'W1': w1.length, 'W3': w3.length, 'W5': w5.length}
    min_wave = min(lengths, key=lengths.get)
    
    passes = min_wave != 'W3'
    
    if passes:
        # Margin: how much longer is W3 than the shortest?
        min_len = min(w1.length, w5.length)
        margin = (w3.length - min_len) / min_len * 100 if min_len > 0 else 0
        return True, f"W3 ({w3.length:.2f}) is not shortest. Shortest: {min_wave} ({lengths[min_wave]:.2f})", margin
    else:
        second_shortest = min(w1.length, w5.length)
        margin = (second_shortest - w3.length) / second_shortest * 100 if second_shortest > 0 else 0
        return False, f"W3 ({w3.length:.2f}) is shortest! W1={w1.length:.2f}, W5={w5.length:.2f}", -margin


def check_impulse_rule_w4_no_overlap(waves: Dict[str, WaveData]) -> Tuple[bool, str, float]:
    """
    Rule: Wave 4 cannot enter the price territory of Wave 1
    Wave 4's low must be above Wave 1's high
    """
    w1 = waves.get('wave1')
    w4 = waves.get('wave4')
    
    if not w1 or not w4:
        return None, "Missing wave data", 0.0
    
    passes = w4.low > w1.high
    
    if passes:
        margin = (w4.low - w1.high) / w1.length * 100 if w1.length > 0 else 0
        return True, f"W4 low ({w4.low:.2f}) > W1 high ({w1.high:.2f})", margin
    else:
        margin = (w1.high - w4.low) / w1.length * 100 if w1.length > 0 else 0
        return False, f"W4 ({w4.low:.2f}) overlaps W1 territory (W1 high: {w1.high:.2f})", -margin


def check_impulse_rule_w3_exceeds_w1(waves: Dict[str, WaveData]) -> Tuple[bool, str, float]:
    """
    Rule: Wave 3 must exceed Wave 1's high
    """
    w1 = waves.get('wave1')
    w3 = waves.get('wave3')
    
    if not w1 or not w3:
        return None, "Missing wave data", 0.0
    
    passes = w3.high > w1.high
    
    if passes:
        margin = (w3.high - w1.high) / w1.length * 100 if w1.length > 0 else 0
        return True, f"W3 high ({w3.high:.2f}) > W1 high ({w1.high:.2f})", margin
    else:
        margin = (w1.high - w3.high) / w1.length * 100 if w1.length > 0 else 0
        return False, f"W3 ({w3.high:.2f}) doesn't exceed W1 high ({w1.high:.2f})", -margin


def check_impulse_rule_w5_exceeds_w3(waves: Dict[str, WaveData]) -> Tuple[bool, str, float]:
    """
    Rule: Wave 5 must exceed Wave 3's high (for non-truncated patterns)
    """
    w3 = waves.get('wave3')
    w5 = waves.get('wave5')
    
    if not w3 or not w5:
        return None, "Missing wave data", 0.0
    
    passes = w5.high > w3.high
    
    if passes:
        margin = (w5.high - w3.high) / w3.length * 100 if w3.length > 0 else 0
        return True, f"W5 high ({w5.high:.2f}) > W3 high ({w3.high:.2f})", margin
    else:
        margin = (w3.high - w5.high) / w3.length * 100 if w3.length > 0 else 0
        return False, f"W5 ({w5.high:.2f}) doesn't exceed W3 high ({w3.high:.2f}) - truncated?", -margin


# =============================================================================
# CORRECTIVE WAVE RULES
# =============================================================================

def check_corrective_rule_b_below_origin(waves: Dict[str, WaveData]) -> Tuple[bool, str, float]:
    """
    Rule: Wave B cannot exceed the origin of Wave A (for zigzag corrections)
    """
    wA = waves.get('wave1')  # Wave A
    wB = waves.get('wave2')  # Wave B
    
    if not wA or not wB:
        return None, "Missing wave data", 0.0
    
    # For downward correction: B high < A high (origin)
    passes = wB.high < wA.high
    
    if passes:
        margin = (wA.high - wB.high) / wA.length * 100 if wA.length > 0 else 0
        return True, f"B high ({wB.high:.2f}) < A origin ({wA.high:.2f})", margin
    else:
        margin = (wB.high - wA.high) / wA.length * 100 if wA.length > 0 else 0
        return False, f"B ({wB.high:.2f}) exceeds A origin ({wA.high:.2f})", -margin


def check_corrective_rule_c_exceeds_a(waves: Dict[str, WaveData]) -> Tuple[bool, str, float]:
    """
    Rule: Wave C must move beyond Wave A's endpoint
    """
    wA = waves.get('wave1')  # Wave A
    wC = waves.get('wave3')  # Wave C
    
    if not wA or not wC:
        return None, "Missing wave data", 0.0
    
    # For downward correction: C low < A low
    passes = wC.low < wA.low
    
    if passes:
        margin = (wA.low - wC.low) / wA.length * 100 if wA.length > 0 else 0
        return True, f"C low ({wC.low:.2f}) < A low ({wA.low:.2f})", margin
    else:
        margin = (wC.low - wA.low) / wA.length * 100 if wA.length > 0 else 0
        return False, f"C ({wC.low:.2f}) doesn't exceed A endpoint ({wA.low:.2f})", -margin


# =============================================================================
# FIBONACCI GUIDELINES (Soft Rules)
# =============================================================================

def check_fib_w2_retracement(waves: Dict[str, WaveData]) -> Tuple[str, float, str]:
    """
    Guideline: Wave 2 typically retraces 38.2% to 78.6% of Wave 1
    """
    w1 = waves.get('wave1')
    w2 = waves.get('wave2')
    
    if not w1 or not w2:
        return "unknown", 0.0, "Missing wave data"
    
    if w1.length == 0:
        return "unknown", 0.0, "W1 has zero length"
    
    retracement = w2.length / w1.length
    
    ideal_low, ideal_high = 0.382, 0.786
    
    if ideal_low <= retracement <= ideal_high:
        return "ideal", retracement, f"W2 retraces {retracement:.1%} of W1 (ideal: 38.2%-78.6%)"
    elif 0.236 <= retracement <= 0.886:
        return "acceptable", retracement, f"W2 retraces {retracement:.1%} of W1 (acceptable)"
    else:
        return "poor", retracement, f"W2 retraces {retracement:.1%} of W1 (outside normal range)"


def check_fib_w3_extension(waves: Dict[str, WaveData]) -> Tuple[str, float, str]:
    """
    Guideline: Wave 3 is often 1.618x to 2.618x the length of Wave 1
    """
    w1 = waves.get('wave1')
    w3 = waves.get('wave3')
    
    if not w1 or not w3:
        return "unknown", 0.0, "Missing wave data"
    
    if w1.length == 0:
        return "unknown", 0.0, "W1 has zero length"
    
    extension = w3.length / w1.length
    
    ideal_low, ideal_high = 1.618, 2.618
    
    if ideal_low <= extension <= ideal_high:
        return "ideal", extension, f"W3 is {extension:.2f}x W1 (ideal: 1.618-2.618x)"
    elif 1.0 <= extension <= 4.236:
        return "acceptable", extension, f"W3 is {extension:.2f}x W1 (acceptable)"
    else:
        return "poor", extension, f"W3 is {extension:.2f}x W1 (outside normal range)"


def check_fib_w4_retracement(waves: Dict[str, WaveData]) -> Tuple[str, float, str]:
    """
    Guideline: Wave 4 typically retraces 23.6% to 50% of Wave 3
    """
    w3 = waves.get('wave3')
    w4 = waves.get('wave4')
    
    if not w3 or not w4:
        return "unknown", 0.0, "Missing wave data"
    
    if w3.length == 0:
        return "unknown", 0.0, "W3 has zero length"
    
    retracement = w4.length / w3.length
    
    ideal_low, ideal_high = 0.236, 0.50
    
    if ideal_low <= retracement <= ideal_high:
        return "ideal", retracement, f"W4 retraces {retracement:.1%} of W3 (ideal: 23.6%-50%)"
    elif 0.146 <= retracement <= 0.618:
        return "acceptable", retracement, f"W4 retraces {retracement:.1%} of W3 (acceptable)"
    else:
        return "poor", retracement, f"W4 retraces {retracement:.1%} of W3 (outside normal range)"


# =============================================================================
# VALIDATION ENGINE
# =============================================================================

def validate_pattern(pattern: dict, ohlcv_df) -> Dict[str, Any]:
    """
    Validate a single pattern against all applicable rules.
    """
    rule_name = pattern['best']['rule_name'].lower()
    
    result = {
        'symbol': pattern['symbol'],
        'rule_name': rule_name,
        'ensemble_score': pattern['ensemble_score'],
        'fib_score': pattern['fib_score'],
        'rule_checks': {},
        'fib_checks': {},
        'is_valid': True,
        'violations': [],
        'warnings': [],
    }
    
    # Extract wave data
    waves = extract_waves_from_pattern(pattern, ohlcv_df)
    
    if waves is None:
        result['is_valid'] = None  # Unknown - can't validate
        result['warnings'].append("Could not extract wave data for validation")
        return result
    
    # Apply rules based on pattern type
    if rule_name == 'impulse':
        # Hard rules
        rules = [
            ('w2_no_full_retrace', check_impulse_rule_w2_no_full_retrace),
            ('w3_not_shortest', check_impulse_rule_w3_not_shortest),
            ('w4_no_overlap', check_impulse_rule_w4_no_overlap),
            ('w3_exceeds_w1', check_impulse_rule_w3_exceeds_w1),
            ('w5_exceeds_w3', check_impulse_rule_w5_exceeds_w3),
        ]
        
        # Fibonacci guidelines
        fib_checks = [
            ('w2_retracement', check_fib_w2_retracement),
            ('w3_extension', check_fib_w3_extension),
            ('w4_retracement', check_fib_w4_retracement),
        ]
        
    elif rule_name == 'corrective':
        rules = [
            ('b_below_origin', check_corrective_rule_b_below_origin),
            ('c_exceeds_a', check_corrective_rule_c_exceeds_a),
        ]
        fib_checks = []
    else:
        result['warnings'].append(f"Unknown pattern type: {rule_name}")
        return result
    
    # Check hard rules
    for rule_id, check_fn in rules:
        passes, message, margin = check_fn(waves)
        result['rule_checks'][rule_id] = {
            'passes': passes,
            'message': message,
            'margin': margin
        }
        if passes is False:
            result['is_valid'] = False
            result['violations'].append(f"{rule_id}: {message}")
        elif passes is None:
            result['warnings'].append(f"{rule_id}: {message}")
    
    # Check Fibonacci guidelines
    for fib_id, check_fn in fib_checks:
        quality, value, message = check_fn(waves)
        result['fib_checks'][fib_id] = {
            'quality': quality,
            'value': value,
            'message': message
        }
        if quality == 'poor':
            result['warnings'].append(f"{fib_id}: {message}")
    
    return result


def validate_all_patterns(patterns: list, ticker_data: dict) -> List[Dict]:
    """Validate all patterns"""
    results = []
    
    print(f"\n🔍 Validating {len(patterns)} patterns...")
    
    # Group patterns by symbol for efficient OHLCV lookup
    patterns_by_symbol = defaultdict(list)
    for p in patterns:
        patterns_by_symbol[p['symbol']].append(p)
    
    validated = 0
    for symbol, symbol_patterns in patterns_by_symbol.items():
        ohlcv_df = ticker_data.get(symbol)
        
        for pattern in symbol_patterns:
            result = validate_pattern(pattern, ohlcv_df)
            results.append(result)
            validated += 1
            
            if validated % 1000 == 0:
                print(f"   Validated {validated}/{len(patterns)} patterns...")
    
    print(f"✅ Validated {len(results)} patterns")
    return results


def generate_validation_report(validation_results: List[Dict]) -> Dict:
    """Generate summary statistics from validation results"""
    
    total = len(validation_results)
    
    # Count by validity
    valid_count = sum(1 for r in validation_results if r['is_valid'] is True)
    invalid_count = sum(1 for r in validation_results if r['is_valid'] is False)
    unknown_count = sum(1 for r in validation_results if r['is_valid'] is None)
    
    # Count violations by rule
    violation_counts = Counter()
    for r in validation_results:
        for v in r['violations']:
            rule_id = v.split(':')[0]
            violation_counts[rule_id] += 1
    
    # Separate by pattern type
    impulse_results = [r for r in validation_results if r['rule_name'] == 'impulse']
    corrective_results = [r for r in validation_results if r['rule_name'] == 'corrective']
    
    def pattern_type_stats(results, name):
        if not results:
            return {'name': name, 'count': 0}
        valid = sum(1 for r in results if r['is_valid'] is True)
        invalid = sum(1 for r in results if r['is_valid'] is False)
        return {
            'name': name,
            'count': len(results),
            'valid': valid,
            'invalid': invalid,
            'valid_pct': valid / len(results) * 100 if results else 0,
        }
    
    # Fibonacci quality distribution
    fib_quality = Counter()
    for r in validation_results:
        for fib_id, fib_check in r.get('fib_checks', {}).items():
            fib_quality[fib_check['quality']] += 1
    
    # Score comparison: valid vs invalid
    valid_scores = [r['ensemble_score'] for r in validation_results if r['is_valid'] is True]
    invalid_scores = [r['ensemble_score'] for r in validation_results if r['is_valid'] is False]
    
    report = {
        'total_patterns': total,
        'validity': {
            'valid': valid_count,
            'invalid': invalid_count,
            'unknown': unknown_count,
            'valid_pct': valid_count / total * 100 if total > 0 else 0,
        },
        'by_pattern_type': {
            'impulse': pattern_type_stats(impulse_results, 'impulse'),
            'corrective': pattern_type_stats(corrective_results, 'corrective'),
        },
        'violation_counts': dict(violation_counts.most_common()),
        'fibonacci_quality': dict(fib_quality),
        'score_comparison': {
            'valid_mean': float(np.mean(valid_scores)) if valid_scores else 0,
            'invalid_mean': float(np.mean(invalid_scores)) if invalid_scores else 0,
        }
    }
    
    return report


def print_validation_report(report: Dict, validation_results: List[Dict]):
    """Print formatted validation report"""
    
    print("\n" + "="*70)
    print("       ELLIOTT WAVE PATTERN VALIDATION REPORT")
    print("="*70)
    
    # Overall validity
    print("\n📊 OVERALL VALIDITY")
    print("-"*50)
    v = report['validity']
    print(f"  Total Patterns:     {report['total_patterns']:,}")
    print(f"  ✅ Valid:           {v['valid']:,} ({v['valid_pct']:.1f}%)")
    print(f"  ❌ Invalid:         {v['invalid']:,} ({100 - v['valid_pct'] - v['unknown']/report['total_patterns']*100:.1f}%)")
    print(f"  ❓ Unknown:         {v['unknown']:,}")
    
    # By pattern type
    print("\n📈 VALIDITY BY PATTERN TYPE")
    print("-"*50)
    for ptype in ['impulse', 'corrective']:
        p = report['by_pattern_type'][ptype]
        if p['count'] > 0:
            print(f"  {ptype.capitalize():12} n={p['count']:,}  valid={p['valid']:,} ({p['valid_pct']:.1f}%)  invalid={p['invalid']:,}")
    
    # Rule violations
    print("\n⚠️  RULE VIOLATIONS (most common)")
    print("-"*50)
    if report['violation_counts']:
        for rule, count in list(report['violation_counts'].items())[:10]:
            pct = count / report['total_patterns'] * 100
            print(f"  {rule:25} {count:6,} ({pct:.1f}%)")
    else:
        print("  No violations recorded")
    
    # Fibonacci quality
    print("\n📐 FIBONACCI ALIGNMENT QUALITY")
    print("-"*50)
    fib = report['fibonacci_quality']
    total_fib = sum(fib.values())
    if total_fib > 0:
        for quality in ['ideal', 'acceptable', 'poor', 'unknown']:
            count = fib.get(quality, 0)
            pct = count / total_fib * 100
            print(f"  {quality.capitalize():12} {count:6,} ({pct:.1f}%)")
    
    # Score comparison
    print("\n⭐ ENSEMBLE SCORE: VALID vs INVALID")
    print("-"*50)
    sc = report['score_comparison']
    print(f"  Valid patterns mean score:    {sc['valid_mean']:.3f}")
    print(f"  Invalid patterns mean score:  {sc['invalid_mean']:.3f}")
    if sc['valid_mean'] > 0 and sc['invalid_mean'] > 0:
        diff = sc['valid_mean'] - sc['invalid_mean']
        print(f"  Difference:                   {diff:+.3f}")
    
    # Sample violations
    print("\n📝 SAMPLE VIOLATIONS (first 10)")
    print("-"*50)
    violations_shown = 0
    for r in validation_results:
        if r['violations'] and violations_shown < 10:
            print(f"  {r['symbol']} ({r['rule_name']}):")
            for v in r['violations'][:2]:
                print(f"    - {v}")
            violations_shown += 1
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-"*50)
    
    valid_pct = report['validity']['valid_pct']
    if valid_pct >= 80:
        print("  ✅ Data quality is GOOD (≥80% valid)")
        print(f"     {report['validity']['valid']:,} patterns ready for training")
    elif valid_pct >= 60:
        print("  ⚠️  Data quality is MODERATE (60-80% valid)")
        print("     Consider filtering out invalid patterns before training")
    else:
        print("  ❌ Data quality needs REVIEW (<60% valid)")
        print("     Investigate rule violations before using for training")
    
    if report['violation_counts']:
        top_violation = list(report['violation_counts'].keys())[0]
        print(f"\n  Most common issue: {top_violation}")
        print("     Review pattern detection algorithm for this rule")
    
    print("\n" + "="*70)
    print("                      END OF REPORT")
    print("="*70)


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Elliott Wave Pattern Validation')
    parser.add_argument('--input', type=str, default='output/all_symbols_results.json',
                        help='Path to results JSON file')
    parser.add_argument('--output', type=str, default='output/validation_report.json',
                        help='Path to save validation report')
    parser.add_argument('--hf-dataset', type=str, 
                        default='usamaahmedsh/financial-markets-dataset-15y-train',
                        help='HuggingFace dataset for OHLCV data')
    parser.add_argument('--export-valid', type=str, default=None,
                        help='Export valid patterns to this file')
    parser.add_argument('--export-invalid', type=str, default=None,
                        help='Export invalid patterns to this file')
    
    args = parser.parse_args()
    
    # Load .env file for HF token
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() and value.strip():
                        os.environ[key.strip()] = value.strip()
    
    # Load results
    data = load_results(args.input)
    patterns = data['patterns']
    
    # Load OHLCV data
    ticker_data = load_ohlcv_data(args.hf_dataset)
    
    # Validate patterns
    validation_results = validate_all_patterns(patterns, ticker_data)
    
    # Generate report
    report = generate_validation_report(validation_results)
    
    # Print report
    print_validation_report(report, validation_results)
    
    # Save report
    print(f"\n💾 Saving validation report to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump({
            'summary': report,
            'detailed_results': validation_results[:100]  # Save first 100 detailed results
        }, f, indent=2, default=str)
    
    # Export valid/invalid patterns if requested
    if args.export_valid:
        valid_patterns = [
            patterns[i] for i, r in enumerate(validation_results) 
            if r['is_valid'] is True
        ]
        print(f"💾 Exporting {len(valid_patterns)} valid patterns to: {args.export_valid}")
        with open(args.export_valid, 'w') as f:
            json.dump({'patterns': valid_patterns}, f, indent=2, default=str)
    
    if args.export_invalid:
        invalid_patterns = [
            patterns[i] for i, r in enumerate(validation_results) 
            if r['is_valid'] is False
        ]
        print(f"💾 Exporting {len(invalid_patterns)} invalid patterns to: {args.export_invalid}")
        with open(args.export_invalid, 'w') as f:
            json.dump({'patterns': invalid_patterns}, f, indent=2, default=str)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
