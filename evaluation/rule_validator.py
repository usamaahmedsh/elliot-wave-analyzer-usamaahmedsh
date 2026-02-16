"""Rule-based validator for Elliott Wave patterns.

Validates detected patterns against Elliott Wave rules:
1. Wave 3 is never the shortest (among waves 1, 3, 5)
2. Wave 2 never retraces beyond the start of Wave 1
3. Wave 4 never overlaps the price territory of Wave 1
4. (Optional) Fibonacci ratio alignment
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class RuleValidator:
    """Validates Elliott Wave patterns against core rules."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize validator.
        
        Args:
            strict_mode: If True, pattern must pass ALL rules. If False, partial compliance allowed.
        """
        self.strict_mode = strict_mode
    
    def validate_pattern(
        self,
        pattern: Dict[str, Any],
        lows: np.ndarray,
        highs: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate a single pattern against Elliott Wave rules.
        
        Args:
            pattern: Pattern dict with 'waves' list
            lows: Price lows array
            highs: Price highs array
            
        Returns:
            (is_valid, details) where details contains rule-by-rule results
        """
        waves = pattern.get('waves', [])
        
        if len(waves) < 5:
            return False, {'error': 'Pattern must have 5 waves'}
        
        # Determine if bullish or bearish
        direction = pattern.get('direction', 'bullish')
        is_bullish = direction in ['bullish', 'up', 1]
        
        # Extract wave endpoints
        wave_indices = []
        for wave in waves[:5]:
            start_idx = wave.get('start_idx', 0)
            end_idx = wave.get('end_idx', 0)
            wave_indices.append((start_idx, end_idx))
        
        # Get prices at wave endpoints
        wave_prices = []
        for start_idx, end_idx in wave_indices:
            if is_bullish:
                start_price = lows[start_idx] if start_idx < len(lows) else 0
                end_price = highs[end_idx] if end_idx < len(highs) else 0
            else:
                start_price = highs[start_idx] if start_idx < len(highs) else 0
                end_price = lows[end_idx] if end_idx < len(lows) else 0
            wave_prices.append((start_price, end_price))
        
        # Calculate wave lengths (magnitudes)
        wave_lengths = []
        for start_price, end_price in wave_prices:
            length = abs(end_price - start_price)
            wave_lengths.append(length)
        
        results = {}
        
        # Rule 1: Wave 3 is never the shortest
        wave1_len = wave_lengths[0]
        wave3_len = wave_lengths[2]
        wave5_len = wave_lengths[4]
        
        rule1_pass = wave3_len >= wave1_len or wave3_len >= wave5_len
        results['rule1_wave3_not_shortest'] = {
            'pass': rule1_pass,
            'wave1_len': float(wave1_len),
            'wave3_len': float(wave3_len),
            'wave5_len': float(wave5_len),
        }
        
        # Rule 2: Wave 2 doesn't retrace beyond Wave 1 start
        wave1_start_price, wave1_end_price = wave_prices[0]
        wave2_start_price, wave2_end_price = wave_prices[1]
        
        if is_bullish:
            # In bullish pattern, wave 2 should not go below wave 1 start
            rule2_pass = wave2_end_price > wave1_start_price
        else:
            # In bearish pattern, wave 2 should not go above wave 1 start
            rule2_pass = wave2_end_price < wave1_start_price
        
        results['rule2_wave2_no_full_retrace'] = {
            'pass': rule2_pass,
            'wave1_start': float(wave1_start_price),
            'wave2_end': float(wave2_end_price),
        }
        
        # Rule 3: Wave 4 doesn't overlap Wave 1
        wave4_start_price, wave4_end_price = wave_prices[3]
        
        if is_bullish:
            # In bullish pattern, wave 4 low should not go below wave 1 high
            rule3_pass = wave4_end_price > wave1_end_price
        else:
            # In bearish pattern, wave 4 high should not go above wave 1 low
            rule3_pass = wave4_end_price < wave1_end_price
        
        results['rule3_wave4_no_overlap'] = {
            'pass': rule3_pass,
            'wave1_end': float(wave1_end_price),
            'wave4_end': float(wave4_end_price),
        }
        
        # Overall validation
        all_rules_pass = rule1_pass and rule2_pass and rule3_pass
        
        results['overall'] = {
            'valid': all_rules_pass,
            'rules_passed': sum([rule1_pass, rule2_pass, rule3_pass]),
            'rules_total': 3,
            'compliance_rate': sum([rule1_pass, rule2_pass, rule3_pass]) / 3.0,
        }
        
        is_valid = all_rules_pass if self.strict_mode else (results['overall']['rules_passed'] >= 2)
        
        return is_valid, results
    
    def validate_batch(
        self,
        patterns: List[Dict[str, Any]],
        lows: np.ndarray,
        highs: np.ndarray
    ) -> Dict[str, Any]:
        """Validate multiple patterns and return aggregate statistics.
        
        Args:
            patterns: List of pattern dicts
            lows: Price lows array
            highs: Price highs array
            
        Returns:
            Dict with validation statistics
        """
        if not patterns:
            return {
                'total_patterns': 0,
                'valid_patterns': 0,
                'invalid_patterns': 0,
                'validation_rate': 0.0,
                'rule_compliance': {},
            }
        
        valid_count = 0
        rule1_pass = 0
        rule2_pass = 0
        rule3_pass = 0
        validation_details = []
        
        for pattern in patterns:
            is_valid, details = self.validate_pattern(pattern, lows, highs)
            
            if is_valid:
                valid_count += 1
            
            if details.get('rule1_wave3_not_shortest', {}).get('pass', False):
                rule1_pass += 1
            if details.get('rule2_wave2_no_full_retrace', {}).get('pass', False):
                rule2_pass += 1
            if details.get('rule3_wave4_no_overlap', {}).get('pass', False):
                rule3_pass += 1
            
            validation_details.append({
                'pattern_id': pattern.get('id', 'unknown'),
                'valid': is_valid,
                'details': details
            })
        
        n = len(patterns)
        
        return {
            'total_patterns': n,
            'valid_patterns': valid_count,
            'invalid_patterns': n - valid_count,
            'validation_rate': valid_count / n if n > 0 else 0.0,
            'rule_compliance': {
                'rule1_wave3_not_shortest': rule1_pass / n if n > 0 else 0.0,
                'rule2_wave2_no_full_retrace': rule2_pass / n if n > 0 else 0.0,
                'rule3_wave4_no_overlap': rule3_pass / n if n > 0 else 0.0,
                'overall': (rule1_pass + rule2_pass + rule3_pass) / (3 * n) if n > 0 else 0.0,
            },
            'details': validation_details if len(validation_details) <= 100 else validation_details[:100],  # Limit details
        }
    
    def print_validation_report(self, validation_results: Dict[str, Any]):
        """Print human-readable validation report.
        
        Args:
            validation_results: Results from validate_batch()
        """
        print("=" * 70)
        print("ELLIOTT WAVE RULE VALIDATION REPORT")
        print("=" * 70)
        print(f"\nTotal Patterns: {validation_results['total_patterns']}")
        print(f"Valid Patterns: {validation_results['valid_patterns']}")
        print(f"Invalid Patterns: {validation_results['invalid_patterns']}")
        print(f"Validation Rate: {validation_results['validation_rate']:.2%}")
        
        print(f"\nRule Compliance:")
        compliance = validation_results['rule_compliance']
        print(f"  Rule 1 (Wave 3 not shortest): {compliance['rule1_wave3_not_shortest']:.2%}")
        print(f"  Rule 2 (Wave 2 no full retrace): {compliance['rule2_wave2_no_full_retrace']:.2%}")
        print(f"  Rule 3 (Wave 4 no overlap): {compliance['rule3_wave4_no_overlap']:.2%}")
        print(f"  Overall Compliance: {compliance['overall']:.2%}")
        
        print("\n" + "=" * 70)
        
        if validation_results['validation_rate'] < 1.0:
            print("\n⚠️  WARNING: Some patterns violate Elliott Wave rules!")
            print("   This may indicate bugs in pattern detection logic.")
        else:
            print("\n✅ All patterns pass Elliott Wave rule validation!")


def validate_pattern_file(
    patterns_file: str,
    data_file: str = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Validate patterns from a JSON file.
    
    Args:
        patterns_file: Path to JSON file with detected patterns
        data_file: Optional path to price data file (if patterns don't include data)
        strict: Whether to use strict mode (all rules must pass)
        
    Returns:
        Validation results dict
    """
    import json
    from pathlib import Path
    
    # Load patterns
    with open(patterns_file) as f:
        data = json.load(f)
    
    # Extract patterns and price data
    if isinstance(data, list):
        patterns = data
        # Need to load price data separately
        if data_file is None:
            raise ValueError("data_file required when patterns file doesn't include price data")
        # Load price data (implement based on your data format)
        lows = np.array([])  # TODO: Load from data_file
        highs = np.array([])
    elif isinstance(data, dict):
        patterns = data.get('patterns', [])
        # Try to extract price data from file
        lows = np.array(data.get('lows', []))
        highs = np.array(data.get('highs', []))
    else:
        raise ValueError(f"Unexpected data format in {patterns_file}")
    
    # Validate
    validator = RuleValidator(strict_mode=strict)
    results = validator.validate_batch(patterns, lows, highs)
    validator.print_validation_report(results)
    
    return results
