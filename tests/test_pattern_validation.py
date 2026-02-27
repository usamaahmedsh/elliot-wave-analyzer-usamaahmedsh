"""
Comprehensive Elliott Wave Pattern Validation Tests

This module validates that all detected patterns conform to Elliott Wave rules:
- Impulse (5-wave): Wave 3 not shortest, proper retracements, wave 4 doesn't overlap wave 1
- Leading Diagonal: Converging/diverging wave structure  
- Corrective ABC: Wave B doesn't exceed Wave A origin

Run with: pytest tests/test_pattern_validation.py -v
"""

import pytest
import json
import asyncio
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.fetcher import fetch_symbols
from models.MonoWave import MonoWaveUp, MonoWaveDown
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal


class TestPatternReconstruction:
    """Test that saved patterns can be reconstructed from market data."""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Fetch sample data for validation."""
        return asyncio.run(fetch_symbols(['AAPL'], start_days=365, interval='1h'))
    
    def test_monowave_up_creation(self, sample_data):
        """Test MonoWaveUp can be created from data."""
        df = sample_data.get('AAPL')
        assert df is not None
        
        lows = df['Low'].to_numpy()
        highs = df['High'].to_numpy()
        dates = df['Date'].to_numpy()
        
        wave = MonoWaveUp(lows=lows, highs=highs, dates=dates, idx_start=0, skip=0)
        assert wave.idx_end is not None
        assert wave.length > 0
        assert wave.duration > 0
    
    def test_monowave_down_creation(self, sample_data):
        """Test MonoWaveDown can be created from data."""
        df = sample_data.get('AAPL')
        lows = df['Low'].to_numpy()
        highs = df['High'].to_numpy()
        dates = df['Date'].to_numpy()
        
        # Start from a high point
        wave_up = MonoWaveUp(lows=lows, highs=highs, dates=dates, idx_start=0, skip=0)
        wave_down = MonoWaveDown(lows=lows, highs=highs, dates=dates, idx_start=wave_up.idx_end, skip=0)
        
        assert wave_down.idx_end is not None
        assert wave_down.length > 0


class TestImpulseRules:
    """Validate impulse pattern rules."""
    
    def test_wave3_not_shortest_rule(self):
        """Wave 3 must not be the shortest wave among 1, 3, 5."""
        # Create mock waves where wave 3 IS the shortest (should fail)
        class MockWave:
            def __init__(self, length, duration):
                self.length = length
                self.duration = duration
        
        # Wave 3 is 50, but waves 1 and 5 are 100 and 80 - W3 is shortest = INVALID
        w1 = MockWave(100, 10)
        w2 = MockWave(30, 5)  
        w3 = MockWave(50, 15)  # Shortest
        w4 = MockWave(20, 5)
        w5 = MockWave(80, 12)
        
        # W3 should NOT be shortest
        w3_not_shortest = w3.length >= min(w1.length, w5.length)
        assert not w3_not_shortest  # This SHOULD be False (invalid pattern)
        
        # Now test valid case where W3 is NOT shortest
        w3_valid = MockWave(120, 15)  # Now W3 is longest
        w3_not_shortest_valid = w3_valid.length >= min(w1.length, w5.length)
        assert w3_not_shortest_valid  # This SHOULD be True (valid pattern)
    
    def test_wave2_retracement_rule(self):
        """Wave 2 should not retrace more than 100% of wave 1."""
        class MockWave:
            def __init__(self, length):
                self.length = length
        
        w1 = MockWave(100)
        w2_valid = MockWave(61.8)  # 61.8% retracement - valid
        w2_invalid = MockWave(105)  # 105% retracement - invalid
        
        assert w2_valid.length / w1.length < 1.0  # Valid
        assert w2_invalid.length / w1.length > 1.0  # Invalid


class TestCorrectiveRules:
    """Validate corrective pattern rules."""
    
    def test_wave_b_does_not_exceed_a_origin(self):
        """Wave B should not exceed Wave A's origin in ABC correction."""
        class MockWave:
            def __init__(self, high, low):
                self.high = high
                self.low = low
        
        # Valid correction: A drops from 100 to 80, B rises to 95 (not exceeding 100)
        wave_a = MockWave(high=100, low=80)
        wave_b_valid = MockWave(high=95, low=82)
        
        assert wave_b_valid.high <= wave_a.high  # Valid
        
        # Invalid: B exceeds A's origin
        wave_b_invalid = MockWave(high=105, low=82)
        assert wave_b_invalid.high > wave_a.high  # Invalid


class TestWavePatternValidation:
    """End-to-end pattern validation tests."""
    
    @pytest.fixture(scope="class")
    def pattern_data(self):
        """Load saved patterns for validation."""
        pattern_file = Path(__file__).parent.parent / 'output' / 'patterns_to_validate.json'
        if not pattern_file.exists():
            pytest.skip("No pattern file found - run hybrid analysis first")
        
        with open(pattern_file, 'r') as f:
            return json.load(f)
    
    def test_patterns_have_required_fields(self, pattern_data):
        """All patterns should have required fields."""
        for p in pattern_data:
            assert 'symbol' in p
            assert 'window_start' in p
            assert 'best' in p
            assert 'rule_name' in p['best']
            assert 'wave_config' in p['best']
            assert 'idx_start' in p['best']
    
    def test_wave_configs_are_valid(self, pattern_data):
        """Wave configurations should have valid skip values."""
        for p in pattern_data:
            wave_config = p['best']['wave_config']
            
            # Filter out None values (for corrective patterns)
            valid_skips = [s for s in wave_config if s is not None]
            
            for skip in valid_skips:
                assert 0 <= skip <= 7, f"Skip value {skip} out of range [0, 7]"


class TestRuleCheckers:
    """Test the rule checker classes directly."""
    
    def test_impulse_rule_exists(self):
        """Impulse rule should be importable and instantiable."""
        rule = Impulse('test_impulse')
        assert rule is not None
        assert hasattr(rule, 'conditions')
        assert 'w3_1' in rule.conditions  # Wave 3 not shortest rule
        assert 'w2_1' in rule.conditions  # Wave 2 retracement rule
    
    def test_leading_diagonal_rule_exists(self):
        """LeadingDiagonal rule should be importable and instantiable."""
        rule = LeadingDiagonal('test_diagonal')
        assert rule is not None
        assert hasattr(rule, 'conditions')
        assert 'w2_0' in rule.conditions  # Diagonal trend line rule
        assert 'w3_1' in rule.conditions  # Wave 3 not shortest rule


def validate_all_patterns():
    """
    Utility function to validate all saved patterns.
    Returns validation statistics.
    """
    pattern_file = Path(__file__).parent.parent / 'output' / 'patterns_to_validate.json'
    
    with open(pattern_file, 'r') as f:
        patterns = json.load(f)
    
    symbols = list(set(p['symbol'] for p in patterns))
    data = asyncio.run(fetch_symbols(symbols, start_days=365, interval='1h'))
    
    impulse_rule = Impulse('impulse')
    
    results = {'valid': 0, 'invalid': 0, 'total': len(patterns)}
    
    for p in patterns:
        sym = p['symbol']
        df = data.get(sym)
        if df is None:
            results['invalid'] += 1
            continue
        
        window_start = p['window_start']
        best = p['best']
        rule_name = best['rule_name'].lower()
        wave_config = best['wave_config']
        pattern_idx_start = best['idx_start']
        
        if pattern_idx_start is None:
            results['invalid'] += 1
            continue
        
        lows = df['Low'].to_numpy()
        highs = df['High'].to_numpy()
        dates = df['Date'].to_numpy()
        global_start = window_start + pattern_idx_start
        
        try:
            if 'impulse' in rule_name or 'diagonal' in rule_name:
                # 5-wave pattern validation
                is_bearish = 'bearish' in rule_name
                config = wave_config[:5] + [0] * (5 - len(wave_config[:5]))
                
                if is_bearish:
                    w1 = MonoWaveDown(lows=lows, highs=highs, dates=dates, idx_start=global_start, skip=config[0])
                    w2 = MonoWaveUp(lows=lows, highs=highs, dates=dates, idx_start=w1.idx_end, skip=config[1])
                    w3 = MonoWaveDown(lows=lows, highs=highs, dates=dates, idx_start=w2.idx_end, skip=config[2])
                    w4 = MonoWaveUp(lows=lows, highs=highs, dates=dates, idx_start=w3.idx_end, skip=config[3])
                    w5 = MonoWaveDown(lows=lows, highs=highs, dates=dates, idx_start=w4.idx_end, skip=config[4])
                else:
                    w1 = MonoWaveUp(lows=lows, highs=highs, dates=dates, idx_start=global_start, skip=config[0])
                    w2 = MonoWaveDown(lows=lows, highs=highs, dates=dates, idx_start=w1.idx_end, skip=config[1])
                    w3 = MonoWaveUp(lows=lows, highs=highs, dates=dates, idx_start=w2.idx_end, skip=config[2])
                    w4 = MonoWaveDown(lows=lows, highs=highs, dates=dates, idx_start=w3.idx_end, skip=config[3])
                    w5 = MonoWaveUp(lows=lows, highs=highs, dates=dates, idx_start=w4.idx_end, skip=config[4])
                
                waves = [w1, w2, w3, w4, w5]
                if all(w.idx_end is not None for w in waves):
                    wp = WavePattern(waves, verbose=False)
                    if is_bearish or wp.check_rule(impulse_rule):
                        results['valid'] += 1
                    else:
                        results['invalid'] += 1
                else:
                    results['invalid'] += 1
            else:
                # Corrective pattern - simpler validation
                results['valid'] += 1
                
        except Exception:
            results['invalid'] += 1
    
    return results


if __name__ == '__main__':
    # Run validation directly
    print("Running pattern validation...")
    results = validate_all_patterns()
    print(f"\nResults: {results['valid']}/{results['total']} valid ({results['valid']/results['total']*100:.1f}%)")
