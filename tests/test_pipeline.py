#!/usr/bin/env python3
"""
Pipeline Integration Tests

Run with: python -m pytest tests/test_pipeline.py -v
Or directly: python tests/test_pipeline.py
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


class TestWaveOptionsGenerator:
    """Test the WaveOptionsGenerator fix"""
    
    def test_generator5_count(self):
        """WaveOptionsGenerator5 should produce 8^5 = 32768 options for up_to=8"""
        from models.WaveOptions import WaveOptionsGenerator5
        gen = WaveOptionsGenerator5(up_to=8)
        assert gen.number == 32768, f"Expected 32768, got {gen.number}"
    
    def test_generator5_independent_skips(self):
        """Previously missing configs like [0,0,2,0,1] should now be present"""
        from models.WaveOptions import WaveOptionsGenerator5
        gen = WaveOptionsGenerator5(up_to=8)
        
        test_configs = [
            [0, 0, 2, 0, 1],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1],
            [2, 0, 3, 0, 4],
        ]
        for cfg in test_configs:
            found = any(opt.values == cfg for opt in gen.options)
            assert found, f"Config {cfg} should be in options"
    
    def test_generator3_count(self):
        """WaveOptionsGenerator3 should produce 8^3 = 512 options for up_to=8"""
        from models.WaveOptions import WaveOptionsGenerator3
        gen = WaveOptionsGenerator3(up_to=8)
        assert gen.number == 512, f"Expected 512, got {gen.number}"


class TestDataFetching:
    """Test multi-interval data fetching"""
    
    def test_daily_fetch(self):
        """Daily data fetching should work"""
        from pipeline.fetcher import fetch_symbols
        data = asyncio.run(fetch_symbols(['AAPL'], start_days=100, interval='1d'))
        assert 'AAPL' in data
        assert len(data['AAPL']) > 50
    
    def test_hourly_fetch(self):
        """Hourly data fetching should work"""
        from pipeline.fetcher import fetch_symbols
        data = asyncio.run(fetch_symbols(['AAPL'], start_days=30, interval='1h'))
        assert 'AAPL' in data
        assert len(data['AAPL']) > 100
    
    def test_weekly_fetch(self):
        """Weekly data fetching should work"""
        from pipeline.fetcher import fetch_symbols
        data = asyncio.run(fetch_symbols(['AAPL'], start_days=1000, interval='1wk'))
        assert 'AAPL' in data
        assert len(data['AAPL']) > 40
    
    def test_data_structure(self):
        """Fetched data should have required columns"""
        from pipeline.fetcher import fetch_symbols
        data = asyncio.run(fetch_symbols(['AAPL'], start_days=100, interval='1d'))
        df = data['AAPL']
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestThreadSafety:
    """Test concurrent fetching thread safety"""
    
    def test_concurrent_fetch_no_contamination(self):
        """Concurrent fetching should not contaminate data between symbols"""
        from pipeline.fetcher import fetch_symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        data = asyncio.run(fetch_symbols(symbols, start_days=100, interval='1d', concurrency=5))
        
        # All symbols should have data
        for sym in symbols:
            assert sym in data, f"Missing data for {sym}"
            assert len(data[sym]) > 0, f"Empty data for {sym}"
        
        # Prices should be different (no contamination)
        prices = {sym: data[sym]['Close'].iloc[-1] for sym in symbols}
        unique_prices = len(set([round(p, 2) for p in prices.values()]))
        assert unique_prices == len(symbols), "Data contamination detected"


class TestPatternDetection:
    """Test pattern detection functionality"""
    
    def test_wave_analyzer_creation(self):
        """WaveAnalyzer should be creatable from DataFrame"""
        from pipeline.fetcher import fetch_symbols
        from models.WaveAnalyzer import WaveAnalyzer
        
        data = asyncio.run(fetch_symbols(['AAPL'], start_days=500, interval='1d'))
        df = data['AAPL'].iloc[:100].reset_index(drop=True)
        wa = WaveAnalyzer(df=df)
        assert len(wa.lows) == 100
        assert len(wa.highs) == 100
    
    def test_scan_methods_exist(self):
        """WaveAnalyzer should have scan methods"""
        from pipeline.fetcher import fetch_symbols
        from models.WaveAnalyzer import WaveAnalyzer
        
        data = asyncio.run(fetch_symbols(['AAPL'], start_days=500, interval='1d'))
        df = data['AAPL'].iloc[:100].reset_index(drop=True)
        wa = WaveAnalyzer(df=df)
        
        assert hasattr(wa, 'scan_impulses')
        assert hasattr(wa, 'scan_all_patterns')
        assert hasattr(wa, 'scan_multi_start')


class TestWorkerScan:
    """Test the worker scan function"""
    
    def test_worker_returns_dict_or_empty(self):
        """Worker scan should return dict with pattern or empty dict"""
        from pipeline.fetcher import fetch_symbols
        from pipeline.executor import _worker_scan_window
        
        data = asyncio.run(fetch_symbols(['AAPL'], start_days=500, interval='1d'))
        df = data['AAPL']
        
        context = {
            'symbol': 'AAPL',
            'lows': df['Low'].to_numpy(),
            'highs': df['High'].to_numpy(),
            'dates': df['Date'].to_numpy()
        }
        cfg_dict = {
            'up_to': 5,
            'top_n': 3,
            'cpu_batch_size': 512,
            'scan_pattern_types': 'all',
            'enable_multi_start': False,
            'max_start_points': 5,
            'max_combinations': 10000
        }
        
        window_tuple = (0, 100, context)
        result = _worker_scan_window(window_tuple, cfg_dict)
        
        assert isinstance(result, dict)


class TestConfig:
    """Test configuration loading"""
    
    def test_config_load(self):
        """Config should load from YAML file"""
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig.load_from_file('configs.yaml')
        assert cfg.up_to > 0
        assert cfg.top_n > 0
        assert cfg.max_combinations > 0


def run_all_tests():
    """Run all tests manually"""
    print("=" * 70)
    print("RUNNING PIPELINE TESTS")
    print("=" * 70)
    
    test_classes = [
        TestWaveOptionsGenerator,
        TestDataFetching,
        TestThreadSafety,
        TestPatternDetection,
        TestWorkerScan,
        TestConfig,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  FAIL: {method_name} - {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)
    
    if failed_tests:
        print("\nFailed tests:")
        for cls, method, error in failed_tests:
            print(f"  {cls}.{method}: {error}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
