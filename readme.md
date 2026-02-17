# Elliott Wave Analyzer

A high-performance Python pipeline for detecting Elliott Wave patterns in financial market data. Automatically identifies both **impulsive (12345)** and **corrective (ABC)** wave patterns across multiple stocks with configurable complexity and quality thresholds.

**Forked from:** [drstevendev/ElliottWaveAnalyzer](https://github.com/drstevendev/ElliottWaveAnalyzer)  
*Original base algorithm developed by drstevendev*

## Overview

Elliott Wave Theory is a form of technical analysis that identifies recurring wave patterns in financial markets. This tool automates the detection of these patterns across hundreds of stocks, scoring them based on:

- **Fibonacci Ratios**: Wave retracements and extensions align with golden ratios (0.382, 0.618, 1.618)
- **Elliott Wave Rules**: Strict compliance with wave structure rules
- **Time Proportions**: Temporal relationships between waves
- **Pattern Complexity**: Simpler patterns (fewer sub-waves) score higher

## Features

### �� Pattern Detection
- **Dual Pattern Types**: Detects both impulsive (5-wave) and corrective (3-wave) patterns
- **Configurable Complexity**: Adjust `up_to` parameter (3-12) to find simple or complex multi-degree patterns
- **Multi-Start Search**: Tries multiple pivot points per window to catch patterns not starting at global extrema
- **Window Overlap**: Configurable overlap (0-80%) to catch patterns spanning window boundaries
- **Ensemble Scoring**: Sophisticated scoring combining Fibonacci analysis, rule compliance, and pattern quality

### ⚡ Performance
- **Cross-Platform**: Auto-detects Mac (MPS), NVIDIA (CUDA), or CPU and optimizes accordingly
- **Parallel Processing**: 10 workers on Mac M4 Pro, 30+ on HPC systems
- **Shared Memory**: Workers use memory-mapped arrays to avoid copying price data
- **Batch Vectorization**: Pre-scores candidates in batches (512-2048) before full evaluation
- **~75 seconds per symbol** (500 windows, 15 years of data, up_to=8)

### 📊 Data & Output
- **15 Years of History**: Fetches daily OHLCV data from yfinance (or HuggingFace datasets)
- **JSON Results**: Comprehensive pattern metadata including scores, wave configurations, date ranges
- **Image Export**: Saves top-N pattern visualizations as PNG files for manual review
- **Checkpoint/Resume**: Can resume interrupted runs from saved progress

## Installation

### Prerequisites
- Python 3.10+ (tested on 3.10-3.14)
- 8GB+ RAM recommended
- Optional: GPU (Apple Silicon MPS or NVIDIA CUDA)

### Setup

```bash
# Clone repository
git clone https://github.com/usamaahmedsh/elliot-wave-analyzer-usamaahmedsh.git
cd elliot-wave-analyzer-usamaahmedsh

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon GPU support (optional)
pip install torch torchvision torchaudio

# For image export (optional)
pip install kaleido
```

## Quick Start

### Basic Usage

```bash
# Analyze a single stock
python scripts/pipeline_run.py --symbols AAPL --verbose

# Analyze multiple stocks
python scripts/pipeline_run.py --symbols "AAPL,MSFT,GOOG" --verbose

# With inline pattern analysis (shows stats per symbol)
python scripts/pipeline_run.py --symbols AAPL --verbose
# (Enable in configs.yaml: analyze_patterns: true)

# Analyze with custom output path
python scripts/pipeline_run.py \
  --symbols "AAPL,MSFT" \
  --output results/my_analysis.json \
  --verbose
```

### With Checkpointing (Recommended for large runs)

```bash
python scripts/pipeline_run.py \
  --symbols "AAPL,MSFT,GOOG,AMZN,TSLA" \
  --checkpoint-dir output/checkpoints \
  --resume \
  --verbose
```

### Results

After running, you'll find:
- **JSON file**: `output/results.json` - All detected patterns with metadata
- **Images**: `output/images/{SYMBOL}/pattern_*.png` - Top 5 visualizations per symbol
- **Checkpoints**: `output/checkpoints/` - Resume data (if enabled)

### Inline Pattern Analysis

Enable `analyze_patterns: true` in `configs.yaml` to see quality statistics during pipeline execution:

```
[1/1] Processing AAPL...
   ✅ Found 18 patterns in 69.0s
   📈 Top score: 0.899
   📊 Pattern Analysis:
      Ensemble: mean=0.551, median=0.530, max=0.899
      Types: impulse:1, Corrective:17
      Quality: Excellent=1, Good=0, Fair=17, Poor=0
```

## Configuration

The pipeline is controlled via `configs.yaml`. Key parameters:

### Pattern Detection Quality

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `up_to` | 8 | 3-12 | Wave complexity (higher = more sub-waves detected) |
| `top_n` | 20 | 5-50 | Number of top candidates kept per window |
| `max_combinations` | 500000 | 50k-1M | Search depth per window |
| `max_start_points` | 8 | 1-10 | Pivot points tried per window |
| `scan_pattern_types` | all | all/impulses | Pattern types to scan |

### Window Configuration

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `max_weeks` | 104 | 12-208 | Maximum window size (weeks) |
| `min_weeks` | 8 | 2-20 | Minimum window size (weeks) |
| `slide_weeks` | 2 | 1-8 | Window slide step (weeks) |
| `max_windows` | 500 | 50-2000 | Windows scanned per symbol |
| `window_overlap_ratio` | 0.5 | 0.0-0.8 | Window overlap (0.5 = 50%) |

### Resource Optimization

| Parameter | Default | Auto-Detected | Effect |
|-----------|---------|---------------|--------|
| `processes` | 10 | ✅ Yes | Worker threads |
| `cpu_batch_size` | 1024 | ✅ Yes | Vectorization batch size |
| `concurrency` | 12 | ✅ Yes | Network fetch parallelism |
| `auto_detect_device` | true | - | Enable hardware auto-detection |

### Image Export & Analysis

| Parameter | Default | Effect |
|-----------|---------|--------|
| `save_images` | true | Enable/disable PNG export |
| `save_images_top_n` | 5 | Images saved per symbol |
| `analyze_patterns` | false | Show inline quality stats during run |

## Understanding the Output

### JSON Structure

```json
{
  "metadata": {
    "total_symbols": 3,
    "total_patterns": 237,
    "runtime_seconds": 223.7
  },
  "patterns": [
    {
      "start_row": 825,
      "window_len": 364,
      "date_start": "2014-06-04",
      "date_end": "2015-11-10",
      "ensemble_score": 0.786,
      "fib_score": 0.633,
      "best": {
        "rule_name": "impulse",
        "score": 1.0,
        "wave_config": [2, 2, 4, 4, 3]
      }
    }
  ]
}
```

### Pattern Scores

- **ensemble_score** (0.0-1.0): Overall quality combining all factors
  - **0.85-1.0**: Excellent patterns, strong Fibonacci alignment
  - **0.70-0.85**: Good patterns, moderate alignment
  - **0.50-0.70**: Fair patterns, weak alignment
  - **<0.50**: Low quality, consider filtering out

- **fib_score** (0.0-1.0): Fibonacci ratio alignment only
- **score** (0.0-1.0): Rule compliance score

### Wave Configuration

Example: `[2, 2, 4, 4, 3]` means:
- **Wave 1**: Skip 2 intermediate highs
- **Wave 2**: Skip 2 intermediate lows
- **Wave 3**: Skip 4 intermediate highs
- **Wave 4**: Skip 4 intermediate lows
- **Wave 5**: Skip 3 intermediate highs

Higher numbers = more complex sub-wave structure.

## Tuning for Your Use Case

### Scenario 1: Fast Screening (10s per symbol)
```yaml
up_to: 5
max_windows: 200
slide_weeks: 4
max_start_points: 3
max_combinations: 100000
```
**Expected**: 20-30 patterns/symbol, lower quality

### Scenario 2: Balanced Quality (75s per symbol) ⭐ Recommended
```yaml
up_to: 8
max_windows: 500
slide_weeks: 2
max_start_points: 8
max_combinations: 500000
```
**Expected**: 70-80 patterns/symbol, high quality (0.85-0.90 scores)

### Scenario 3: Maximum Quality (300s per symbol)
```yaml
up_to: 12
max_windows: 1000
slide_weeks: 1
max_start_points: 10
max_combinations: 1000000
```
**Expected**: 150-250 patterns/symbol, exhaustive search

### Scenario 4: Quick Test (3s per symbol)
```yaml
up_to: 5
max_windows: 50
slide_weeks: 8
max_start_points: 1
enable_multi_start: false
```
**Expected**: 5-15 patterns/symbol, fast validation

## Performance Benchmarks

### Mac M4 Pro (12 cores, 16 GPU cores)

| Symbols | Time | Patterns | Avg/Symbol | Windows |
|---------|------|----------|------------|---------|
| 1 (AAPL) | 72s | 79 | 79 | 500 |
| 3 (AAPL,GOOG,MSFT) | 224s | 237 | 79 | 1,500 |
| 315 (S&P 500) | ~6.5 hrs | ~25,000 | 79 | 157,500 |

**Hardware Utilization**: ~90% of available cores

### HPC System (32 cores, Tesla A100)

| Symbols | Time | Patterns | Avg/Symbol | Workers |
|---------|------|----------|------------|---------|
| 315 (estimated) | ~2.5 hrs | ~25,000 | 79 | 30 |

**Note**: Actual performance depends on `up_to`, `max_windows`, and hardware specs.

## Architecture

### Pipeline Flow

```
1. Data Fetching (yfinance or HuggingFace)
   ↓
2. Window Generation (sliding windows with overlap)
   ↓
3. Parallel Pattern Detection (multi-process)
   ├─ Pre-scoring (vectorized batches)
   ├─ Top-K selection
   └─ Full evaluation (Elliott Wave rules)
   ↓
4. Ensemble Scoring (Fibonacci + rules + time)
   ↓
5. Image Export (top-N patterns)
   ↓
6. JSON Output
```

### Directory Structure

```
elliot-wave-analyzer-usamaahmedsh/
├── configs.yaml              # Main configuration file
├── requirements.txt          # Python dependencies
├── readme.md                 # This file
├── CHANGELOG.md             # Version history
│
├── data/                     # Stock ticker lists
│   ├── sp500_tickers.txt
│   └── international_tickers.txt
│
├── models/                   # Core detection logic
│   ├── WaveAnalyzer.py      # Main analyzer
│   ├── WavePattern.py       # Pattern representation
│   ├── MonoWave.py          # Single wave detection
│   ├── WaveRules.py         # Elliott Wave rules
│   ├── EnsembleScoring.py   # Scoring algorithms
│   └── helpers.py           # Plotting utilities
│
├── pipeline/                 # Pipeline infrastructure
│   ├── config.py            # Configuration loading
│   ├── executor.py          # Parallel execution
│   ├── device.py            # Hardware detection
│   ├── fetcher.py           # Data fetching
│   └── shared_memory.py     # Memory optimization
│
├── scripts/                  # Executable scripts
│   └── pipeline_run.py      # Main entry point
│
├── tools/                    # Performance tools
│   └── sweep_cpu_batch.py   # Parameter tuning
│
└── output/                   # Results (created at runtime)
    ├── results.json
    ├── images/
    │   ├── AAPL/
    │   ├── MSFT/
    │   └── GOOG/
    └── checkpoints/
```

## Advanced Usage

### Custom Data Sources

Use HuggingFace datasets instead of yfinance:

```bash
python scripts/pipeline_run.py \
  --symbols "AAPL,MSFT" \
  --hf-dataset "usamaahmedsh/financial-markets-dataset-15y-train" \
  --verbose
```

### Filtering Results by Score

```python
import json

# Load results
with open('output/results.json', 'r') as f:
    data = json.load(f)

# Filter high-quality patterns only
high_quality = [
    p for p in data['patterns']
    if p.get('ensemble_score', 0) > 0.85
]

print(f"Found {len(high_quality)} high-quality patterns")
```

### Batch Processing Script

```bash
#!/bin/bash
# process_sp500.sh

source venv/bin/activate

# Read symbols from file (50 at a time)
BATCH_SIZE=50
TOTAL=$(wc -l < data/sp500_tickers.txt)

for i in $(seq 0 $BATCH_SIZE $TOTAL); do
    SYMBOLS=$(tail -n +$((i+1)) data/sp500_tickers.txt | head -$BATCH_SIZE | tr '\n' ',' | sed 's/,$//')
    
    echo "Processing batch $((i/BATCH_SIZE + 1))..."
    
    python scripts/pipeline_run.py \
        --symbols "$SYMBOLS" \
        --output "output/batch_$((i/BATCH_SIZE + 1)).json" \
        --verbose
done

echo "All batches complete!"
```

## Evaluation Tools

The toolkit includes two evaluation tools to analyze pattern quality and profitability.

### 1. Pattern Quality Analysis

Analyze score distributions, quality tiers, and pattern type breakdowns:

```bash
# Basic analysis
python tools/analyze_patterns.py

# With per-symbol breakdown
python tools/analyze_patterns.py --by-symbol

# Filter to high-quality patterns only
python tools/analyze_patterns.py --min-score 0.70

# Export filtered patterns
python tools/analyze_patterns.py --min-score 0.85 --export-filtered output/high_quality.json
```

**Output includes:**
- Score distributions (ensemble, Fibonacci, rule compliance)
- Quality tier breakdown (Excellent/Good/Fair/Poor)
- Pattern type counts (impulsive vs corrective)
- Per-symbol statistics

**Example output:**
```
OVERALL SCORE DISTRIBUTION
  Ensemble Score:
    Count:   18
    Min:     0.530
    Median:  0.530
    Mean:    0.551
    Max:     0.899
    StdDev:  0.087

QUALITY TIERS
  Excellent (0.85-1.0)     1 patterns (5.6%)
  Good (0.70-0.85)         0 patterns (0.0%)
  Fair (0.50-0.70)        17 patterns (94.4%)
  Poor (<0.50)             0 patterns (0.0%)
```

### 2. Backtesting Framework

Simulate trading based on detected patterns and measure profitability:

```bash
# Basic backtest (default: 20-day hold, 5% stop-loss, 10% take-profit)
python tools/backtest_patterns.py

# Backtest specific symbol only
python tools/backtest_patterns.py --symbol AAPL

# Backtest specific pattern type
python tools/backtest_patterns.py --pattern-type impulse --min-score 0.85

# Custom parameters
python tools/backtest_patterns.py \
  --symbol AAPL \
  --min-score 0.70 \
  --holding-days 30 \
  --stop-loss 0.03 \
  --take-profit 0.15 \
  --position-size 10000

# Export trade log to CSV
python tools/backtest_patterns.py --export-trades output/trades.csv
```
  --holding-days 30 \
  --stop-loss 0.03 \
  --take-profit 0.15 \
  --position-size 10000

# Export trade log to CSV
python tools/backtest_patterns.py --export-trades output/trades.csv
```

**Trading Strategy:**
- **Entry:** At pattern end date (after detection)
- **Position:** Long on impulsive patterns (skip corrective)
- **Exit:** First of:
  - Stop-loss hit (default: -5%)
  - Take-profit hit (default: +10%)
  - Maximum holding period (default: 20 days)

**Performance Metrics:**
- Win rate, average return, median return
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown
- Profit factor (total profit / total loss)
- Best/worst trades

**Example output:**
```
BACKTEST RESULTS
📊 Trade Statistics:
  Total Trades:     1
  Winning Trades:   1 (100.0%)
  Avg Holding:      43.0 days

💰 Returns:
  Total Return:     4.21%
  Average Return:   4.21%
  Best Trade:       4.21%

📈 Performance Metrics:
  Sharpe Ratio:     0.000
  Max Drawdown:     0.00%
  Profit Factor:    inf
  Net P/L:          $42.14

🏆 Top 5 Best Trades:
  1. AAPL: +4.21% (score=0.899, 2017-04-04 → 2017-05-17)
```

**Note:** Backtesting uses historical data and does not guarantee future performance. Results include survivorship bias and assume perfect execution.

## Troubleshooting

### "No patterns found"

1. Check `up_to` value (try 5-8 for most stocks)
2. Increase `max_windows` (try 500-1000)
3. Enable `enable_multi_start: true`
4. Increase `window_overlap_ratio` (try 0.5)

### "Too slow"

1. Reduce `max_windows` (try 200-300)
2. Reduce `up_to` (try 5-6)
3. Increase `slide_weeks` (try 3-4)
4. Reduce `max_combinations` (try 200000)

### "Out of memory"

1. Reduce `processes` (try 6-8 on 12-core Mac)
2. Reduce `max_windows`
3. Set `use_shared_memory: true` (default)
4. Process fewer symbols at once

### "ModuleNotFoundError"

```bash
pip install -r requirements.txt
```

## Roadmap

### Real-Time & Emerging Pattern Detection 🚀

The next major feature is **real-time pattern detection mode** to identify incomplete Elliott Waves as they form:

**Planned Features:**
- **Streaming Data Integration**: Connect to live market data feeds (WebSocket, REST APIs)
- **Incomplete Wave Detection**: Identify patterns in progress (e.g., Wave 3 forming, expecting Wave 4-5)
- **Market Trend Analysis**: Real-time trend classification (bullish/bearish/neutral)
- **Alert System**: Notifications when high-probability patterns emerge
- **Confidence Scoring**: Probability estimates for incomplete patterns
- **Continuation Prediction**: Forecast likely completion points for in-progress waves

**Use Cases:**
- Monitor watchlist for emerging patterns throughout trading day
- Get alerts when Wave 3 completes (strongest wave, highest profit potential)
- Identify potential entry points before Wave 5 completion
- Track pattern invalidation (e.g., Wave 4 overlaps Wave 1)

**Technical Approach:**
- Sliding window analysis on most recent N bars
- Incremental pattern matching as new bars arrive
- Lower `up_to` values for faster detection (3-5 vs 8-12)
- Probabilistic scoring for incomplete patterns
- Pattern lifecycle tracking (forming → maturing → complete → invalidated)

To contribute or discuss this feature, see [GitHub Issues](https://github.com/usamaahmedsh/elliot-wave-analyzer-usamaahmedsh/issues).

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional pattern types (triangles, flats, zigzags)
- [ ] **Real-time & emerging pattern detection** 🚀 *(see Roadmap above)*
- [ ] Web dashboard for results
- [ ] Machine learning-based scoring
- [x] Backtesting framework ✅
- [x] Pattern quality analysis ✅
- [x] Inline pattern statistics ✅

## Credits

**Original Algorithm:** [drstevendev/ElliottWaveAnalyzer](https://github.com/drstevendev/ElliottWaveAnalyzer)  
This project is forked from and builds upon the foundational Elliott Wave detection algorithm developed by **drstevendev**.

**Enhancements & Extensions:**
- Cross-platform optimization (Mac MPS, CUDA, CPU)
- Multi-pattern type support (impulsive + corrective)
- Ensemble scoring with Fibonacci analysis
- Backtesting framework
- Pattern quality analysis tools
- Production-ready pipeline infrastructure

## License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this software under the terms of the Apache 2.0 license. See the [LICENSE](LICENSE) file for the full license text.

**Key Points:**
- ✅ Free for commercial and private use
- ✅ Modification and distribution permitted
- ✅ Patent grant included
- ⚠️ Must include copyright notice and license
- ⚠️ Changes must be documented
- ❌ No trademark rights granted
- ❌ No warranty provided

For more information, visit: https://www.apache.org/licenses/LICENSE-2.0

## Citation

If you use this tool in research, please cite:

```
Elliott Wave Analyzer (2026)
https://github.com/usamaahmedsh/elliot-wave-analyzer-usamaahmedsh
```

## Contact

- **Issues**: https://github.com/usamaahmedsh/elliot-wave-analyzer-usamaahmedsh/issues
- **Author**: Usama Ahmed

---

**Happy Pattern Hunting! 📈**
