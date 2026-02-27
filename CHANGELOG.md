# Changelog

All notable changes to the Elliott Wave Analyzer project.

## [2.1.0] - 2026-02-27

### Sequential Scanning with Skip-Ahead

#### New Features
- **Skip-Ahead Algorithm**: When a valid pattern is found, the scanner skips past the pattern's end date before searching for the next pattern. This eliminates duplicate patterns found in overlapping windows.
- **Thread-Safe Data Fetching**: Added threading lock to yfinance downloads to prevent data contamination when fetching multiple symbols concurrently.

#### Bug Fixes
- **Fixed yfinance Concurrent Download Bug**: yfinance is not thread-safe when downloading multiple symbols via ThreadPoolExecutor. Added `threading.Lock()` to serialize downloads, ensuring each symbol receives correct data.
- **Fixed Pattern Duplication**: Previously, 98% window overlap caused the same Elliott Wave pattern to be detected dozens of times with slightly different starting points. The skip-ahead approach ensures each unique pattern is found only once.

#### Configuration Changes
- Changed `slide_weeks` from 2 to 1 for finer granularity pattern detection
- Removed deduplication post-processing (no longer needed with skip-ahead)

#### Performance
- Reduced redundant window scans by skipping past found patterns
- More efficient use of compute resources by avoiding duplicate pattern detection

---

## [2.0.0] - 2026-02-16

### Major Refactor - Production Ready Pipeline

#### New Features
- **Cross-Platform Device Detection**: Automatically detects and optimizes for Mac (MPS), NVIDIA (CUDA), or CPU
- **Image Export**: Saves top-N pattern visualizations as PNG images for manual review
- **Enhanced Pattern Scoring**: Ensemble scoring system combining Fibonacci ratios, time proportions, and rule satisfaction
- **Multi-Start Search**: Attempts pattern detection from multiple pivot points within each window
- **Checkpoint/Resume Support**: Can resume interrupted runs from saved checkpoints
- **15-Year Historical Data**: Fetches and analyzes 15 years of daily OHLCV data per symbol

#### Performance Optimizations
- **10x Faster Pattern Detection**: Optimized from GPU-focused to CPU-vectorized approach
- **Shared Memory Arrays**: Workers use memory-mapped views instead of copying price data
- **Batch Vectorization**: Pre-scores candidates in batches before full evaluation
- **Parallel Processing**: 10 workers on Mac M4 Pro, 32+ on HPC systems
- **Approximately 75 seconds per symbol** with optimized configuration (500 windows, up_to=8)

#### Configuration System
- **Auto-Detection**: Hardware capabilities automatically set optimal worker count and batch sizes
- **Comprehensive Tuning Guide**: 50+ parameters for fine-tuning pattern detection
- **Preset Scenarios**: Fast screening, balanced quality, maximum quality configurations
- **Window Overlap**: Configurable overlap ratio to catch patterns spanning window boundaries

#### Bug Fixes
- Fixed overly defensive guards in `find_impulsive_wave()` that rejected valid patterns
- Fixed Python 3.14 compatibility issues with `from __future__ import annotations`
- Fixed corrective pattern scoring (removed non-existent `WavePattern.corrective()` method)
- Fixed None value handling in complexity calculations for ABC patterns
- Disabled overly strict `count_extrema` pre-filter

#### Pattern Detection Improvements
- **3x More Patterns**: Optimized configuration finds 70-80 patterns per symbol (vs 20-25 previously)
- **Higher Quality Scores**: Pattern scores now range 0.85-0.90 (vs 0.0-0.5 previously)
- **Both Pattern Types**: Scans impulsive (12345) and corrective (ABC) patterns
- **Flexible Complexity**: Configurable `up_to` parameter (3-12) for simple to complex patterns

#### Project Structure
- Cleaned up test files and legacy code
- Removed GPU-specific references (now CPU-first with GPU acceleration)
- Consolidated documentation into README.md
- Organized output: `output/results.json` and `output/images/{SYMBOL}/`

#### Dependencies
- Added PyTorch for MPS (Apple Silicon) GPU acceleration
- Added Kaleido for Plotly image export
- Added yfinance for data fetching
- Removed CUDA-specific dependencies

---

## [1.x.x] - Previous Versions

Earlier versions focused on GPU-accelerated batch processing and experimental features. See git history for details.

---

## Version Numbering

- **Major**: Breaking API changes or significant architecture shifts
- **Minor**: New features, non-breaking improvements
- **Patch**: Bug fixes, documentation updates
