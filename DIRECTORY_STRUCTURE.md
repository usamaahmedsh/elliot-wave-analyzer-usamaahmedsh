# Directory Structure

```
elliott-wave-analyzer/
├── README.md                       # Main project README
├── CHANGELOG.md                    # Version history
├── requirements.txt                # Python dependencies
├── configs.yaml                    # Main configuration
├── configs_high_perf.yaml          # High-performance configuration
│
├── bin/                            # Executable scripts (user-facing)
│   ├── run_pipeline.sh            # Main pipeline runner
│   └── quickstart.sh              # Quick test script
│
├── scripts/                        # Core pipeline scripts
│   └── pipeline_run.py            # Main pipeline orchestrator
│
├── models/                         # Elliott Wave pattern models
│   ├── __init__.py
│   ├── functions.py               # Wave calculation functions
│   ├── helpers.py                 # Helper utilities
│   ├── MonoWave.py                # Single wave representation
│   ├── Trend.py                   # Trend analysis
│   ├── WaveAnalyzer.py            # Main analyzer engine
│   ├── WaveCycle.py               # Wave cycle management
│   ├── WaveOptions.py             # Wave combination generator
│   ├── WavePattern.py             # Pattern matching
│   └── WaveRules.py               # Elliott Wave rules
│
├── pipeline/                       # Pipeline infrastructure
│   ├── config.py                  # Configuration management
│   ├── executor.py                # Parallel execution engine
│   ├── fetcher.py                 # Data fetching utilities
│   ├── numba_warm.py              # Numba JIT pre-warming
│   └── ... (other pipeline modules)
│
├── evaluation/                     # Pattern evaluation framework
│   ├── __init__.py
│   ├── metrics.py                 # Evaluation metrics (P/R/F1, IoU)
│   ├── rule_validator.py          # Rule compliance validation
│   └── predictive_evaluator.py    # Predictive power analysis
│
├── utils/                          # Utility scripts and tools
│   ├── hpc/                       # HPC-specific scripts
│   │   ├── run_hpc.sh            # Quick HPC runner
│   │   └── submit_hpc.sh         # SLURM batch submission
│   └── evaluation/                # Evaluation utilities
│       ├── evaluate_patterns.py   # Main evaluation CLI
│       └── build_validation_set.py # Validation set builder
│
├── examples/                       # Example scripts and demos
│   ├── usage/                     # Usage examples
│   │   ├── example_monowave.py
│   │   ├── example_waveoptions.py
│   │   ├── example_12345_impulsive_wave.py
│   │   └── example_evaluation.py
│   └── data/                      # Data management examples
│       ├── fetch_data.py          # yfinance data fetcher
│       ├── upload_to_hf.py        # HF dataset uploader
│       ├── push_parquet_as_dataset.py
│       └── sanity_check.py        # Data validation
│
├── docs/                           # Documentation
│   ├── guides/                    # Comprehensive guides
│   │   ├── AUTOMATION_GUIDE.md    # Pipeline automation
│   │   ├── HPC_GUIDE.md           # HPC/SLURM usage
│   │   ├── GPU_ACCELERATION_GUIDE.md # GPU setup
│   │   ├── EVALUATION_QUICKSTART.md # Evaluation methods
│   │   └── EVALUATION_FRAMEWORK.md  # Evaluation overview
│   ├── summaries/                 # Technical summaries
│   │   ├── HARDWARE_OPTIMIZATION_SUMMARY.md
│   │   ├── PERFORMANCE_OPTIMIZATIONS.md
│   │   ├── PIPELINE_FLOW.md       # Visual pipeline flow
│   │   ├── ACCURACY_IMPROVEMENTS.md
│   │   ├── ADVANCED_ACCURACY_FEATURES.md
│   │   └── RECENT_UPDATES_2026-02-13.md
│   ├── references/                # Quick references
│   │   ├── HPC_QUICK_REF.md       # HPC quick reference
│   │   ├── HPC_COMPLETE.md        # HPC feature summary
│   │   ├── AUTOMATION_QUICK_REF.md
│   │   ├── AUTOMATION_COMPLETE.md
│   │   ├── EVALUATION_COMPLETE.md
│   │   ├── ENHANCEMENTS_COMPLETE.md
│   │   ├── ACCELERATION_COMPLETE.md
│   │   ├── OPTIMIZATION_SUMMARY.md
│   │   ├── QUICK_REFERENCE.md
│   │   └── PIPELINE_GPU_MIGRATION.md
│   └── img/                       # Documentation images
│
├── tests/                          # Unit and integration tests
│   ├── test_fetch_data.py
│   └── test_monowave.py
│
├── tools/                          # Development tools
│   └── sweep_cpu_batch.py         # Performance tuning tool
│
├── data/                           # Data storage
│   ├── README_dataset.md          # Dataset documentation
│   ├── all_markets_15y_metadata.json
│   ├── market_manifest.json
│   ├── fetch_failures.json
│   ├── sp500_tickers.txt
│   └── international_tickers.txt
│
├── cache/                          # Cached data
│   └── tickers/                   # Per-ticker cache
│
├── output/                         # Pipeline outputs
│   └── TIMESTAMP/                 # Timestamped runs
│       ├── results.json           # Detected patterns
│       ├── checkpoints/           # Checkpoints for resume
│       ├── evaluation_report.html # Evaluation report
│       └── evaluation_metrics.json # Metrics data
│
├── backups/                        # Backup scripts
└── images/                         # Generated pattern images
```

## Key Directories

### User-Facing
- **`bin/`** - Scripts you run directly (`run_pipeline.sh`, `quickstart.sh`)
- **`docs/guides/`** - Start here for comprehensive documentation
- **`docs/references/`** - Quick reference cards and summaries

### Core Code
- **`models/`** - Elliott Wave pattern detection logic
- **`pipeline/`** - Pipeline execution infrastructure
- **`evaluation/`** - Pattern quality evaluation

### Utilities
- **`utils/hpc/`** - HPC/SLURM scripts for cluster deployment
- **`utils/evaluation/`** - Evaluation tools and validation

### Examples & Learning
- **`examples/usage/`** - Example code showing how to use components
- **`examples/data/`** - Data management utilities

### Configuration
- **`configs.yaml`** - Main configuration (CPU-optimized)
- **`configs_high_perf.yaml`** - High-performance config (30 workers + GPU)

## Quick Start

```bash
# Quick test (30 seconds)
bin/quickstart.sh

# Full pipeline
bin/run_pipeline.sh AAPL,MSFT,GOOG

# HPC batch job
sbatch utils/hpc/submit_hpc.sh
```

## Documentation

- **Getting Started**: `docs/guides/AUTOMATION_GUIDE.md`
- **HPC Usage**: `docs/guides/HPC_GUIDE.md`
- **Evaluation**: `docs/guides/EVALUATION_QUICKSTART.md`
- **Quick Ref**: `docs/references/HPC_QUICK_REF.md`
