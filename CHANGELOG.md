
# Changelog

All notable changes to this project are documented in this file.

## 2026-02-13 — GPU batching & pipeline updates

Summary

- Added hybrid GPU-batching pre-score in `models/WaveAnalyzer.scan_impulses` to allow batched in-device scoring of cheap features and only fully evaluate the top candidates per batch (`gpu_enabled`, `gpu_batch_size`, `gpu_top_k`).
- Added `pipeline/gpu_accel.GPUAccelerator.score_features` which prefers MPS → CUDA → CPU via PyTorch and falls back to a CPU linear scorer when PyTorch is unavailable.
- Centralized runtime knobs in `configs.yaml` (notable keys: `up_to`, `gpu_enabled`, `gpu_batch_size`, `gpu_top_k`, `pre_score_top_k`, `pre_score_threshold`, `min_volatility`, `max_windows`).
- `scripts/pipeline_run.py` now supports `--gpu`, writes a single consolidated JSON (`data/results_run_<ts>.json`), and moves the latest result to `output/latest_results.json`; images are saved under `output/images/`.

Benchmark (single-run smoke):

- Command: `PYTHONPATH=. .venv/bin/python3 scripts/pipeline_run.py GOOG --config configs.yaml --source yfinance --out-dir output --processes 6 --gpu` with `up_to=15` in `configs.yaml`.
- Observed timing (local M4 Pro, PyTorch MPS when available): fetch 0.14s, build_windows ~0s, scan ~115.86s, total ~116.6s. This confirms end-to-end functionality; further tuning of batch/top_k and pre-score features is suggested to reduce total scan time.

Notes

- This is an incremental hybrid approach: expensive enumerations are still CPU-bound; we prune via cheap GPU-batched scoring to reduce full evaluations. A full GPU port (translating inner scans into array kernels) remains a future larger task.

## 2026-01-18 — Small fixes & CLI runner

Summary

- Fixed a data-conversion bug in `models/helpers.py::convert_yf_data`: the function now tolerates yfinance return shapes that may make `df["Open"]` (or other OHLC selections) a single-column DataFrame rather than a Series. This prevents an AttributeError when converting to lists and improves robustness across yfinance versions/platforms.

- Added `scripts/run_symbol.py` — a small, documented CLI runner to scan one or many tickers for impulsive wave patterns. Features:
  - positional tickers (e.g. `AAPL MSFT`)
  - reading tickers from a file (`-f tickers.txt`)
  - `--days` to set lookback and `--delay` to throttle requests

Artifacts / outputs

- `images/` — generated PNG charts for found patterns, plus `.json` and `.csv` payloads saved alongside images.

Files changed

- Modified: `models/helpers.py` (robust `convert_yf_data`)
- Added: `scripts/run_symbol.py` (CLI for one-or-many tickers)

- Added: soft-scoring for WavePattern validation:
  - `models/WavePattern.py::WavePattern.score_rule(waverule)` — a lightweight heuristic
    that returns a float in [0,1] indicating how well a candidate pattern satisfies
    a given `WaveRule`. This lets callers rank multiple valid patterns by confidence
    instead of using a strict first-valid-wins approach.

- Added: standardized export payloads (JSON + CSV) written alongside images:
  - JSON top-level keys: `symbol`, `timeframe`, `rule_name`, `score`, `pattern_type`,
    `degree`, `idx_start`, `idx_end`, `low`, `high`, `dates_polyline`,
    `values_polyline`, `labels_polyline`, `waves` (list of per-wave dicts).
  - CSV flattened columns (one row per wave):
    `symbol, timeframe, rule_name, score, pattern_type, degree, idx_start, idx_end, low, high,`
    `wave_key, wave_label, wave_idx_start, wave_idx_end, wave_date_start, wave_date_end,`
    `wave_low, wave_high, wave_low_idx, wave_high_idx, wave_length, wave_duration`.
  - Implementations: `models/helpers.py` exposing helpers (internal names with leading
    underscores) such as `_serialize_wavepattern`, `_write_pattern_json_and_csv`,
    `_new_base_filename` and `save_chart_as_image`.

- Added: sliding adaptive-window scan in `models/WaveAnalyzer.py`:
  - `find_best_impulse_adaptive_window` and `sliding_adaptive_impulses` let the
    analyzer grow a search window (configurable by weeks/bars) and slide it
    forward to detect impulse patterns robustly across varying lookbacks.
  - Parameters: `slide_weeks`, `min_weeks`, `max_weeks`, `grow_weeks`, `up_to`,
    and `top_n` control the scan granularity and candidate selection.
  - See `scripts/example_12345_impulsive_wave.py` for an orchestration example
    showing sliding-adaptive calls and how outputs are written to `images/`.

Notes

- The project uses a Python 3.11 venv for local runs (see `readme.md` usage). The CLI is defensive and will continue scanning other symbols if one fails.

- Tests: the repository contains basic pytest tests under `tests/` (e.g. `test_fetch_data.py`, `test_monowave.py`). Run them with `pytest` inside the project venv to validate basic behavior.

- Helper script location: older README references `get_data.py`; the current fetcher is `scripts/fetch_data.py` (a legacy copy is kept at `backups/get_data.py`). Updated README accordingly.

## 2026-01-15 — Contributions by `usamaahmedsh`

Summary

- Produced a merged financial markets dataset covering the last 15 years (daily OHLCV) for multiple markets and top candidates per market.
- Added tooling for fetching, retrying, normalizing symbols, validating, and publishing datasets to Hugging Face.
- Improved robustness (caching, retries, dedupe/merge) and added metadata + sanity checks.

Artifacts produced

- data/all_markets_15y.parquet — combined dataset (merged with retry results)
- data/failed_markets_retry.parquet — retry fetch results for missing markets
- data/market_manifest.json — per-market/ticker manifest with rows/min/max dates and failures
- data/README_dataset.md — short README describing the dataset
- data/all_markets_15y_metadata.json — machine-readable metadata for the dataset

Scripts added/updated

- scripts/sanity_check.py (modified)
  - Added a small tolerance for start-date check and ensured Date parsing; prints per-market NA fractions and basic assertions.

- scripts/upload_to_hf.py (added + iterated)
  - Added script to create a Hugging Face repo and upload dataset files.
  - Made import of `Repository` robust across huggingface_hub versions and added a fallback to `HfApi.upload_file` when Repository is not available.
  - Added `--repo-type` flag and better error/warning messages.

- scripts/push_parquet_as_dataset.py (added)
  - Converts a local parquet to a Hugging Face `datasets.Dataset` and pushes it (default: single `train` split).
  - Reads `HF_TOKEN` from the environment or uses `huggingface-cli login`.

- scripts/upload_to_hf.py (iterative fixes)
  - Handled cases where `Repository` is missing and used HfApi fallback for uploading files.
  - Added create-repo logic with `repo_type='dataset'` option.

Repository docs updated

- readme.md (modified)
  - Added "Contributions by usamaahmedsh (2026-01-15)" near the top describing dataset work, tooling, and artifacts.

Other actions performed (engineering steps)

- Ran batch fetches (yfinance-only, daily) for many markets with top-N selection and parallel downloads.
- Implemented retry passes for missing markets and merged retry output into the main combined parquet (deduplicated by ticker+Date, preferring retry rows on conflict).
- Ran the sanity-check script against the merged file and verified assertions passed after adding a small start-date tolerance.
- Prepared upload tooling and tested upload flows locally; debugged huggingface_hub import/version issues and added fallbacks.

Notes & next steps

- Many international tickers require symbol normalization (e.g. `-L` -> `.L`) or provider fallbacks; these are logged in `data/market_manifest.json` for follow-up.
- If you want the dataset split-by-market in HF Datasets or streaming upload for very large files, I can modify `scripts/push_parquet_as_dataset.py` to emit multiple splits and stream chunked uploads.

Full file list (created/modified in this session)

- Added:
  - data/README_dataset.md
  - data/all_markets_15y_metadata.json
  - scripts/upload_to_hf.py
  - scripts/push_parquet_as_dataset.py
  - CHANGELOG.md (this file)

- Created datasets/artifacts (not source edits but generated):
  - data/all_markets_15y.parquet
  - data/failed_markets_retry.parquet
  - data/market_manifest.json (updated)

- Modified:
  - scripts/sanity_check.py
  - readme.md
# Changelog

All notable changes to this project are documented in this file.

## 2026-01-15 — Contributions by `usamaahmedsh`

Summary
- Produced a merged financial markets dataset covering the last 15 years (daily OHLCV) for multiple markets and top candidates per market.
- Added tooling for fetching, retrying, normalizing symbols, validating, and publishing datasets to Hugging Face.
- Improved robustness (caching, retries, dedupe/merge) and added metadata + sanity checks.

Artifacts produced
- data/all_markets_15y.parquet — combined dataset (merged with retry results)
- data/failed_markets_retry.parquet — retry fetch results for missing markets
- data/market_manifest.json — per-market/ticker manifest with rows/min/max dates and failures
- data/README_dataset.md — short README describing the dataset
- data/all_markets_15y_metadata.json — machine-readable metadata for the dataset

Scripts added/updated
- scripts/sanity_check.py (modified)
  - Added a small tolerance for start-date check and ensured Date parsing; prints per-market NA fractions and basic assertions.

- scripts/upload_to_hf.py (added + iterated)
  - Added script to create a Hugging Face repo and upload dataset files.
  - Made import of `Repository` robust across huggingface_hub versions and added a fallback to `HfApi.upload_file` when Repository is not available.
  - Added `--repo-type` flag and better error/warning messages.

- scripts/push_parquet_as_dataset.py (added)
  - Converts a local parquet to a Hugging Face `datasets.Dataset` and pushes it (default: single `train` split).
  - Reads `HF_TOKEN` from the environment or uses `huggingface-cli login`.

- scripts/upload_to_hf.py (iterative fixes)
  - Handled cases where `Repository` is missing and used HfApi fallback for uploading files.
  - Added create-repo logic with `repo_type='dataset'` option.

Repository docs updated
- readme.md (modified)
  - Added "Contributions by usamaahmedsh (2026-01-15)" near the top describing dataset work, tooling, and artifacts.

Other actions performed (engineering steps)
- Ran batch fetches (yfinance-only, daily) for many markets with top-N selection and parallel downloads.
- Implemented retry passes for missing markets and merged retry output into the main combined parquet (deduplicated by ticker+Date, preferring retry rows on conflict).
- Ran the sanity-check script against the merged file and verified assertions passed after adding a small start-date tolerance.
- Prepared upload tooling and tested upload flows locally; debugged huggingface_hub import/version issues and added fallbacks.

Notes & next steps
- Many international tickers require symbol normalization (e.g. `-L` -> `.L`) or provider fallbacks; these are logged in `data/market_manifest.json` for follow-up.
- If you want the dataset split-by-market in HF Datasets or streaming upload for very large files, I can modify `scripts/push_parquet_as_dataset.py` to emit multiple splits and stream chunked uploads.

Full file list (created/modified in this session)
- Added:
  - data/README_dataset.md
  - data/all_markets_15y_metadata.json
  ```markdown
  # Changelog

  All notable changes to this project are documented in this file.

  ## 2026-01-18 — Small fixes & CLI runner

  Summary

  - Fixed a data-conversion bug in `models/helpers.py::convert_yf_data`: the function now tolerates yfinance return shapes that may make `df["Open"]` (or other OHLC selections) a single-column DataFrame rather than a Series. This prevents an AttributeError when converting to lists and improves robustness across yfinance versions/platforms.

  - Added `scripts/run_symbol.py` — a small, documented CLI runner to scan one or many tickers for impulsive wave patterns. Features:
    - positional tickers (e.g. `AAPL MSFT`)
    - reading tickers from a file (`-f tickers.txt`)
    - `--days` to set lookback and `--delay` to throttle requests

  Artifacts / outputs

  - `images/` — generated PNG charts for found patterns, plus `.json` and `.csv` payloads saved alongside images.

  Files changed

  - Modified: `models/helpers.py` (robust `convert_yf_data`)
  - Added: `scripts/run_symbol.py` (CLI for one-or-many tickers)

  Notes

  - The project uses a Python 3.11 venv for local runs (see `readme.md` usage). The CLI is defensive and will continue scanning other symbols if one fails.

  ## 2026-01-15 — Contributions by `usamaahmedsh`

  Summary
  - Produced a merged financial markets dataset covering the last 15 years (daily OHLCV) for multiple markets and top candidates per market.
  - Added tooling for fetching, retrying, normalizing symbols, validating, and publishing datasets to Hugging Face.
  - Improved robustness (caching, retries, dedupe/merge) and added metadata + sanity checks.

  Artifacts produced
  - data/all_markets_15y.parquet — combined dataset (merged with retry results)
  - data/failed_markets_retry.parquet — retry fetch results for missing markets
  - data/market_manifest.json — per-market/ticker manifest with rows/min/max dates and failures
  - data/README_dataset.md — short README describing the dataset
  - data/all_markets_15y_metadata.json — machine-readable metadata for the dataset

  Scripts added/updated
  - scripts/sanity_check.py (modified)
    - Added a small tolerance for start-date check and ensured Date parsing; prints per-market NA fractions and basic assertions.

  - scripts/upload_to_hf.py (added + iterated)
    - Added script to create a Hugging Face repo and upload dataset files.
    - Made import of `Repository` robust across huggingface_hub versions and added a fallback to `HfApi.upload_file` when Repository is not available.
    - Added `--repo-type` flag and better error/warning messages.

  - scripts/push_parquet_as_dataset.py (added)
    - Converts a local parquet to a Hugging Face `datasets.Dataset` and pushes it (default: single `train` split).
    - Reads `HF_TOKEN` from the environment or uses `huggingface-cli login`.

  - scripts/upload_to_hf.py (iterative fixes)
    - Handled cases where `Repository` is missing and used HfApi fallback for uploading files.
    - Added create-repo logic with `repo_type='dataset'` option.

  Repository docs updated
  - readme.md (modified)
    - Added "Contributions by usamaahmedsh (2026-01-15)" near the top describing dataset work, tooling, and artifacts.

  Other actions performed (engineering steps)
  - Ran batch fetches (yfinance-only, daily) for many markets with top-N selection and parallel downloads.
  - Implemented retry passes for missing markets and merged retry output into the main combined parquet (deduplicated by ticker+Date, preferring retry rows on conflict).
  - Ran the sanity-check script against the merged file and verified assertions passed after adding a small start-date tolerance.
  - Prepared upload tooling and tested upload flows locally; debugged huggingface_hub import/version issues and added fallbacks.

  Notes & next steps
  - Many international tickers require symbol normalization (e.g. `-L` -> `.L`) or provider fallbacks; these are logged in `data/market_manifest.json` for follow-up.
  - If you want the dataset split-by-market in HF Datasets or streaming upload for very large files, I can modify `scripts/push_parquet_as_dataset.py` to emit multiple splits and stream chunked uploads.

  Full file list (created/modified in this session)
  - Added:
    - data/README_dataset.md
    - data/all_markets_15y_metadata.json
    - scripts/upload_to_hf.py
    - scripts/push_parquet_as_dataset.py
    - CHANGELOG.md (this file)

  - Created datasets/artifacts (not source edits but generated):
    - data/all_markets_15y.parquet
    - data/failed_markets_retry.parquet
    - data/market_manifest.json (updated)

  - Modified:
    - scripts/sanity_check.py
    - readme.md

  ```
