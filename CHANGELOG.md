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

If you'd like a git commit message suggestion for these changes, or want me to create a tidy PR that groups the code edits and dataset artifacts, tell me how you'd like the commits organized (single commit vs separate commits for code/docs/data).