
# ElliottWaveAnalyzer
First Version of an (not yet) iterative Elliott Wave scanner in financial data.

## 

- Forked from the repository https://github.com/drstevendev/ElliottWaveAnalyzer
- HF Link: https://huggingface.co/datasets/usamaahmedsh/financial-markets-dataset-15y-train

## Contributions by usamaahmedsh (2026-01-15)

- Dataset engineering and ingestion:
	- Implemented a yfinance-based, daily-only batch fetcher to download OHLCV histories (top-N per market) and produce unified parquet outputs (`data/all_markets_15y.parquet`, `data/all_markets_20y.parquet`).
	- Added per-ticker parquet caching, aggressive parallel downloads, and OHLC normalization to fill missing Open/High/Low values from Close when necessary.

- Coverage and recovery:
	- Expanded market candidate lists across equities, crypto, ETFs, commodities, bonds, FX, indices, REITs, sectors, futures and volatility instruments.
	- Implemented retry logic, symbol-normalization heuristics, and a manifest to record per-ticker success/failure and date ranges.
	- Merged retry results into the final combined dataset and added a sanity-check script (`scripts/sanity_check.py`) to validate date ranges and completeness.

- Tooling and reproducibility:
	- Created upload tooling and helpers to publish the dataset to the Hugging Face Hub (`scripts/upload_to_hf.py` and `scripts/push_parquet_as_dataset.py`).
	- Added dataset metadata and a short dataset README at `data/README_dataset.md` and `data/all_markets_15y_metadata.json` to make the artifact reproducible and discoverable.

These contributions expanded the project's data pipeline, improved robustness for large batch downloads, and produced a reusable market dataset for downstream analysis.

## Recent updates (2026-01-18)

- Fix: `models/helpers.py::convert_yf_data` now robustly handles yfinance outputs where selecting a single OHLC column may return a DataFrame instead of a Series. This prevents an AttributeError when converting to lists and makes data conversion tolerant to varying yfinance return shapes.
- New CLI: `scripts/run_symbol.py` — a small command-line runner that scans one or many tickers for impulsive wave patterns using the same analyzer logic as `example_12345_impulsive_wave.py`.

- New: Sliding adaptive-window scan: `WaveAnalyzer.sliding_adaptive_impulses` and
	`find_best_impulse_adaptive_window` (implemented in `models/WaveAnalyzer.py`).
	These helpers let the analyzer slide an adaptive time-window across a series
	and grow the window until an impulse candidate is found (configurable by
	weeks/bars). Typical parameters include `slide_weeks`, `min_weeks`,
	`max_weeks`, `grow_weeks`, `up_to` and `top_n`. The example script
	`scripts/example_12345_impulsive_wave.py` demonstrates usage and orchestration
	of the sliding-adaptive scan. Outputs are written to `images/` the same way as
	the single-run examples.

Usage examples

- Create and activate a Python 3.11 venv and install pinned dependencies (recommended):

```bash
cd /path/to/elliot-wave-analyzer-usamaahmedsh
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- Run the built-in examples (repo root on PYTHONPATH so local `models` package resolves):

```bash
# single symbol (defaults to AAPL when none provided)
PYTHONPATH=$(pwd) .venv/bin/python3 scripts/run_symbol.py AAPL

# multiple symbols
PYTHONPATH=$(pwd) .venv/bin/python3 scripts/run_symbol.py AAPL MSFT GOOGL

# read symbols from file (one per line)
PYTHONPATH=$(pwd) .venv/bin/python3 scripts/run_symbol.py -f tickers.txt

# adjust lookback and delay between symbols
PYTHONPATH=$(pwd) .venv/bin/python3 scripts/run_symbol.py AAPL --days 365 --delay 0.5
```

Notes

- The scanner writes plotting artifacts (PNG) and serialized payloads (JSON/CSV) into the `images/` directory. Each plotted pattern will also produce a `.json` and `.csv` file next to the image.
- `scripts/run_symbol.py` deduplicates symbols, uppercases them, and protects each symbol run with a try/except so a single failure doesn't stop a batch.
- If you prefer the older example, `scripts/example_12345_impulsive_wave.py` is still available and behaves the same for a single symbol.

### Soft scoring (confidence)

- The `WavePattern` class now exposes a heuristic scoring API to rank candidates:
	- `WavePattern.score_rule(waverule) -> float` returns a score in [0.0, 1.0]
		representing how well the pattern satisfies the supplied `WaveRule`.
	- Use the returned `score` to sort or filter multiple valid patterns (higher = more confidence).

### Exports / payload format

When a plot is generated the code now writes three files alongside one another:

- `<base>.png`  — chart image
- `<base>.json` — full payload describing the detected pattern
- `<base>.csv`  — flattened CSV with one row per wave (easy to ingest for ML)

JSON top-level keys (example):

- `symbol`, `timeframe`, `rule_name`, `score`, `pattern_type`, `degree`,
	`idx_start`, `idx_end`, `low`, `high`, `dates_polyline`, `values_polyline`,
	`labels_polyline`, `waves`

`waves` is a list of per-wave dictionaries with keys such as: `key`, `label`,
`idx_start`, `idx_end`, `date_start`, `date_end`, `low`, `high`, `low_idx`,
`high_idx`, `length`, `duration`.

CSV columns (one row per wave):

`symbol, timeframe, rule_name, score, pattern_type, degree, idx_start, idx_end, low, high,`
`wave_key, wave_label, wave_idx_start, wave_idx_end, wave_date_start, wave_date_end,`
`wave_low, wave_high, wave_low_idx, wave_high_idx, wave_length, wave_duration`

Quick example: load the flattened CSV from a run into pandas for ML preprocessing:

```python
import pandas as pd
df = pd.read_csv('images/20260118_123456_000001.csv')  # use the real filename generated
print(df.columns)
```

Implementation notes:

- Helpers that produce these exports live in `models/helpers.py` (`_serialize_wavepattern`,
	`_write_pattern_json_and_csv`, `_new_base_filename`, `save_chart_as_image`). These are small
	internal helpers but deliberately keep stable field names for downstream ML use.

--------------------------------------- ORIGINAL README.md ----------------------------------------------
## Setup
use Python 3.9 environment and install all packages via
`pip install -r requirements.txt`

## Quickstart
Start with `example_monowave.py` to see how the basic concept (finding monowaves) works and play with the parameter `skip_n`.

Then have a look into `example_12345_impulsive_wave.py` to see how the algorithm works for finding 12345 impulsive movements.

## Helper
Use `scripts/fetch_data.py` to download data directly from Yahoo Finance (historical OHLCV). A legacy copy also exists at `backups/get_data.py`.

## Tests
There are a couple of fast unit tests under the `tests/` directory to exercise basic fetch and monowave logic:

- `tests/test_fetch_data.py` — small sanity tests for the data fetcher/normalizer.
- `tests/test_monowave.py` — basic tests for monowave detection logic.

Run tests inside the recommended Python 3.11 venv with pytest:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

# Algorithm / Idea
The basic idea of the algorithm is to try **a lot** of combinations of possible wave
patterns for a given OHLC chart and validate each one against a given
set of rules (e.g. against an 12345 impulsive movement).

# Class Structure
## MonoWave
The smallest element in a chart (or a trend) is called a MonoWave: 
The impulsive movement from a given low (or high) to the next high 
(or down to the low), where each candle (exactly: high / low) 
forms a new high (or new low respectively). 

The MonoWave ends, once a candle breaks this "micro trend".

There is `MonoWaveUp` and the `MonoWaveDown`, denoting the direction of the wave.

### WaveOptions
`WaveOptions` are a set of integers denoting how many of the (local) highs or lows should be
skipped to form a MonoWave.

### Parameters
The essential idea is, that with the parameter `skip=`, smaller corrections can be skipped. In case of an upwards trend, 
e.g. `skip=2` will skip the next 2 maxima.

![](doc/img/monowave_skip.png)

## WavePattern
A `WavePattern` is the chaining of e.g. in case for an Impulse 5 `MonoWaves` (alternating between up and down direction). It is initialized with a list of `MonoWave`.

## WaveRule
`WavePattern` can be validated against a set of rules. E. g. form a valid 12345 impulsive waves, certain rules have to apply for the 
monowaves, e.g. wave 3 must not be the shortest wave, top of wave 3 must be over the top of wave 1 etc. 

Own rules can be created via inheritance from the base class. There are rules
implemented for 12345 Impulse. Leading Triangle and for ABC Corrections.

To create an own rule, the `.set_conditions()` method has to be implemented for every inherited rule. The method has a `dict`, having
arbitrarily named keys, having `{'waves': list 'function': ..., 'message': ...}` as value.

For `waves` you pass a list of waves which are used to validate a specific rule, e.g. `[wave1, wave2]`.

For `function` you use a `lambda` function to check, e.g. `lambda wave1, wave2: wave2.low > wave1.low`

For `message` you enter a message to display (in case `WavePattern(..., verbose=True)` is set).

Note that only if all rules in the `conditions` are `True` the whole `WaveRule` is valid.

### Check WavePattern against Rule
Once you have a `WavePattern` (chaining of 5 `MonoWave` for an impulse or 3 `MonoWave` for a correction)
 You can check against a `WaveRule` via the `.check_rule(waverule: WaveRule)` method.

## WaveCycle
A `WaveCycle` is the combination of an impulsive (12345) and a corrective (ABC) movement.
Not working atm.

## WaveAnalyzer
Is used to find impulsive and corrective movements.
Not working atm.

### WaveOptionsGenerator
There are three `WaveOptionsGenerators` available at the moment to fit the needs for creating
tuples of 2, 3 and 5 integers (for a 12 `TDWave`, an ABC `Correction` and a 12345 `Impulse`).

The generators already remove invalid combinations, e.g. [1,2,0,4,5], as after selecting the next minimum (3rd index is 0), for the 4th and 5th wave skipping is not allowed.

As unordered sets are used, the generators have the `.options_sorted` property to go from low numbers to high ones. This means that
first, the shortest (time wise) movements will be found.

## Helpers
Contains some plotting functions to plot a `MonoWave` (a single movement), a `WavePattern` (e.g. 12345 or ABC) and a `WaveCycle` (12345-ABC).

# Plotting
For different models there are plotting functions. E.g. use `plot_monowave` to plot a `MonoWave` instance or `plot_pattern` for a `WavePattern`.
