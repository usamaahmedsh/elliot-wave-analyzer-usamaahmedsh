Market dataset — 15 years, top-200 per market (merged)

This repository file contains a combined parquet dataset produced by the project's batch fetcher.

Filename: `all_markets_15y.parquet`
Produced: 2026-01-15
Rows: 1,134,666
Unique tickers: 315
Markets included: bond, commodity, continuous, crypto, equity, etf, fx_futures, index, international, reit, sector, volatility

Notes:
- This dataset was built by selecting top N=200 candidates per market where available and fetching daily OHLCV via yfinance.
- Some international tickers required retries; remaining missing tickers are documented in `data/market_manifest.json`.
- The dataset is daily-only (no intraday data).

License: please set a license when you create the Hugging Face dataset repo (e.g., CC-BY-4.0).