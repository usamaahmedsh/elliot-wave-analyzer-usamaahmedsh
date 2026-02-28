#!/usr/bin/env python3
"""
Comprehensive Market Data Fetcher for Elliott Wave Neural Network Training

Fetches diverse market data from multiple sources:
- Stocks: S&P 500, International stocks
- Crypto: Top cryptocurrencies  
- ETFs: Major sector and index ETFs
- Commodities: Gold, Silver, Oil, etc.
- Forex: Major currency pairs

Timeframes: 1h, 4h (synthetic), 1d, 1wk

Outputs to Parquet files and pushes to HuggingFace Hub.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_fetch.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# TICKER UNIVERSE - Comprehensive list of instruments
# ============================================================================

# Top Cryptocurrencies (yfinance format)
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "SOL-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
    "SHIB-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD",
    "UNI-USD", "XMR-USD", "ETC-USD", "XLM-USD", "BCH-USD",
    "ALGO-USD", "VET-USD", "MANA-USD", "SAND-USD", "AXS-USD",
    "AAVE-USD", "FTM-USD", "NEAR-USD", "THETA-USD", "FIL-USD",
    "ICP-USD", "HBAR-USD", "EGLD-USD", "XTZ-USD", "EOS-USD",
    "FLOW-USD", "CHZ-USD", "LRC-USD", "ENJ-USD", "GALA-USD",
    "ONE-USD", "ZEC-USD", "DASH-USD", "NEO-USD", "WAVES-USD",
    "KSM-USD", "CAKE-USD", "RUNE-USD", "ZIL-USD", "CRV-USD",
]

# Major ETFs covering different sectors and asset classes
ETFS = [
    # US Broad Market
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV",
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
    # Tech focused
    "ARKK", "ARKW", "ARKF", "ARKG", "ARKQ", "SOXX", "SMH", "IGV", "FTEC",
    # International
    "EFA", "EEM", "VEA", "VWO", "IEFA", "IEMG", "FXI", "EWJ", "EWZ", "EWG",
    "EWU", "EWC", "EWA", "EWH", "EWS", "EWT", "EWY", "EWQ", "EWI", "EWP",
    # Fixed Income
    "TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "JNK", "TIP", "MUB",
    # Commodities
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GDX", "GDXJ", "XME",
    # Real Estate
    "VNQ", "IYR", "SCHH", "RWR", "REET",
    # Volatility
    "VIXY", "UVXY", "VXX",
    # Leveraged/Inverse (for pattern diversity)
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SPDN",
]

# Forex pairs (via yfinance)
FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "NZDUSD=X", "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "AUDJPY=X", "CHFJPY=X", "CADJPY=X", "NZDJPY=X", "EURAUD=X",
    "EURNZD=X", "EURCHF=X", "EURCAD=X", "GBPAUD=X", "GBPNZD=X",
    "GBPCHF=X", "GBPCAD=X", "AUDNZD=X", "AUDCHF=X", "AUDCAD=X",
]

# Commodities Futures (via yfinance)
COMMODITIES = [
    "GC=F",  # Gold
    "SI=F",  # Silver
    "CL=F",  # Crude Oil
    "NG=F",  # Natural Gas
    "HG=F",  # Copper
    "PL=F",  # Platinum
    "PA=F",  # Palladium
    "ZC=F",  # Corn
    "ZS=F",  # Soybeans
    "ZW=F",  # Wheat
    "KC=F",  # Coffee
    "SB=F",  # Sugar
    "CC=F",  # Cocoa
    "CT=F",  # Cotton
    "LE=F",  # Live Cattle
]

# Indices (via yfinance)
INDICES = [
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones
    "^IXIC",   # NASDAQ
    "^RUT",    # Russell 2000
    "^VIX",    # VIX
    "^FTSE",   # FTSE 100
    "^GDAXI",  # DAX
    "^FCHI",   # CAC 40
    "^N225",   # Nikkei 225
    "^HSI",    # Hang Seng
    "^STI",    # Straits Times
    "^AXJO",   # ASX 200
    "^BVSP",   # Bovespa
    "^MXX",    # IPC Mexico
    "^STOXX50E",  # Euro Stoxx 50
]

# Timeframe configurations
TIMEFRAME_CONFIG = {
    "1h": {"interval": "1h", "period": "730d", "description": "Hourly (2 years)"},
    "1d": {"interval": "1d", "period": "max", "description": "Daily (max history)"},
    "1wk": {"interval": "1wk", "period": "max", "description": "Weekly (max history)"},
}


def load_sp500_tickers(filepath: str = "data/sp500_tickers.txt") -> List[str]:
    """Load S&P 500 tickers from file."""
    try:
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.warning(f"SP500 file not found: {filepath}")
        return []


def load_international_tickers(filepath: str = "data/international_tickers.txt") -> List[str]:
    """Load international tickers from file."""
    try:
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.warning(f"International tickers file not found: {filepath}")
        return []


def fetch_single_ticker(
    ticker: str, 
    interval: str, 
    period: str,
    retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single ticker with retries.
    
    Args:
        ticker: Ticker symbol
        interval: Timeframe interval (1h, 1d, 1wk)
        period: Historical period to fetch
        retries: Number of retry attempts
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    for attempt in range(retries):
        try:
            # Use yfinance download with proper settings
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                prepost=False,
                threads=False,  # Single-threaded per call
            )
            
            if df.empty:
                return None
            
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                return None
            
            # Keep only OHLCV
            df = df[required].copy()
            
            # Remove any NaN rows
            df = df.dropna()
            
            # Add metadata
            df['ticker'] = ticker
            df['interval'] = interval
            
            # Reset index to make datetime a column
            df = df.reset_index()
            df.rename(columns={'index': 'datetime', 'Date': 'datetime', 'Datetime': 'datetime'}, inplace=True)
            
            # Ensure datetime column exists
            if 'datetime' not in df.columns:
                df = df.reset_index()
                df.columns = ['datetime'] + list(df.columns[1:])
            
            return df
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            logger.debug(f"Failed to fetch {ticker} ({interval}): {e}")
            return None
    
    return None


def resample_to_4h(df_1h: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Resample 1h data to 4h timeframe.
    
    Args:
        df_1h: DataFrame with 1h OHLCV data
        
    Returns:
        DataFrame with 4h OHLCV data
    """
    if df_1h is None or df_1h.empty:
        return None
    
    try:
        df = df_1h.copy()
        ticker = df['ticker'].iloc[0]
        
        # Set datetime as index for resampling
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        # Resample OHLCV to 4h
        df_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if df_4h.empty:
            return None
        
        # Add metadata
        df_4h['ticker'] = ticker
        df_4h['interval'] = '4h'
        df_4h = df_4h.reset_index()
        
        return df_4h
        
    except Exception as e:
        logger.debug(f"Failed to resample to 4h: {e}")
        return None


def fetch_ticker_all_timeframes(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch all timeframes for a single ticker.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        Dict mapping timeframe to DataFrame
    """
    results = {}
    
    # Fetch each base timeframe
    for tf_name, tf_config in TIMEFRAME_CONFIG.items():
        df = fetch_single_ticker(ticker, tf_config["interval"], tf_config["period"])
        if df is not None and len(df) >= 50:  # Minimum 50 bars
            results[tf_name] = df
    
    # Create 4h by resampling 1h
    if "1h" in results:
        df_4h = resample_to_4h(results["1h"])
        if df_4h is not None and len(df_4h) >= 50:
            results["4h"] = df_4h
    
    return results


def fetch_all_data(
    tickers: List[str],
    category: str,
    output_dir: Path,
    max_workers: int = 5,
    progress_callback=None
) -> Dict[str, int]:
    """
    Fetch data for all tickers in a category.
    
    Args:
        tickers: List of ticker symbols
        category: Category name (e.g., "stocks", "crypto")
        output_dir: Output directory for Parquet files
        max_workers: Number of parallel workers
        progress_callback: Optional callback for progress updates
        
    Returns:
        Statistics dict
    """
    stats = {"total": len(tickers), "success": 0, "failed": 0, "bars": {}}
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = {tf: [] for tf in ["1h", "4h", "1d", "1wk"]}
    
    logger.info(f"Fetching {category}: {len(tickers)} tickers...")
    
    # Process tickers with rate limiting
    for i, ticker in enumerate(tickers):
        try:
            data = fetch_ticker_all_timeframes(ticker)
            
            if data:
                stats["success"] += 1
                for tf, df in data.items():
                    all_data[tf].append(df)
                    if tf not in stats["bars"]:
                        stats["bars"][tf] = 0
                    stats["bars"][tf] += len(df)
            else:
                stats["failed"] += 1
                
        except Exception as e:
            stats["failed"] += 1
            logger.debug(f"Error fetching {ticker}: {e}")
        
        # Rate limiting: small delay between tickers
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
            
        # Progress update
        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(category, i + 1, len(tickers))
    
    # Save combined DataFrames as Parquet
    for tf, dfs in all_data.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            output_path = category_dir / f"{category}_{tf}.parquet"
            combined.to_parquet(output_path, engine="pyarrow", compression="snappy")
            logger.info(f"Saved {output_path}: {len(combined)} rows, {combined['ticker'].nunique()} tickers")
    
    return stats


def upload_to_huggingface(
    data_dir: Path,
    repo_name: str = "elliott-wave-market-data",
    private: bool = False
) -> bool:
    """
    Upload dataset to HuggingFace Hub.
    
    Args:
        data_dir: Directory containing Parquet files
        repo_name: Name of the HuggingFace dataset repository
        private: Whether to make the dataset private
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi, create_repo
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN not found in environment variables")
            return False
        
        api = HfApi(token=hf_token)
        
        # Get username
        user_info = api.whoami()
        username = user_info.get("name", user_info.get("fullname", "user"))
        repo_id = f"{username}/{repo_name}"
        
        logger.info(f"Creating/updating HuggingFace repo: {repo_id}")
        
        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=hf_token,
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Repo creation note: {e}")
        
        # Upload all Parquet files
        for parquet_file in data_dir.rglob("*.parquet"):
            relative_path = parquet_file.relative_to(data_dir)
            logger.info(f"Uploading {relative_path}...")
            
            api.upload_file(
                path_or_fileobj=str(parquet_file),
                path_in_repo=str(relative_path),
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
        
        # Create and upload README
        readme_content = generate_dataset_readme(data_dir)
        readme_path = data_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        
        # Upload metadata JSON
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            api.upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo="metadata.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
        
        logger.info(f"✅ Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error(f"Failed to upload to HuggingFace: {e}")
        return False


def generate_dataset_readme(data_dir: Path) -> str:
    """Generate README.md for the HuggingFace dataset."""
    
    # Collect statistics
    stats = {}
    total_rows = 0
    total_tickers = set()
    
    for parquet_file in data_dir.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            category = parquet_file.parent.name
            tf = parquet_file.stem.split("_")[-1]
            
            if category not in stats:
                stats[category] = {}
            
            stats[category][tf] = {
                "rows": len(df),
                "tickers": df['ticker'].nunique() if 'ticker' in df.columns else 0
            }
            total_rows += len(df)
            if 'ticker' in df.columns:
                total_tickers.update(df['ticker'].unique())
                
        except Exception as e:
            logger.debug(f"Error reading {parquet_file}: {e}")
    
    readme = f"""---
license: mit
task_categories:
  - time-series-forecasting
tags:
  - finance
  - stocks
  - crypto
  - forex
  - commodities
  - elliott-wave
  - technical-analysis
  - ohlcv
pretty_name: Elliott Wave Market Data
size_categories:
  - 1M<n<10M
---

# Elliott Wave Market Data

Comprehensive OHLCV market data for training Elliott Wave pattern recognition neural networks.

## Dataset Description

This dataset contains historical OHLCV (Open, High, Low, Close, Volume) data across multiple asset classes and timeframes, specifically curated for Elliott Wave analysis and pattern recognition.

### Asset Classes
- **Stocks**: S&P 500 components and international equities
- **Crypto**: Top 50 cryptocurrencies by market cap
- **ETFs**: Sector ETFs, index ETFs, commodity ETFs
- **Forex**: Major and cross currency pairs
- **Commodities**: Precious metals, energy, agriculture futures
- **Indices**: Major global market indices

### Timeframes
- **1h**: Hourly data (up to 2 years history)
- **4h**: 4-hour data (resampled from 1h)
- **1d**: Daily data (maximum available history)
- **1wk**: Weekly data (maximum available history)

## Dataset Statistics

- **Total Rows**: {total_rows:,}
- **Total Unique Tickers**: {len(total_tickers):,}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

### By Category

| Category | 1h Bars | 4h Bars | 1d Bars | 1wk Bars | Tickers |
|----------|---------|---------|---------|----------|---------|
"""
    
    for category in sorted(stats.keys()):
        row = f"| {category} |"
        tickers = 0
        for tf in ["1h", "4h", "1d", "1wk"]:
            if tf in stats[category]:
                row += f" {stats[category][tf]['rows']:,} |"
                tickers = max(tickers, stats[category][tf].get('tickers', 0))
            else:
                row += " - |"
        row += f" {tickers} |"
        readme += row + "\n"
    
    readme += """
## Data Schema

Each Parquet file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| datetime | datetime64 | Timestamp of the bar |
| open | float64 | Opening price |
| high | float64 | Highest price |
| low | float64 | Lowest price |
| close | float64 | Closing price |
| volume | float64 | Trading volume |
| ticker | string | Ticker symbol |
| interval | string | Timeframe interval |

## Usage

```python
import pandas as pd
from datasets import load_dataset

# Load specific file
df = pd.read_parquet("hf://datasets/YOUR_USERNAME/elliott-wave-market-data/stocks/stocks_1d.parquet")

# Or load all data
dataset = load_dataset("YOUR_USERNAME/elliott-wave-market-data")
```

## Data Sources

- Primary source: Yahoo Finance (via yfinance)
- All data is adjusted for splits and dividends

## License

MIT License - Free for academic and commercial use.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{elliott_wave_market_data,
  title={Elliott Wave Market Data},
  author={Elliott Wave Analyzer},
  year={2025},
  publisher={HuggingFace}
}
```
"""
    
    return readme


def progress_callback(category: str, current: int, total: int):
    """Print progress updates."""
    pct = (current / total) * 100
    logger.info(f"[{category}] Progress: {current}/{total} ({pct:.1f}%)")


def main():
    """Main entry point for data fetching."""
    start_time = time.time()
    
    # Output directory
    output_dir = Path("data/hf_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ELLIOTT WAVE MARKET DATA FETCHER")
    logger.info("=" * 60)
    
    all_stats = {}
    
    # 1. Fetch S&P 500 Stocks
    sp500_tickers = load_sp500_tickers()
    if sp500_tickers:
        logger.info(f"\n📈 S&P 500 Stocks: {len(sp500_tickers)} tickers")
        all_stats["stocks"] = fetch_all_data(
            sp500_tickers, "stocks", output_dir, 
            progress_callback=progress_callback
        )
    
    # 2. Fetch International Stocks
    intl_tickers = load_international_tickers()
    if intl_tickers:
        logger.info(f"\n🌍 International Stocks: {len(intl_tickers)} tickers")
        all_stats["international"] = fetch_all_data(
            intl_tickers, "international", output_dir,
            progress_callback=progress_callback
        )
    
    # 3. Fetch Cryptocurrencies
    logger.info(f"\n₿ Cryptocurrencies: {len(CRYPTO_TICKERS)} tickers")
    all_stats["crypto"] = fetch_all_data(
        CRYPTO_TICKERS, "crypto", output_dir,
        progress_callback=progress_callback
    )
    
    # 4. Fetch ETFs
    logger.info(f"\n📊 ETFs: {len(ETFS)} tickers")
    all_stats["etfs"] = fetch_all_data(
        ETFS, "etfs", output_dir,
        progress_callback=progress_callback
    )
    
    # 5. Fetch Forex
    logger.info(f"\n💱 Forex Pairs: {len(FOREX_PAIRS)} pairs")
    all_stats["forex"] = fetch_all_data(
        FOREX_PAIRS, "forex", output_dir,
        progress_callback=progress_callback
    )
    
    # 6. Fetch Commodities
    logger.info(f"\n🛢️ Commodities: {len(COMMODITIES)} tickers")
    all_stats["commodities"] = fetch_all_data(
        COMMODITIES, "commodities", output_dir,
        progress_callback=progress_callback
    )
    
    # 7. Fetch Indices
    logger.info(f"\n📉 Indices: {len(INDICES)} indices")
    all_stats["indices"] = fetch_all_data(
        INDICES, "indices", output_dir,
        progress_callback=progress_callback
    )
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "stats": all_stats,
        "timeframes": list(TIMEFRAME_CONFIG.keys()) + ["4h"],
        "categories": list(all_stats.keys()),
        "total_tickers": {
            "sp500": len(sp500_tickers),
            "international": len(intl_tickers),
            "crypto": len(CRYPTO_TICKERS),
            "etfs": len(ETFS),
            "forex": len(FOREX_PAIRS),
            "commodities": len(COMMODITIES),
            "indices": len(INDICES),
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 60)
    
    total_success = sum(s.get("success", 0) for s in all_stats.values())
    total_failed = sum(s.get("failed", 0) for s in all_stats.values())
    total_bars = sum(
        sum(s.get("bars", {}).values()) 
        for s in all_stats.values()
    )
    
    logger.info(f"Total tickers fetched: {total_success}")
    logger.info(f"Total tickers failed: {total_failed}")
    logger.info(f"Total data bars: {total_bars:,}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Output directory: {output_dir}")
    
    # Upload to HuggingFace
    logger.info("\n" + "=" * 60)
    logger.info("UPLOADING TO HUGGINGFACE")
    logger.info("=" * 60)
    
    success = upload_to_huggingface(output_dir)
    
    if success:
        logger.info("🎉 All done! Dataset available on HuggingFace Hub.")
    else:
        logger.warning("⚠️ HuggingFace upload failed. Data saved locally.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
