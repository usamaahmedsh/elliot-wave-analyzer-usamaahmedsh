"""
fetch_data.py

Unified fetcher for free data sources with provider fallbacks.
Supports:
 - Binance (crypto, high-resolution intraday)
 - CoinGecko (crypto historical)
 - Alpha Vantage (stocks/FX/crypto; requires ALPHAVANTAGE_API_KEY env var)
 - Stooq (stocks daily; via pandas-datareader)
 - yfinance (Yahoo fallback)

Usage examples:
  # fetch recent BTC/USDT 1m candles (Binance)
  python fetch_data.py --symbol BTCUSDT --provider binance --interval 1m --limit 500

  # fetch daily AAPL using Alpha Vantage (needs env ALPHAVANTAGE_API_KEY)
  python fetch_data.py --symbol AAPL --provider alpha --interval daily

The script returns a pandas DataFrame with Date/Time index and OHLCV columns.
Note: this tool operates in daily-only mode (1d). Intraday/minute data has been removed.
"""

import os
import sys
import time
import argparse
from datetime import datetime
import requests
import pandas as pd
import concurrent.futures
from pathlib import Path
import math
import json

# optional imports guarded by try/except
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

try:
    import yfinance as yf
except Exception:
    yf = None


def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 500):
    """Fetch klines/candles from Binance public REST API.
    symbol: e.g. BTCUSDT, ETHUSDT
    interval: 1m,3m,5m,15m,30m,1h,4h,1d
    limit: number of candles (max 1000)
    """
    base = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
    r = requests.get(base + endpoint, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    # each kline: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("open_time", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "Date"
    return df


def fetch_binance_klines_full(symbol: str, interval: str = "1d"):
    """Fetch full available klines from Binance by paging backwards in time.
    This requests the most recent batch (limit=1000) then repeatedly requests
    earlier batches using endTime until no more data is returned.
    Returns a DataFrame sorted ascending by Date.
    """
    cache_dir = Path('cache') / 'binance'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol.upper()}_{interval}.parquet"
    # if cache exists, return cached data to avoid re-downloading
    if cache_file.exists():
        try:
            df_cached = pd.read_parquet(cache_file)
            df_cached.index = pd.to_datetime(df_cached.index)
            return df_cached.sort_index()
        except Exception:
            pass
    base = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    limit = 1000
    all_frames = []
    end_time = None
    while True:
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(base + endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("open_time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "Date"
        all_frames.append(df)
        # prepare next page: set end_time to earliest open_time - 1 ms
        earliest = df.index.min()
        end_time = int(earliest.tz_localize(None).timestamp() * 1000) - 1
        # be polite to the API
        time.sleep(0.34)
    if not all_frames:
        return pd.DataFrame()
    full = pd.concat(reversed(all_frames)) if len(all_frames) > 1 else all_frames[0]
    full = full[~full.index.duplicated()]
    full = full.sort_index()
    # save cache
    try:
        full.to_parquet(cache_file)
    except Exception:
        pass
    return full


def fetch_coingecko_history(coin_id: str, vs_currency: str = "usd", days: int = 365):
    """Fetch CoinGecko market_chart data (prices/market_caps/total_volumes).
    coin_id: e.g. 'bitcoin', 'ethereum'
    days: number of days (or 'max')
    Returns prices as DataFrame with Date index and price column.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    prices = j.get("prices", [])
    if not prices:
        raise ValueError("No price data returned from CoinGecko")
    df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
    df["Date"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df.set_index("Date", inplace=True)
    return df[["price"]]


# Alpha Vantage removed as a provider (prefer yfinance for equities)


def fetch_stooq(symbol: str):
    """Fetch daily data from Stooq via pandas-datareader (no API key).
    symbol examples: 'aapl.us' or 'btc.us' (stooq uses suffixes)
    """
    if pdr is None:
        raise RuntimeError("pandas-datareader is not installed")
    df = pdr.DataReader(symbol, "stooq")
    df = df.sort_index()
    df.index.name = "Date"
    return df


def fetch_fred(symbol: str):
    """Fetch series from FRED (requires FRED_API_KEY in env). Returns a DataFrame with Date index."""
    if pdr is None:
        raise RuntimeError("pandas-datareader is not installed")
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        raise RuntimeError('FRED_API_KEY not found in environment')
    df = pdr.DataReader(symbol, 'fred', api_key=api_key)
    df = df.sort_index()
    df.index.name = 'Date'
    return df


def fetch_yfinance(symbol: str, interval: str = "1d", period: str | None = None):
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    tk = yf.Ticker(symbol)
    if period:
        df = tk.history(period=period, interval=interval)
    else:
        df = tk.history(interval=interval)
    if df is None or df.empty:
        raise ValueError("No data returned from yfinance")
    df.index.name = "Date"
    return df


######### New: yfinance-only, batch fetching utilities #########


def clean_previous_csvs(data_dir: str = 'data'):
    """Remove existing CSV files in data_dir to clear previous test fetches."""
    p = Path(data_dir)
    if not p.exists():
        return
    removed = []
    for f in p.glob('*.csv'):
        try:
            f.unlink()
            removed.append(str(f))
        except Exception:
            continue
    return removed


def get_top_crypto_coins(n: int = 100):
    """Discover candidate crypto tickers and return top-n tickers by average volume using yfinance.
    Since we operate yfinance-only, we maintain a static candidate list of common crypto tickers
    (Yahoo format like BTC-USD). We then rank them by recent volume and return the top-n.
    """
    # Expanded curated candidate list of common crypto tickers on Yahoo Finance (format: SYMBOL-USD).
    # This list is intentionally large so yfinance ranking can pick the true top-N by volume.
    candidates = [
        'BTC-USD','ETH-USD','USDT-USD','BNB-USD','XRP-USD','ADA-USD','SOL-USD','DOGE-USD','DOT-USD','MATIC-USD',
        'LTC-USD','BCH-USD','LINK-USD','TRX-USD','ETC-USD','XLM-USD','FIL-USD','NEAR-USD','APT-USD','AVAX-USD',
        'SHIB-USD','ATOM-USD','EGLD-USD','SAND-USD','AAVE-USD','MKR-USD','FTM-USD','XTZ-USD','KSM-USD','CHZ-USD',
        'GRT-USD','RUNE-USD','ALGO-USD','CRV-USD','QNT-USD','ZEC-USD','XMR-USD','FTT-USD','BSV-USD','OKB-USD',
        'NEO-USD','BAT-USD','ENJ-USD','COMP-USD','CEL-USD','ZIL-USD','VET-USD','DASH-USD','WAVES-USD','KLAY-USD',
        'RPL-USD','SXP-USD','MANA-USD','ICX-USD','NEXO-USD','LRC-USD','1INCH-USD','HNT-USD','ANKR-USD','ONT-USD',
        'NANO-USD','ZRX-USD','AR-USD','CHSB-USD','HOT-USD','BSW-USD','KAVA-USD','TON-USD','LEO-USD','GNO-USD',
        'ZIL-USD','OCEAN-USD','SUSHI-USD','YFI-USD','CELR-USD','RAY-USD','FLOW-USD','BAT-USD','OMG-USD','STX-USD',
        'QTUM-USD','PAXG-USD','BTT-USD','KNC-USD','ANKR-USD','ANKR-USD','IOTA-USD','DCR-USD','CELO-USD','WAXP-USD',
        'XEM-USD','ENS-USD','ARPA-USD','KSM-USD','KAVA-USD','CHSB-USD','SXP-USD','FET-USD','GALA-USD','MINA-USD',
        'HNT-USD','BTG-USD','SC-USD','ZRX-USD','RLC-USD','KDA-USD','AMP-USD','ICP-USD','MOVR-USD','BDX-USD',
        'GMX-USD','CRO-USD','SAPE-USD','XCH-USD','ERD-USD','ANKR-USD','SNX-USD','CVX-USD','LDO-USD','DYDX-USD',
        'NEAR-USD','IMX-USD','OP-USD','ARB-USD','PEPE-USD','WBTC-USD','WETH-USD','USDC-USD','LUNC-USD','CEL-USD'
    ]
    # ensure we have enough candidates; if n > len(candidates), we still rank what's available
    ranked = rank_equities_by_volume(candidates, lookback='1mo', top_n=n, chunk_size=50)
    return ranked[:n]


def get_sp500_tickers():
    """Scrape Wikipedia S&P 500 constituents table to get tickers list."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # normalize tickers for yfinance (dots -> - sometimes)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception:
        # fallback: read bundled list in data/sp500_tickers.txt if present
        fallback = Path('data') / 'sp500_tickers.txt'
        if fallback.exists():
            try:
                with fallback.open() as fh:
                    lines = [l.strip() for l in fh.readlines() if l.strip()]
                return [t.replace('.', '-').upper() for t in lines]
            except Exception:
                return []
        return []


def rank_equities_by_volume(tickers: list, lookback: str = '1mo', top_n: int = 100, chunk_size: int = 100):
    """Batch download recent data for tickers and rank by average daily volume. Returns top_n tickers.
    Uses yfinance.download with threads for performance.
    """
    if not tickers:
        return []
    # work in chunks to avoid huge single requests
    avg_vol = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            # group_by='ticker' gives a dict-like panel when multiple tickers
            data = yf.download(chunk, period=lookback, interval='1d', group_by='ticker', threads=True, progress=False)
        except Exception:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            for t in chunk:
                try:
                    sub = data[t]
                    if 'Volume' in sub.columns:
                        vols = sub['Volume'].dropna()
                        if not vols.empty:
                            avg_vol[t] = float(vols.mean())
                except Exception:
                    continue
        else:
            # single ticker returned
            if 'Volume' in data.columns:
                vols = data['Volume'].dropna()
                if not vols.empty:
                    avg_vol[chunk[0]] = float(vols.mean())
    # sort and return top_n
    ranked = sorted(avg_vol.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, v in ranked[:top_n]]


def _ohlc_fill(sub: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Open/High/Low values using Close when available.
    Modifies a copy and returns it.
    """
    if sub is None or sub.empty:
        return sub
    sub = sub.copy()
    if 'Close' in sub.columns:
        for col in ('Open', 'High', 'Low'):
            if col in sub.columns:
                sub[col] = sub[col].where(sub[col].notna(), sub['Close'])
    return sub


def batch_fetch_yfinance(tickers: list, start: str = None, end: str = None, threads: int = 8, chunk_size: int = 100):
    """Fetch daily history for a list of tickers using parallel batch downloads and return a long-form DataFrame.
    Returns DataFrame with columns: ['ticker','Date','Open','High','Low','Close','Volume']
    """
    out_rows = []
    failures = {}
    cache_dir = Path('cache') / 'tickers'
    cache_dir.mkdir(parents=True, exist_ok=True)
    def fetch_chunk(chunk):
        # for each ticker in chunk, use cache if it satisfies the requested range; otherwise include in download list
        to_dl = []
        result = {}
        for t in chunk:
            cache_file = cache_dir / f"{t.replace('/','_').replace('^','').replace(':','_')}.parquet"
            if cache_file.exists():
                try:
                    cached = pd.read_parquet(cache_file)
                    if 'Date' in cached.columns:
                        cached['Date'] = pd.to_datetime(cached['Date'])
                        min_date = cached['Date'].min()
                        max_date = cached['Date'].max()
                        use_cache = True
                        if start:
                            try:
                                sdt = pd.to_datetime(start)
                                if sdt < min_date:
                                    use_cache = False
                            except Exception:
                                use_cache = False
                        if end:
                            try:
                                edt = pd.to_datetime(end)
                                if edt > max_date:
                                    use_cache = False
                            except Exception:
                                use_cache = False
                        if use_cache:
                            # cached covers requested range
                            cached = cached[['ticker','Date','Open','High','Low','Close','Volume']]
                            result[t] = cached
                            continue
                except Exception:
                    # broken cache -> ignore and re-download
                    pass
            to_dl.append(t)

        if to_dl:
            try:
                df = yf.download(to_dl, start=start, end=end, interval='1d', group_by='ticker', threads=True, progress=False)
            except Exception as e:
                # record failure for this chunk
                for t in to_dl:
                    failures[t] = str(e)
                df = None

            if df is not None:
                if isinstance(df.columns, pd.MultiIndex):
                    for t in to_dl:
                        try:
                            sub = df[t].dropna(how='all')
                            if sub.empty:
                                failures[t] = 'empty'
                                continue
                            sub = sub[['Open','High','Low','Close','Volume']]
                            sub = sub.reset_index()
                            sub['ticker'] = t
                            sub = _ohlc_fill(sub)
                            result[t] = sub
                            # write per-ticker cache
                            try:
                                result[t].to_parquet(cache_dir / f"{t.replace('/','_').replace('^','').replace(':','_')}.parquet", index=False)
                            except Exception:
                                pass
                        except Exception as e:
                            failures[t] = str(e)
                else:
                    # single ticker returned
                    try:
                        t = to_dl[0]
                        sub = df[['Open','High','Low','Close','Volume']].dropna(how='all').reset_index()
                        sub['ticker'] = t
                        sub = _ohlc_fill(sub)
                        result[t] = sub
                        try:
                            result[t].to_parquet(cache_dir / f"{t.replace('/','_').replace('^','').replace(':','_')}.parquet", index=False)
                        except Exception:
                            pass
                    except Exception as e:
                        failures[to_dl[0]] = str(e)

        return result

    # parallel execution of chunked downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        futures = []
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            futures.append(ex.submit(fetch_chunk, chunk))
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            for t, sub in res.items():
                out_rows.append(sub)
    # failures collected in out_rows absence
    if not out_rows:
        return pd.DataFrame(columns=['ticker','Date','Open','High','Low','Close','Volume'])
    df_all = pd.concat(out_rows, ignore_index=True)
    # normalize column names
    df_all.rename(columns={'index':'Date'}, inplace=True)
    # ensure Date column is datetime
    if 'Date' in df_all.columns:
        df_all['Date'] = pd.to_datetime(df_all['Date'])
    # reorder
    cols = ['ticker','Date','Open','High','Low','Close','Volume']
    for c in cols:
        if c not in df_all.columns:
            df_all[c] = pd.NA
    # persist failures if any
    try:
        if failures:
            write_failure_report(failures)
    except Exception:
        pass
    return df_all[cols]


def write_out(df: pd.DataFrame, out_path: str, fmt: str = 'parquet'):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'parquet':
        df.to_parquet(p, index=False)
    else:
        # write json lines
        with p.open('w', encoding='utf8') as fh:
            for _, row in df.iterrows():
                fh.write(json.dumps({k: (None if pd.isna(v) else (v.isoformat() if isinstance(v, pd.Timestamp) else v)) for k, v in row.items()}, default=str))
                fh.write('\n')


def write_failure_report(failures: dict, out_path: str = 'data/fetch_failures.json'):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open('w', encoding='utf8') as fh:
            json.dump(failures, fh, indent=2, default=str)
    except Exception:
        pass



def normalize_symbol_for_binance(symbol: str) -> str:
    # try to convert common forms to Binance symbol
    s = symbol.replace('-', '').replace('/', '')
    # if symbol ends with 'USD' and not 'USDT', prefer USDT pair
    if s.upper().endswith('USD') and not s.upper().endswith('USDT'):
        base = s[:-3]
        return base.upper() + 'USDT'
    return s.upper()


def detect_market_type(symbol: str) -> str:
    """Heuristic detector for market type: 'crypto', 'equity', 'commodity', 'bond', or 'unknown'.
    - crypto: contains 'USDT'/'USDC' or a dash like BTC-USD
    - commodity: common futures tickers (ends with =F) or contains 'F=' style (e.g. GC=F)
    - bond: common yield tickers (starts with '^') or contains 'TY'/'TNX'
    - equity: otherwise assume equity when symbol is alphabetic / contains dots
    This is a heuristic to choose provider priority; users can still pass --provider.
    """
    s = symbol.strip()
    up = s.upper()
    # crypto heuristics
    if 'USDT' in up or 'USDC' in up or '-' in s or '/' in s:
        return 'crypto'
    # commodity futures on Yahoo like GC=F, CL=F
    if '=' in s and s.endswith('F'):
        return 'commodity'
    # bonds / yields (Yahoo uses ^TNX, ^IRX etc)
    if s.startswith('^') or 'TNX' in up or 'TY' in up:
        return 'bond'
    # default to equity if alpha characters present
    if any(c.isalpha() for c in s):
        return 'equity'
    return 'unknown'


# common commodity ticker variants to try when Yahoo or Stooq fails
COMMODITY_VARIANTS = [
    # common Yahoo futures format and simple variations
    lambda s: s,
    lambda s: s.replace('=', '.'),
    lambda s: s.replace('=',''),
    lambda s: s.split('=')[0] if '=' in s else s,
]


def try_variants_fetch(fetch_fn, symbol, variants):
    """Try fetch_fn(symbol_variant) for each variant in variants and return first non-empty DataFrame."""
    last_exc = None
    for fn in variants:
        try:
            sym = fn(symbol)
            df = fetch_fn(sym)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    return pd.DataFrame()


def get_data(symbol: str, provider_priority=None, **kwargs):
    """Unified getter. provider_priority is a list like ['binance','coingecko','alpha','stooq','yahoo']
    kwargs forwarded to provider functions.
    """
    # if the user supplied an explicit priority list, use it
    if provider_priority is not None:
        # allow single-provider strings
        if isinstance(provider_priority, str):
            provider_priority = [provider_priority]
    else:
        # auto-select default provider order based on detected market type
        mtype = kwargs.get('market_type') or detect_market_type(symbol)
        if mtype == 'crypto':
            provider_priority = ['binance', 'coingecko', 'yahoo', 'stooq']
        elif mtype == 'commodity':
            provider_priority = ['yahoo', 'stooq']
        elif mtype == 'bond':
            # try FRB / FRED via pandas-datareader if available, then Yahoo
            if os.environ.get('FRED_API_KEY') and pdr is not None:
                provider_priority = ['fred', 'stooq', 'yahoo']
            else:
                provider_priority = ['stooq', 'yahoo']
        else:
            # default: equities and unknown -> Stooq then Yahoo
            provider_priority = ['stooq', 'yahoo']

    last_exc = None
    for provider in provider_priority:
        try:
            # if user requested full history via yfinance --period max, prefer yahoo early
            if kwargs.get('period') == 'max' and provider != 'yahoo' and 'yahoo' in provider_priority:
                # move yahoo to front for long-history requests
                provider_priority = ['yahoo'] + [p for p in provider_priority if p != 'yahoo']
                # restart loop with updated order
                last_exc = None
                break
            if provider == 'binance':
                sym = normalize_symbol_for_binance(symbol)
                print(f"Trying Binance for {sym}")
                # if user asked for full history, use paginated fetch
                if kwargs.get('period') == 'max':
                    df = fetch_binance_klines_full(sym, interval=kwargs.get('interval', '1d'))
                else:
                    df = fetch_binance_klines(sym, interval=kwargs.get('interval', '1d'), limit=kwargs.get('limit', 500))
                return df
            elif provider == 'coingecko':
                coin = kwargs.get('coin_id') or symbol.lower()
                print(f"Trying CoinGecko for {coin}")
                df = fetch_coingecko_history(coin_id=coin, vs_currency=kwargs.get('vs', 'usd'), days=kwargs.get('days', 365))
                return df
            # alpha provider intentionally removed; prefer yfinance ("yahoo") for equities
            elif provider == 'stooq':
                print(f"Trying Stooq for {symbol}")
                # attempt common suffixes and commodity variants
                if detect_market_type(symbol) == 'commodity':
                    df = try_variants_fetch(fetch_stooq, symbol, COMMODITY_VARIANTS)
                else:
                    try:
                        df = fetch_stooq(symbol)
                    except Exception:
                        df = fetch_stooq(symbol + '.us')
                return df
            elif provider == 'yahoo':
                print(f"Trying yfinance for {symbol}")
                # for commodities, try a few symbol variants if the direct ticker fails
                if detect_market_type(symbol) == 'commodity':
                    df = try_variants_fetch(lambda s: fetch_yfinance(s, interval=kwargs.get('interval', '1d'), period=kwargs.get('period')), symbol, COMMODITY_VARIANTS)
                else:
                    df = fetch_yfinance(symbol, interval=kwargs.get('interval', '1d'), period=kwargs.get('period'))
                return df
        except Exception as e:
            last_exc = e
            print(f"Provider {provider} failed: {e}")
            continue
    raise RuntimeError(f"All providers failed. Last error: {last_exc}")


def main():
    parser = argparse.ArgumentParser(description="Fetch market data (daily-only) — supports single-symbol and batch top-N modes")
    sub = parser.add_subparsers(dest='command')

    # single symbol (legacy) mode
    p1 = sub.add_parser('single', help='Fetch single symbol (daily)')
    p1.add_argument('--symbol', required=True, help='Symbol to fetch (e.g. AAPL, GC=F, BTC-USD)')
    p1.add_argument('--provider', help='Provider name (ignored — uses yfinance)')
    p1.add_argument('--period', help="Period for yfinance (e.g. 'max' or '1y')")
    p1.add_argument('--start', help='Start date YYYY-MM-DD')
    p1.add_argument('--end', help='End date YYYY-MM-DD (defaults to today)')
    p1.add_argument('--out', help='Output path (csv/jsonl/parquet). Default: data/<symbol>.<parquet>')

    # batch mode: fetch top-N per market
    p2 = sub.add_parser('batch', help='Fetch top-N entities per market and combine into one dataframe')
    p2.add_argument('--markets', help='Comma separated markets to fetch: crypto,equity,commodity,bond', default='crypto,equity,commodity,bond')
    p2.add_argument('--n', type=int, default=100, help='Top N entities per market')
    p2.add_argument('--start', help='Start date YYYY-MM-DD (inclusive)')
    p2.add_argument('--end', help='End date YYYY-MM-DD (inclusive). Defaults to today')
    p2.add_argument('--out', required=True, help='Output path (jsonl or parquet)')
    p2.add_argument('--format', choices=['parquet','jsonl'], default='parquet')
    p2.add_argument('--remove-csv', action='store_true', help='Remove previous CSV fetch artifacts in data/ before running')
    p2.add_argument('--threads', type=int, default=8, help='Number of threads for parallel downloads')

    args = parser.parse_args()
    args = parser.parse_args()

    if args.command == 'single':
        outp = args.out or os.path.join('data', f"{args.symbol.replace('/','_')}.parquet")
        # fetch single symbol via yfinance
        try:
            df = fetch_yfinance(args.symbol, interval='1d', period=args.period) if not args.start else yf.Ticker(args.symbol).history(start=args.start, end=args.end or None, interval='1d')
            if df is None or df.empty:
                print('No data returned')
                sys.exit(2)
            df = df.reset_index()
            df['ticker'] = args.symbol
            df = df[['ticker','Date','Open','High','Low','Close','Volume']]
            write_out(df, outp, fmt='parquet' if outp.endswith('.parquet') else 'jsonl')
            print('Wrote', outp)
        except Exception as e:
            print('Failed to fetch single symbol:', e)
            sys.exit(2)

    elif args.command == 'batch':
        if args.remove_csv:
            removed = clean_previous_csvs('data')
            print('Removed CSVs:', removed)
        markets = [m.strip() for m in args.markets.split(',') if m.strip()]
        per_market_dfs = []
        overall_failures = {}
        # Fetch each market separately to get top-N PER MARKET (no cross-market dedupe)
        for m in markets:
            tickers = []
            try:
                if m == 'crypto':
                    coins = get_top_crypto_coins(args.n)
                    crypto_symbols = []
                    for c in coins:
                        cu = c.upper()
                        if cu.endswith('-USD') or cu.endswith('USD'):
                            crypto_symbols.append(cu)
                        else:
                            crypto_symbols.append(cu + '-USD')
                    tickers = crypto_symbols[:args.n]
                elif m == 'equity':
                    sps = get_sp500_tickers()
                    if sps:
                        tickers = rank_equities_by_volume(sps, lookback='1mo', top_n=args.n)
                elif m == 'commodity':
                    # Expanded candidate commodity futures (Yahoo tickers). This list includes
                    # energy, metals, agriculturals and softs commonly available on Yahoo Finance.
                    commodity_candidates = [
                        'CL=F','GC=F','SI=F','NG=F','ZC=F','ZS=F','ZL=F','HG=F','PL=F','PA=F','HO=F','RB=F',
                        'KC=F','SB=F','CT=F','LH=F','OJ=F','CC=F','SM=F','W=F','ZW=F','ZB=F','ZN=F','ZF=F',
                        'ZQ=F','ZC=F','KE=F','LBS=F','HE=F','FC=F','SB=F','RBOB=F'
                    ]
                    # attempt to rank these by volume (best-effort)
                    tickers = rank_equities_by_volume(commodity_candidates, lookback='1mo', top_n=args.n)
                elif m == 'bond' or m == 'bonds':
                    # Expanded bond/yield candidates: common Yahoo tickers for yields and indices
                    bond_candidates = ['^IRX','^FVX','^TNX','^TYX','^VIX','^GSPC']
                    # also include some FRED series via ticker-like names if desired; for now use these Yahoo series
                    tickers = bond_candidates[:args.n]
                elif m == 'etf':
                    etf_candidates = [
                        'SPY','QQQ','IWM','VTI','VOO','EEM','GLD','SLV','VNQ','XLF','XLE','XLK','XLY','XLP','XLV','XLI','XLU','XLB','XLC','XBI',
                        'LQD','HYG','AGG','GDX','TBT','SHY','BND','SPYG','SPYD','VONG','VIG','SCHD','DGRO','VYM','IJR','IWF','IWD','VO','VB'
                    ]
                    tickers = rank_equities_by_volume(etf_candidates, lookback='1mo', top_n=args.n)
                elif m == 'fx':
                    # Expanded FX candidate list: majors, minors and cross pairs in Yahoo format (SYMBOL=X)
                    fx_candidates = [
                        'EURUSD=X','GBPUSD=X','USDJPY=X','USDCHF=X','USDCAD=X','AUDUSD=X','NZDUSD=X','USDSGD=X','USDHKD=X','USDMXN=X',
                        'USDZAR=X','USDBRL=X','USDINR=X','USDKRW=X','USDTRY=X','USDNOK=X','USDSEK=X','USDDKK=X','EURGBP=X','EURJPY=X',
                        'EURCHF=X','EURAUD=X','EURCAD=X','EURNZD=X','EURPLN=X','EURHUF=X','EURCZK=X','GBPAUD=X','GBPCAD=X','GBPNZD=X',
                        'GBPCHF=X','GBPJPY=X','GBPPLN=X','AUDJPY=X','AUDNZD=X','AUDCAD=X','AUDCHF=X','CADJPY=X','CADCHF=X','NZDJPY=X',
                        'NZDCAD=X','NZDCHF=X','SGDJPY=X','SGDUSD=X','HKDUSD=X','CHFJPY=X','CHFCAD=X','CHFGBP=X','MXNUSD=X','BRLUSD=X',
                        'ZARUSD=X','INRUSD=X','TRYUSD=X','RUBUSD=X','PLNUSD=X','HUFUSD=X','CZKUSD=X','SEKUSD=X','NOKUSD=X','DKKUSD=X',
                        'EURSEK=X','EURNOK=X','EURDKK=X','EURTRY=X','EURMXN=X','EURBRL=X','EURZAR=X','GBPSEK=X','GBPNOK=X','GBPDKK=X',
                        'GBPMXN=X','GBRTRY=X','AUDSGD=X','AUDHKD=X','AUDTRY=X','CADSGD=X','CADHKD=X','NZDSGD=X','NZDHKD=X','SGDHKD=X',
                        'EURSGD=X','EURHKD=X','GBPSGD=X','GBPHKD=X','USDBYN=X','USDPLN=X','USDHUF=X','USDCZK=X','USDILS=X','USDTHB=X',
                        'USDIDR=X','USDPHP=X','USDMYR=X','USDKRW=X','USDCNY=X','USDCNH=X','CNHUSD=X','CNXUSD=X','JPYUSD=X','CNYUSD=X',
                        # include raw pairs without '=X' where Yahoo uses different forms (kept for best-effort retrieval)
                        'EURUSD','GBPUSD','USDJPY','AUDUSD','NZDUSD','USDCAD','USDCHF'
                    ]
                    # FX usually has no volume; just try to fetch the listed pairs (truncate to requested n)
                    tickers = fx_candidates[:args.n]
                elif m == 'index':
                    index_candidates = ['^GSPC','^IXIC','^RUT','^FTSE','^N225','^GDAXI','^FCHI','^HSI','^SSEC','^STOXX50E']
                    tickers = index_candidates[:args.n]
                elif m == 'reit':
                    reit_candidates = ['VNQ','IYR','XLRE','SPG','O','NNN','EQIX','AVB','EQR','WELL','VTR','ESS']
                    tickers = rank_equities_by_volume(reit_candidates, lookback='1mo', top_n=args.n)
                elif m == 'sector':
                    sector_candidates = ['XLF','XLK','XLE','XLY','XLP','XLV','XLI','XLU','XLB','XLC']
                    tickers = rank_equities_by_volume(sector_candidates, lookback='1mo', top_n=args.n)
                elif m == 'international':
                    # Use bundled international tickers file for non-US equities
                    intl_file = Path('data') / 'international_tickers.txt'
                    tickers = []
                    if intl_file.exists():
                        try:
                            with intl_file.open() as fh:
                                lines = [l.strip() for l in fh.readlines() if l.strip()]
                                tickers = [t.replace('.', '-').upper() for t in lines][:args.n]
                        except Exception:
                            tickers = []
                elif m == 'continuous':
                    # continuous futures: reuse commodity continuous tickers (Yahoo =F series)
                    tickers = ['GC=F','CL=F','SI=F','NG=F','PL=F','PA=F','HG=F','ZC=F','ZS=F','ZL=F'][:args.n]
                elif m == 'fx_futures':
                    fxf = ['6E=F','6B=F','6A=F','6J=F','6C=F','6S=F','6N=F','DX=F','EURUSD0=X','GBPUSD0=X']
                    tickers = fxf[:args.n]
                elif m == 'volatility':
                    vol_candidates = ['^VIX','^VVIX','^VXV','^VIX3M','^VIX1Y']
                    tickers = vol_candidates[:args.n]
                elif m == 'options':
                    # placeholder: options surfaces are stored separately; we fetch volatility indices instead
                    tickers = ['^VIX','^VVIX'][:args.n]
                else:
                    # unknown market label: try to use it as a plain list of tickers (comma-separated)
                    # or skip
                    print(f'Skipping unknown market: {m}')
                    continue
            except Exception as e:
                print(f'Failed to prepare tickers for market {m}:', e)
                continue

            if not tickers:
                print(f'No tickers found for market {m} (skipping)')
                continue

            print(f'Fetching market {m} top {len(tickers)} tickers...')
            df_m = batch_fetch_yfinance(tickers, start=args.start, end=args.end, threads=args.threads)
            if df_m is None or df_m.empty:
                print(f'No data fetched for market {m}')
                # record missing tickers as failures
                for t in tickers:
                    overall_failures.setdefault(m, {})[t] = 'no-data'
                continue
            df_m['market'] = m
            # build per-ticker stats and record missing tickers
            present = set(df_m['ticker'].unique())
            market_stats = {}
            for t in tickers:
                if t in present:
                    s = df_m[df_m['ticker'] == t]
                    market_stats[t] = {
                        'rows': int(len(s)),
                        'min_date': str(s['Date'].min().date()),
                        'max_date': str(s['Date'].max().date())
                    }
                else:
                    market_stats[t] = {'rows': 0, 'min_date': None, 'max_date': None}
                    overall_failures.setdefault(m, {})[t] = 'no-data'
            # attach stats to DataFrame as attribute for manifest building later
            df_m._market_stats = market_stats
            per_market_dfs.append(df_m)

        if not per_market_dfs:
            print('No data fetched for any market')
            sys.exit(2)

        # concatenate and write combined output
        df_all = pd.concat(per_market_dfs, ignore_index=True)
        write_out(df_all, args.out, fmt=args.format)

        # build and persist manifest summarizing per-market, per-ticker stats and failures
        manifest = {'markets': {}, 'failures': overall_failures}
        for df_m in per_market_dfs:
            m = df_m['market'].iloc[0]
            stats = getattr(df_m, '_market_stats', None)
            if stats is None:
                # compute simple stats
                stats = {}
                for t in sorted(df_m['ticker'].unique()):
                    s = df_m[df_m['ticker'] == t]
                    stats[t] = {'rows': int(len(s)), 'min_date': str(s['Date'].min().date()), 'max_date': str(s['Date'].max().date())}
            manifest['markets'][m] = stats

        manifest_path = Path('data') / 'market_manifest.json'
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open('w', encoding='utf8') as fh:
                json.dump(manifest, fh, indent=2)
            print('Wrote manifest:', str(manifest_path))
        except Exception as e:
            print('Failed to write manifest:', e)

        print('Wrote', args.out)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
