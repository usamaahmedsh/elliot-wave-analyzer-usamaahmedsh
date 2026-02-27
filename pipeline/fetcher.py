import asyncio
from typing import List, Dict
import pandas as pd
import yfinance as yf
from models.helpers import convert_yf_data
import threading

# Lock to prevent concurrent yfinance downloads (yfinance is not thread-safe)
_yf_lock = threading.Lock()


async def _download_to_df(symbol: str, start_days: int) -> pd.DataFrame:
    # Offload the blocking yfinance.download call to a thread
    loop = asyncio.get_running_loop()

    def _download():
        # Use lock to serialize yfinance downloads (thread-safety issue)
        with _yf_lock:
            end_date = pd.Timestamp.now()
            
            # If start_days is 0 or negative, fetch ALL available history
            if start_days <= 0:
                # yfinance default: fetch maximum available history
                raw = yf.download(symbol, period="max", progress=False)
            else:
                start_date = end_date - pd.DateOffset(days=start_days)
                raw = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            return convert_yf_data(raw)

    df = await loop.run_in_executor(None, _download)
    return df


def _load_hf_dataset_to_pdf(hf_id: str = "usamaahmedsh/financial-markets-dataset-15y-train") -> pd.DataFrame:
    """Blocking helper to load the HF dataset into a pandas DataFrame.

    This is run in a threadpool from async code to avoid blocking the event loop.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets library is required to fetch from HuggingFace") from e

    ds = load_dataset(hf_id, split="train")
    pdf = ds.to_pandas()
    # Ensure expected column names and types
    if "Date" in pdf.columns:
        pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce")
    elif "date" in pdf.columns:
        pdf["Date"] = pd.to_datetime(pdf["date"], errors="coerce")

    return pdf


async def fetch_symbols(symbols: List[str], start_days: int = 720, concurrency: int = 8, source: str = "yfinance") -> Dict[str, pd.DataFrame]:
    """Fetch many symbols concurrently. Supports two sources:
    - 'yfinance' (default): use yfinance async wrapper (threadpool)
    - 'hf': download the HuggingFace dataset and slice it per-symbol (runs in threadpool)

    Returns a dict symbol->DataFrame. Failures are returned as empty DataFrames.
    """
    sem = asyncio.Semaphore(concurrency)

    if source == "hf":
        loop = asyncio.get_running_loop()

        # Load dataset once in a thread
        try:
            pdf = await loop.run_in_executor(None, _load_hf_dataset_to_pdf)
        except Exception as e:
            print(f"error loading HF dataset: {e}")
            return {s: pd.DataFrame() for s in symbols}

        # helper to slice per-symbol and apply time-window
        def _slice_symbol(s: str):
            try:
                now = pd.Timestamp.now()
                
                # detect the column that contains the ticker/symbol name
                possible_symbol_cols = [c for c in pdf.columns if c.lower() in ("ticker", "symbol", "tickers", "ticker_symbol")]
                symbol_col = possible_symbol_cols[0] if possible_symbol_cols else None
                if symbol_col is None:
                    # fallback: try 'market' or 'exchange' or assume index
                    symbol_col = "symbol" if "symbol" in pdf.columns else None

                if symbol_col is None:
                    print("HF dataset does not contain a ticker/symbol column; cannot slice per-symbol")
                    return s, pd.DataFrame()

                sub = pdf[pdf[symbol_col] == s]
                if sub.empty:
                    return s, pd.DataFrame()
                sub = sub.sort_values(by="Date")
                
                # If start_days <= 0, use ALL available data
                if start_days <= 0:
                    # No date filtering, use all data
                    df_filtered = sub
                else:
                    start_date = now - pd.DateOffset(days=start_days)
                    df_filtered = sub[(sub["Date"] >= start_date) & (sub["Date"] <= now)]
                
                # keep columns Date/Open/High/Low/Close if present
                cols = [c for c in ["Date", "Open", "High", "Low", "Close"] if c in df_filtered.columns]
                df_out = df_filtered[cols].reset_index(drop=True)
                return s, df_out
                return s, df_out
            except Exception as e:
                print(f"hf slice error for {s}: {e}")
                return s, pd.DataFrame()

        # schedule slice tasks but report progress as they finish
        tasks = [asyncio.create_task(loop.run_in_executor(None, _slice_symbol, s)) for s in symbols]
        results = []
        done = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            res = await coro
            done += 1
            print(f"[fetcher:hf] fetched {done}/{total} ({res[0]})")
            results.append(res)
        return {k: v for k, v in results}

    # default: yfinance
    async def _safe_fetch(s: str):
        async with sem:
            try:
                df = await _download_to_df(s, start_days)
                return s, df
            except Exception as e:
                print(f"fetch error for {s}: {e}")
                return s, pd.DataFrame()

    tasks = [asyncio.create_task(_safe_fetch(s)) for s in symbols]
    results = []
    done = 0
    total = len(tasks)
    for coro in asyncio.as_completed(tasks):
        res = await coro
        done += 1
        print(f"[fetcher:yf] fetched {done}/{total} ({res[0]})")
        results.append(res)
    return {k: v for k, v in results}
