import time
import logging
from pathlib import Path

import yfinance as yf

from models.helpers import convert_yf_data


def fetch_history_with_retries(ticker: str, interval: str = "1d", start: str = "2022-12-01",
                               retries: int = 3, backoff_factor: float = 2.0):
    """Fetch history for a ticker with simple retry/backoff.

    Raises the last exception if all retries fail.
    """
    attempt = 1
    while True:
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(interval=interval, start=start)
            if df is None or df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            return df
        except Exception as exc:
            logging.warning("Fetch attempt %d for %s failed: %s", attempt, ticker, exc)
            if attempt >= retries:
                logging.error("All %d attempts failed for %s", retries, ticker)
                raise
            sleep_for = backoff_factor ** (attempt - 1)
            logging.info("Retrying in %.1f seconds...", sleep_for)
            time.sleep(sleep_for)
            attempt += 1


def main():
    out_path = Path(__file__).parent / "data" / "aapl_1d_2020.csv"
    ticker = "AAPL"
    try:
        df = fetch_history_with_retries(ticker=ticker, interval="1d", start="2022-12-01", retries=3)
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return

    converted = convert_yf_data(df)
    converted.to_csv(out_path, sep=",", index=False)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
