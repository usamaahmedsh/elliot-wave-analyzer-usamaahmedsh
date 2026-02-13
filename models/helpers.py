from models.WavePattern import WavePattern
import pandas as pd
import time
import plotly.graph_objects as go
import os
import random
import string
import json
from typing import Dict, Any

# configurable images directory (can be changed at runtime by pipeline)
IMAGES_DIR = "images"


def set_images_dir(path: str) -> None:
    """Set a new images directory for plotting exports."""
    global IMAGES_DIR
    IMAGES_DIR = path


def timeit(func):
    def wrapper(*arg, **kw):
        t1 = time.perf_counter_ns()
        res = func(*arg, **kw)
        t2 = time.perf_counter_ns()
        print("took:", t2 - t1, "ns")
        return res
    return wrapper


def _ensure_images_dir() -> None:
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR, exist_ok=True)


def _new_base_filename(prefix: str = "") -> str:
    _ensure_images_dir()
    current_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    rand = generate_random_string(6)
    base = f"{current_timestamp}_{rand}"
    if prefix:
        base = f"{prefix}_{base}"
    # path without extension
    return os.path.join(IMAGES_DIR, base)


def save_chart_as_image(fig, base_path: str = None) -> str:
    """
    Save chart as PNG and return the base path (without extension).
    """
    if base_path is None:
        base_path = _new_base_filename()
    png_path = base_path + ".png"
    fig.write_image(png_path)
    return base_path


def generate_random_string(length) -> str:
    characters = string.digits
    return "".join(random.choice(characters) for _ in range(length))


def convert_yf_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a yahoo finance OHLC DataFrame to column names used in this project

    old_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    new_names = ['Date', 'Open', 'High', 'Low', 'Close']
    """
    df_output = pd.DataFrame()

    # Use the DataFrame index as the Date column
    df_output["Date"] = list(df.index)
    # normalize to pandas Timestamp
    df_output["Date"] = pd.to_datetime(df_output["Date"], errors="coerce")

    # Helper to robustly convert a column (Series or single-column DataFrame) to a list
    def _col_to_list(col):
        # If selection returned a DataFrame (possible with certain yfinance outputs),
        # try to squeeze to a Series. If multiple columns exist, take the first one.
        if isinstance(col, pd.DataFrame):
            if col.shape[1] == 1:
                col = col.iloc[:, 0]
            else:
                # multi-column: pick the first column
                col = col.iloc[:, 0]
        # At this point `col` should be a Series; fall back to converting via list()
        try:
            return col.to_list()
        except Exception:
            return list(col)

    df_output["Open"] = _col_to_list(df["Open"])
    df_output["High"] = _col_to_list(df["High"])
    df_output["Low"] = _col_to_list(df["Low"])
    df_output["Close"] = _col_to_list(df["Close"])

    return df_output


def _serialize_wavepattern(wave_pattern: WavePattern,
                           symbol: str = "",
                           timeframe: str = "1D",
                           rule_name: str = "",
                           score: float = None) -> Dict[str, Any]:
    """
    Turn a WavePattern into a JSON-friendly dict with per-wave endpoints.
    """
    waves_payload = []
    for key, wave in wave_pattern.waves.items():
        waves_payload.append(
            {
                "key": key,
                "label": getattr(wave, "label", None),
                "idx_start": getattr(wave, "idx_start", None),
                "idx_end": getattr(wave, "idx_end", None),
                "date_start": getattr(wave, "date_start", None),
                "date_end": getattr(wave, "date_end", None),
                "low": getattr(wave, "low", None),
                "high": getattr(wave, "high", None),
                "low_idx": getattr(wave, "low_idx", None),
                "high_idx": getattr(wave, "high_idx", None),
                "length": getattr(wave, "length", None),
                "duration": getattr(wave, "duration", None),
            }
        )

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "rule_name": rule_name,
        "score": score,
        "pattern_type": wave_pattern.type if isinstance(wave_pattern.type, str) else "unknown",
        "degree": wave_pattern.degree,
        "idx_start": wave_pattern.idx_start,
        "idx_end": wave_pattern.idx_end,
        "low": wave_pattern.low,
        "high": wave_pattern.high,
        "dates_polyline": [str(d) for d in wave_pattern.dates],
        "values_polyline": wave_pattern.values,
        "labels_polyline": wave_pattern.labels,
        "waves": waves_payload,
    }
    return payload


def _write_pattern_json_and_csv(base_path: str,
                                payload: Dict[str, Any]) -> None:
    """
    Given a base path (images/xxx_yyy), write .json and .csv files.
    """
    # Only write a JSON payload now; CSV sidecars were removed per new output policy.
    json_path = base_path + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def plot_cycle(df, wave_cycle, title: str = "",
               symbol: str = "",
               timeframe: str = "1D",
               rule_name: str = "",
               score: float = None):

    data = go.Ohlc(x=df["Date"],
                   open=df["Open"],
                   high=df["High"],
                   low=df["Low"],
                   close=df["Close"])

    monowaves = go.Scatter(x=wave_cycle.dates,
                           y=wave_cycle.values,
                           text=wave_cycle.labels,
                           mode="lines+markers+text",
                           textposition="middle right",
                           textfont=dict(size=15, color="#2c3035"),
                           line=dict(
                               color=("rgb(111, 126, 130)"),
                               width=3),
                           )
    layout = dict(title=title)
    fig = go.Figure(data=[data, monowaves], layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)

    base = _new_base_filename(prefix="cycle")
    base = save_chart_as_image(fig, base_path=base)

    # build JSON payload only (no CSV)
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "rule_name": rule_name,
        "score": score,
        "pattern_type": "cycle",
        "degree": wave_cycle.degree,
        "idx_start": wave_cycle.start_idx,
        "idx_end": wave_cycle.end_idx,
        "dates_polyline": [str(d) for d in wave_cycle.dates],
        "values_polyline": wave_cycle.values,
        "labels_polyline": wave_cycle.labels,
        "waves": [],  # could be filled with underlying wpup/wpdown if needed
    }
    return payload


def plot_pattern(df: pd.DataFrame,
                 wave_pattern: WavePattern,
                 title: str = "",
                 symbol: str = "",
                 timeframe: str = "1D",
                 rule_name: str = "",
                 score: float = None):
    data = go.Ohlc(x=df["Date"],
                   open=df["Open"],
                   high=df["High"],
                   low=df["Low"],
                   close=df["Close"])

    monowaves = go.Scatter(x=wave_pattern.dates,
                           y=wave_pattern.values,
                           text=wave_pattern.labels,
                           mode="lines+markers+text",
                           textposition="middle right",
                           textfont=dict(size=15, color="#2c3035"),
                           line=dict(
                               color=("rgb(111, 126, 130)"),
                               width=3),
                           )
    layout = dict(title=title)
    fig = go.Figure(data=[data, monowaves], layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)

    base = _new_base_filename(prefix="pattern")
    base = save_chart_as_image(fig, base_path=base)

    payload = _serialize_wavepattern(
        wave_pattern=wave_pattern,
        symbol=symbol,
        timeframe=timeframe,
        rule_name=rule_name,
        score=score,
    )
    return payload


def plot_monowave(df, monowave, title: str = "", symbol: str = "", timeframe: str = "1D"):
    data = go.Ohlc(x=df["Date"],
                   open=df["Open"],
                   high=df["High"],
                   low=df["Low"],
                   close=df["Close"])

    monowaves = go.Scatter(x=monowave.dates,
                           y=monowave.points,
                           mode="lines+markers+text",
                           textposition="middle right",
                           textfont=dict(size=15, color="#2c3035"),
                           line=dict(
                               color=("rgb(111, 126, 130)"),
                               width=3),
                           )
    layout = dict(title=title)
    fig = go.Figure(data=[data, monowaves], layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)

    base = _new_base_filename(prefix="monowave")
    save_chart_as_image(fig, base_path=base)
    # monowave does not currently emit JSON payloads
    return None
