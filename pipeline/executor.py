from typing import List, Dict, Any, Tuple
import multiprocessing as mp
from functools import partial
import math

from models.WaveAnalyzer import WaveAnalyzer


def _worker_scan_window(window_tuple: Tuple[int, int, dict], cfg: dict) -> Dict[str, Any]:
    """Worker function executed in a separate process.

    window_tuple: (start_row, window_len, context) where context can include symbol etc.
    cfg: configuration dict for scan_impulses
    """
    start_row, window_len, context = window_tuple

    # Prefer operating on numpy arrays passed in the context to avoid pandas slicing/pickling overhead.
    lows = context.get('lows')
    highs = context.get('highs')
    dates = context.get('dates')

    if lows is None or highs is None or dates is None:
        # fallback to DataFrame slicing if arrays are not available
        df_window = context['base_df'].iloc[start_row : start_row + window_len].reset_index(drop=True)
        wa = WaveAnalyzer(df=df_window, verbose=False)
        import numpy as _np
        try:
            local_idx_start = int(_np.argmin(_np.array(list(df_window['Low']))))
        except Exception:
            return {}
    else:
        # slice arrays without copying large DataFrames
        wnd_lows = lows[start_row : start_row + window_len]
        wnd_highs = highs[start_row : start_row + window_len]
        wnd_dates = dates[start_row : start_row + window_len]
        wa = WaveAnalyzer(df=None, lows=wnd_lows, highs=wnd_highs, dates=wnd_dates, verbose=False)
        import numpy as _np
        try:
            local_idx_start = int(_np.argmin(wnd_lows))
        except Exception:
            return {}

    candidates = wa.scan_impulses(
        idx_start=local_idx_start,
        up_to=cfg.get('up_to', 8),
        top_n=cfg.get('top_n', 1),
        max_combinations=cfg.get('max_combinations', None),
    )

    if not candidates:
        return {}

    best = candidates[0]
    # return minimal information; avoid returning DataFrame across process boundary
    date_start = dates[start_row] if dates is not None else context['base_df'].iloc[start_row]['Date']
    date_end = dates[min(start_row + window_len - 1, len(dates) - 1)] if dates is not None else context['base_df'].iloc[min(start_row + window_len - 1, len(context['base_df']) - 1)]['Date']
    return {
        'start_row': start_row,
        'window_len': window_len,
        'date_start': date_start,
        'date_end': date_end,
        'best': best,
        'all': candidates,
    }


def parallel_scan_windows(windows: List[Tuple[int, int, dict]], cfg: dict, processes: int = None) -> List[Dict[str, Any]]:
    """Run windows in parallel using multiprocessing Pool.

    windows: list of tuples (start_row, window_len, context)
    cfg: dict with scan params (up_to, top_n)
    """
    if processes is None:
        processes = max(1, mp.cpu_count() - 1)

    # Pre-warm numba-compiled functions in main process to avoid JIT cost in child forks.
    try:
        from models.functions import hi, lo  # noqa: F401
        import numpy as _np
        # call hi/lo on a tiny array to trigger compilation
        _ = hi(_np.array([1.0, 2.0, 1.5]), _np.array([1.0, 2.0, 1.5]), 0)
        _ = lo(_np.array([1.0, 0.5, 0.8]), _np.array([1.0, 0.5, 0.8]), 0)
    except Exception:
        # ignore warmup failures
        pass

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=processes) as pool:
        worker = partial(_worker_scan_window, cfg=cfg)
        # use imap_unordered for better throughput and chunksize heuristic
        # allow override via cfg['chunk_size'] (0 or missing -> heuristic)
        cfg_chunk = cfg.get('chunk_size', 0) if isinstance(cfg, dict) else 0
        if cfg_chunk and isinstance(cfg_chunk, int) and cfg_chunk > 0:
            chunksize = cfg_chunk
        else:
            chunksize = max(1, len(windows) // (processes * 4))
        results_iter = pool.imap_unordered(worker, windows, chunksize=chunksize)

        results = []
        total = len(windows)
        done = 0
        # iterate and report progress
        for res in results_iter:
            if res:
                results.append(res)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print(f"[executor] scanned {done}/{total} windows")

    # filter empty results (already handled)
    return results
