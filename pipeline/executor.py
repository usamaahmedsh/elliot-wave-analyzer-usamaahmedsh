from typing import List, Dict, Any, Tuple
import multiprocessing as mp
from functools import partial
import math

from models.WaveAnalyzer import WaveAnalyzer
from typing import Optional

# Per-process cache for attached shared-memory views. Keys are shared array names.
_SHM_CACHE: Dict[str, Dict[str, Any]] = {}


def _worker_scan_window(window_tuple: Tuple[int, int, dict], cfg: dict) -> Dict[str, Any]:
    """Worker function executed in a separate process.

    window_tuple: (start_row, window_len, context) where context can include symbol etc.
    cfg: configuration dict for scan_impulses
    """
    start_row, window_len, context = window_tuple

    # Prefer operating on numpy arrays passed in the context to avoid pandas slicing/pickling overhead.
    # If shared memory metadata is provided in the context, attach shared views (cached per-process).
    lows = context.get('lows')
    highs = context.get('highs')
    dates = context.get('dates')
    shared_meta = context.get('shared')

    if shared_meta is not None:
        # lazy import to avoid requiring multiprocessing.shared_memory in callers
        try:
            from pipeline.shared_memory import attach_shared_view
        except Exception:
            attach_shared_view = None

        if attach_shared_view is not None:
            # Attach each shared buffer once per process and cache the numpy views
            for key, meta in shared_meta.items():
                # use name as cache key (unique across arrays)
                name = meta.get('name')
                if name not in _SHM_CACHE:
                    try:
                        view = attach_shared_view(meta)
                        _SHM_CACHE[name] = {'view': view, 'meta': meta}
                    except Exception:
                        # fallback to leaving _SHM_CACHE untouched
                        _SHM_CACHE[name] = None

            # rebind local vars to views if available
            if 'lows' in shared_meta:
                sm = shared_meta['lows']['name']
                lows = None if _SHM_CACHE.get(sm) is None else _SHM_CACHE[sm]['view']
            if 'highs' in shared_meta:
                sm = shared_meta['highs']['name']
                highs = None if _SHM_CACHE.get(sm) is None else _SHM_CACHE[sm]['view']
            if 'dates' in shared_meta:
                sm = shared_meta['dates']['name']
                dates = None if _SHM_CACHE.get(sm) is None else _SHM_CACHE[sm]['view']

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

    # Choose scan method based on config
    scan_mode = cfg.get('scan_pattern_types', 'all')
    enable_multi_start = cfg.get('enable_multi_start', False)
    max_start_points = cfg.get('max_start_points', 5)
    
    scan_cfg = {
        'cpu_batch_size': cfg.get('cpu_batch_size', 512),
        'cpu_top_k': cfg.get('cpu_top_k', 64)
    }
    
    # Use multi-start search if enabled (tries multiple pivot points)
    if enable_multi_start:
        candidates = wa.scan_multi_start(
            up_to=cfg.get('up_to', 8),
            top_n=cfg.get('top_n', 1),
            max_combinations=cfg.get('max_combinations', None),
            scan_cfg=scan_cfg,
            max_starts=max_start_points,
            pattern_types=scan_mode
        )
    elif scan_mode == 'all':
        # Scan ALL pattern types (impulsive + corrective) for maximum recall
        candidates = wa.scan_all_patterns(
            idx_start=local_idx_start,
            up_to=cfg.get('up_to', 8),
            top_n=cfg.get('top_n', 1),
            max_combinations=cfg.get('max_combinations', None),
            scan_cfg=scan_cfg
        )
    else:
        # Default: scan only impulsive patterns
        candidates = wa.scan_impulses(
            idx_start=local_idx_start,
            up_to=cfg.get('up_to', 8),
            top_n=cfg.get('top_n', 1),
            max_combinations=cfg.get('max_combinations', None),
            scan_cfg=scan_cfg
        )


    if not candidates:
        return {}

    best = candidates[0]
    # return minimal information; avoid returning DataFrame across process boundary
    date_start = dates[start_row] if dates is not None else context['base_df'].iloc[start_row]['Date']
    date_end = dates[min(start_row + window_len - 1, len(dates) - 1)] if dates is not None else context['base_df'].iloc[min(start_row + window_len - 1, len(context['base_df']) - 1)]['Date']
    # if dates were stored as int64 (datetime64 view in shared memory), convert back
    try:
        import numpy as _np
        if hasattr(date_start, '__int__') and getattr(_np, 'issubdtype')(_np.dtype(type(date_start)), _np.integer):
            date_start = _np.datetime64(int(date_start), 'ns')
        if hasattr(date_end, '__int__') and getattr(_np, 'issubdtype')(_np.dtype(type(date_end)), _np.integer):
            date_end = _np.datetime64(int(date_end), 'ns')
    except Exception:
        pass
    return {
        'start_row': start_row,
        'window_len': window_len,
        'date_start': date_start,
        'date_end': date_end,
        'best': best,
        'all': candidates,
        'scan_stats': getattr(wa, '_last_scan_stats', None),
        # Include ensemble scoring details
        'ensemble_score': best.ensemble_score if hasattr(best, 'ensemble_score') else None,
        'fib_score': best.fib_score if hasattr(best, 'fib_score') else None,
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
        from models.functions import hi, lo, next_hi, next_lo, count_extrema  # noqa: F401
        import numpy as _np
        # call numba functions on tiny arrays to trigger compilation in main process
        _ = hi(_np.array([1.0, 2.0, 1.5]), _np.array([1.0, 2.0, 1.5]), 0)
        _ = lo(_np.array([1.0, 0.5, 0.8]), _np.array([1.0, 0.5, 0.8]), 0)
        _ = next_hi(_np.array([1.0, 2.0, 1.5]), _np.array([1.0, 2.0, 1.5]), 0, 0.0)
        _ = next_lo(_np.array([1.0, 0.5, 0.8]), _np.array([1.0, 0.5, 0.8]), 0, 0.0)
        _ = count_extrema(_np.array([1.0, 2.0, 1.0]))
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
