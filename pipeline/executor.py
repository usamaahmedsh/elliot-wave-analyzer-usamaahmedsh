from typing import List, Dict, Any, Tuple, Optional
import multiprocessing as mp
from functools import partial
import math
import os
import traceback

from models.WaveAnalyzer import WaveAnalyzer

# Per-process cache for attached shared-memory views. Keys are shared array names.
_SHM_CACHE: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# WINDOW-LEVEL WORKER  (unchanged — called by both paths)
# =============================================================================

def _worker_scan_window(window_tuple: Tuple[int, int, dict], cfg: dict) -> Dict[str, Any]:
    """
    Worker function: scan a single window slice for wave patterns.
    window_tuple: (start_row, window_len, context)
    cfg: scan configuration dict
    """
    start_row, window_len, context = window_tuple

    lows        = context.get('lows')
    highs       = context.get('highs')
    dates       = context.get('dates')
    shared_meta = context.get('shared')

    if shared_meta is not None:
        try:
            from pipeline.shared_memory import attach_shared_view
        except Exception:
            attach_shared_view = None

        if attach_shared_view is not None:
            for key, meta in shared_meta.items():
                name = meta.get('name')
                if name not in _SHM_CACHE:
                    try:
                        view = attach_shared_view(meta)
                        _SHM_CACHE[name] = {'view': view, 'meta': meta}
                    except Exception:
                        _SHM_CACHE[name] = None

            if 'lows' in shared_meta:
                sm   = shared_meta['lows']['name']
                lows = None if _SHM_CACHE.get(sm) is None else _SHM_CACHE[sm]['view']
            if 'highs' in shared_meta:
                sm    = shared_meta['highs']['name']
                highs = None if _SHM_CACHE.get(sm) is None else _SHM_CACHE[sm]['view']
            if 'dates' in shared_meta:
                sm    = shared_meta['dates']['name']
                dates = None if _SHM_CACHE.get(sm) is None else _SHM_CACHE[sm]['view']

    import numpy as _np

    if lows is None or highs is None or dates is None:
        df_window = context['base_df'].iloc[start_row: start_row + window_len].reset_index(drop=True)
        wa = WaveAnalyzer(df=df_window, verbose=False)
        try:
            local_idx_start = int(_np.argmin(_np.array(list(df_window['Low']))))
        except Exception:
            return {}
    else:
        wnd_lows  = lows [start_row: start_row + window_len]
        wnd_highs = highs[start_row: start_row + window_len]
        wnd_dates = dates[start_row: start_row + window_len]
        wa = WaveAnalyzer(df=None, lows=wnd_lows, highs=wnd_highs,
                          dates=wnd_dates, verbose=False)
        try:
            local_idx_start = int(_np.argmin(wnd_lows))
        except Exception:
            return {}

    scan_mode         = cfg.get('scan_pattern_types', 'all')
    enable_multi_start = cfg.get('enable_multi_start', False)
    max_start_points  = cfg.get('max_start_points', 5)

    scan_cfg = {
        'cpu_batch_size':       cfg.get('cpu_batch_size', 512),
        'cpu_top_k':            cfg.get('cpu_top_k', 64),
        'max_seconds_per_scan': cfg.get('max_seconds_per_scan', 1e9),
    }

    if enable_multi_start:
        candidates = wa.scan_multi_start(
            up_to            = cfg.get('up_to', 8),
            top_n            = cfg.get('top_n', 1),
            max_combinations = cfg.get('max_combinations', None),
            scan_cfg         = scan_cfg,
            max_starts       = max_start_points,
            pattern_types    = scan_mode,
        )
    elif scan_mode == 'all':
        candidates = wa.scan_all_patterns(
            idx_start        = local_idx_start,
            up_to            = cfg.get('up_to', 8),
            top_n            = cfg.get('top_n', 1),
            max_combinations = cfg.get('max_combinations', None),
            scan_cfg         = scan_cfg,
        )
    else:
        candidates = wa.scan_impulses(
            idx_start        = local_idx_start,
            up_to            = cfg.get('up_to', 8),
            top_n            = cfg.get('top_n', 1),
            max_combinations = cfg.get('max_combinations', None),
            scan_cfg         = scan_cfg,
        )

    if not candidates:
        return {}

    best       = candidates[0]
    date_start = dates[start_row] if dates is not None else context['base_df'].iloc[start_row]['Date']
    date_end   = dates[min(start_row + window_len - 1, len(dates) - 1)] \
                 if dates is not None \
                 else context['base_df'].iloc[min(start_row + window_len - 1,
                                                   len(context['base_df']) - 1)]['Date']

    try:
        if hasattr(date_start, '__int__') and \
                _np.issubdtype(_np.dtype(type(date_start)), _np.integer):
            date_start = _np.datetime64(int(date_start), 'ns')
        if hasattr(date_end, '__int__') and \
                _np.issubdtype(_np.dtype(type(date_end)), _np.integer):
            date_end = _np.datetime64(int(date_end), 'ns')
    except Exception:
        pass

    best_dict = {
        'rule_name':   best.rule_name   if hasattr(best, 'rule_name')   else 'unknown',
        'score':       float(best.score) if hasattr(best, 'score')      else 0.0,
        'wave_config': best.wave_config  if hasattr(best, 'wave_config') else [],
        'idx_start':   best.idx_start    if hasattr(best, 'idx_start')   else 0,
        'idx_end':     best.idx_end      if hasattr(best, 'idx_end')     else 0,
    }

    wave_boundaries = {}
    try:
        if hasattr(best, 'pattern') and hasattr(best.pattern, 'waves'):
            for wave_name, wave in best.pattern.waves.items():
                wave_boundaries[wave_name] = {
                    'idx_start': int(wave.idx_start)  if hasattr(wave, 'idx_start') else 0,
                    'idx_end':   int(wave.idx_end)    if hasattr(wave, 'idx_end')   else 0,
                    'low':       float(wave.low)      if hasattr(wave, 'low')       else 0.0,
                    'high':      float(wave.high)     if hasattr(wave, 'high')      else 0.0,
                    'low_idx':   int(wave.low_idx)    if hasattr(wave, 'low_idx')   else 0,
                    'high_idx':  int(wave.high_idx)   if hasattr(wave, 'high_idx') else 0,
                    'length':    float(wave.length)   if hasattr(wave, 'length')   else 0.0,
                    'duration':  int(wave.duration)   if hasattr(wave, 'duration') else 0,
                }
    except Exception:
        pass
    best_dict['waves'] = wave_boundaries

    return {
        'symbol':         context.get('symbol', 'UNKNOWN'),
        'start_row':      start_row,
        'window_len':     window_len,
        'date_start':     date_start,
        'date_end':       date_end,
        'best':           best_dict,
        'scan_stats':     getattr(wa, '_last_scan_stats', None),
        'ensemble_score': float(best.ensemble_score) if hasattr(best, 'ensemble_score') else 0.0,
        'fib_score':      float(best.fib_score)      if hasattr(best, 'fib_score')      else 0.0,
        '_pattern_obj':   best if hasattr(best, 'pattern') else None,
    }


# =============================================================================
# TICKER-LEVEL WORKER  (new — the right parallelism axis for HPC)
# =============================================================================

def _worker_scan_ticker(args: Tuple) -> Dict[str, Any]:
    """
    Top-level worker: scan ALL windows for a single ticker.
    Each worker process owns one full ticker — no nested parallelism.
    Uses 'fork' on Linux so imports/numba cache are inherited instantly.

    Returns:
        {symbol, results: List[Dict], wins_scanned: int, error: str|None}
    """
    symbol, lows, highs, dates, cfg_dict = args

    slide_bars  = int(cfg_dict.get('slide_bars',   7))
    min_bars    = int(cfg_dict.get('min_bars',     56))
    max_bars    = int(cfg_dict.get('max_bars',    728))
    max_windows = int(cfg_dict.get('max_windows', 1000))

    import numpy as _np

    n_total      = len(lows)
    results      = []
    current_idx  = 0
    wins_scanned = 0

    try:
        while current_idx < n_total and wins_scanned < max_windows:
            end_idx    = min(current_idx + max_bars, n_total)
            window_len = end_idx - current_idx
            if window_len < min_bars:
                break

            context = {
                'symbol': symbol,
                'lows':   lows,
                'highs':  highs,
                'dates':  dates,
            }
            try:
                result = _worker_scan_window(
                    (current_idx, window_len, context), cfg_dict)
            except Exception:
                result = {}
            wins_scanned += 1

            if result and result.get('best', {}).get('score', 0) > 0:
                results.append(result)
                pattern_end = result['best'].get('idx_end', 0)
                current_idx = current_idx + pattern_end + slide_bars
            else:
                current_idx += slide_bars

    except Exception as e:
        return {
            'symbol':       symbol,
            'results':      results,
            'wins_scanned': wins_scanned,
            'error':        f"{e}\n{traceback.format_exc()}",
        }

    return {
        'symbol':       symbol,
        'results':      results,
        'wins_scanned': wins_scanned,
        'error':        None,
    }


def parallel_scan_tickers(
    ticker_data:   Dict[str, Any],   # {symbol: pd.DataFrame}
    cfg_dict:      Dict[str, Any],
    n_workers:     int  = None,
    checkpoint_fn  = None,           # callable(symbol, results, error=None)
) -> List[Dict]:
    """
    Parallelize wave scanning across tickers using ProcessPoolExecutor.

    Each worker handles one full ticker (all its windows) independently.
    On Linux HPC nodes, 'fork' start method means workers inherit the
    already-warmed Numba JIT cache — no re-compilation overhead.

    n_workers:     defaults to SLURM_NTASKS → SLURM_CPUS_ON_NODE → os.cpu_count()
    checkpoint_fn: called in the main process after each ticker completes.
                   Signature: checkpoint_fn(symbol: str, results: List[Dict],
                                            error: Optional[str])
    """
    import numpy as _np
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    if n_workers is None:
        n_workers = int(
            os.environ.get('SLURM_NTASKS',
            os.environ.get('SLURM_CPUS_ON_NODE',
            os.cpu_count() or 1))
        )

    # Convert DataFrames → numpy arrays before pickling.
    # float32 halves memory; numpy arrays pickle ~10× faster than DataFrames.
    args_list = []
    for symbol, df in ticker_data.items():
        try:
            lows  = df['Low'].to_numpy().astype(_np.float32)
            highs = df['High'].to_numpy().astype(_np.float32)
            dates = df['Date'].to_numpy()
        except Exception:
            continue
        args_list.append((symbol, lows, highs, dates, cfg_dict))

    all_results  = []
    n_total      = len(args_list)
    n_patterns   = 0

    # 'fork' on Linux: workers inherit everything (numba cache, imports).
    # Falls back to 'spawn' on macOS/Windows automatically.
    try:
        mp_ctx = mp.get_context('fork')
    except ValueError:
        mp_ctx = mp.get_context('spawn')

    bar = tqdm(
        total         = n_total,
        desc          = "tickers",
        unit          = "ticker",
        dynamic_ncols = True,
        bar_format    = (
            "{desc} |{bar}| {n}/{total} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
    )

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as exe:
        fut_to_sym = {
            exe.submit(_worker_scan_ticker, args): args[0]
            for args in args_list
        }

        for fut in as_completed(fut_to_sym):
            symbol = fut_to_sym[fut]
            try:
                res = fut.result(timeout=600)   # 10-min hard cap per ticker
            except Exception as e:
                err = f"{e}\n{traceback.format_exc()}"
                bar.set_postfix(ticker=symbol, status="❌error", refresh=False)
                bar.update(1)
                if checkpoint_fn:
                    checkpoint_fn(symbol, [], error=err)
                continue

            ticker_results  = res.get('results', [])
            n_patterns     += len(ticker_results)
            all_results.extend(ticker_results)

            bar.set_postfix(
                ticker = symbol,
                pats   = n_patterns,
                wins   = res.get('wins_scanned', 0),
                err    = "⚠️" if res.get('error') else "",
                refresh = False,
            )
            bar.update(1)

            if checkpoint_fn:
                checkpoint_fn(symbol, ticker_results, error=res.get('error'))

    bar.close()
    return all_results


# =============================================================================
# WINDOW-LEVEL POOL  (kept for backward-compat — not used in main pipeline)
# =============================================================================

def parallel_scan_windows(
    windows:   List[Tuple[int, int, dict]],
    cfg:       dict,
    processes: int = None,
) -> List[Dict[str, Any]]:
    """
    Parallelise across windows within a single ticker.
    NOTE: this is the *wrong* parallelism axis for large ticker universes —
    use parallel_scan_tickers() instead.  Kept here for backward compatibility.
    """
    global _SHM_CACHE
    _SHM_CACHE.clear()

    if processes is None:
        processes = max(1, mp.cpu_count() - 1)

    try:
        from models.functions import hi, lo, next_hi, next_lo, count_extrema
        import numpy as _np
        _ = hi(_np.array([1.0, 2.0, 1.5]),    _np.array([1.0, 2.0, 1.5]),    0)
        _ = lo(_np.array([1.0, 0.5, 0.8]),     _np.array([1.0, 0.5, 0.8]),    0)
        _ = next_hi(_np.array([1.0, 2.0, 1.5]),_np.array([1.0, 2.0, 1.5]),    0, 0.0)
        _ = next_lo(_np.array([1.0, 0.5, 0.8]),_np.array([1.0, 0.5, 0.8]),    0, 0.0)
        _ = count_extrema(_np.array([1.0, 2.0, 1.0]))
    except Exception:
        pass

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=processes) as pool:
        worker     = partial(_worker_scan_window, cfg=cfg)
        cfg_chunk  = cfg.get('chunk_size', 0) if isinstance(cfg, dict) else 0
        chunksize  = (cfg_chunk if (cfg_chunk and isinstance(cfg_chunk, int) and cfg_chunk > 0)
                      else max(1, len(windows) // (processes * 4)))
        results_iter = pool.imap_unordered(worker, windows, chunksize=chunksize)

        results = []
        total   = len(windows)
        done    = 0
        for res in results_iter:
            if res:
                results.append(res)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print(f"[executor] scanned {done}/{total} windows")

    return results
