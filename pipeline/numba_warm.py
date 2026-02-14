"""Helpers to pre-warm numba JITs in the main process before spawning workers.

Call prewarm_numba() early in the orchestrator to avoid JIT overhead inside
worker processes.
"""
def prewarm_numba():
    try:
        import numpy as _np
        from models.functions import hi, lo, next_hi, next_lo, count_extrema
        # Also precompute WaveOptions combinatorics for common up_to values to
        # avoid paying the Python-side combinatorics cost during the first
        # scan_impulses call.
        try:
            from models.WaveOptions import WaveOptionsGenerator5, WaveOptionsGenerator3
            # build and cache sorted options for a typical up_to value so the
            # WaveAnalyzer scan path doesn't incur the combinatorics/sort cost
            # on the first real scan.
            opts5 = list(WaveOptionsGenerator5(up_to=8).options_sorted)
            opts3 = list(WaveOptionsGenerator3(up_to=8).options_sorted)
            try:
                # populate WaveAnalyzer internal cache directly
                from models import WaveAnalyzer as _wa
                _wa._options_cache[8] = opts5
                _wa._options_cache[3] = opts3
            except Exception:
                # if cache isn't accessible, ignore
                pass
        except Exception:
            # ignore failures - this is a best-effort warm-up
            pass

        a = _np.array([1.0, 2.0, 1.5], dtype=_np.float64)
        b = _np.array([1.0, 0.5, 0.8], dtype=_np.float64)
        # call each function once to trigger compilation
        try:
            _ = hi(a, a, 0)
        except Exception:
            pass
        try:
            _ = lo(b, b, 0)
        except Exception:
            pass
        try:
            _ = next_hi(a, a, 0, 0.0)
        except Exception:
            pass
        try:
            _ = next_lo(b, b, 0, 0.0)
        except Exception:
            pass
        try:
            _ = count_extrema(a)
        except Exception:
            pass
    except Exception:
        # best-effort: if numba isn't available or compilation fails, ignore
        return
