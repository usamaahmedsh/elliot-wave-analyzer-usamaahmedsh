from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import time as _time

import numpy as np
import pandas as pd

from models.MonoWave import MonoWaveUp, MonoWaveDown
from models.WaveOptions import (WaveOptionsGenerator5, WaveOptionsGenerator3,
                                 get_options_array)
from models.WaveCycle import WaveCycle
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, BearishImpulse, Correction, TDWave, LeadingDiagonal
from models.EnsembleScoring import EnsembleScorer


@dataclass
class FoundPattern:
    pattern: WavePattern
    rule_name: str
    score: float
    wave_config: List[int]
    idx_start: int
    idx_end: int
    ensemble_score: Optional[float] = None
    fib_score: Optional[float] = None


_options_cache = {}
_ensemble_scorer = EnsembleScorer(
    fib_weight=0.5, rule_weight=0.3,
    time_weight=0.1, complexity_weight=0.1
)
_window_features_cache = {}


def _get_or_create_options(up_to: int, pattern_type: str = 'impulse'):
    """Cache WaveOptions to avoid repeated generation."""
    cache_key = (up_to, pattern_type)
    if cache_key in _options_cache:
        return _options_cache[cache_key]
    if pattern_type == 'impulse':
        gen = WaveOptionsGenerator5(up_to=up_to)
        options = list(gen.options_sorted)
    elif pattern_type == 'corrective':
        gen = WaveOptionsGenerator3(up_to=up_to)
        options = list(gen.options_sorted)
    else:
        options = []
    _options_cache[cache_key] = options
    return options


def _vectorized_prescore(opts_arr: np.ndarray, up_to: int,
                          base_score: float) -> np.ndarray:
    """
    Compute pre-scores for ALL options at once via numpy.
    Returns float32 array of shape (N,).
    """
    n_waves = opts_arr.shape[1]
    complexity = opts_arr.sum(axis=1).astype(np.float32) / (n_waves * max(1, up_to))
    return base_score + 0.1 * complexity


class WaveAnalyzer:
    """
    Find impulse or corrective waves for given dataframe.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None,
                 lows: Optional[np.ndarray] = None,
                 highs: Optional[np.ndarray] = None,
                 dates: Optional[np.ndarray] = None,
                 verbose: bool = False):
        self.verbose = verbose
        if df is not None:
            self.df = df
            try:
                self.lows  = df['Low'].to_numpy()
                self.highs = df['High'].to_numpy()
                self.dates = df['Date'].to_numpy()
            except Exception:
                self.lows  = np.array(list(df['Low']))
                self.highs = np.array(list(df['High']))
                self.dates = np.array(list(df['Date']))
        else:
            if lows is None or highs is None or dates is None:
                raise ValueError('Either df or arrays (lows, highs, dates) must be provided')
            self.df    = None
            self.lows  = np.asarray(lows)
            self.highs = np.asarray(highs)
            self.dates = np.asarray(dates)

        self.impulse_rules    = list()
        self.correction_rules = list()
        self.__waveoptions_up: WaveOptionsGenerator5
        self.__waveoptions_down: WaveOptionsGenerator3
        self.set_combinatorial_limits()

    def get_absolute_low(self):
        return np.min(self.lows)

    def find_local_extrema(self, window_size: int = 5,
                           min_distance: int = 3) -> List[int]:
        n = len(self.lows)
        if n < window_size:
            return [int(np.argmin(self.lows))]
        extrema = []
        for i in range(window_size, n - window_size):
            window_lows = self.lows[i - window_size:i + window_size + 1]
            if self.lows[i] == np.min(window_lows):
                if not extrema or (i - extrema[-1]) >= min_distance:
                    extrema.append(i)
        global_min = int(np.argmin(self.lows))
        if global_min not in extrema:
            extrema.append(global_min)
            extrema.sort()
        return extrema

    def set_combinatorial_limits(self, n_up: int = 10, n_down: int = 10):
        self.__waveoptions_up   = WaveOptionsGenerator5(n_up)
        self.__waveoptions_down = WaveOptionsGenerator3(n_down)

    # -------------------------------------------------------------------------
    # Wave builders (unchanged)
    # -------------------------------------------------------------------------

    def find_impulsive_wave(self, idx_start: int, wave_config: list = None):
        if wave_config is None:
            wave_config = [0, 0, 0, 0, 0]
        wave1 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates,
                           idx_start=idx_start, skip=wave_config[0])
        wave1.label = "1"
        wave1_end = wave1.idx_end
        if wave1_end is None:
            return False
        wave2 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates,
                             idx_start=wave1_end, skip=wave_config[1])
        wave2.label = "2"
        wave2_end = wave2.idx_end
        if wave2_end is None:
            return False
        wave3 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates,
                           idx_start=wave2_end, skip=wave_config[2])
        wave3.label = "3"
        wave3_end = wave3.idx_end
        if wave3_end is None:
            return False
        wave4 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates,
                             idx_start=wave3_end, skip=wave_config[3])
        wave4.label = "4"
        wave4_end = wave4.idx_end
        if wave4_end is None:
            return False
        if wave2.low > np.min(self.lows[wave2.low_idx:wave4.low_idx]):
            return False
        wave5 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates,
                           idx_start=wave4_end, skip=wave_config[4])
        wave5.label = "5"
        wave5_end = wave5.idx_end
        if wave5_end is None:
            return False
        if (self.lows[wave4.low_idx:wave5.high_idx].any() and
                wave4.low > np.min(self.lows[wave4.low_idx:wave5.high_idx])):
            return False
        return [wave1, wave2, wave3, wave4, wave5]

    def find_bearish_impulsive_wave(self, idx_start: int, wave_config: list = None):
        if wave_config is None:
            wave_config = [0, 0, 0, 0, 0]
        wave1 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates,
                             idx_start=idx_start, skip=wave_config[0])
        wave1.label = "1"
        wave1_end = wave1.idx_end
        if wave1_end is None:
            return False
        wave2 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates,
                           idx_start=wave1_end, skip=wave_config[1])
        wave2.label = "2"
        wave2_end = wave2.idx_end
        if wave2_end is None:
            return False
        wave3 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates,
                             idx_start=wave2_end, skip=wave_config[2])
        wave3.label = "3"
        wave3_end = wave3.idx_end
        if wave3_end is None:
            return False
        wave4 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates,
                           idx_start=wave3_end, skip=wave_config[3])
        wave4.label = "4"
        wave4_end = wave4.idx_end
        if wave4_end is None:
            return False
        if wave2.high < np.max(self.highs[wave2.high_idx:wave4.high_idx]):
            return False
        wave5 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates,
                             idx_start=wave4_end, skip=wave_config[4])
        wave5.label = "5"
        wave5_end = wave5.idx_end
        if wave5_end is None:
            return False
        if (self.highs[wave4.high_idx:wave5.low_idx].any() and
                wave4.high < np.max(self.highs[wave4.high_idx:wave5.low_idx])):
            return False
        return [wave1, wave2, wave3, wave4, wave5]

    def find_corrective_wave(self, idx_start: int, wave_config: list = None):
        if wave_config is None:
            wave_config = [0, 0, 0]
        waveA = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates,
                             idx_start=idx_start, skip=wave_config[0])
        waveA.label = "A"
        waveA_end = waveA.idx_end
        if waveA_end is None:
            return False
        waveB = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates,
                           idx_start=waveA_end, skip=wave_config[1])
        waveB.label = "B"
        waveB_end = waveB.idx_end
        if waveB_end is None:
            return False
        waveC = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates,
                             idx_start=waveB_end, skip=wave_config[2])
        waveC.label = "C"
        waveC_end = waveC.idx_end
        if waveC_end is None:
            return False
        return [waveA, waveB, waveC]

    def find_td_wave(self, idx_start: int, wave_config: list = None):
        if wave_config is None:
            wave_config = [0, 0]
        wave1 = MonoWaveUp(self.df, idx_start=idx_start, skip=wave_config[0])
        wave1.label = "1"
        wave1_end = wave1.idx_end
        if wave1_end is None:
            return False
        wave2 = MonoWaveDown(self.df, idx_start=wave1_end, skip=wave_config[1])
        wave2.label = "2"
        wave2_end = wave2.idx_end
        if wave2_end is None:
            return False
        return [wave1, wave2]

    def next_cycle(self, start_idx: int):
        impulse    = Impulse("impulse")
        correction = Correction("correction")
        wave_cycles = set()
        for new_option_impulse in self.__waveoptions_up.options_sorted:
            cycle_complete = False
            waves_up = self.find_impulsive_wave(idx_start=start_idx,
                                                wave_config=new_option_impulse.values)
            if waves_up:
                wavepattern_up = WavePattern(waves_up, verbose=False)
                if wavepattern_up.check_rule(impulse):
                    end = waves_up[4].idx_end
                    for new_option_correction in self.__waveoptions_down.options_sorted:
                        waves = self.find_corrective_wave(
                            idx_start=end,
                            wave_config=new_option_correction.values
                        )
                        if waves:
                            wavepattern = WavePattern(waves, verbose=False)
                            if wavepattern.check_rule(correction):
                                cycle_complete = True
                                wave_cycle = WaveCycle(wavepattern_up, wavepattern)
                                wave_cycles.add(wave_cycle)
                    if cycle_complete:
                        yield wave_cycle
                        return None

    # -------------------------------------------------------------------------
    # Core scan helpers — shared vectorized pre-score + global top-k
    # -------------------------------------------------------------------------

    def _compute_base_score(self, idx_start: int, use_highs: bool = False) -> float:
        """Cheap window-level proxy score for pre-filtering."""
        try:
            sl = self.lows[idx_start: idx_start + 10]
            sh = self.highs[idx_start: idx_start + 10]
            if sl.size == 0:
                return 0.0
            lo_val  = float(np.min(sl))
            hi_val  = float(np.max(sh))
            ref     = hi_val if use_highs else lo_val
            vol     = float(np.std(sl)) if sl.size > 1 else 0.0
            return 0.4 * vol + 0.3 * ((hi_val - lo_val) / (ref + 1e-9))
        except Exception:
            return 0.0

    def _global_topk_indices(self, opts_arr: np.ndarray, up_to: int,
                              base_score: float, top_k: int) -> np.ndarray:
        """
        Vectorized pre-score over ALL options at once, return top_k indices.
        This replaces the per-batch top-k loop — far more accurate filtering.
        """
        pre_scores = _vectorized_prescore(opts_arr, up_to, base_score)
        k = min(top_k, len(opts_arr))
        if k < len(opts_arr):
            top_idx = np.argpartition(-pre_scores, k - 1)[:k]
            return top_idx[np.argsort(-pre_scores[top_idx])]
        return np.argsort(-pre_scores)

    # -------------------------------------------------------------------------
    # scan_impulses  (rewritten — global top-k, numpy options array)
    # -------------------------------------------------------------------------

    def scan_impulses(self, idx_start: int, up_to: int = 10, top_n: int = 5,
                      max_combinations: int = None,
                      scan_cfg: dict = None) -> List['FoundPattern']:
        scan_cfg   = scan_cfg or {}
        # top_k_global: how many candidates to fully evaluate (global, not per-batch)
        top_k      = int(scan_cfg.get('cpu_top_k', 64))
        max_combos = max_combinations or 1_000_000

        # Cheap extrema pre-filter
        try:
            from models.functions import count_extrema
            if count_extrema(self.lows[idx_start:]) < 4:
                return []
        except Exception:
            pass

        opts_arr   = get_options_array(up_to, n_waves=5)  # cached numpy array
        base_score = self._compute_base_score(idx_start, use_highs=False)
        top_indices = self._global_topk_indices(opts_arr, up_to, base_score, top_k)

        impulse          = Impulse("impulse")
        leading_diagonal = LeadingDiagonal("leading_diagonal")
        rules_to_check   = [impulse, leading_diagonal]

        found: List[FoundPattern] = []
        seen  = set()
        n_full_evals = 0
        t_full_eval  = 0.0
        t_start_wall = _time.time()
        max_seconds  = float(scan_cfg.get('max_seconds_per_scan', 1e9))

        for idx in top_indices:
            if n_full_evals >= max_combos:
                break
            if _time.time() - t_start_wall > max_seconds:
                break

            wave_config = opts_arr[idx].tolist()
            t1 = _time.time()
            waves_up = self.find_impulsive_wave(idx_start=idx_start,
                                                wave_config=wave_config)
            t_full_eval  += _time.time() - t1
            n_full_evals += 1

            if not waves_up:
                continue

            wp = WavePattern(waves_up, verbose=False)
            for rule in rules_to_check:
                if wp.check_rule(rule):
                    if wp in seen:
                        continue
                    seen.add(wp)
                    rule_score = float(wp.score_rule(rule)) if hasattr(wp, 'score_rule') else 0.0
                    ed = _ensemble_scorer.score_with_details(wp, rule_score=rule_score)
                    found.append(FoundPattern(
                        pattern=wp, rule_name=rule.name, score=rule_score,
                        wave_config=wave_config,
                        idx_start=wp.idx_start, idx_end=wp.idx_end,
                        ensemble_score=ed['ensemble_score'],
                        fib_score=ed.get('fibonacci_score', 0.5),
                    ))

        try:
            self._last_scan_stats = {
                'n_options':    len(opts_arr),
                'n_full_evals': n_full_evals,
                'time_full_eval': float(t_full_eval),
                'pattern_type': 'impulse',
            }
        except Exception:
            pass

        found.sort(key=lambda x: (x.ensemble_score or x.score, x.idx_end), reverse=True)
        return found[:top_n]

    # -------------------------------------------------------------------------
    # scan_bearish_impulses  (rewritten — global top-k, numpy options array)
    # -------------------------------------------------------------------------

    def scan_bearish_impulses(self, idx_start: int, up_to: int = 10, top_n: int = 5,
                               max_combinations: int = None,
                               scan_cfg: dict = None) -> List['FoundPattern']:
        scan_cfg   = scan_cfg or {}
        top_k      = int(scan_cfg.get('cpu_top_k', 64))
        max_combos = max_combinations or 1_000_000

        opts_arr   = get_options_array(up_to, n_waves=5)
        base_score = self._compute_base_score(idx_start, use_highs=True)
        top_indices = self._global_topk_indices(opts_arr, up_to, base_score, top_k)

        bearish_impulse = BearishImpulse("bearish_impulse")
        rules_to_check  = [bearish_impulse]

        found: List[FoundPattern] = []
        seen  = set()
        n_full_evals = 0
        t_full_eval  = 0.0
        t_start_wall = _time.time()
        max_seconds  = float(scan_cfg.get('max_seconds_per_scan', 1e9))

        for idx in top_indices:
            if n_full_evals >= max_combos:
                break
            if _time.time() - t_start_wall > max_seconds:
                break

            wave_config  = opts_arr[idx].tolist()
            t1 = _time.time()
            waves_down   = self.find_bearish_impulsive_wave(idx_start=idx_start,
                                                            wave_config=wave_config)
            t_full_eval  += _time.time() - t1
            n_full_evals += 1

            if not waves_down:
                continue

            wp = WavePattern(waves_down, verbose=False)
            for rule in rules_to_check:
                if wp.check_rule(rule):
                    if wp in seen:
                        continue
                    seen.add(wp)
                    rule_score = float(wp.score_rule(rule)) if hasattr(wp, 'score_rule') else 0.0
                    ed = _ensemble_scorer.score_with_details(wp, rule_score=rule_score)
                    found.append(FoundPattern(
                        pattern=wp, rule_name=rule.name, score=rule_score,
                        wave_config=wave_config,
                        idx_start=wp.idx_start, idx_end=wp.idx_end,
                        ensemble_score=ed['ensemble_score'],
                        fib_score=ed.get('fibonacci_score', 0.5),
                    ))

        try:
            self._last_scan_stats = {
                'n_options':    len(opts_arr),
                'n_full_evals': n_full_evals,
                'time_full_eval': float(t_full_eval),
                'pattern_type': 'bearish_impulse',
            }
        except Exception:
            pass

        found.sort(key=lambda x: (x.ensemble_score or x.score, x.idx_end), reverse=True)
        return found[:top_n]

    # -------------------------------------------------------------------------
    # scan_correctives  (rewritten — global top-k, numpy options array)
    # -------------------------------------------------------------------------

    def scan_correctives(self, idx_start: int, up_to: int = 10, top_n: int = 5,
                         max_combinations: int = None,
                         scan_cfg: dict = None) -> List['FoundPattern']:
        scan_cfg   = scan_cfg or {}
        top_k      = int(scan_cfg.get('cpu_top_k', 64))
        max_combos = max_combinations or 1_000_000

        opts_arr   = get_options_array(up_to, n_waves=3)
        base_score = self._compute_base_score(idx_start, use_highs=False)
        top_indices = self._global_topk_indices(opts_arr, up_to, base_score, top_k)

        correction     = Correction("corrective")
        found: List[FoundPattern] = []
        seen  = set()
        n_full_evals = 0
        t_start_wall = _time.time()
        max_seconds  = float(scan_cfg.get('max_seconds_per_scan', 1e9))

        for idx in top_indices:
            if n_full_evals >= max_combos:
                break
            if _time.time() - t_start_wall > max_seconds:
                break

            wave_config = opts_arr[idx].tolist()
            waves       = self.find_corrective_wave(idx_start=idx_start,
                                                    wave_config=wave_config)
            n_full_evals += 1

            if not waves:
                continue

            pat = WavePattern(waves, verbose=False)
            if not pat.check_rule(correction):
                continue
            if pat in seen:
                continue
            seen.add(pat)

            rule_score = float(pat.score_rule(correction)) if hasattr(pat, 'score_rule') else 0.0
            ed = _ensemble_scorer.score_with_details(pat, rule_score=rule_score)
            found.append(FoundPattern(
                pattern=pat, rule_name='Corrective', score=rule_score,
                wave_config=wave_config,
                idx_start=idx_start, idx_end=waves[-1].idx_end,
                ensemble_score=ed['ensemble_score'],
                fib_score=ed.get('fibonacci_score', 0.5),
            ))

        try:
            self._last_scan_stats = {
                'n_full_evals': n_full_evals,
                'pattern_type': 'corrective',
            }
        except Exception:
            pass

        found.sort(key=lambda x: (x.ensemble_score or x.score, x.idx_end), reverse=True)
        return found[:top_n]

    # -------------------------------------------------------------------------
    # scan_all_patterns  (unchanged logic, calls rewritten scan methods)
    # -------------------------------------------------------------------------

    def scan_all_patterns(self, idx_start: int, up_to: int = 10, top_n: int = 5,
                          max_combinations: int = None,
                          scan_cfg: dict = None) -> List['FoundPattern']:
        all_found = []
        all_found.extend(self.scan_impulses(
            idx_start, up_to, top_n=top_n * 2,
            max_combinations=max_combinations, scan_cfg=scan_cfg))
        all_found.extend(self.scan_bearish_impulses(
            idx_start, up_to, top_n=top_n * 2,
            max_combinations=max_combinations, scan_cfg=scan_cfg))
        all_found.extend(self.scan_correctives(
            idx_start, up_to, top_n=top_n * 2,
            max_combinations=max_combinations, scan_cfg=scan_cfg))
        all_found.sort(key=lambda x: (x.ensemble_score or x.score, x.idx_end), reverse=True)
        return all_found[:top_n]

    # -------------------------------------------------------------------------
    # scan_multi_start  (unchanged logic)
    # -------------------------------------------------------------------------

    def scan_multi_start(self, up_to: int = 10, top_n: int = 5,
                         max_combinations: int = None, scan_cfg: dict = None,
                         max_starts: int = 5,
                         pattern_types: str = 'all') -> List['FoundPattern']:
        extrema = self.find_local_extrema(window_size=5, min_distance=10)
        if len(extrema) > max_starts:
            global_min  = int(np.argmin(self.lows))
            extrema_set = set(extrema[:max_starts - 1])
            extrema_set.add(global_min)
            extrema = sorted(list(extrema_set))[:max_starts]

        seen_patterns  = set()
        unique_patterns = []

        scan_funcs = {
            'all':        self.scan_all_patterns,
            'impulses':   self.scan_impulses,
            'correctives': self.scan_correctives,
        }
        scan_func = scan_funcs.get(pattern_types, self.scan_all_patterns)

        for start_idx in extrema:
            try:
                patterns = scan_func(
                    idx_start=start_idx,
                    up_to=up_to,
                    top_n=top_n,
                    max_combinations=max_combinations,
                    scan_cfg=scan_cfg,
                )
                for p in patterns:
                    if p.pattern not in seen_patterns:
                        seen_patterns.add(p.pattern)
                        unique_patterns.append(p)
            except Exception:
                continue

        unique_patterns.sort(key=lambda x: (x.ensemble_score or x.score, x.idx_end), reverse=True)
        return unique_patterns[:top_n]

    # -------------------------------------------------------------------------
    # Adaptive window methods (unchanged)
    # -------------------------------------------------------------------------

    @staticmethod
    def _bars_per_week(timeframe: str) -> int:
        if timeframe.upper() == "1W":
            return 1
        return 7

    @staticmethod
    def slice_df(df: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
        return df.iloc[start: start + length].reset_index(drop=True)

    def find_best_impulse_adaptive_window(self, base_df, start_row, timeframe="1D",
                                          min_weeks=4, max_weeks=12, up_to=10,
                                          grow_weeks=1, top_n=3):
        bars_per_week = self._bars_per_week(timeframe)
        min_len  = min_weeks  * bars_per_week
        max_len  = max_weeks  * bars_per_week
        step_len = grow_weeks * bars_per_week
        for window_len in range(min_len, max_len + 1, step_len):
            if start_row + window_len > len(base_df):
                break
            window_df        = self.slice_df(base_df, start_row, window_len)
            wa               = WaveAnalyzer(window_df, verbose=False)
            local_idx_start  = int(np.argmin(np.array(list(window_df["Low"]))))
            candidates       = wa.scan_impulses(idx_start=local_idx_start,
                                                up_to=up_to, top_n=top_n)
            if candidates:
                best = candidates[0]
                return {
                    "window_len":   window_len,
                    "window_weeks": window_len / bars_per_week,
                    "start_row":    start_row,
                    "date_start":   base_df.iloc[start_row]["Date"],
                    "date_end":     base_df.iloc[start_row + window_len - 1]["Date"],
                    "best":         best,
                    "all":          candidates,
                    "window_df":    window_df,
                }
        return None

    def sliding_adaptive_impulses(self, df, symbol, timeframe="1D", slide_weeks=1,
                                  min_weeks=4, max_weeks=12, up_to=10,
                                  grow_weeks=1, top_n=1):
        bars_per_week = self._bars_per_week(timeframe)
        slide_step = slide_weeks * bars_per_week
        min_len    = min_weeks  * bars_per_week
        out        = []
        start_row  = 0
        while start_row <= len(df) - min_len:
            result = self.find_best_impulse_adaptive_window(
                base_df=df, start_row=start_row, timeframe=timeframe,
                min_weeks=min_weeks, max_weeks=max_weeks, up_to=up_to,
                grow_weeks=grow_weeks, top_n=top_n,
            )
            if result is None:
                start_row += slide_step
                continue
            out.append(result)
            start_row += slide_step
        return out
