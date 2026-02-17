from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from models.MonoWave import MonoWaveUp, MonoWaveDown
from models.WaveOptions import WaveOptionsGenerator5, WaveOptionsGenerator3
from models.WaveCycle import WaveCycle
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, Correction, TDWave, LeadingDiagonal
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
_ensemble_scorer = EnsembleScorer(fib_weight=0.5, rule_weight=0.3, time_weight=0.1, complexity_weight=0.1)

# Reusable pre-computed window features cache (avoid recomputing per pattern type)
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


class WaveAnalyzer:
    """
    Find impulse or corrective waves for given dataframe
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, lows: Optional[np.ndarray] = None, highs: Optional[np.ndarray] = None, dates: Optional[np.ndarray] = None, verbose: bool = False):
        """
        Construct WaveAnalyzer either from a pandas DataFrame (old behavior) or directly from numpy arrays.
        Passing arrays avoids repeated DataFrame slicing and conversion when used inside worker processes.
        """
        self.verbose = verbose
        if df is not None:
            self.df = df
            # keep arrays as numpy (avoid list->array conversion when possible)
            try:
                self.lows = df['Low'].to_numpy()
                self.highs = df['High'].to_numpy()
                self.dates = df['Date'].to_numpy()
            except Exception:
                self.lows = np.array(list(df['Low']))
                self.highs = np.array(list(df['High']))
                self.dates = np.array(list(df['Date']))
        else:
            # expect numpy arrays
            if lows is None or highs is None or dates is None:
                raise ValueError('Either df or arrays (lows, highs, dates) must be provided')
            self.df = None
            self.lows = np.asarray(lows)
            self.highs = np.asarray(highs)
            self.dates = np.asarray(dates)

        self.impulse_rules = list()
        self.correction_rules = list()

        self.__waveoptions_up: WaveOptionsGenerator5
        self.__waveoptions_down: WaveOptionsGenerator3
        self.set_combinatorial_limits()

    def get_absolute_low(self):
        """
        find the absolute low in the dataframe. Can be used to start the wave analysis from this low.
        """
        return np.min(self.lows)

    def find_local_extrema(self, window_size: int = 5, min_distance: int = 3) -> List[int]:
        """
        Find local minima (potential wave start points) using a simple rolling window approach.
        
        Args:
            window_size: Size of window to check for local minimum
            min_distance: Minimum distance between extrema
            
        Returns:
            List of indices representing local minima
        """
        n = len(self.lows)
        if n < window_size:
            return [int(np.argmin(self.lows))]
        
        extrema = []
        
        for i in range(window_size, n - window_size):
            # Check if current point is lower than neighbors
            window_lows = self.lows[i - window_size:i + window_size + 1]
            if self.lows[i] == np.min(window_lows):
                # Check minimum distance from previous extrema
                if not extrema or (i - extrema[-1]) >= min_distance:
                    extrema.append(i)
        
        # Always include global minimum if not already there
        global_min = int(np.argmin(self.lows))
        if global_min not in extrema:
            extrema.append(global_min)
            extrema.sort()
        
        return extrema

    def set_combinatorial_limits(self, n_up: int = 10, n_down: int = 10):
        """
        Change the limit to skip min / maxima for the WaveOptionsGenerators.
        """
        self.__waveoptions_up = WaveOptionsGenerator5(n_up)
        self.__waveoptions_down = WaveOptionsGenerator3(n_down)

    # -------------------------
    # Existing wave builders
    # -------------------------

    def find_impulsive_wave(self, idx_start: int, wave_config: list = None):
        """
        Tries to find 5 consecutive waves (up, down, up, down, up) to build an impulsive 12345 wave
        """
        if wave_config is None:
            wave_config = [0, 0, 0, 0, 0]

        wave1 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        wave1.label = "1"
        wave1_end = wave1.idx_end
        if wave1_end is None:
            if self.verbose:
                print("Wave 1 has no End in Data")
            return False

        wave2 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave1_end, skip=wave_config[1])
        wave2.label = "2"
        wave2_end = wave2.idx_end
        if wave2_end is None:
            if self.verbose:
                print("Wave 2 has no End in Data")
            return False

        wave3 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave2_end, skip=wave_config[2])
        wave3.label = "3"
        wave3_end = wave3.idx_end
        if wave3_end is None:
            if self.verbose:
                print("Wave 3 has no End in Data")
            return False

        wave4 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave3_end, skip=wave_config[3])
        wave4.label = "4"
        wave4_end = wave4.idx_end
        if wave4_end is None:
            if self.verbose:
                print("Wave 4 has no End in Data")
            return False

        if wave2.low > np.min(self.lows[wave2.low_idx:wave4.low_idx]):
            return False

        wave5 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave4_end, skip=wave_config[4])
        wave5.label = "5"
        wave5_end = wave5.idx_end
        if wave5_end is None:
            if self.verbose:
                print("Wave 5 has no End in Data")
            return False

        if self.lows[wave4.low_idx:wave5.high_idx].any() and wave4.low > np.min(self.lows[wave4.low_idx:wave5.high_idx]):
            if self.verbose:
                print("Low of Wave 4 higher than a low between Wave 4 and Wave 5")
            return False

        return [wave1, wave2, wave3, wave4, wave5]

    def find_corrective_wave(self, idx_start: int, wave_config: list = None):
        """
        Tries to find a corrective movement (ABC)
        """
        if wave_config is None:
            wave_config = [0, 0, 0]

        waveA = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        waveA.label = "A"
        waveA_end = waveA.idx_end
        if waveA_end is None:
            return False

        waveB = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveA_end, skip=wave_config[1])
        waveB.label = "B"
        waveB_end = waveB.idx_end
        if waveB_end is None:
            return False

        waveC = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveB_end, skip=wave_config[2])
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
            if self.verbose:
                print("Wave 1 has no End in Data")
            return False

        wave2 = MonoWaveDown(self.df, idx_start=wave1_end, skip=wave_config[1])
        wave2.label = "2"
        wave2_end = wave2.idx_end
        if wave2_end is None:
            if self.verbose:
                print("Wave 2 has no End in Data")
            return False

        return [wave1, wave2]

    def next_cycle(self, start_idx: int):
        impulse = Impulse("impulse")
        correction = Correction("correction")

        wave_cycles = set()

        for new_option_impulse in self.__waveoptions_up.options_sorted:
            cycle_complete = False

            waves_up = self.find_impulsive_wave(idx_start=start_idx, wave_config=new_option_impulse.values)

            if waves_up:
                wavepattern_up = WavePattern(waves_up, verbose=False)
                if wavepattern_up.check_rule(impulse):
                    end = waves_up[4].idx_end

                    for new_option_correction in self.__waveoptions_down.options_sorted:
                        waves = self.find_corrective_wave(idx_start=end, wave_config=new_option_correction.values)
                        if waves:
                            wavepattern = WavePattern(waves, verbose=False)
                            if wavepattern.check_rule(correction):
                                cycle_complete = True
                                wave_cycle = WaveCycle(wavepattern_up, wavepattern)
                                wave_cycles.add(wave_cycle)

                    if cycle_complete:
                        yield wave_cycle
                        return None

    # -------------------------
    # New: Impulse scanning helpers
    # -------------------------

    def scan_impulses(self, idx_start: int, up_to: int = 10, top_n: int = 5, max_combinations: int = None, scan_cfg: dict = None) -> List[FoundPattern]:
        """
        Scan impulse candidates from idx_start for a given up_to and return top-N patterns by score.
        """
        # Use cached precomputed options for the given up_to to avoid regenerating combinatorial options repeatedly.
        if up_to in _options_cache:
            wave_options_impulse_options = _options_cache[up_to]
        else:
            wave_options_impulse = WaveOptionsGenerator5(up_to=up_to)
            wave_options_impulse_options = list(wave_options_impulse.options_sorted)
            _options_cache[up_to] = wave_options_impulse_options

        # read batching options from scan_cfg if provided (CPU-only optimized path)
        if scan_cfg is None:
            scan_cfg = {}
        # Read CPU-only batching knobs from scan_cfg (defaults tuned for CPU runs)
        batch_size = int(scan_cfg.get('cpu_batch_size', 512))
        top_k = int(scan_cfg.get('cpu_top_k', 64))

        impulse = Impulse("impulse")
        leading_diagonal = LeadingDiagonal("leading_diagonal")
        rules_to_check = [impulse, leading_diagonal]

        # Cheap numba-accelerated pre-filter: skip windows that do not contain
        # enough local extrema (peaks/troughs) to form a 5-wave impulsive structure.
        # DISABLED FOR DEBUGGING - this may be rejecting valid patterns
        # try:
        #     from models.functions import count_extrema
        #     # count extrema in the tail of the arrays from idx_start
        #     n_ext = count_extrema(self.lows[idx_start:])
        #     if n_ext < 4:
        #         return []
        # except Exception:
        #     # if the fast filter isn't available for some reason, continue normally
        #     pass

        found: List[FoundPattern] = []
        seen = set()

        processed = 0
        # CPU-batched path: process candidate options in batches, compute a cheap
        # vectorized score per candidate using numpy, then fully evaluate only the
        # top_k candidates per batch. This reduces the number of expensive
        # find_impulsive_wave / WavePattern validations.
        n_opts = len(wave_options_impulse_options)
        i = 0

        # instrumentation counters for this scan call
        n_pre_scored = 0
        n_full_evals = 0
        t_pre_score = 0.0
        t_full_eval = 0.0

        # precompute window-level proxies (cheap) once per scan
        try:
            slice_window = self.lows[idx_start: idx_start + 10]
            lo_val = float(np.min(slice_window)) if slice_window.size > 0 else 0.0
            hi_val = float(np.max(self.highs[idx_start: idx_start + 10])) if slice_window.size > 0 else 0.0
            vol_proxy = float(np.std(slice_window)) if slice_window.size > 1 else 0.0
        except Exception:
            lo_val = 0.0
            hi_val = 0.0
            vol_proxy = 0.0

        base_score = 0.4 * vol_proxy + 0.3 * ((hi_val - lo_val) / (lo_val + 1e-9))

        while i < n_opts:
            batch = wave_options_impulse_options[i : i + batch_size]
            nb = len(batch)
            if nb == 0:
                break

            # vectorized compute of complexity (only per-candidate varying part)
            import time as _time
            t0 = _time.time()
            complexities = np.array([float(sum(opt.values)) / (len(opt.values) * max(1, up_to)) for opt in batch], dtype=np.float32)
            # compute scores vectorized
            scores = base_score + 0.1 * complexities
            t_pre_score += _time.time() - t0
            n_pre_scored += nb

            # select top_k in this batch (fast numpy argsort)
            if nb > top_k:
                top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
                # order them by score descending
                top_idx = top_idx[np.argsort(-scores[top_idx])]
            else:
                top_idx = np.argsort(-scores)

            top_candidates = [batch[int(j)] for j in top_idx]

            # fully evaluate top candidates
            for opt in top_candidates:
                processed += 1
                n_full_evals += 1
                if max_combinations is not None and processed > max_combinations:
                    if self.verbose:
                        print(f"scan_impulses reached max_combinations={max_combinations}, stopping early")
                    i = n_opts
                    break

                t1 = _time.time()
                waves_up = self.find_impulsive_wave(idx_start=idx_start, wave_config=opt.values)
                t_full_eval += _time.time() - t1
                if not waves_up:
                    continue

                wp = WavePattern(waves_up, verbose=False)

                for rule in rules_to_check:
                    if wp.check_rule(rule):
                        if wp in seen:
                            continue
                        seen.add(wp)

                        # Base rule satisfaction score
                        rule_score = 0.0
                        if hasattr(wp, "score_rule"):
                            rule_score = float(wp.score_rule(rule))

                        # Compute ensemble score (Fibonacci + time + complexity)
                        ensemble_details = _ensemble_scorer.score_with_details(wp, rule_score=rule_score)
                        ensemble_score = ensemble_details['ensemble_score']
                        fib_score = ensemble_details.get('fibonacci_score', 0.5)

                        found.append(
                            FoundPattern(
                                pattern=wp,
                                rule_name=rule.name,
                                score=rule_score,
                                wave_config=opt.values,
                                idx_start=wp.idx_start,
                                idx_end=wp.idx_end,
                                ensemble_score=ensemble_score,
                                fib_score=fib_score
                            )
                        )

            i += batch_size

        # attach last-scan instrumentation so callers can inspect (optional)
        try:
            self._last_scan_stats = {
                'n_options': n_opts,
                'n_pre_scored': int(n_pre_scored),
                'n_full_evals': int(n_full_evals),
                'time_pre_score': float(t_pre_score),
                'time_full_eval': float(t_full_eval),
            }
        except Exception:
            pass

        # Sort by ensemble_score (desc), then by longer coverage (idx_end desc)
        found.sort(key=lambda x: (x.ensemble_score if x.ensemble_score is not None else x.score, x.idx_end), reverse=True)
        return found[:top_n]

    def scan_correctives(self, idx_start: int, up_to: int = 10, top_n: int = 5, max_combinations: int = None, scan_cfg: dict = None) -> List[FoundPattern]:
        """Scan for corrective wave patterns (ABC, etc.) starting from idx_start."""
        scan_cfg = scan_cfg or {}
        batch_size = int(scan_cfg.get('cpu_batch_size', 512))
        top_k = int(scan_cfg.get('cpu_top_k', 64))
        max_combinations = max_combinations or 1_000_000
        
        # Use cached options
        options = _get_or_create_options(up_to, 'corrective')
        
        # Use cached window features
        base_score = 0.0
        cache_key = (id(self.lows), idx_start)
        if cache_key in _window_features_cache:
            base_score = _window_features_cache[cache_key]
        else:
            try:
                slice_window = self.lows[idx_start: idx_start + 10]
                if slice_window.size > 0:
                    lo_val = float(np.min(slice_window))
                    hi_val = float(np.max(self.highs[idx_start: idx_start + 10]))
                    vol_proxy = float(np.std(slice_window)) if slice_window.size > 1 else 0.0
                    base_score = 0.4 * vol_proxy + 0.3 * ((hi_val - lo_val) / (lo_val + 1e-9))
                _window_features_cache[cache_key] = base_score
            except Exception:
                base_score = 0.0
        
        found = []
        seen = set()
        n_pre_scored = 0
        n_full_evals = 0
        
        for i in range(0, len(options), batch_size):
            if n_full_evals >= max_combinations:
                break
            
            batch = options[i:i + batch_size]
            if not batch:
                break

            # Vectorized pre-score (filter out None values for corrective patterns)
            complexities = np.array([
                float(sum(v for v in opt.values if v is not None)) / (sum(1 for v in opt.values if v is not None) * max(1, up_to))
                for opt in batch
            ], dtype=np.float32)
            scores = base_score + 0.1 * complexities
            n_pre_scored += len(batch)

            # Top-k selection
            nb = len(batch)
            top_idx = np.argpartition(-scores, min(top_k - 1, nb - 1))[:top_k] if nb > top_k else np.arange(nb)
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            
            # Evaluate top candidates
            for j in top_idx:
                if n_full_evals >= max_combinations:
                    break
                    
                opt = batch[int(j)]
                waves = self.find_corrective_wave(idx_start=idx_start, wave_config=opt.values)
                n_full_evals += 1
                
                if not waves or len(waves) == 0:
                    continue
                
                pat = WavePattern(waves, verbose=False)
                # Note: corrective patterns don't have the same validation as impulsive
                # For now, just check if pattern was created
                    
                if pat in seen:
                    continue
                seen.add(pat)
                
                # Compute scores (corrective patterns don't have strict rules like impulsive)
                # Use a simple scoring based on pattern validity
                rule_score = 0.5  # Default score for corrective patterns
                ensemble_details = _ensemble_scorer.score_with_details(pat, rule_score=rule_score)
                
                found.append(FoundPattern(
                    pattern=pat,
                    rule_name='Corrective',
                    score=rule_score,
                    wave_config=opt.values,
                    idx_start=idx_start,
                    idx_end=waves[-1].idx_end,
                    ensemble_score=ensemble_details['ensemble_score'],
                    fib_score=ensemble_details.get('fibonacci_score', 0.5)
                ))
        
        self._last_scan_stats = {
            'n_pre_scored': n_pre_scored,
            'n_full_evals': n_full_evals,
            'pattern_type': 'corrective'
        }
        
        found.sort(key=lambda x: (x.ensemble_score if x.ensemble_score else x.score, x.idx_end), reverse=True)
        return found[:top_n]

    def scan_all_patterns(self, idx_start: int, up_to: int = 10, top_n: int = 5, max_combinations: int = None, scan_cfg: dict = None) -> List[FoundPattern]:
        """
        Scan for ALL pattern types (impulsive, corrective) and return combined top_n results.
        This maximizes recall by checking every pattern type.
        """
        all_found = []
        
        # Scan each pattern type (ask for more from each, then combine)
        impulses = self.scan_impulses(idx_start, up_to, top_n=top_n*2, max_combinations=max_combinations, scan_cfg=scan_cfg)
        all_found.extend(impulses)
        
        correctives = self.scan_correctives(idx_start, up_to, top_n=top_n*2, max_combinations=max_combinations, scan_cfg=scan_cfg)
        all_found.extend(correctives)
        
        # Sort by ensemble_score and return top_n
        all_found.sort(key=lambda x: (x.ensemble_score if x.ensemble_score is not None else x.score, x.idx_end), reverse=True)
        return all_found[:top_n]

    def scan_multi_start(self, up_to: int = 10, top_n: int = 5, max_combinations: int = None, scan_cfg: dict = None, 
                        max_starts: int = 5, pattern_types: str = 'all') -> List[FoundPattern]:
        """
        Multi-start search: try multiple pivot points (local extrema) as potential wave starts.
        Optimized with early deduplication to avoid redundant scans.
        """
        # Find local extrema (potential start points)
        extrema = self.find_local_extrema(window_size=5, min_distance=10)
        
        # Limit and prioritize start points
        if len(extrema) > max_starts:
            global_min = int(np.argmin(self.lows))
            # Keep first max_starts-1 plus global min
            extrema_set = set(extrema[:max_starts - 1])
            extrema_set.add(global_min)
            extrema = sorted(list(extrema_set))[:max_starts]
        
        # Use set for O(1) deduplication
        seen_patterns = set()
        unique_patterns = []
        
        # Map pattern_types to scan function (avoid repeated conditionals)
        scan_funcs = {
            'all': self.scan_all_patterns,
            'impulses': self.scan_impulses,
            'correctives': self.scan_correctives
        }
        scan_func = scan_funcs.get(pattern_types, self.scan_all_patterns)
        
        for start_idx in extrema:
            try:
                patterns = scan_func(
                    idx_start=start_idx,
                    up_to=up_to,
                    top_n=top_n,
                    max_combinations=max_combinations,
                    scan_cfg=scan_cfg
                )
                
                # Deduplicate on-the-fly
                for p in patterns:
                    if p.pattern not in seen_patterns:
                        seen_patterns.add(p.pattern)
                        unique_patterns.append(p)
            except Exception:
                continue
        
        # Sort by ensemble score and return top-N
        unique_patterns.sort(key=lambda x: (x.ensemble_score if x.ensemble_score else x.score, x.idx_end), reverse=True)
        return unique_patterns[:top_n]

    # -------------------------
    # New: Adaptive window growth
    # -------------------------

    @staticmethod
    def _bars_per_week(timeframe: str) -> int:
        # This is only used for window math in the example orchestration.
        # For 1D, 1 week ~ 7 bars. For 1W, 1 week ~ 1 bar.
        if timeframe.upper() == "1W":
            return 1
        return 7

    @staticmethod
    def slice_df(df: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
        return df.iloc[start : start + length].reset_index(drop=True)

    def find_best_impulse_adaptive_window(
        self,
        base_df: pd.DataFrame,
        start_row: int,
        timeframe: str = "1D",
        min_weeks: int = 4,
        max_weeks: int = 12,
        up_to: int = 10,
        grow_weeks: int = 1,
        top_n: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        For a given start_row in base_df:
        - Start with min_weeks window
        - If no impulse found, grow by grow_weeks
        - Stop at max_weeks or end of data
        """
        bars_per_week = self._bars_per_week(timeframe)
        min_len = min_weeks * bars_per_week
        max_len = max_weeks * bars_per_week
        step_len = grow_weeks * bars_per_week

        for window_len in range(min_len, max_len + 1, step_len):
            if start_row + window_len > len(base_df):
                break

            window_df = self.slice_df(base_df, start_row, window_len)

            # Build a local analyzer for this window
            wa = WaveAnalyzer(window_df, verbose=False)

            # Choose a local start inside the window (current heuristic)
            local_idx_start = int(np.argmin(np.array(list(window_df["Low"]))))

            candidates = wa.scan_impulses(idx_start=local_idx_start, up_to=up_to, top_n=top_n)

            if candidates:
                best = candidates[0]
                return {
                    "window_len": window_len,
                    "window_weeks": window_len / bars_per_week,
                    "start_row": start_row,
                    "date_start": base_df.iloc[start_row]["Date"],
                    "date_end": base_df.iloc[start_row + window_len - 1]["Date"],
                    "best": best,
                    "all": candidates,
                    "window_df": window_df,
                }

        return None

    def sliding_adaptive_impulses(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1D",
        slide_weeks: int = 1,
        min_weeks: int = 4,
        max_weeks: int = 12,
        up_to: int = 10,
        grow_weeks: int = 1,
        top_n: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Slide across df. At each start position, try adaptive window growth.
        Returns a list of detections including the window slice and best candidate.
        """
        bars_per_week = self._bars_per_week(timeframe)
        slide_step = slide_weeks * bars_per_week
        min_len = min_weeks * bars_per_week

        out = []
        start_row = 0

        while start_row <= len(df) - min_len:
            result = self.find_best_impulse_adaptive_window(
                base_df=df,
                start_row=start_row,
                timeframe=timeframe,
                min_weeks=min_weeks,
                max_weeks=max_weeks,
                up_to=up_to,
                grow_weeks=grow_weeks,
                top_n=top_n,
            )

            if result is None:
                start_row += slide_step
                continue

            out.append(result)

            # Move forward (choose overlap behavior)
            # Overlapping: slide fixed step
            start_row += slide_step

        return out
