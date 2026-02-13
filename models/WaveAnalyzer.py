from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from models.MonoWave import MonoWaveUp, MonoWaveDown
from models.WaveOptions import WaveOptionsGenerator5, WaveOptionsGenerator3
from models.WaveCycle import WaveCycle
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, Correction, TDWave, LeadingDiagonal


@dataclass
class FoundPattern:
    pattern: WavePattern
    rule_name: str
    score: float
    wave_config: List[int]
    idx_start: int
    idx_end: int


_options_cache = {}


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

        # Guard against invalid index ranges which can produce zero-length slices
        try:
            s_start = int(wave2.low_idx) if wave2.low_idx is not None else None
            s_end = int(wave4.low_idx) if wave4.low_idx is not None else None
        except Exception:
            s_start = None
            s_end = None

        if s_start is None or s_end is None or s_end <= s_start:
            # invalid range -> reject this configuration
            return False

        if wave2.low > np.min(self.lows[s_start:s_end]):
            return False

        wave5 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave4_end, skip=wave_config[4])
        wave5.label = "5"
        wave5_end = wave5.idx_end
        if wave5_end is None:
            if self.verbose:
                print("Wave 5 has no End in Data")
            return False

        # Guard the slice between wave4.low_idx and wave5.high_idx
        try:
            s4 = int(wave4.low_idx) if wave4.low_idx is not None else None
            e5 = int(wave5.high_idx) if wave5.high_idx is not None else None
        except Exception:
            s4 = None
            e5 = None

        if s4 is None or e5 is None or e5 <= s4:
            # invalid range -> reject
            return False

        slice_vals = self.lows[s4:e5]
        if slice_vals.size > 0 and slice_vals.any() and wave4.low > np.min(slice_vals):
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

        # read batching options from scan_cfg if provided
        if scan_cfg is None:
            scan_cfg = {}
        gpu_enabled = bool(scan_cfg.get('gpu_enabled', False))
        gpu_batch = int(scan_cfg.get('gpu_batch_size', 512))
        gpu_top_k = int(scan_cfg.get('gpu_top_k', 64))

        impulse = Impulse("impulse")
        leading_diagonal = LeadingDiagonal("leading_diagonal")
        rules_to_check = [impulse, leading_diagonal]

        # Cheap numba-accelerated pre-filter: skip windows that do not contain
        # enough local extrema (peaks/troughs) to form a 5-wave impulsive structure.
        try:
            from models.functions import count_extrema
            # count extrema in the tail of the arrays from idx_start
            n_ext = count_extrema(self.lows[idx_start:])
            if n_ext < 4:
                return []
        except Exception:
            # if the fast filter isn't available for some reason, continue normally
            pass

        found: List[FoundPattern] = []
        seen = set()

        processed = 0
        # If GPU batching is enabled, process options in batches: score cheaply and only
        # fully evaluate the top-k candidates per batch to reduce heavy WavePattern work.
        if gpu_enabled:
            try:
                from pipeline.gpu_accel import GPUAccelerator
                gpu = GPUAccelerator()
                use_gpu = True
            except Exception:
                gpu = None
                use_gpu = False

            n_opts = len(wave_options_impulse_options)
            i = 0
            while i < n_opts:
                batch = wave_options_impulse_options[i : i + gpu_batch]
                # compute cheap features per candidate: complexity metric
                feats = []
                for opt in batch:
                    # complexity: normalized sum of skips
                    cfg_vals = opt.values
                    complexity = float(sum(cfg_vals)) / (len(cfg_vals) * max(1, up_to))
                    # window-level proxies: use lows/highs around idx_start for quick range
                    try:
                        lo_val = float(np.min(self.lows[idx_start: idx_start + 10]))
                        hi_val = float(np.max(self.highs[idx_start: idx_start + 10]))
                        vol_proxy = float(np.std(self.lows[idx_start: idx_start + 10]))
                    except Exception:
                        lo_val = 0.0
                        hi_val = 0.0
                        vol_proxy = 0.0
                    feats.append([vol_proxy, (hi_val - lo_val) / (lo_val + 1e-9), 0.0, complexity])

                if use_gpu and len(feats) > 0:
                    scores = gpu.score_features(feats)
                else:
                    # fallback scoring
                    scores = [0.4 * f[0] + 0.3 * f[1] + 0.2 * f[2] + 0.1 * f[3] for f in feats]

                # pick top-k candidates in this batch
                indexed = list(enumerate(batch))
                scored = list(zip(indexed, scores))
                scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                top_candidates = [x[0][1] for x in scored_sorted[:gpu_top_k]]

                # evaluate the top candidates fully
                for opt in top_candidates:
                    processed += 1
                    if max_combinations is not None and processed > max_combinations:
                        if self.verbose:
                            print(f"scan_impulses reached max_combinations={max_combinations}, stopping early")
                        i = n_opts
                        break
                    waves_up = self.find_impulsive_wave(idx_start=idx_start, wave_config=opt.values)
                    if not waves_up:
                        continue

                    wp = WavePattern(waves_up, verbose=False)

                    for rule in rules_to_check:
                        if wp.check_rule(rule):
                            if wp in seen:
                                continue
                            seen.add(wp)

                            score = 0.0
                            if hasattr(wp, "score_rule"):
                                score = float(wp.score_rule(rule))

                            found.append(
                                FoundPattern(
                                    pattern=wp,
                                    rule_name=rule.name,
                                    score=score,
                                    wave_config=opt.values,
                                    idx_start=wp.idx_start,
                                    idx_end=wp.idx_end,
                                )
                            )

                i += gpu_batch

        else:
            for opt in wave_options_impulse_options:
                processed += 1
                if max_combinations is not None and processed > max_combinations:
                    # safety stop - return what we've found so far
                    if self.verbose:
                        print(f"scan_impulses reached max_combinations={max_combinations}, stopping early")
                    break
                waves_up = self.find_impulsive_wave(idx_start=idx_start, wave_config=opt.values)
                if not waves_up:
                    continue

                wp = WavePattern(waves_up, verbose=False)

                for rule in rules_to_check:
                    if wp.check_rule(rule):
                        if wp in seen:
                            continue
                        seen.add(wp)

                        score = 0.0
                        if hasattr(wp, "score_rule"):
                            score = float(wp.score_rule(rule))

                        found.append(
                            FoundPattern(
                                pattern=wp,
                                rule_name=rule.name,
                                score=score,
                                wave_config=opt.values,
                                idx_start=wp.idx_start,
                                idx_end=wp.idx_end,
                            )
                        )

        # Sort by score (desc), then by longer coverage (idx_end desc)
        found.sort(key=lambda x: (x.score, x.idx_end), reverse=True)
        return found[:top_n]

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
