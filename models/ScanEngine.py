"""
Optimized scanning engine for Elliott Wave patterns.
Consolidates common logic to reduce code duplication and improve performance.
"""
import numpy as np
from typing import List, Set, Callable, Optional
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal


def _compute_ensemble_score(pattern: WavePattern, rule_score: float, ensemble_scorer) -> tuple:
    """Compute ensemble score once and return (ensemble_score, fib_score)."""
    ensemble_details = ensemble_scorer.score_with_details(pattern, rule_score=rule_score)
    return ensemble_details['ensemble_score'], ensemble_details.get('fibonacci_score', 0.5)


def _scan_with_batching(
    wave_options: list,
    idx_start: int,
    up_to: int,
    top_n: int,
    max_combinations: int,
    batch_size: int,
    top_k: int,
    find_wave_func: Callable,
    pattern_creator: Callable,
    rules_to_check: list,
    base_score: float,
    ensemble_scorer,
    FoundPattern: type
) -> tuple:
    """
    Unified batching logic for both impulse and corrective patterns.
    Returns (found_patterns, n_pre_scored, n_full_evals)
    """
    found: List = []
    seen: Set = set()
    n_pre_scored = 0
    n_full_evals = 0

    for i in range(0, len(wave_options), batch_size):
        if n_full_evals >= max_combinations:
            break
        
        batch = wave_options[i : i + batch_size]
        if not batch:
            break

        # Vectorized pre-score
        complexities = np.array(
            [float(sum(opt.values)) / (len(opt.values) * max(1, up_to)) for opt in batch],
            dtype=np.float32
        )
        scores = base_score + 0.1 * complexities
        n_pre_scored += len(batch)

        # Top-k selection
        nb = len(batch)
        if nb > top_k:
            top_idx = np.argpartition(-scores, min(top_k - 1, nb - 1))[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)

        # Evaluate top candidates
        for j in top_idx:
            if n_full_evals >= max_combinations:
                break
            
            opt = batch[int(j)]
            waves = find_wave_func(idx_start=idx_start, wave_config=opt.values)
            n_full_evals += 1
            
            if not waves:
                continue

            # Create pattern and check rules
            pattern = pattern_creator(waves)
            
            for rule in rules_to_check:
                if pattern.check_rule(rule):
                    if pattern in seen:
                        continue
                    seen.add(pattern)

                    # Compute scores
                    rule_score = float(pattern.score_rule(rule)) if hasattr(pattern, "score_rule") else pattern.score()
                    ensemble_score, fib_score = _compute_ensemble_score(pattern, rule_score, ensemble_scorer)

                    found.append(
                        FoundPattern(
                            pattern=pattern,
                            rule_name=rule.name,
                            score=rule_score,
                            wave_config=opt.values,
                            idx_start=pattern.idx_start,
                            idx_end=pattern.idx_end,
                            ensemble_score=ensemble_score,
                            fib_score=fib_score
                        )
                    )

    return found, n_pre_scored, n_full_evals


def _precompute_window_features(lows: np.ndarray, highs: np.ndarray, idx_start: int, cache: dict) -> float:
    """
    Compute cheap window-level features once and cache.
    Returns base_score for pre-scoring.
    """
    cache_key = (id(lows), idx_start)
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        slice_window = lows[idx_start: idx_start + 10]
        if slice_window.size == 0:
            base_score = 0.0
        else:
            lo_val = float(np.min(slice_window))
            hi_val = float(np.max(highs[idx_start: idx_start + 10]))
            vol_proxy = float(np.std(slice_window)) if slice_window.size > 1 else 0.0
            base_score = 0.4 * vol_proxy + 0.3 * ((hi_val - lo_val) / (lo_val + 1e-9))
        cache[cache_key] = base_score
        return base_score
    except Exception:
        return 0.0
