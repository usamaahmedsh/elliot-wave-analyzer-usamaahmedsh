"""
Ensemble scoring for Elliott Wave patterns.
Combines multiple scoring signals to improve accuracy and ranking.
"""
import numpy as np
from typing import List, Optional
from models.WavePattern import WavePattern
from models.MonoWave import MonoWaveUp, MonoWaveDown


class FibonacciScorer:
    """
    Score patterns based on Fibonacci ratio alignment.
    Elliott Wave theory expects specific ratios between waves (0.382, 0.5, 0.618, 1.0, 1.618, 2.618).
    """
    
    # Golden ratios commonly seen in Elliott Waves
    GOLDEN_RATIOS = [0.382, 0.5, 0.618, 1.0, 1.272, 1.618, 2.618]
    
    # Cache for ratio distance lookups (avoid repeated min() calls)
    _ratio_distance_cache = {}
    
    @staticmethod
    def _closest_fib_distance(ratio: float) -> float:
        """Return distance to closest Fibonacci ratio. Cached for performance."""
        if ratio <= 0:
            return 999.0
        
        # Round to 3 decimals for cache key
        cache_key = round(ratio, 3)
        if cache_key in FibonacciScorer._ratio_distance_cache:
            return FibonacciScorer._ratio_distance_cache[cache_key]
        
        distances = [abs(ratio - fib) for fib in FibonacciScorer.GOLDEN_RATIOS]
        min_dist = min(distances)
        
        # Cache result
        FibonacciScorer._ratio_distance_cache[cache_key] = min_dist
        return min_dist
    
    @staticmethod
    def score_retracement(wave2_length: float, wave1_length: float) -> float:
        """
        Wave 2 typically retraces 50-61.8% of Wave 1.
        Score higher if retracement aligns with Fibonacci levels.
        """
        if wave1_length <= 0:
            return 0.0
        
        retracement = abs(wave2_length) / abs(wave1_length)
        
        # Ideal retracements for Wave 2: 0.382, 0.5, 0.618
        ideal_retracements = [0.382, 0.5, 0.618]
        min_dist = min([abs(retracement - ideal) for ideal in ideal_retracements])
        
        # Score decays with distance from ideal
        if min_dist < 0.05:
            return 1.0
        elif min_dist < 0.15:
            return 0.8
        elif min_dist < 0.25:
            return 0.5
        else:
            return 0.2
    
    @staticmethod
    def score_extension(wave3_length: float, wave1_length: float) -> float:
        """
        Wave 3 is often the longest and typically 1.618x Wave 1.
        """
        if wave1_length <= 0:
            return 0.0
        
        extension = abs(wave3_length) / abs(wave1_length)
        
        # Wave 3 should be >= Wave 1 (rule) and often 1.618x
        if extension < 1.0:
            return 0.0  # Rule violation
        
        # Distance to 1.618
        dist_to_golden = abs(extension - 1.618)
        
        if dist_to_golden < 0.1:
            return 1.0
        elif dist_to_golden < 0.3:
            return 0.8
        elif dist_to_golden < 0.5:
            return 0.6
        else:
            return 0.4
    
    @staticmethod
    def score_wave5_projection(wave5_length: float, wave1_length: float, wave3_length: float) -> float:
        """
        Wave 5 often equals Wave 1 or is 0.618 * Wave 3.
        """
        if wave1_length <= 0 or wave3_length <= 0:
            return 0.5
        
        # Check both projections
        ratio_to_w1 = abs(wave5_length) / abs(wave1_length)
        ratio_to_w3 = abs(wave5_length) / abs(wave3_length)
        
        dist_to_w1_equal = abs(ratio_to_w1 - 1.0)
        dist_to_w3_golden = abs(ratio_to_w3 - 0.618)
        
        score1 = 1.0 if dist_to_w1_equal < 0.1 else max(0.3, 1.0 - dist_to_w1_equal)
        score2 = 1.0 if dist_to_w3_golden < 0.1 else max(0.3, 1.0 - dist_to_w3_golden)
        
        return max(score1, score2)
    
    @staticmethod
    def score_pattern(pattern: WavePattern) -> float:
        """
        Compute overall Fibonacci alignment score for a pattern.
        Returns score in [0, 1].
        """
        waves = pattern.waves
        
        # Check if we have a 5-wave impulse
        if 'wave5' in waves:
            try:
                w1_len = abs(waves['wave1'].length) if waves['wave1'].length else 0
                w2_len = abs(waves['wave2'].length) if waves['wave2'].length else 0
                w3_len = abs(waves['wave3'].length) if waves['wave3'].length else 0
                w5_len = abs(waves['wave5'].length) if waves['wave5'].length else 0
                
                scores = []
                
                if w1_len > 0 and w2_len > 0:
                    scores.append(FibonacciScorer.score_retracement(w2_len, w1_len))
                
                if w1_len > 0 and w3_len > 0:
                    scores.append(FibonacciScorer.score_extension(w3_len, w1_len))
                
                if w1_len > 0 and w3_len > 0 and w5_len > 0:
                    scores.append(FibonacciScorer.score_wave5_projection(w5_len, w1_len, w3_len))
                
                return np.mean(scores) if scores else 0.5
            except Exception:
                return 0.5
        
        # For 3-wave corrections (ABC)
        elif 'wave3' in waves and 'wave5' not in waves:
            try:
                w1_len = abs(waves['wave1'].length) if waves['wave1'].length else 0
                w2_len = abs(waves['wave2'].length) if waves['wave2'].length else 0
                w3_len = abs(waves['wave3'].length) if waves['wave3'].length else 0
                
                # Wave C often equals Wave A or is 0.618 * Wave A
                if w1_len > 0 and w3_len > 0:
                    ratio = w3_len / w1_len
                    dist_equal = abs(ratio - 1.0)
                    dist_golden = abs(ratio - 0.618)
                    
                    if min(dist_equal, dist_golden) < 0.1:
                        return 0.9
                    elif min(dist_equal, dist_golden) < 0.2:
                        return 0.7
                    else:
                        return 0.5
            except Exception:
                return 0.5
        
        return 0.5


class TimeProportionScorer:
    """
    Score patterns based on time relationships between waves.
    Elliott Wave theory suggests time proportions also follow Fibonacci ratios.
    """
    
    @staticmethod
    def score_pattern(pattern: WavePattern) -> float:
        """
        Score based on time symmetry and Fibonacci time ratios.
        """
        try:
            waves = pattern.waves
            
            # Extract time durations (number of bars)
            durations = []
            for key in sorted(waves.keys()):
                wave = waves[key]
                duration = wave.idx_end - wave.idx_start
                durations.append(duration)
            
            if len(durations) < 3:
                return 0.5
            
            # Check time ratios
            scores = []
            
            # Wave 2 vs Wave 1 time
            if len(durations) >= 2 and durations[0] > 0:
                time_ratio = durations[1] / durations[0]
                dist = FibonacciScorer._closest_fib_distance(time_ratio)
                scores.append(max(0.3, 1.0 - dist))
            
            # Wave 3 vs Wave 1 time
            if len(durations) >= 3 and durations[0] > 0:
                time_ratio = durations[2] / durations[0]
                dist = FibonacciScorer._closest_fib_distance(time_ratio)
                scores.append(max(0.3, 1.0 - dist))
            
            return np.mean(scores) if scores else 0.5
        except Exception:
            return 0.5


class ComplexityScorer:
    """
    Score patterns based on wave degree complexity.
    Simpler patterns (lower degree) are often more reliable.
    """
    
    @staticmethod
    def score_pattern(pattern: WavePattern) -> float:
        """
        Lower degree patterns score higher (they're clearer).
        """
        try:
            degree = pattern.degree
            # Normalize: degree 0 = 1.0, degree 10 = 0.5, degree 20+ = 0.3
            if degree <= 0:
                return 1.0
            elif degree <= 10:
                return 1.0 - (degree * 0.05)
            else:
                return max(0.3, 0.5 - (degree - 10) * 0.01)
        except Exception:
            return 0.5


class EnsembleScorer:
    """
    Combine multiple scoring signals into a single ensemble score.
    """
    
    def __init__(self, 
                 fib_weight: float = 0.5,
                 rule_weight: float = 0.3,
                 time_weight: float = 0.1,
                 complexity_weight: float = 0.1):
        """
        Initialize ensemble scorer with configurable weights.
        
        Args:
            fib_weight: Weight for Fibonacci ratio alignment
            rule_weight: Weight for Elliott Wave rule satisfaction
            time_weight: Weight for time proportion analysis
            complexity_weight: Weight for pattern complexity/clarity
        """
        self.fib_weight = fib_weight
        self.rule_weight = rule_weight
        self.time_weight = time_weight
        self.complexity_weight = complexity_weight
        
        # Normalize weights
        total = fib_weight + rule_weight + time_weight + complexity_weight
        if total > 0:
            self.fib_weight /= total
            self.rule_weight /= total
            self.time_weight /= total
            self.complexity_weight /= total
    
    def score(self, pattern: WavePattern, rule_score: float = None) -> float:
        """
        Compute ensemble score for a pattern.
        
        Args:
            pattern: WavePattern to score
            rule_score: Pre-computed rule satisfaction score (if available)
        
        Returns:
            Ensemble score in [0, 1]
        """
        try:
            # Get individual scores
            fib_score = FibonacciScorer.score_pattern(pattern)
            time_score = TimeProportionScorer.score_pattern(pattern)
            complexity_score = ComplexityScorer.score_pattern(pattern)
            
            # If rule_score not provided, use a default
            if rule_score is None:
                rule_score = 0.5
            
            # Weighted combination
            ensemble = (
                self.fib_weight * fib_score +
                self.rule_weight * rule_score +
                self.time_weight * time_score +
                self.complexity_weight * complexity_score
            )
            
            return max(0.0, min(1.0, ensemble))
        except Exception:
            return 0.5
    
    def score_with_details(self, pattern: WavePattern, rule_score: float = None) -> dict:
        """
        Compute ensemble score and return detailed breakdown.
        """
        try:
            fib_score = FibonacciScorer.score_pattern(pattern)
            time_score = TimeProportionScorer.score_pattern(pattern)
            complexity_score = ComplexityScorer.score_pattern(pattern)
            
            if rule_score is None:
                rule_score = 0.5
            
            ensemble = (
                self.fib_weight * fib_score +
                self.rule_weight * rule_score +
                self.time_weight * time_score +
                self.complexity_weight * complexity_score
            )
            
            return {
                'ensemble_score': max(0.0, min(1.0, ensemble)),
                'fibonacci_score': fib_score,
                'rule_score': rule_score,
                'time_score': time_score,
                'complexity_score': complexity_score,
                'weights': {
                    'fib': self.fib_weight,
                    'rule': self.rule_weight,
                    'time': self.time_weight,
                    'complexity': self.complexity_weight
                }
            }
        except Exception:
            return {
                'ensemble_score': 0.5,
                'error': 'scoring_failed'
            }
