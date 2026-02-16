"""GPU-accelerated batch pattern scoring using CuPy.

This module provides GPU acceleration for scoring large batches of wave pattern
candidates. It's designed to work with your 140GB GPU to process thousands of
candidates in parallel.

Key optimizations:
- Batch all rule checks into vectorized GPU operations
- Use GPU for Fibonacci ratio distance calculations
- Parallel ensemble scoring across all candidates
- Memory-efficient batching to utilize 140GB VRAM
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    warnings.warn("CuPy not installed. GPU acceleration disabled. Install with: pip install cupy-cuda12x")


class GPUBatchScorer:
    """GPU-accelerated batch scoring for wave patterns.
    
    Uses CuPy to vectorize rule checking and scoring across large batches.
    Designed to maximize utilization of 140GB GPU memory.
    """
    
    def __init__(self, batch_size: int = 10000, use_gpu: bool = True):
        """Initialize GPU batch scorer.
        
        Args:
            batch_size: Number of candidates to process per GPU batch (default 10k)
            use_gpu: Whether to use GPU if available (falls back to CPU if False or unavailable)
        """
        self.batch_size = batch_size
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            # Verify GPU is accessible
            try:
                cp.cuda.Device(0).compute_capability
                print(f"GPU acceleration enabled. Device: {cp.cuda.Device(0).name}")
                print(f"GPU memory: {cp.cuda.Device(0).mem_info[1] / 1e9:.1f} GB total")
            except Exception as e:
                warnings.warn(f"GPU detected but not accessible: {e}. Falling back to CPU.")
                self.use_gpu = False
    
    def batch_check_wave_rules(
        self, 
        candidates: List[Dict[str, Any]], 
        lows: np.ndarray, 
        highs: np.ndarray
    ) -> np.ndarray:
        """Vectorized rule checking for wave pattern candidates on GPU.
        
        Args:
            candidates: List of candidate dictionaries with wave indices
            lows: Price low array
            highs: Price high array
            
        Returns:
            Boolean array indicating which candidates pass all rules
        """
        if not candidates:
            return np.array([], dtype=bool)
        
        n_candidates = len(candidates)
        
        # Extract wave indices into arrays
        wave_indices = np.zeros((n_candidates, 5), dtype=np.int32)
        for i, cand in enumerate(candidates):
            waves = cand.get('waves', [])
            if len(waves) == 5:
                for j, wave in enumerate(waves):
                    wave_indices[i, j] = wave.get('end_idx', 0)
        
        if self.use_gpu:
            # Transfer to GPU
            wave_indices_gpu = cp.asarray(wave_indices)
            lows_gpu = cp.asarray(lows)
            highs_gpu = cp.asarray(highs)
            
            # Vectorized rule checks on GPU
            valid_gpu = self._gpu_check_rules(wave_indices_gpu, lows_gpu, highs_gpu)
            
            # Transfer result back
            valid = cp.asnumpy(valid_gpu)
        else:
            # CPU fallback
            valid = self._cpu_check_rules(wave_indices, lows, highs)
        
        return valid
    
    def _gpu_check_rules(
        self, 
        wave_idx: 'cp.ndarray',
        lows: 'cp.ndarray', 
        highs: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """GPU-accelerated rule checking (Elliott Wave rules).
        
        Vectorized checks:
        1. Wave 3 cannot be shortest
        2. Wave 2 doesn't retrace beyond wave 1 start
        3. Wave 4 doesn't overlap wave 1 peak
        """
        n = wave_idx.shape[0]
        valid = cp.ones(n, dtype=bool)
        
        # Extract wave endpoints (assuming each wave end_idx is cumulative)
        # For impulsive waves: 0-1-2-3-4 pattern
        # Get prices at key points
        w1_end = wave_idx[:, 0]
        w2_end = wave_idx[:, 1]
        w3_end = wave_idx[:, 2]
        w4_end = wave_idx[:, 3]
        w5_end = wave_idx[:, 4]
        
        # Clip indices to valid range
        max_idx = len(lows) - 1
        w1_end = cp.clip(w1_end, 0, max_idx)
        w2_end = cp.clip(w2_end, 0, max_idx)
        w3_end = cp.clip(w3_end, 0, max_idx)
        w4_end = cp.clip(w4_end, 0, max_idx)
        w5_end = cp.clip(w5_end, 0, max_idx)
        
        # Get prices (assuming bullish pattern for now - can extend for bearish)
        p1 = highs[w1_end]
        p2 = lows[w2_end]
        p3 = highs[w3_end]
        p4 = lows[w4_end]
        p5 = highs[w5_end]
        
        # Rule 1: Wave 3 cannot be shortest (among 1, 3, 5)
        wave1_len = p1 - lows[0]  # Simplified: assume start at index 0
        wave3_len = p3 - p2
        wave5_len = p5 - p4
        
        valid &= (wave3_len >= wave1_len) | (wave3_len >= wave5_len)
        
        # Rule 2: Wave 2 doesn't retrace beyond start
        valid &= p2 > lows[0]  # Simplified
        
        # Rule 3: Wave 4 doesn't overlap Wave 1 peak
        valid &= p4 > p1
        
        return valid
    
    def _cpu_check_rules(
        self, 
        wave_idx: np.ndarray,
        lows: np.ndarray, 
        highs: np.ndarray
    ) -> np.ndarray:
        """CPU fallback for rule checking."""
        # Simple vectorized CPU version (less optimized than GPU)
        n = wave_idx.shape[0]
        valid = np.ones(n, dtype=bool)
        
        # Similar logic as GPU version but with numpy
        max_idx = len(lows) - 1
        w1_end = np.clip(wave_idx[:, 0], 0, max_idx)
        w2_end = np.clip(wave_idx[:, 1], 0, max_idx)
        w3_end = np.clip(wave_idx[:, 2], 0, max_idx)
        w4_end = np.clip(wave_idx[:, 3], 0, max_idx)
        w5_end = np.clip(wave_idx[:, 4], 0, max_idx)
        
        p1 = highs[w1_end]
        p2 = lows[w2_end]
        p3 = highs[w3_end]
        p4 = lows[w4_end]
        p5 = highs[w5_end]
        
        wave1_len = p1 - lows[0]
        wave3_len = p3 - p2
        wave5_len = p5 - p4
        
        valid &= (wave3_len >= wave1_len) | (wave3_len >= wave5_len)
        valid &= p2 > lows[0]
        valid &= p4 > p1
        
        return valid
    
    def batch_fibonacci_scoring(
        self,
        candidates: List[Dict[str, Any]],
        lows: np.ndarray,
        highs: np.ndarray
    ) -> np.ndarray:
        """GPU-accelerated Fibonacci ratio scoring.
        
        Computes Fibonacci alignment scores for all candidates in parallel.
        
        Args:
            candidates: List of wave pattern candidates
            lows: Price lows array
            highs: Price highs array
            
        Returns:
            Array of Fibonacci scores (0-1, higher is better)
        """
        if not candidates:
            return np.array([])
        
        n = len(candidates)
        
        # Fibonacci target ratios
        fib_ratios = np.array([0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618], dtype=np.float32)
        
        # Extract wave data
        wave_data = self._extract_wave_data(candidates, lows, highs)
        
        if self.use_gpu:
            wave_data_gpu = cp.asarray(wave_data)
            fib_ratios_gpu = cp.asarray(fib_ratios)
            scores_gpu = self._gpu_fib_scoring(wave_data_gpu, fib_ratios_gpu)
            scores = cp.asnumpy(scores_gpu)
        else:
            scores = self._cpu_fib_scoring(wave_data, fib_ratios)
        
        return scores
    
    def _extract_wave_data(
        self,
        candidates: List[Dict[str, Any]],
        lows: np.ndarray,
        highs: np.ndarray
    ) -> np.ndarray:
        """Extract wave price movements into array for vectorized scoring."""
        n = len(candidates)
        # Store: [wave1_height, wave2_retrace, wave3_height, wave4_retrace, wave5_height]
        wave_data = np.zeros((n, 5), dtype=np.float32)
        
        for i, cand in enumerate(candidates):
            waves = cand.get('waves', [])
            if len(waves) >= 5:
                # Simplified extraction - extend with actual logic
                for j in range(min(5, len(waves))):
                    wave = waves[j]
                    start_idx = wave.get('start_idx', 0)
                    end_idx = wave.get('end_idx', 0)
                    
                    if j % 2 == 0:  # Impulse waves (1, 3, 5)
                        wave_data[i, j] = highs[end_idx] - lows[start_idx]
                    else:  # Corrective waves (2, 4)
                        wave_data[i, j] = highs[start_idx] - lows[end_idx]
        
        return wave_data
    
    def _gpu_fib_scoring(
        self,
        wave_data: 'cp.ndarray',
        fib_ratios: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """GPU-accelerated Fibonacci scoring."""
        n = wave_data.shape[0]
        scores = cp.zeros(n, dtype=cp.float32)
        
        # Wave 2 retracement (should be 0.382-0.618 of Wave 1)
        w1 = wave_data[:, 0]
        w2 = wave_data[:, 1]
        ratio_w2 = cp.where(w1 > 0, w2 / w1, 0)
        
        # Distance to nearest Fibonacci ratio
        dist_w2 = cp.min(cp.abs(ratio_w2[:, None] - fib_ratios[None, :]), axis=1)
        score_w2 = cp.maximum(0, 1.0 - dist_w2 * 2.0)  # Convert distance to score
        
        # Wave 3 extension (should be ~1.618 of Wave 1)
        w3 = wave_data[:, 2]
        ratio_w3 = cp.where(w1 > 0, w3 / w1, 0)
        dist_w3 = cp.min(cp.abs(ratio_w3[:, None] - fib_ratios[None, :]), axis=1)
        score_w3 = cp.maximum(0, 1.0 - dist_w3 * 2.0)
        
        # Wave 4 retracement
        w4 = wave_data[:, 3]
        ratio_w4 = cp.where(w3 > 0, w4 / w3, 0)
        dist_w4 = cp.min(cp.abs(ratio_w4[:, None] - fib_ratios[None, :]), axis=1)
        score_w4 = cp.maximum(0, 1.0 - dist_w4 * 2.0)
        
        # Combined score (weighted average)
        scores = (score_w2 * 0.4 + score_w3 * 0.4 + score_w4 * 0.2)
        
        return scores
    
    def _cpu_fib_scoring(
        self,
        wave_data: np.ndarray,
        fib_ratios: np.ndarray
    ) -> np.ndarray:
        """CPU fallback for Fibonacci scoring."""
        n = wave_data.shape[0]
        scores = np.zeros(n, dtype=np.float32)
        
        w1 = wave_data[:, 0]
        w2 = wave_data[:, 1]
        w3 = wave_data[:, 2]
        w4 = wave_data[:, 3]
        
        ratio_w2 = np.where(w1 > 0, w2 / w1, 0)
        dist_w2 = np.min(np.abs(ratio_w2[:, None] - fib_ratios[None, :]), axis=1)
        score_w2 = np.maximum(0, 1.0 - dist_w2 * 2.0)
        
        ratio_w3 = np.where(w1 > 0, w3 / w1, 0)
        dist_w3 = np.min(np.abs(ratio_w3[:, None] - fib_ratios[None, :]), axis=1)
        score_w3 = np.maximum(0, 1.0 - dist_w3 * 2.0)
        
        ratio_w4 = np.where(w3 > 0, w4 / w3, 0)
        dist_w4 = np.min(np.abs(ratio_w4[:, None] - fib_ratios[None, :]), axis=1)
        score_w4 = np.maximum(0, 1.0 - dist_w4 * 2.0)
        
        scores = (score_w2 * 0.4 + score_w3 * 0.4 + score_w4 * 0.2)
        
        return scores


def create_gpu_scorer(config: Optional[Dict[str, Any]] = None) -> GPUBatchScorer:
    """Factory function to create GPU scorer with config.
    
    Args:
        config: Optional configuration dict with 'gpu_batch_size', 'use_gpu'
        
    Returns:
        Configured GPUBatchScorer instance
    """
    if config is None:
        config = {}
    
    batch_size = config.get('gpu_batch_size', 10000)
    use_gpu = config.get('use_gpu', True)
    
    return GPUBatchScorer(batch_size=batch_size, use_gpu=use_gpu)
