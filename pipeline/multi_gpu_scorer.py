"""Multi-GPU support for distributed pattern scoring.

This module extends GPU acceleration to support multiple GPUs when available.
With multiple GPUs, work can be distributed across devices for additional speedup.

Usage:
    # Auto-detect and use all available GPUs
    scorer = create_multi_gpu_scorer()
    
    # Use specific GPUs
    scorer = create_multi_gpu_scorer(device_ids=[0, 1])
"""

import numpy as np
from typing import List, Dict, Any, Optional
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class MultiGPUScorer:
    """Distribute batch scoring across multiple GPUs.
    
    Automatically detects available GPUs and distributes workload.
    Falls back to single GPU or CPU if multiple GPUs unavailable.
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None, batch_size_per_gpu: int = 10000):
        """Initialize multi-GPU scorer.
        
        Args:
            device_ids: List of GPU device IDs to use (None = auto-detect all)
            batch_size_per_gpu: Batch size per GPU device
        """
        self.batch_size_per_gpu = batch_size_per_gpu
        
        if not GPU_AVAILABLE:
            warnings.warn("CuPy not available. Multi-GPU disabled.")
            self.devices = []
            self.num_gpus = 0
            return
        
        # Auto-detect GPUs
        try:
            if device_ids is None:
                self.num_gpus = cp.cuda.runtime.getDeviceCount()
                self.devices = list(range(self.num_gpus))
            else:
                self.devices = device_ids
                self.num_gpus = len(device_ids)
            
            # Verify each device
            verified_devices = []
            for dev_id in self.devices:
                try:
                    with cp.cuda.Device(dev_id):
                        # Test device access
                        _ = cp.array([1, 2, 3])
                        verified_devices.append(dev_id)
                        
                        # Print device info
                        dev = cp.cuda.Device(dev_id)
                        mem_info = dev.mem_info
                        print(f"GPU {dev_id}: {dev.name}, {mem_info[1]/1e9:.1f}GB total")
                except Exception as e:
                    warnings.warn(f"GPU {dev_id} not accessible: {e}")
            
            self.devices = verified_devices
            self.num_gpus = len(verified_devices)
            
            if self.num_gpus == 0:
                warnings.warn("No GPUs accessible. Falling back to CPU.")
            else:
                print(f"Multi-GPU enabled: {self.num_gpus} devices")
                
        except Exception as e:
            warnings.warn(f"Multi-GPU initialization failed: {e}")
            self.devices = []
            self.num_gpus = 0
    
    def batch_score_distributed(
        self,
        candidates: List[Dict[str, Any]],
        lows: np.ndarray,
        highs: np.ndarray,
        score_fn: str = 'fibonacci'
    ) -> np.ndarray:
        """Score candidates distributed across multiple GPUs.
        
        Args:
            candidates: List of pattern candidates
            lows: Price lows array
            highs: Price highs array
            score_fn: Scoring function to use ('fibonacci', 'ensemble', etc.)
            
        Returns:
            Array of scores for all candidates
        """
        if self.num_gpus == 0:
            # CPU fallback
            return self._cpu_scoring(candidates, lows, highs, score_fn)
        
        if self.num_gpus == 1:
            # Single GPU - no distribution needed
            return self._single_gpu_scoring(candidates, lows, highs, score_fn, self.devices[0])
        
        # Multi-GPU distribution
        n_candidates = len(candidates)
        candidates_per_gpu = (n_candidates + self.num_gpus - 1) // self.num_gpus
        
        # Split candidates across GPUs
        results = []
        for i, dev_id in enumerate(self.devices):
            start_idx = i * candidates_per_gpu
            end_idx = min((i + 1) * candidates_per_gpu, n_candidates)
            
            if start_idx >= n_candidates:
                break
            
            chunk = candidates[start_idx:end_idx]
            
            # Score on this GPU
            chunk_scores = self._single_gpu_scoring(chunk, lows, highs, score_fn, dev_id)
            results.append(chunk_scores)
        
        # Concatenate results
        all_scores = np.concatenate(results)
        return all_scores
    
    def _single_gpu_scoring(
        self,
        candidates: List[Dict[str, Any]],
        lows: np.ndarray,
        highs: np.ndarray,
        score_fn: str,
        device_id: int
    ) -> np.ndarray:
        """Score candidates on a single GPU device."""
        with cp.cuda.Device(device_id):
            # Transfer data to GPU
            lows_gpu = cp.asarray(lows)
            highs_gpu = cp.asarray(highs)
            
            # Extract candidate data
            wave_data = self._extract_wave_data_gpu(candidates, lows_gpu, highs_gpu)
            
            # Score based on function
            if score_fn == 'fibonacci':
                scores_gpu = self._gpu_fibonacci_scoring(wave_data)
            elif score_fn == 'ensemble':
                scores_gpu = self._gpu_ensemble_scoring(wave_data)
            else:
                raise ValueError(f"Unknown score function: {score_fn}")
            
            # Transfer back to CPU
            scores = cp.asnumpy(scores_gpu)
            return scores
    
    def _extract_wave_data_gpu(
        self,
        candidates: List[Dict[str, Any]],
        lows_gpu: 'cp.ndarray',
        highs_gpu: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """Extract wave data on GPU."""
        n = len(candidates)
        wave_data = cp.zeros((n, 5), dtype=cp.float32)
        
        # Extract wave heights/retracements
        for i, cand in enumerate(candidates):
            waves = cand.get('waves', [])
            for j in range(min(5, len(waves))):
                wave = waves[j]
                start_idx = wave.get('start_idx', 0)
                end_idx = wave.get('end_idx', 0)
                
                if j % 2 == 0:  # Impulse
                    wave_data[i, j] = highs_gpu[end_idx] - lows_gpu[start_idx]
                else:  # Corrective
                    wave_data[i, j] = highs_gpu[start_idx] - lows_gpu[end_idx]
        
        return wave_data
    
    def _gpu_fibonacci_scoring(self, wave_data: 'cp.ndarray') -> 'cp.ndarray':
        """GPU Fibonacci scoring."""
        fib_ratios = cp.array([0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618], dtype=cp.float32)
        
        n = wave_data.shape[0]
        scores = cp.zeros(n, dtype=cp.float32)
        
        w1 = wave_data[:, 0]
        w2 = wave_data[:, 1]
        w3 = wave_data[:, 2]
        
        # Wave 2 retracement
        ratio_w2 = cp.where(w1 > 0, w2 / w1, 0)
        dist_w2 = cp.min(cp.abs(ratio_w2[:, None] - fib_ratios[None, :]), axis=1)
        score_w2 = cp.maximum(0, 1.0 - dist_w2 * 2.0)
        
        # Wave 3 extension
        ratio_w3 = cp.where(w1 > 0, w3 / w1, 0)
        dist_w3 = cp.min(cp.abs(ratio_w3[:, None] - fib_ratios[None, :]), axis=1)
        score_w3 = cp.maximum(0, 1.0 - dist_w3 * 2.0)
        
        # Combined
        scores = score_w2 * 0.5 + score_w3 * 0.5
        return scores
    
    def _gpu_ensemble_scoring(self, wave_data: 'cp.ndarray') -> 'cp.ndarray':
        """GPU ensemble scoring (Fibonacci + time + complexity)."""
        # Simplified ensemble - extend with full logic
        fib_scores = self._gpu_fibonacci_scoring(wave_data)
        
        # Add time and complexity scoring here
        # For now, just return Fibonacci component
        return fib_scores
    
    def _cpu_scoring(
        self,
        candidates: List[Dict[str, Any]],
        lows: np.ndarray,
        highs: np.ndarray,
        score_fn: str
    ) -> np.ndarray:
        """CPU fallback for scoring."""
        # Simple CPU implementation
        n = len(candidates)
        scores = np.zeros(n, dtype=np.float32)
        
        # Placeholder - implement actual CPU scoring
        for i, cand in enumerate(candidates):
            scores[i] = cand.get('score', 0.0)
        
        return scores
    
    def get_device_memory_info(self) -> List[Dict[str, Any]]:
        """Get memory info for all GPU devices.
        
        Returns:
            List of dicts with device info
        """
        if not GPU_AVAILABLE or self.num_gpus == 0:
            return []
        
        info = []
        for dev_id in self.devices:
            try:
                with cp.cuda.Device(dev_id):
                    dev = cp.cuda.Device(dev_id)
                    mem_info = dev.mem_info
                    
                    info.append({
                        'device_id': dev_id,
                        'name': dev.name,
                        'total_memory_gb': mem_info[1] / 1e9,
                        'free_memory_gb': mem_info[0] / 1e9,
                        'used_memory_gb': (mem_info[1] - mem_info[0]) / 1e9,
                        'utilization_pct': (1 - mem_info[0] / mem_info[1]) * 100
                    })
            except Exception as e:
                warnings.warn(f"Could not get info for GPU {dev_id}: {e}")
        
        return info


def create_multi_gpu_scorer(
    device_ids: Optional[List[int]] = None,
    batch_size_per_gpu: int = 10000
) -> MultiGPUScorer:
    """Factory function to create multi-GPU scorer.
    
    Args:
        device_ids: GPU device IDs to use (None = auto-detect)
        batch_size_per_gpu: Batch size per GPU
        
    Returns:
        Configured MultiGPUScorer
    """
    return MultiGPUScorer(device_ids=device_ids, batch_size_per_gpu=batch_size_per_gpu)


def print_gpu_info():
    """Print information about available GPUs."""
    if not GPU_AVAILABLE:
        print("CuPy not available. No GPU support.")
        return
    
    try:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"Available GPUs: {num_gpus}")
        print("")
        
        for i in range(num_gpus):
            with cp.cuda.Device(i):
                dev = cp.cuda.Device(i)
                mem_info = dev.mem_info
                
                print(f"GPU {i}:")
                print(f"  Name: {dev.name}")
                print(f"  Total Memory: {mem_info[1]/1e9:.1f} GB")
                print(f"  Free Memory: {mem_info[0]/1e9:.1f} GB")
                print(f"  Compute Capability: {dev.compute_capability}")
                print("")
    except Exception as e:
        print(f"Error getting GPU info: {e}")


if __name__ == '__main__':
    # Print GPU info when run as script
    print_gpu_info()
