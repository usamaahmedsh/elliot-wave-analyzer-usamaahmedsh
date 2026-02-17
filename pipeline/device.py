"""
Automatic device detection and configuration for cross-platform execution.

This module detects available hardware (CPU, CUDA GPU, Apple Silicon GPU) and
configures optimal settings for the current environment.
"""
import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceConfig:
    """Hardware configuration detected for this system"""
    device_type: str  # 'cuda', 'mps', 'cpu'
    device_name: str
    num_cores: int
    num_workers: int
    batch_size: int
    use_gpu_acceleration: bool
    system_type: str  # 'hpc', 'mac', 'linux', 'windows'
    
    def __str__(self):
        gpu_status = f"✓ {self.device_name}" if self.use_gpu_acceleration else "✗ CPU only"
        return f"{self.system_type.upper()} | {self.num_cores} cores | {self.num_workers} workers | GPU: {gpu_status}"


def detect_device() -> DeviceConfig:
    """
    Automatically detect the best available hardware configuration.
    
    Priority:
    1. CUDA GPU (NVIDIA on HPC/Linux)
    2. MPS (Apple Silicon GPU on Mac)
    3. CPU (fallback)
    
    Returns:
        DeviceConfig with optimal settings for detected hardware
    """
    system = platform.system()
    num_cores = os.cpu_count() or 4
    
    # Detect system type
    if 'SLURM_JOB_ID' in os.environ or 'PBS_JOBID' in os.environ:
        system_type = 'hpc'
    elif system == 'Darwin':
        system_type = 'mac'
    elif system == 'Linux':
        system_type = 'linux'
    elif system == 'Windows':
        system_type = 'windows'
    else:
        system_type = 'unknown'
    
    # Try CUDA first (NVIDIA GPUs)
    cuda_available = False
    cuda_device_name = "No CUDA GPU"
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            cuda_device_name = torch.cuda.get_device_name(0)
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                cuda_device_name = f"{cuda_device_name} (×{num_gpus})"
    except (ImportError, Exception):
        pass
    
    if cuda_available:
        # HPC with CUDA - use aggressive settings
        return DeviceConfig(
            device_type='cuda',
            device_name=cuda_device_name,
            num_cores=num_cores,
            num_workers=min(num_cores - 2, 30),  # Leave 2 cores for system
            batch_size=2048 if num_cores > 16 else 1024,
            use_gpu_acceleration=True,
            system_type=system_type
        )
    
    # Try Apple Silicon MPS (Mac GPU)
    mps_available = False
    mps_device_name = "No MPS GPU"
    if system == 'Darwin':
        # First check if we're on Apple Silicon at all
        is_apple_silicon = False
        cpu_brand = ""
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=2)
            cpu_brand = result.stdout.strip()
            is_apple_silicon = 'Apple' in cpu_brand
        except Exception:
            # Fallback: check architecture
            is_apple_silicon = platform.machine() == 'arm64'
        
        # If Apple Silicon, check for MPS via PyTorch
        if is_apple_silicon:
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    mps_available = True
                    mps_device_name = f"{cpu_brand} (MPS GPU)" if cpu_brand else "Apple Silicon GPU (MPS)"
                else:
                    # PyTorch found but MPS not available
                    mps_device_name = f"{cpu_brand} (MPS not available)" if cpu_brand else "Apple Silicon"
            except ImportError:
                # PyTorch not installed - can't use MPS but we know it's Apple Silicon
                if cpu_brand:
                    mps_device_name = f"{cpu_brand} (PyTorch not installed - install for GPU support)"
                else:
                    mps_device_name = "Apple Silicon (install PyTorch for GPU support)"
    
    if mps_available:
        # Mac with Apple Silicon GPU
        return DeviceConfig(
            device_type='mps',
            device_name=mps_device_name,
            num_cores=num_cores,
            num_workers=max(num_cores - 2, 1),  # Leave 2 for system
            batch_size=512,  # Conservative for Mac
            use_gpu_acceleration=True,
            system_type=system_type
        )
    
    # Fallback to CPU only
    cpu_info = "CPU"
    try:
        if system == 'Darwin':
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=2)
            cpu_info = result.stdout.strip() or "CPU"
        elif system == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_info = line.split(':')[1].strip()
                        break
    except Exception:
        pass
    
    # CPU-only configuration
    if system_type == 'hpc':
        # HPC without GPU - use many cores
        workers = min(num_cores - 2, 30)
        batch_size = 1024
    elif system_type == 'mac':
        # Mac without GPU detected (Intel Mac or old system)
        workers = max(num_cores - 2, 1)
        batch_size = 512
    else:
        # Generic Linux/Windows
        workers = max(num_cores - 2, 2)
        batch_size = 512
    
    return DeviceConfig(
        device_type='cpu',
        device_name=cpu_info,
        num_cores=num_cores,
        num_workers=workers,
        batch_size=batch_size,
        use_gpu_acceleration=False,
        system_type=system_type
    )


def get_optimal_config() -> DeviceConfig:
    """Get optimal configuration with caching"""
    global _cached_config
    if _cached_config is None:
        _cached_config = detect_device()
    return _cached_config


def print_device_info(config: Optional[DeviceConfig] = None):
    """Print detected device information"""
    if config is None:
        config = get_optimal_config()
    
    print("=" * 70)
    print("🖥️  DEVICE CONFIGURATION")
    print("=" * 70)
    print(f"System:        {config.system_type.upper()} ({platform.system()} {platform.machine()})")
    print(f"CPU Cores:     {config.num_cores}")
    print(f"Device Type:   {config.device_type.upper()}")
    print(f"Device:        {config.device_name}")
    print(f"Workers:       {config.num_workers}")
    print(f"Batch Size:    {config.batch_size}")
    print(f"GPU Accel:     {'✓ Enabled' if config.use_gpu_acceleration else '✗ CPU only'}")
    print("=" * 70)


# Global cache
_cached_config: Optional[DeviceConfig] = None


if __name__ == '__main__':
    # Test device detection
    config = detect_device()
    print_device_info(config)
    
    print("\n📊 Recommended settings for configs.yaml:")
    print(f"processes: {config.num_workers}")
    print(f"cpu_batch_size: {config.batch_size}")
    print(f"# Detected: {config}")
