"""
CPU device configuration for pipeline execution.

Detects the number of available CPU cores and configures optimal worker
and batch-size settings for the current machine.
"""
import os
import platform
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceConfig:
    """Hardware configuration detected for this system"""
    device_name: str
    num_cores: int
    num_workers: int
    batch_size: int
    system_type: str  # 'hpc', 'mac', 'linux', 'windows'

    def __str__(self):
        return (
            f"{self.system_type.upper()} | {self.num_cores} cores | "
            f"{self.num_workers} workers | {self.device_name}"
        )


def detect_device() -> DeviceConfig:
    """
    Detect CPU configuration and return optimal worker/batch-size settings.

    Returns:
        DeviceConfig with settings appropriate for this machine's CPU.
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

    # Get a human-readable CPU name
    cpu_info = "CPU"
    try:
        if system == 'Darwin':
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=2
            )
            cpu_info = result.stdout.strip() or "CPU"
        elif system == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_info = line.split(':')[1].strip()
                        break
    except Exception:
        pass

    # Worker / batch-size heuristics per system type
    if system_type == 'hpc':
        workers = min(num_cores - 2, 30)
        batch_size = 1024
    elif system_type == 'mac':
        workers = max(num_cores - 2, 1)
        batch_size = 512
    else:
        workers = max(num_cores - 2, 2)
        batch_size = 512

    return DeviceConfig(
        device_name=cpu_info,
        num_cores=num_cores,
        num_workers=workers,
        batch_size=batch_size,
        system_type=system_type,
    )


def get_optimal_config() -> DeviceConfig:
    """Return optimal CPU configuration (cached after first call)."""
    global _cached_config
    if _cached_config is None:
        _cached_config = detect_device()
    return _cached_config


def print_device_info(config: Optional[DeviceConfig] = None):
    """Print detected device information."""
    if config is None:
        config = get_optimal_config()

    print("=" * 70)
    print("🖥️  DEVICE CONFIGURATION")
    print("=" * 70)
    print(f"System:        {config.system_type.upper()} ({platform.system()} {platform.machine()})")
    print(f"CPU Cores:     {config.num_cores}")
    print(f"Device:        {config.device_name}")
    print(f"Workers:       {config.num_workers}")
    print(f"Batch Size:    {config.batch_size}")
    print("=" * 70)


# Global cache
_cached_config: Optional[DeviceConfig] = None


if __name__ == '__main__':
    config = detect_device()
    print_device_info(config)

    print("\n📊 Recommended settings for configs.yaml:")
    print(f"processes: {config.num_workers}")
    print(f"cpu_batch_size: {config.batch_size}")
    print(f"# Detected: {config}")
