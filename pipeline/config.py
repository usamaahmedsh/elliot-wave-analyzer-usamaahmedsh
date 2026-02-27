from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineConfig:
    days: int = 720
    slide_weeks: int = 1
    min_weeks: int = 4
    max_weeks: int = 12
    up_to: int = 8
    grow_weeks: int = 1
    top_n: int = 1
    delay: float = 0.5
    processes: int = 4
    max_combinations: int = 200_000  # safety cap
    concurrency: int = 8
    max_windows: int = 1000
    # Data interval: '1h' (hourly), '1d' (daily), '1wk' (weekly), etc.
    interval: str = '1d'
    # New knobs
    save_images: bool = True
    save_images_top_n: int = 1
    chunk_size: int = 0  # 0 means use heuristic
    min_volatility: float = 0.0
    skip_flat_windows: bool = False
    profile: bool = False
    out_dir: str = 'output'
    # Pre-score knobs
    pre_score_top_k: int = 0  # 0 means don't use top-k pre-score
    pre_score_threshold: float = 0.0  # score threshold to keep windows
    pre_score_weights: tuple = (0.4, 0.3, 0.2, 0.1)  # weights: volatility, range, extrema, abs(slope)
    # Shared memory usage for price arrays (lows/highs/dates). When True the
    # orchestrator will allocate shared buffers which worker processes will map
    # by name to avoid copying large arrays via IPC.
    use_shared_memory: bool = False
    # CPU batching knobs (used by scan_impulses)
    cpu_batch_size: int = 512
    cpu_top_k: int = 64
    # Additional pipeline knobs
    window_overlap_ratio: float = 0.2
    enable_multi_start: bool = False
    max_start_points: int = 3
    scan_pattern_types: str = 'all'  # 'all', 'impulses', or 'corrective'
    # Auto-detect device (overrides processes and batch_size if True)
    auto_detect_device: bool = True
    # Pattern analysis during pipeline run
    analyze_patterns: bool = False

    @classmethod
    def load_from_file(cls, path: str = "configs.yaml", auto_detect: bool = True) -> "PipelineConfig":
        """Load configuration from a YAML file if present, falling back to defaults.

        The YAML keys map directly to the dataclass fields.
        
        Args:
            path: Path to YAML config file
            auto_detect: If True, auto-detect device and override processes/batch_size
        """
        p = Path(path)
        if not p.exists():
            cfg = cls()
        else:
            try:
                import yaml

                with p.open("r", encoding="utf-8") as f:
                    data: Dict[str, Any] = yaml.safe_load(f) or {}
                # Only pass known fields
                valid = {k: v for k, v in data.items() if k in cls.__annotations__}
                cfg = cls(**valid)
            except Exception:
                # if anything goes wrong, return defaults
                cfg = cls()
        
        # Auto-detect device if enabled
        if auto_detect and cfg.auto_detect_device:
            try:
                from pipeline.device import get_optimal_config
                device_cfg = get_optimal_config()
                cfg.processes = device_cfg.num_workers
                cfg.cpu_batch_size = device_cfg.batch_size
                cfg.concurrency = device_cfg.num_workers
            except Exception:
                # If device detection fails, keep config as-is
                pass
        
        return cfg
