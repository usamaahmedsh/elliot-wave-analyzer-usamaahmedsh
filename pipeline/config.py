from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


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
    gpu_enabled: bool = False
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
    # GPU batching knobs
    gpu_batch_size: int = 512
    gpu_top_k: int = 64

    @classmethod
    def load_from_file(cls, path: str = "configs.yaml") -> "PipelineConfig":
        """Load configuration from a YAML file if present, falling back to defaults.

        The YAML keys map directly to the dataclass fields.
        """
        p = Path(path)
        if not p.exists():
            return cls()

        try:
            import yaml

            with p.open("r", encoding="utf-8") as f:
                data: Dict[str, Any] = yaml.safe_load(f) or {}
            # Only pass known fields
            valid = {k: v for k, v in data.items() if k in cls.__annotations__}
            return cls(**valid)
        except Exception:
            # if anything goes wrong, return defaults
            return cls()
