# Recent updates (2026-02-13)

This document summarizes the incremental GPU-batching and pipeline changes implemented today.

What changed

- GPU-batched pre-scoring:
  - `models/WaveAnalyzer.scan_impulses` now supports a `scan_cfg` dict and respects `gpu_enabled`, `gpu_batch_size`, and `gpu_top_k`.
  - When enabled, WaveOptions are processed in batches; cheap features are computed per candidate and scored with `pipeline/gpu_accel.GPUAccelerator.score_features`.
  - Only the top `gpu_top_k` candidates per batch are fully evaluated using the existing CPU enumerator (`find_impulsive_wave` + `WavePattern` checks).

- `pipeline/gpu_accel.py`:
  - Added `score_features` which uses PyTorch tensors on `mps`/`cuda`/`cpu` when available and falls back to CPU scoring.

- Orchestrator & config:
  - `scripts/pipeline_run.py` accepts `--gpu` to enable GPU scoring and writes consolidated JSON to `data/results_run_<ts>.json` (moved to `output/latest_results.json` after run). Images are saved under `output/images/`.
  - `configs.yaml` now exposes `up_to`, `gpu_enabled`, `gpu_batch_size`, `gpu_top_k`, `pre_score_top_k`, `pre_score_threshold`, `pre_score_weights`, `min_volatility`, `max_windows`, and other runtime knobs.

Benchmark (single-run smoke)

- Command used:

```bash
export PYTHONPATH=.
.venv/bin/python3 scripts/pipeline_run.py GOOG --config configs.yaml --source yfinance --out-dir output --processes 6 --gpu
```

- `configs.yaml` was set with `up_to: 15` for this test.
- Observed timings on local Apple M4 Pro (PyTorch MPS used when available):
  - fetch: 0.14s
  - build_windows: ~0s
  - scan: ~115.86s
  - total: ~116.6s

Notes & next steps

- This is a hybrid pruning strategy — it'll reduce the number of CPU enumerations but the inner enumerator remains CPU-bound.
- Recommended follow-ups:
  1. Run a parameter sweep over `gpu_batch_size` and `gpu_top_k` to find a practical sweet spot on your machine.
  2. Improve pre-score features (longer-range proxies, extrema counts, slope metrics) to improve pruning precision.
  3. Begin porting numeric inner loops (some MonoWave helpers) to numba and/or PyTorch kernels for further acceleration.

If you'd like, I can run the parameter sweep now and report per-run timings and memory use.
