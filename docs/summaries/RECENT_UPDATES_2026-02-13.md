# Recent updates (2026-02-13)

This document summarizes today's pipeline changes and the shift to a CPU-first,
high-throughput scanning path.

What changed

- CPU-batched pre-scoring:
  - `models/WaveAnalyzer.scan_impulses` now uses a CPU-optimized batching path: it
    computes cheap numpy features per candidate (volatility proxy, range, extrema
    count, slope), ranks candidates per batch, and fully evaluates only the top-k
    survivors.

- Shared-memory arrays:
  - The orchestrator can allocate shared numpy buffers for `lows`/`highs`/`dates`
    so worker processes map lightweight views instead of receiving full copies.

- Numba pre-warm & combinatorics cache:
  - `pipeline/numba_warm.py` pre-warms numba-compiled numeric primitives and
    precomputes common `WaveOptions` combinatorics to eliminate first-call JIT
    and combinatorics overhead in worker processes.

- Config / CLI:
  - The pipeline uses CPU-centric knobs: `cpu_batch_size`, `cpu_top_k`, and
    `use_shared_memory` in `configs.yaml`. The `--gpu` CLI and GPU-specific
    knobs have been removed from the active run-path.

Quick-run example (CPU-first):

```bash
PYTHONPATH=. .venv/bin/python3 scripts/pipeline_run.py GOOG --config configs.yaml --source yfinance --out-dir output --processes 6
```

Notes & next steps

- The pipeline keeps the existing `WaveAnalyzer` detection semantics while
  improving throughput via batching, shared buffers, and numba-accelerated
  primitives.
- Recommended follow-ups:
  1. Run a short parameter sweep (cpu_batch_size x cpu_top_k) to find a good
     sweet-spot for your machine.
  2. Collect a multi-worker profile to identify remaining per-candidate hotspots
     for targeted numba ports.
  3. If you want to resume a GPU-focused path later, the archived migration
     notes remain available but are not part of the active pipeline.

If you'd like, I can run the parameter sweep now and report per-run timings and
aggregated `scan_stats` (n_pre_scored, n_full_evals, phase times).
