GPU Migration Plan — high level

Goal
----
Move the combinatorial inner search of Elliott Wave candidates to a GPU-friendly, array-first kernel so we can evaluate many more candidate configurations (orders of magnitude) per second.

Constraints
-----------
- Must preserve existing detection semantics (rules, scoring, outputs).
- Keep solution GPU-agnostic: support CUDA GPUs (V100), and Apple Silicon (via PyTorch MPS backend) where possible.
- Keep CPU parallelism for orchestration and IO so GPU stays fed.

Phased plan
-----------
1) Prepare data model (2-3 days)
   - Define compact array representations for candidate configurations:
     - Candidate start indices (N candidates)
     - Candidate skip-configurations encoded as small ints per wave (shape: N x 5)
   - Decide on candidate generation strategy: produce coarse candidates from heuristic pivots (peak detection) or enumerate compactly.

2) Implement GPU-friendly scoring kernels (3-5 days)
   - Port simple numeric checks (span, length ratios, duration ratios) into PyTorch kernels.
   - Implement vectorized versions of a subset of WaveRules to filter candidates entirely on-device.
   - Return surviving candidate indices to CPU.

3) Reimplement inner search in batches (5-10 days)
   - Create batched kernel that, given base DF arrays (Open, High, Low, Close) and a batch of candidate configs, computes mono-wave endpoints using array scans.
   - This is the most complex piece: requires translating `hi/next_hi/lo/next_lo` into parallel prefix-like operations or windowed scans.

4) Validation and integration (3-5 days)
   - For a subset of windows, compare GPU results vs existing CPU implementation; ensure identical candidates and rule outcomes.

5) Optimize and scale (ongoing)
   - Tune batch sizes, memory transfers, and use mixed-precision if needed.
   - Add fallback path for CPU if GPU not available.

Estimated effort
----------------
- Minimal usable GPU scoring (phase 2) to accelerate filtering: ~1 week (40 hrs)
- Full GPU inner-loop rewrite (phase 3) to get orders-of-magnitude improvement: ~2-3 weeks (80-120 hrs)
- QA, tuning, CI: additional 1 week.

Engineering risks & mitigations
------------------------------
- Porting `next_hi`/`next_lo` to GPUs is non-trivial due to control flow; mitigate by implementing batch heuristics (candidate pivots) and limiting per-candidate scan windows.
- Memory transfer overhead: minimize round-trips by batching large numbers of candidates and returning only indices/short summaries.
- Validation complexity: build a test harness that compares CPU vs GPU outputs for many windows.

Integration notes
-----------------
- The `pipeline/gpu_accel.py` currently contains a scoring stub; we'll expand it incrementally.
- Keep current CPU path intact and gated behind a flag `--gpu` so we can compare results and fall back if necessary.

Next immediate step
-------------------
- Implement Phase 1 (data model) and Phase 2 (GPU scoring kernel) prototype. I can start that now if you want — it'll produce a PyTorch kernel that filters many candidates quickly and returns survivors to the CPU for final object construction.

```

### Updates (2026-02-13)

Today we implemented a practical, incremental GPU-batching approach to accelerate the inner search without a full GPU port.

- Added a hybrid pre-score + batch evaluation path in `models/WaveAnalyzer.scan_impulses`:
   - `scan_impulses(..., scan_cfg={...})` accepts `gpu_enabled`, `gpu_batch_size`, and `gpu_top_k`.
   - When `gpu_enabled` is true the scanner:
      1. Processes combinatorial `WaveOptions` in batches of `gpu_batch_size`.
      2. Computes cheap per-candidate features (local volatility proxy, normalized range, normalized "complexity" from skip-config) for each candidate in the batch.
      3. Uses `pipeline.gpu_accel.GPUAccelerator.score_features` (PyTorch; prefers MPS / CUDA when available) to score the batch in-device.
      4. Fully evaluates only the top `gpu_top_k` candidates from each batch using the existing `find_impulsive_wave` + `WavePattern` checks.

- Added `pipeline/gpu_accel.py::GPUAccelerator.score_features` as a batched scoring path that falls back to CPU if PyTorch is not available.
- Centralized knobs in `configs.yaml` (we exposed `gpu_enabled`, `up_to`, `pre_score_*`, `gpu_batch_size`, `gpu_top_k`) and added example profiles for local vs HPC runs.
- The pipeline runner `scripts/pipeline_run.py` now:
   - Accepts `--gpu` to enable GPU scoring.
   - Writes a single consolidated JSON `data/results_run_<ts>.json` and moves the latest copy to `output/latest_results.json`.
   - Stores images under `output/images/` (the runner clears the images dir on start to keep results tidy).

Benchmark (single-run smoke):

- Run: GOOG, `up_to=15`, local Apple M4 Pro, `--gpu` enabled (PyTorch MPS used when available)
- Result: scan phase ~115.9s; total run ~116.6s (fetch: 0.14s, build_windows: 0.00s). This demonstrates the hybrid approach works end-to-end; further tuning (batch sizes, pre-score features, gpu_top_k) is expected to reduce scan time substantially.

Notes / next steps

- The current implementation is a hybrid pruning strategy: the heavy enumerator remains CPU-bound. For larger speedups we'll need to port more of the enumerator / mono-wave endpoint computation to vectorized kernels (numba for CPU or PyTorch kernels for GPU).
- We should run a small parameter sweep (gpu_batch_size x gpu_top_k) to find a good sweet-spot on your M4 Pro (I can run that and report timings).

If you'd like, I'll now (a) run a short sweep to find good batch/top_k values on your machine, or (b) start a targeted numba port of inner numeric helpers as a next step.
