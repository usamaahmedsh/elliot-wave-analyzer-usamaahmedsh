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
ARCHIVED: GPU migration notes

This document described an optional GPU migration plan for the project. The
project has since moved to a CPU-first implementation (batching, shared-memory
buffers, numba pre-warm) and the active pipeline no longer exposes a GPU path.

The migration notes are retained here for historical reference but are not part
of the active pipeline documentation. If you want to resume a GPU-focused
effort in the future, these notes may be a helpful starting point.

