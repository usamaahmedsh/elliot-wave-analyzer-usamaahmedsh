"""DEPRECATED: GPU accelerator stub.

This module previously contained an optional PyTorch-based scoring helper used
by an experimental GPU-path of the pipeline. The project has since been
re-focused on a CPU-first pipeline (batching, shared-memory, numba). The GPU
stub is retained as a historical artifact but is intentionally inert.

Do not import or rely on this module; the pipeline no longer uses a GPU path.
"""

__all__ = []
