"""GPU-accelerator abstraction (optional). This module provides a thin wrapper
that will use PyTorch (if available) to compute vectorized scoring of candidates.

This is intentionally small — it only provides an example pathway to move scoring
and simple numeric checks to GPU. The full rewrite of the core search to GPU
requires reworking the innermost loops into array kernels.
"""
from typing import List, Any, Dict

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class GPUAccelerator:
    def __init__(self, device: str = None):
        if TORCH_AVAILABLE:
            if device is None:
                # Prefer MPS on Apple Silicon, then CUDA, then CPU
                if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                elif torch.cuda.is_available():
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
            else:
                # allow device strings like 'mps', 'cuda', 'cpu'
                self.device = torch.device(device)
        else:
            self.device = None

    def score_candidates(self, candidates: List[Any]) -> List[float]:
        """Placeholder: vectorized scoring on GPU if torch is available.

        candidates: list of FoundPattern or similar
        returns list of floats (same length)
        """
        if not TORCH_AVAILABLE or self.device is None:
            # noop: return existing .score if present
            out = []
            for c in candidates:
                try:
                    out.append(float(c.score))
                except Exception:
                    out.append(0.0)
            return out

        # Example: take a numeric feature like (idx_end - idx_start) and length range
        import torch
        feats = []
        for c in candidates:
            try:
                span = float(c.idx_end - c.idx_start)
                score = float(c.score) if hasattr(c, 'score') else 0.0
                feats.append([span, score])
            except Exception:
                feats.append([0.0, 0.0])

        t = torch.tensor(feats, dtype=torch.float32, device=self.device)
        # crude combined score: normalized span + existing score
        spans = t[:, 0]
        s = t[:, 1]
        spans_norm = (spans - spans.min()) / (spans.max() - spans.min() + 1e-6)
        combined = 0.5 * spans_norm + 0.5 * s
        return combined.cpu().tolist()

    def score_features(self, features: List[List[float]]) -> List[float]:
        """Score a batch of feature vectors on the selected device.

        features: list of [volatility, range, extrema_count, abs_slope, span, existing_score?]
        Returns list of floats.
        """
        if not TORCH_AVAILABLE or self.device is None:
            # fallback to simple CPU scoring
            out = []
            for f in features:
                # naive linear combination
                vol, ran, ext, slope = f[0], f[1], f[2], f[3]
                out.append(0.4 * vol + 0.3 * ran + 0.2 * min(ext / 10.0, 1.0) + 0.1 * slope)
            return out

        import torch
        t = torch.tensor(features, dtype=torch.float32, device=self.device)
        # simple learned-ish linear weights (same as default pre-score), then normalize
        weights = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32, device=self.device)
        vals = t[:, :4]
        scores = (vals * weights).sum(dim=1)
        # normalize to 0..1 by min/max
        mn = scores.min()
        mx = scores.max()
        norm = (scores - mn) / (mx - mn + 1e-6)
        return norm.cpu().tolist()
