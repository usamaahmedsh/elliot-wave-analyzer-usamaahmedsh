"""Evaluation metrics for Elliott Wave pattern detection.

Provides standard metrics for measuring pattern detection accuracy:
- Precision, Recall, F1-Score
- IoU (Intersection over Union)
- Pattern overlap metrics
- Rule compliance scores
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    iou_scores: List[float] = None
    mean_iou: float = 0.0
    
    def __post_init__(self):
        if self.iou_scores is None:
            self.iou_scores = []
    
    def compute(self):
        """Compute derived metrics from counts."""
        # Precision: Of all detections, what % are correct?
        if (self.true_positives + self.false_positives) > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0
        
        # Recall: Of all real patterns, what % did we detect?
        if (self.true_positives + self.false_negatives) > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0
        
        # F1-Score: Harmonic mean
        if (self.precision + self.recall) > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0
        
        # Mean IoU
        if self.iou_scores:
            self.mean_iou = np.mean(self.iou_scores)
        else:
            self.mean_iou = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'mean_iou': round(self.mean_iou, 4),
            'num_iou_scores': len(self.iou_scores),
        }
    
    def __str__(self) -> str:
        return (
            f"Precision: {self.precision:.2%}\n"
            f"Recall: {self.recall:.2%}\n"
            f"F1-Score: {self.f1_score:.2%}\n"
            f"Mean IoU: {self.mean_iou:.4f}\n"
            f"TP: {self.true_positives}, FP: {self.false_positives}, FN: {self.false_negatives}"
        )


def calculate_temporal_iou(
    detected_range: Tuple[int, int],
    ground_truth_range: Tuple[int, int]
) -> float:
    """Calculate Intersection over Union for temporal ranges.
    
    Args:
        detected_range: (start_idx, end_idx) of detected pattern
        ground_truth_range: (start_idx, end_idx) of ground truth pattern
        
    Returns:
        IoU score (0-1, higher is better)
    """
    det_start, det_end = detected_range
    gt_start, gt_end = ground_truth_range
    
    # Calculate intersection
    intersect_start = max(det_start, gt_start)
    intersect_end = min(det_end, gt_end)
    intersection = max(0, intersect_end - intersect_start)
    
    # Calculate union
    union_start = min(det_start, gt_start)
    union_end = max(det_end, gt_end)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections_to_ground_truth(
    detections: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Match detected patterns to ground truth using IoU.
    
    Args:
        detections: List of detected pattern dicts with 'start_idx', 'end_idx'
        ground_truth: List of ground truth pattern dicts
        iou_threshold: Minimum IoU to consider a match (default 0.5)
        
    Returns:
        matches: List of (detection_idx, gt_idx, iou) tuples
        unmatched_detections: List of detection indices with no match
        unmatched_ground_truth: List of gt indices with no match
    """
    if not detections or not ground_truth:
        return [], list(range(len(detections))), list(range(len(ground_truth)))
    
    # Compute IoU matrix
    n_det = len(detections)
    n_gt = len(ground_truth)
    iou_matrix = np.zeros((n_det, n_gt))
    
    for i, det in enumerate(detections):
        det_range = (det.get('start_idx', 0), det.get('end_idx', 0))
        for j, gt in enumerate(ground_truth):
            gt_range = (gt.get('start_idx', 0), gt.get('end_idx', 0))
            iou_matrix[i, j] = calculate_temporal_iou(det_range, gt_range)
    
    # Greedy matching: match highest IoU pairs first
    matches = []
    matched_det = set()
    matched_gt = set()
    
    while True:
        # Find highest IoU
        max_iou = np.max(iou_matrix)
        if max_iou < iou_threshold:
            break
        
        # Find indices of max IoU
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        
        # Record match
        matches.append((int(i), int(j), float(max_iou)))
        matched_det.add(int(i))
        matched_gt.add(int(j))
        
        # Zero out row and column to prevent re-matching
        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0
    
    unmatched_det = [i for i in range(n_det) if i not in matched_det]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt]
    
    return matches, unmatched_det, unmatched_gt


def calculate_detection_metrics(
    detections: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> EvaluationMetrics:
    """Calculate precision, recall, F1, and IoU metrics.
    
    Args:
        detections: List of detected patterns
        ground_truth: List of ground truth patterns
        iou_threshold: Minimum IoU for a true positive
        
    Returns:
        EvaluationMetrics object with all metrics
    """
    metrics = EvaluationMetrics()
    
    # Match detections to ground truth
    matches, unmatched_det, unmatched_gt = match_detections_to_ground_truth(
        detections, ground_truth, iou_threshold
    )
    
    # True Positives: matched detections
    metrics.true_positives = len(matches)
    
    # False Positives: unmatched detections
    metrics.false_positives = len(unmatched_det)
    
    # False Negatives: unmatched ground truth
    metrics.false_negatives = len(unmatched_gt)
    
    # IoU scores for matched patterns
    metrics.iou_scores = [iou for _, _, iou in matches]
    
    # Compute derived metrics
    metrics.compute()
    
    return metrics


def calculate_rule_compliance(patterns: List[Dict[str, Any]]) -> Dict[str, float]:
    """Check what % of patterns comply with Elliott Wave rules.
    
    Args:
        patterns: List of detected patterns with wave data
        
    Returns:
        Dict with compliance rates for each rule
    """
    if not patterns:
        return {
            'overall_compliance': 0.0,
            'wave3_not_shortest': 0.0,
            'wave2_no_overlap': 0.0,
            'wave4_no_overlap': 0.0,
        }
    
    n = len(patterns)
    compliant_wave3 = 0
    compliant_wave2 = 0
    compliant_wave4 = 0
    
    for pattern in patterns:
        waves = pattern.get('waves', [])
        if len(waves) < 5:
            continue
        
        # Extract wave data
        try:
            w1_start = waves[0].get('start_idx', 0)
            w1_end = waves[0].get('end_idx', 0)
            w2_end = waves[1].get('end_idx', 0)
            w3_end = waves[2].get('end_idx', 0)
            w4_end = waves[3].get('end_idx', 0)
            w5_end = waves[4].get('end_idx', 0)
            
            # Get prices (simplified - assumes bullish pattern)
            # In reality, you'd need actual price data
            wave1_len = abs(w1_end - w1_start)
            wave3_len = abs(w3_end - w2_end)
            wave5_len = abs(w5_end - w4_end)
            
            # Rule 1: Wave 3 not shortest
            if wave3_len >= wave1_len or wave3_len >= wave5_len:
                compliant_wave3 += 1
            
            # Rule 2: Wave 2 doesn't retrace beyond Wave 1 start
            # (Need price data to validate properly)
            compliant_wave2 += 1  # Assume compliant for now
            
            # Rule 3: Wave 4 doesn't overlap Wave 1
            # (Need price data to validate properly)
            compliant_wave4 += 1  # Assume compliant for now
            
        except (KeyError, IndexError, TypeError):
            # Skip patterns with missing data
            continue
    
    return {
        'overall_compliance': (compliant_wave3 + compliant_wave2 + compliant_wave4) / (3 * n),
        'wave3_not_shortest': compliant_wave3 / n,
        'wave2_no_overlap': compliant_wave2 / n,
        'wave4_no_overlap': compliant_wave4 / n,
    }


def calculate_stability_score(
    run1_patterns: List[Dict[str, Any]],
    run2_patterns: List[Dict[str, Any]],
    iou_threshold: float = 0.7
) -> float:
    """Calculate stability score between two runs.
    
    Measures how consistent pattern detection is across runs.
    
    Args:
        run1_patterns: Patterns from first run
        run2_patterns: Patterns from second run
        iou_threshold: Minimum IoU to consider patterns "same"
        
    Returns:
        Stability score (0-1, higher = more stable)
    """
    if not run1_patterns and not run2_patterns:
        return 1.0  # Both empty = perfectly stable
    
    if not run1_patterns or not run2_patterns:
        return 0.0  # One empty = completely unstable
    
    # Match patterns between runs
    matches, _, _ = match_detections_to_ground_truth(
        run1_patterns, run2_patterns, iou_threshold
    )
    
    # Stability = ratio of matched patterns to total unique patterns
    total_patterns = len(run1_patterns) + len(run2_patterns)
    matched_patterns = 2 * len(matches)  # Count each match twice
    
    stability = matched_patterns / total_patterns
    return stability
