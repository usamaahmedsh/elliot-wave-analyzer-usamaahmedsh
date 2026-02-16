"""Evaluation module for Elliott Wave pattern detection accuracy.

Provides tools for:
- Rule compliance validation
- Predictive power analysis
- Supervised evaluation (vs ground truth)
- Stability testing
- Performance metrics
"""

from evaluation.metrics import (
    EvaluationMetrics,
    calculate_temporal_iou,
    match_detections_to_ground_truth,
    calculate_detection_metrics,
    calculate_rule_compliance,
    calculate_stability_score,
)

from evaluation.rule_validator import RuleValidator, validate_pattern_file

from evaluation.predictive_evaluator import (
    PredictionResult,
    PredictiveEvaluator,
    backtest_patterns,
)

__all__ = [
    'EvaluationMetrics',
    'calculate_temporal_iou',
    'match_detections_to_ground_truth',
    'calculate_detection_metrics',
    'calculate_rule_compliance',
    'calculate_stability_score',
    'RuleValidator',
    'validate_pattern_file',
    'PredictionResult',
    'PredictiveEvaluator',
    'backtest_patterns',
]
