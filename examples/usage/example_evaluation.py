#!/usr/bin/env python3
"""Example: Run evaluation on pipeline results.

This script demonstrates how to evaluate Elliott Wave pattern detection accuracy
using the evaluation framework.

Usage:
    python scripts/example_evaluation.py
"""

import asyncio
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import (
    RuleValidator,
    PredictiveEvaluator,
    calculate_detection_metrics,
    EvaluationMetrics,
)


def create_mock_patterns():
    """Create mock patterns for demonstration."""
    patterns = []
    
    for i in range(5):
        pattern = {
            'id': f'pattern_{i}',
            'pattern_type': 'impulsive',
            'direction': 'bullish',
            'start_idx': i * 50,
            'end_idx': i * 50 + 45,
            'score': 0.75 + np.random.rand() * 0.2,
            'ensemble_score': 0.70 + np.random.rand() * 0.25,
            'waves': [
                {'wave': 1, 'start_idx': i*50, 'end_idx': i*50 + 8},
                {'wave': 2, 'start_idx': i*50 + 8, 'end_idx': i*50 + 15},
                {'wave': 3, 'start_idx': i*50 + 15, 'end_idx': i*50 + 30},
                {'wave': 4, 'start_idx': i*50 + 30, 'end_idx': i*50 + 37},
                {'wave': 5, 'start_idx': i*50 + 37, 'end_idx': i*50 + 45},
            ]
        }
        patterns.append(pattern)
    
    return patterns


def example_rule_validation():
    """Example 1: Rule Compliance Validation."""
    print("\n" + "="*70)
    print("EXAMPLE 1: RULE COMPLIANCE VALIDATION")
    print("="*70 + "\n")
    
    # Create mock patterns
    patterns = create_mock_patterns()
    
    # Create mock price data
    n_bars = 300
    lows = 100 + np.cumsum(np.random.randn(n_bars) * 2)
    highs = lows + np.random.rand(n_bars) * 5
    
    # Validate
    validator = RuleValidator(strict_mode=True)
    results = validator.validate_batch(patterns, lows, highs)
    
    # Print report
    validator.print_validation_report(results)
    
    return results


def example_detection_metrics():
    """Example 2: Detection Metrics (Precision, Recall, F1)."""
    print("\n" + "="*70)
    print("EXAMPLE 2: DETECTION METRICS")
    print("="*70 + "\n")
    
    # Mock detections
    detections = [
        {'start_idx': 10, 'end_idx': 50},
        {'start_idx': 60, 'end_idx': 100},
        {'start_idx': 110, 'end_idx': 145},
        {'start_idx': 200, 'end_idx': 240},  # False positive
    ]
    
    # Mock ground truth
    ground_truth = [
        {'start_idx': 12, 'end_idx': 48},   # Match to detection 1
        {'start_idx': 62, 'end_idx': 98},   # Match to detection 2
        {'start_idx': 112, 'end_idx': 143}, # Match to detection 3
        {'start_idx': 160, 'end_idx': 185}, # Missed (false negative)
    ]
    
    # Calculate metrics
    metrics = calculate_detection_metrics(detections, ground_truth, iou_threshold=0.5)
    
    print("Detection Metrics:")
    print(metrics)
    print("\nMetric Details:")
    print(f"  True Positives: {metrics.true_positives}")
    print(f"  False Positives: {metrics.false_positives}")
    print(f"  False Negatives: {metrics.false_negatives}")
    print(f"  Precision: {metrics.precision:.2%}")
    print(f"  Recall: {metrics.recall:.2%}")
    print(f"  F1-Score: {metrics.f1_score:.2%}")
    print(f"  Mean IoU: {metrics.mean_iou:.4f}")
    
    return metrics


def example_predictive_evaluation():
    """Example 3: Predictive Power Evaluation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: PREDICTIVE POWER EVALUATION")
    print("="*70 + "\n")
    
    # Create mock patterns
    patterns = create_mock_patterns()
    
    # Create mock price data (trending up then down)
    n_bars = 300
    prices = 100 + np.cumsum(np.concatenate([
        np.ones(150) * 0.5,    # Uptrend
        np.ones(150) * -0.3,   # Downtrend
    ]) + np.random.randn(n_bars) * 0.5)
    
    dates = [f"2024-01-{i+1:02d}" for i in range(n_bars)]
    
    # Evaluate
    evaluator = PredictiveEvaluator(forward_horizon_days=30)
    results = evaluator.evaluate_batch(patterns, prices, dates)
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    return results


def example_save_results():
    """Example 4: Save Evaluation Results to File."""
    print("\n" + "="*70)
    print("EXAMPLE 4: SAVE RESULTS")
    print("="*70 + "\n")
    
    # Run evaluations
    rule_results = example_rule_validation()
    metric_results = example_detection_metrics()
    predictive_results = example_predictive_evaluation()
    
    # Combine results
    all_results = {
        'rule_compliance': rule_results,
        'detection_metrics': metric_results.to_dict(),
        'predictive_power': predictive_results,
    }
    
    # Save to JSON
    output_file = Path('evaluation_results_example.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"\nResult summary:")
    print(f"  Rule Compliance: {rule_results['rule_compliance']['overall']:.2%}")
    print(f"  Detection F1-Score: {metric_results.f1_score:.2%}")
    print(f"  Prediction Accuracy: {predictive_results['accuracy']:.2%}")


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ELLIOTT WAVE PATTERN EVALUATION - EXAMPLES")
    print("="*70)
    
    # Run examples
    example_rule_validation()
    example_detection_metrics()
    example_predictive_evaluation()
    example_save_results()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run evaluation on real pipeline results:")
    print("   python scripts/evaluate_patterns.py --mode all --input your_results.json")
    print("\n2. Create labeled dataset for supervised evaluation")
    print("\n3. Test predictive power on historical data with forward-looking period")
    print("="*70 + "\n")


if __name__ == '__main__':
    asyncio.run(main())
