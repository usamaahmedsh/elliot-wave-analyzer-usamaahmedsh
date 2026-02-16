# Elliott Wave Pattern Evaluation Framework

## Overview

Evaluating Elliott Wave pattern accuracy is challenging because:
1. **Subjective interpretation**: Different analysts may label the same chart differently
2. **Multiple valid patterns**: A chart may contain several valid wave structures
3. **Future validation**: True accuracy is often only known after price continuation
4. **Time horizon dependency**: Patterns valid at one time scale may not be at another

## Evaluation Approaches

We implement **4 complementary evaluation methods**:

### 1. **Rule Compliance Testing** (Objective, Fast) ✅
Verify detected patterns follow Elliott Wave rules:
- Wave 3 is never the shortest
- Wave 2 doesn't retrace beyond Wave 1 start
- Wave 4 doesn't overlap Wave 1
- Fibonacci ratio alignment

**Metric**: Rule violation rate (should be 0%)

### 2. **Expert-Labeled Validation Set** (Semi-Supervised) 📊
Create ground truth by manual labeling:
- Expert traders annotate patterns in historical charts
- Compare detections against labeled data
- Metrics: Precision, Recall, F1-Score, IoU (Intersection over Union)

**Challenge**: Labor-intensive, subjective

### 3. **Predictive Power Analysis** (Empirical) 📈
Evaluate if patterns predict future price movement:
- After detecting a Wave 5 completion, does price reverse as expected?
- Measure prediction accuracy, Sharpe ratio, win rate
- Compare against baseline (random, moving average)

**Metric**: Prediction accuracy, profitability of pattern-based signals

### 4. **Stability & Consistency Testing** (Regression) 🔄
Ensure detection is stable across parameter changes:
- Same pattern detected with slightly different configs
- Patterns persist when retested on same data
- Deterministic results (same config = same output)

**Metric**: Detection stability rate, variance across runs

---

## Implementation

See:
- `evaluation/rule_validator.py` - Rule compliance checker
- `evaluation/labeled_dataset.py` - Expert annotation tools
- `evaluation/predictive_evaluator.py` - Forward-looking validation
- `evaluation/stability_tester.py` - Regression testing
- `evaluation/metrics.py` - Evaluation metrics
- `scripts/evaluate_patterns.py` - Main evaluation script

---

## Quick Start

### 1. Rule Compliance Test (Immediate)
```bash
python scripts/evaluate_patterns.py --mode rules --input output/results.json
```

### 2. Create Labeled Dataset
```bash
# Interactive labeling tool
python scripts/label_patterns.py --symbols AAPL,MSFT --output labeled_data.json

# Evaluate against labels
python scripts/evaluate_patterns.py --mode supervised --labels labeled_data.json
```

### 3. Predictive Power Test
```bash
# Backtest pattern predictions
python scripts/evaluate_patterns.py --mode predictive --symbols AAPL --days 730
```

### 4. Stability Test
```bash
# Run regression suite
python scripts/evaluate_patterns.py --mode stability --config configs.yaml
```

---

## Metrics Explained

### Precision
Of all patterns we detected, what % are correct?
```
Precision = True Positives / (True Positives + False Positives)
```

### Recall
Of all real patterns, what % did we detect?
```
Recall = True Positives / (True Positives + False Negatives)
```

### F1-Score
Harmonic mean of Precision and Recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### IoU (Intersection over Union)
Pattern overlap with ground truth
```
IoU = Overlap Area / Union Area
```

### Prediction Accuracy
How often does the pattern correctly predict price direction?
```
Accuracy = Correct Predictions / Total Predictions
```

---

## Example Workflow

```bash
# 1. Run pipeline and generate detections
python scripts/pipeline_run.py --symbols AAPL,MSFT --output detections.json

# 2. Validate rule compliance (should be 100%)
python scripts/evaluate_patterns.py --mode rules --input detections.json

# 3. Test predictive power
python scripts/evaluate_patterns.py --mode predictive --input detections.json --forward-days 30

# 4. Generate evaluation report
python scripts/evaluate_patterns.py --mode all --input detections.json --report evaluation_report.html
```

---

## Benchmarks

### Rule Compliance
**Target**: 100% (all detected patterns must follow Elliott Wave rules)

### Supervised Metrics (with expert labels)
- **Precision**: Target >70% (most detections are correct)
- **Recall**: Target >50% (find at least half of all patterns)
- **F1-Score**: Target >60%

### Predictive Power
- **Direction Accuracy**: Target >55% (better than random)
- **Sharpe Ratio**: Target >0.5 (positive risk-adjusted returns)
- **Win Rate**: Target >50%

### Stability
- **Config Stability**: Target >90% (same patterns with minor config changes)
- **Temporal Stability**: Target >95% (deterministic on same data)

---

## Next Steps

1. ✅ Implement rule validator (automated)
2. ⏳ Create labeled dataset (manual, 50-100 patterns)
3. ⏳ Build predictive backtester (automated)
4. ⏳ Add stability regression tests (automated)
5. ⏳ Generate comprehensive evaluation report (automated)
