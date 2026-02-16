# Elliott Wave Pattern Evaluation - Complete Guide

## 📊 Measuring Pattern Detection Accuracy

Evaluating Elliott Wave patterns is challenging because they're subjective and forward-looking. We've implemented **4 complementary evaluation methods** to rigorously measure accuracy.

---

## Evaluation Methods

### 1. **Rule Compliance Validation** ✅ (Objective)

**What it measures**: Do detected patterns follow Elliott Wave rules?

**Core Elliott Wave Rules:**
1. Wave 3 is never the shortest (among waves 1, 3, 5)
2. Wave 2 never retraces beyond Wave 1 start
3. Wave 4 never overlaps Wave 1 price territory

**Target**: 100% compliance (all patterns must be valid)

**Usage:**
```bash
python scripts/evaluate_patterns.py --mode rules --input output/results.json
```

**Metrics:**
- Validation Rate: % of patterns passing all rules
- Per-Rule Compliance: % passing each individual rule
- Overall Compliance: Average across all rules

**Example Output:**
```
Total Patterns: 150
Valid Patterns: 147
Invalid Patterns: 3
Validation Rate: 98.00%

Rule Compliance:
  Rule 1 (Wave 3 not shortest): 99.33%
  Rule 2 (Wave 2 no full retrace): 100.00%
  Rule 3 (Wave 4 no overlap): 96.67%
  Overall Compliance: 98.67%
```

---

### 2. **Predictive Power Analysis** 📈 (Empirical)

**What it measures**: Do patterns predict future price movement?

**Approach:**
- After detecting Wave 5 completion → Expect price reversal
- Measure actual price movement N days forward
- Calculate prediction accuracy

**Target**: >55% accuracy (better than random), Sharpe ratio >0.5

**Usage:**
```bash
python scripts/evaluate_patterns.py --mode predictive --input results.json --forward-days 30
```

**Metrics:**
- **Direction Accuracy**: % of correct direction predictions
- **Win Rate**: % of profitable predictions
- **Average Return**: Mean return per prediction
- **Sharpe Ratio**: Risk-adjusted returns
- **By Pattern Type**: Separate metrics for impulsive vs corrective

**Example Output:**
```
Forward Horizon: 30 days
Total Patterns: 89
Evaluable Predictions: 82

Prediction Metrics:
  Direction Accuracy: 58.54%
  Win Rate: 56.10%
  Average Return: 2.34%
  Std Return: 4.52%
  Sharpe Ratio: 0.518

By Pattern Type:
  Impulsive:
    Count: 67
    Accuracy: 61.19%
    Avg Return: 2.81%
  Corrective:
    Count: 15
    Accuracy: 46.67%
    Avg Return: 0.92%
```

**Interpretation:**
- Accuracy >60% = Strong predictive power ✅
- Accuracy 55-60% = Moderate predictive power ✓
- Accuracy 50-55% = Weak predictive power ⚠️
- Accuracy <50% = No predictive power ❌

---

### 3. **Supervised Evaluation** 📋 (Semi-Supervised)

**What it measures**: How well do detections match expert-labeled patterns?

**Approach:**
- Expert traders manually label patterns in charts
- Compare automated detections against labels
- Calculate precision, recall, F1-score, IoU

**Target**: Precision >70%, Recall >50%, F1 >60%

**Usage:**
```bash
# First, create labeled dataset (manual)
python scripts/label_patterns.py --symbols AAPL --output labels.json

# Then evaluate
python scripts/evaluate_patterns.py --mode supervised --input results.json --labels labels.json
```

**Metrics:**
- **Precision**: Of all detections, what % are correct?
  ```
  Precision = True Positives / (True Positives + False Positives)
  ```
  
- **Recall**: Of all real patterns, what % did we detect?
  ```
  Recall = True Positives / (True Positives + False Negatives)
  ```
  
- **F1-Score**: Harmonic mean of Precision and Recall
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```
  
- **IoU (Intersection over Union)**: Pattern overlap score
  ```
  IoU = Overlap / Union (higher = better match)
  ```

**Example Output:**
```
Precision: 73.21%
Recall: 58.90%
F1-Score: 65.28%
Mean IoU: 0.6834
TP: 89, FP: 33, FN: 62
```

---

### 4. **Stability Testing** 🔄 (Regression)

**What it measures**: Are detections consistent across parameter changes?

**Approach:**
- Run detection with slightly different configs
- Compare patterns detected in both runs
- Measure overlap/consistency

**Target**: >90% stability (same patterns with minor config changes)

**Usage:**
```bash
# Run with config A
python scripts/pipeline_run.py --config config_a.yaml --output results_a.json

# Run with config B (minor changes)
python scripts/pipeline_run.py --config config_b.yaml --output results_b.json

# Compare stability
python scripts/evaluate_patterns.py --mode stability --input results_a.json --compare results_b.json
```

**Metrics:**
- **Stability Score**: Ratio of consistent detections (0-1)
- **Detection Variance**: How much results vary
- **Determinism Check**: Same config = same results?

---

## Quick Start

### Step 1: Run Pipeline
```bash
python scripts/pipeline_run.py --symbols AAPL,MSFT,GOOG --output detections.json
```

### Step 2: Rule Validation (Immediate)
```bash
python scripts/evaluate_patterns.py --mode rules --input detections.json
```
✅ Should show 100% validation rate

### Step 3: Predictive Power (If have enough data)
```bash
python scripts/evaluate_patterns.py --mode predictive --input detections.json --forward-days 30
```
🎯 Target: >55% accuracy

### Step 4: Generate Report
```bash
python scripts/evaluate_patterns.py --mode all --input detections.json --report evaluation_report.html
```
📄 Opens comprehensive HTML report

---

## Benchmarks & Targets

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| **Rule Compliance** | 100% | 100% | 100% |
| **Direction Accuracy** | >55% | >60% | >65% |
| **Win Rate** | >50% | >55% | >60% |
| **Sharpe Ratio** | >0.5 | >0.75 | >1.0 |
| **Precision** | >70% | >75% | >80% |
| **Recall** | >50% | >60% | >70% |
| **F1-Score** | >60% | >67% | >75% |
| **Stability** | >90% | >93% | >95% |

---

## File Structure

```
evaluation/
├── __init__.py                  # Module exports
├── metrics.py                   # Precision, recall, F1, IoU metrics
├── rule_validator.py            # Elliott Wave rule compliance checker
└── predictive_evaluator.py      # Predictive power & backtesting

scripts/
└── evaluate_patterns.py         # Main evaluation script

doc/
├── EVALUATION_FRAMEWORK.md      # This guide
└── EVALUATION_QUICKSTART.md     # Quick reference
```

---

## Common Workflows

### Workflow 1: Quick Sanity Check
```bash
# Just check rules (fast)
python scripts/evaluate_patterns.py --mode rules --input results.json
```
**Use case**: Verify detection logic has no bugs

---

### Workflow 2: Full Evaluation
```bash
# All evaluation modes
python scripts/evaluate_patterns.py \
  --mode all \
  --input results.json \
  --labels ground_truth.json \
  --forward-days 30 \
  --report full_eval.html
```
**Use case**: Comprehensive accuracy assessment

---

### Workflow 3: Parameter Tuning
```bash
# Test with config A
python scripts/pipeline_run.py --config config_a.yaml --output results_a.json
python scripts/evaluate_patterns.py --mode predictive --input results_a.json

# Test with config B
python scripts/pipeline_run.py --config config_b.yaml --output results_b.json
python scripts/evaluate_patterns.py --mode predictive --input results_b.json

# Compare which config gives better predictive power
```
**Use case**: Find optimal configuration

---

### Workflow 4: Regression Testing
```bash
# Before code changes
python scripts/pipeline_run.py --symbols AAPL --output baseline.json

# After code changes
python scripts/pipeline_run.py --symbols AAPL --output new_version.json

# Verify stability
python scripts/evaluate_patterns.py --mode stability --input baseline.json --compare new_version.json
```
**Use case**: Ensure code changes don't break detection

---

## Creating Ground Truth Labels

Ground truth is essential for supervised evaluation. Here's how to create it:

### Manual Labeling (High Quality)
1. Select representative chart samples (20-50 charts)
2. Expert trader manually identifies Elliott Wave patterns
3. Record pattern locations (start/end indices, wave counts)
4. Save as JSON in standardized format

### Semi-Automated Labeling
1. Run detection algorithm
2. Expert reviews and corrects detections
3. Approved patterns become ground truth
4. Rejected patterns marked as false positives

### Labeled Dataset Format
```json
{
  "symbol": "AAPL",
  "date_range": ["2023-01-01", "2024-01-01"],
  "patterns": [
    {
      "id": "pattern_001",
      "type": "impulsive",
      "direction": "bullish",
      "start_idx": 45,
      "end_idx": 178,
      "waves": [
        {"wave": 1, "start_idx": 45, "end_idx": 67},
        {"wave": 2, "start_idx": 67, "end_idx": 89},
        {"wave": 3, "start_idx": 89, "end_idx": 134},
        {"wave": 4, "start_idx": 134, "end_idx": 156},
        {"wave": 5, "start_idx": 156, "end_idx": 178}
      ],
      "confidence": "high",
      "annotator": "expert_trader_1"
    }
  ]
}
```

---

## Interpretation Guidelines

### Rule Compliance
- **100%**: ✅ Perfect - algorithm is bug-free
- **95-99%**: ⚠️ Good but has edge cases to fix
- **<95%**: ❌ Critical issues in detection logic

### Predictive Power
- **Accuracy >60%**: ✅ Strong signal - patterns have predictive value
- **Accuracy 55-60%**: ✓ Moderate signal - useful with other indicators
- **Accuracy 50-55%**: ⚠️ Weak signal - marginal utility
- **Accuracy <50%**: ❌ No signal - worse than random

### Sharpe Ratio
- **>1.0**: ✅ Excellent - patterns generate strong risk-adjusted returns
- **0.5-1.0**: ✓ Good - viable for trading with risk management
- **0-0.5**: ⚠️ Marginal - requires additional filters
- **<0**: ❌ Negative - patterns lose money

---

## Troubleshooting

### "No patterns found to evaluate"
- Check input file format
- Verify patterns were actually detected
- Ensure JSON structure is correct

### "Not enough future data"
- Increase `--days` parameter when fetching data
- Reduce `--forward-days` prediction horizon
- Use data with longer history

### "Low predictive accuracy (<50%)"
Possible causes:
- Patterns may need additional filtering (ensemble score threshold)
- Forward horizon may be too long or too short
- Market conditions may not favor Elliott Wave patterns
- Detection logic may have issues

### "Poor precision/recall"
- May need to tune detection parameters
- Ground truth labels may be inconsistent
- IoU threshold may be too strict/loose

---

## Next Steps

1. ✅ **Rule Validation** - Implement and run immediately
2. ⏳ **Create labeled dataset** - Manual effort (50-100 patterns)
3. ⏳ **Predictive backtesting** - Requires long-term data
4. ⏳ **Stability regression suite** - Automate as CI tests

---

## Summary

### What You Can Measure:
1. ✅ **Rule compliance** (objective, automated)
2. 📈 **Predictive power** (empirical, automated)
3. 📋 **Detection accuracy** (vs labels, semi-automated)
4. 🔄 **Stability** (regression testing, automated)

### Expected Results (Well-Tuned System):
- Rule compliance: **100%**
- Direction accuracy: **55-65%**
- Sharpe ratio: **0.5-1.5**
- Precision: **70-80%**
- Recall: **50-70%**
- F1-Score: **60-75%**

### Tools Available:
- `scripts/evaluate_patterns.py` - Main evaluation script
- `evaluation/` - Metrics and validators
- `doc/EVALUATION_FRAMEWORK.md` - This guide

**Start with rule validation, then add predictive evaluation as data permits!**
