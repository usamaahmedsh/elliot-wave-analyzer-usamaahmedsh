# Elliott Wave GAN: Adversarial Pattern Detection & Generation

## Overview

A three-model adversarial system for Elliott Wave pattern detection, validation, and synthetic data generation. This approach combines rule-based financial analysis with modern deep learning to create an interpretable, self-improving pattern recognition system.

---

## Core Concept

Traditional GANs learn to generate realistic data by playing a game between a Generator and Discriminator. Our system extends this by replacing the simple "real/fake" discriminator with a **Rule-Aware Critic** that provides structured feedback on *why* a pattern is invalid.

```
┌─────────────────────────────────────────────────────────────────┐
│                        ELLIOTT WAVE GAN                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────┐       ┌─────────────────────┐       │
│   │     GENERATOR       │       │       CRITIC        │       │
│   │  (Pattern Creator)  │       │  (Rule Validator)   │       │
│   │                     │       │                     │       │
│   │  Input: Noise +     │       │  Input: OHLCV       │       │
│   │         conditions   │       │         pattern     │       │
│   │                     │       │                     │       │
│   │  Output: Synthetic  │  ───► │  Output:            │       │
│   │          OHLCV with │       │  - Real/Fake        │       │
│   │          wave       │       │  - Rule violations  │       │
│   │          pattern    │       │  - Fib alignment    │       │
│   └─────────────────────┘       └─────────────────────┘       │
│            ▲                              │                    │
│            │         Adversarial          │                    │
│            └────────── Feedback ──────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Three Models

### Model 1: Pattern Detector

**Purpose:** Find Elliott Wave patterns in market data

**Training Data:** Real market OHLCV with labeled patterns (from pipeline output)

**Architecture:** Temporal Fusion Transformer or CNN-LSTM

**Output:**
- Pattern type (Impulse / Corrective / None)
- Wave positions (indices of wave points)
- Current wave state (for partial patterns)
- Confidence score

---

### Model 2: Pattern Critic

**Purpose:** Validate patterns and explain violations

**Training Data:** 
- Valid patterns (positive samples)
- Invalid patterns with known rule violations (negative samples)
- Synthetically modified patterns with targeted rule breaks

**Architecture:** Multi-task classifier with rule-specific heads

**Output:**
```python
{
    "real_probability": 0.34,
    "pattern_type": "impulse",
    "rule_scores": {
        "w2_1": {"passes": True,  "confidence": 0.95, "margin": 12.3},
        "w3_1": {"passes": False, "confidence": 0.88, "margin": -2.1},
        "w4_1": {"passes": True,  "confidence": 0.92, "margin": 5.7},
        ...
    },
    "fibonacci_alignment": {
        "w2_retracement": {"actual": 0.72, "ideal_range": [0.382, 0.618], "score": 0.4},
        "w3_extension": {"actual": 1.45, "ideal_range": [1.618, 2.618], "score": 0.6},
        ...
    },
    "overall_validity": False,
    "primary_violations": ["w3_1: Wave 3 is shortest wave"]
}
```

---

### Model 3: Pattern Generator

**Purpose:** Generate synthetic OHLCV data with valid Elliott Wave patterns

**Training:** Adversarial training against the Critic

**Architecture:** Conditional GAN / Diffusion Model / Autoregressive Transformer

**Conditioning Inputs:**
- Pattern type (impulse / corrective)
- Target duration (bars)
- Volatility level
- Specific Fibonacci ratios
- Starting price

**Output:** Synthetic OHLCV sequence with embedded wave pattern

---

## Why This Approach?

### Traditional Discriminator:
```
Discriminator: "Fake" (0.34 probability real)
Generator: ??? (no idea what to fix)
```

### Rule-Aware Critic:
```
Critic: "Fake because:
  - Wave 3 is 5.2 points, shorter than Wave 1 (8.1) and Wave 5 (6.3)
  - Wave 2 retracement is 82%, should be 38.2%-61.8%
  - Wave 4 low is 2.3% into Wave 1 territory"

Generator: Targeted fixes for each violation
```

---

## Training Pipeline

### Phase 1: Detector Training
```
Input: Real market data with labeled patterns
Output: Model that finds patterns in OHLCV
```

### Phase 2: Critic Training
```
Input: Valid + invalid patterns (with known violations)
Output: Model that validates and explains pattern quality
```

### Phase 3: Generator Training (Adversarial)
```
Loop:
  1. Generator creates synthetic OHLCV with pattern
  2. Critic evaluates and provides rule-by-rule feedback
  3. Generator updates to fix specific violations
  4. Repeat until Critic can't distinguish real from generated
```

### Phase 4: Self-Improvement Loop
```
Generator produces synthetic data
  ↓
Detector trains on synthetic + real data
  ↓
Better Detector finds more/better patterns
  ↓
Critic trains on expanded dataset
  ↓
Better Critic provides sharper feedback
  ↓
Generator improves further
  ↓
(Repeat)
```

---

## Generating Invalid Samples for Critic Training

For each valid pattern, create targeted violations:

| Modification | Rule Violated | How to Generate |
|-------------|---------------|-----------------|
| Extend Wave 2 below Wave 0 | w2_1: W2 > 100% retracement | Scale W2 down by 1.1x-1.5x |
| Shorten Wave 3 | w3_1: W3 shortest | Compress W3 to be < min(W1, W5) |
| Lower Wave 4 into Wave 1 | w4_1: W4/W1 overlap | Translate W4 down until overlap |
| Truncate Wave 5 | w5_1: W5 < W3 high | Cap W5 below W3 endpoint |
| Wrong Fibonacci ratios | Soft guideline | Scale waves to wrong ratios |
| Time distortion | Duration rules | Stretch/compress wave durations |

---

## Loss Functions

### Critic Loss
```python
L_critic = L_classification + λ1 * L_rule_detection + λ2 * L_fib_alignment

# L_classification: Binary cross-entropy (real vs fake)
# L_rule_detection: Multi-label BCE for each rule violation
# L_fib_alignment: MSE on Fibonacci ratio predictions
```

### Generator Loss
```python
L_generator = L_adversarial + λ1 * L_rule_compliance + λ2 * L_realism

# L_adversarial: Fool the critic (standard GAN loss)
# L_rule_compliance: Penalize predicted rule violations
# L_realism: Penalize invalid OHLCV (e.g., High < Low)
```

---

## Key Benefits

1. **Interpretability**: The Critic explains exactly why a pattern is invalid
2. **Targeted Learning**: Generator receives specific guidance on what to fix
3. **Rule Enforcement**: Hard constraints from Elliott Wave theory are explicitly modeled
4. **Self-Improvement**: System continuously improves as models train on each other's outputs
5. **Synthetic Data Quality**: Generated patterns are guaranteed to follow Elliott Wave rules

---

## Next Steps

- [ ] Implement Pattern Detector baseline
- [ ] Build Critic with rule-specific heads
- [ ] Design Generator architecture (GAN vs Diffusion)
- [ ] Create invalid pattern generation pipeline
- [ ] Set up adversarial training loop
- [ ] Implement self-improvement cycle
- [ ] Evaluate on held-out real market data

---

## Requirements

```
python >= 3.8
pytorch >= 2.0
numpy
pandas
ta-lib  # Technical analysis library
```

---

## License

[Your License Here]

---

## Citation

If you use this work, please cite:

```bibtex
@software{elliott_wave_gan,
  title={Elliott Wave GAN: Adversarial Pattern Detection and Generation},
  author={Your Name},
  year={2026}
}
```
