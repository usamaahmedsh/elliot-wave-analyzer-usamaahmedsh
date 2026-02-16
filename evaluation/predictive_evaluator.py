"""Predictive power evaluation for Elliott Wave patterns.

Evaluates whether detected patterns have predictive power for future price movements.
After detecting a Wave 5 completion, does price actually reverse as expected?

Metrics:
- Direction accuracy: Does price move in predicted direction?
- Magnitude accuracy: Does price move by predicted amount?
- Win rate: % of profitable trades if acting on signals
- Sharpe ratio: Risk-adjusted returns
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PredictionResult:
    """Result of a single pattern prediction."""
    pattern_id: str
    completion_date: str
    completion_idx: int
    predicted_direction: str  # 'up' or 'down'
    predicted_reversal: bool
    
    # Forward-looking results (measured after N days)
    forward_days: int
    actual_direction: str
    actual_return: float
    prediction_correct: bool
    
    # Pattern metadata
    pattern_type: str = 'impulsive'
    confidence_score: float = 0.0


class PredictiveEvaluator:
    """Evaluate predictive power of Elliott Wave patterns."""
    
    def __init__(self, forward_horizon_days: int = 30):
        """Initialize evaluator.
        
        Args:
            forward_horizon_days: How many days ahead to measure results (default 30)
        """
        self.forward_horizon_days = forward_horizon_days
    
    def predict_from_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction from completed wave pattern.
        
        For impulsive 5-wave pattern: Expect reversal after Wave 5
        For corrective ABC pattern: Expect continuation after C
        
        Args:
            pattern: Detected pattern dict
            
        Returns:
            Prediction dict with expected direction and magnitude
        """
        pattern_type = pattern.get('pattern_type', 'impulsive')
        direction = pattern.get('direction', 'bullish')
        waves = pattern.get('waves', [])
        
        if pattern_type == 'impulsive' and len(waves) >= 5:
            # After completing 5 waves, expect reversal
            if direction in ['bullish', 'up']:
                predicted_direction = 'down'
                predicted_reversal = True
            else:
                predicted_direction = 'up'
                predicted_reversal = True
        
        elif pattern_type == 'corrective' and len(waves) >= 3:
            # After completing ABC correction, expect continuation of prior trend
            # (Need prior trend direction - simplified assumption)
            predicted_direction = direction  # Assume continuation
            predicted_reversal = False
        
        else:
            # Unknown pattern type
            predicted_direction = 'unknown'
            predicted_reversal = False
        
        return {
            'predicted_direction': predicted_direction,
            'predicted_reversal': predicted_reversal,
            'pattern_type': pattern_type,
            'confidence': pattern.get('ensemble_score', pattern.get('score', 0.5)),
        }
    
    def evaluate_prediction(
        self,
        pattern: Dict[str, Any],
        prices: np.ndarray,
        dates: np.ndarray,
        completion_idx: int
    ) -> Optional[PredictionResult]:
        """Evaluate a pattern's prediction against actual future price movement.
        
        Args:
            pattern: Detected pattern dict
            prices: Price array (close prices)
            dates: Dates array
            completion_idx: Index where pattern completed
            
        Returns:
            PredictionResult if evaluation possible, None otherwise
        """
        # Generate prediction
        prediction = self.predict_from_pattern(pattern)
        
        if prediction['predicted_direction'] == 'unknown':
            return None
        
        # Find forward index (N days ahead)
        if completion_idx + self.forward_horizon_days >= len(prices):
            return None  # Not enough future data
        
        forward_idx = completion_idx + self.forward_horizon_days
        
        # Measure actual price movement
        completion_price = prices[completion_idx]
        forward_price = prices[forward_idx]
        
        actual_return = (forward_price - completion_price) / completion_price
        actual_direction = 'up' if actual_return > 0 else 'down'
        
        # Check if prediction was correct
        prediction_correct = (actual_direction == prediction['predicted_direction'])
        
        return PredictionResult(
            pattern_id=pattern.get('id', 'unknown'),
            completion_date=str(dates[completion_idx]) if completion_idx < len(dates) else 'unknown',
            completion_idx=completion_idx,
            predicted_direction=prediction['predicted_direction'],
            predicted_reversal=prediction['predicted_reversal'],
            forward_days=self.forward_horizon_days,
            actual_direction=actual_direction,
            actual_return=actual_return,
            prediction_correct=prediction_correct,
            pattern_type=pattern.get('pattern_type', 'impulsive'),
            confidence_score=prediction['confidence'],
        )
    
    def evaluate_batch(
        self,
        patterns: List[Dict[str, Any]],
        prices: np.ndarray,
        dates: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate predictive power across multiple patterns.
        
        Args:
            patterns: List of detected patterns
            prices: Price array (close prices)
            dates: Dates array
            
        Returns:
            Dict with evaluation metrics
        """
        results = []
        
        for pattern in patterns:
            # Get completion index (end of last wave)
            waves = pattern.get('waves', [])
            if not waves:
                continue
            
            completion_idx = waves[-1].get('end_idx', 0)
            
            result = self.evaluate_prediction(pattern, prices, dates, completion_idx)
            if result is not None:
                results.append(result)
        
        if not results:
            return {
                'total_predictions': 0,
                'evaluable_predictions': 0,
                'accuracy': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
            }
        
        # Calculate metrics
        n = len(results)
        correct = sum(1 for r in results if r.prediction_correct)
        accuracy = correct / n
        
        # Returns
        returns = [r.actual_return for r in results]
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0.0
        
        # Win rate (positive returns)
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / n
        
        # By pattern type
        by_type = {}
        for pattern_type in ['impulsive', 'corrective']:
            type_results = [r for r in results if r.pattern_type == pattern_type]
            if type_results:
                type_correct = sum(1 for r in type_results if r.prediction_correct)
                by_type[pattern_type] = {
                    'count': len(type_results),
                    'accuracy': type_correct / len(type_results),
                    'avg_return': np.mean([r.actual_return for r in type_results]),
                }
        
        return {
            'total_predictions': len(patterns),
            'evaluable_predictions': n,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe_ratio,
            'forward_horizon_days': self.forward_horizon_days,
            'by_pattern_type': by_type,
            'sample_predictions': [
                {
                    'pattern_id': r.pattern_id,
                    'predicted': r.predicted_direction,
                    'actual': r.actual_direction,
                    'return': round(r.actual_return, 4),
                    'correct': r.prediction_correct,
                }
                for r in results[:10]  # First 10 samples
            ],
        }
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print human-readable evaluation report.
        
        Args:
            results: Results from evaluate_batch()
        """
        print("=" * 70)
        print("ELLIOTT WAVE PREDICTIVE POWER EVALUATION")
        print("=" * 70)
        print(f"\nForward Horizon: {results['forward_horizon_days']} days")
        print(f"Total Patterns: {results['total_predictions']}")
        print(f"Evaluable Predictions: {results['evaluable_predictions']}")
        
        print(f"\nPrediction Metrics:")
        print(f"  Direction Accuracy: {results['accuracy']:.2%}")
        print(f"  Win Rate: {results['win_rate']:.2%}")
        print(f"  Average Return: {results['avg_return']:.2%}")
        print(f"  Std Return: {results['std_return']:.2%}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        
        print(f"\nBy Pattern Type:")
        for ptype, metrics in results.get('by_pattern_type', {}).items():
            print(f"  {ptype.capitalize()}:")
            print(f"    Count: {metrics['count']}")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    Avg Return: {metrics['avg_return']:.2%}")
        
        print("\n" + "=" * 70)
        
        # Interpretation
        accuracy = results['accuracy']
        if accuracy > 0.60:
            print("\n✅ Strong predictive power (>60% accuracy)")
        elif accuracy > 0.55:
            print("\n✓ Moderate predictive power (55-60% accuracy)")
        elif accuracy > 0.50:
            print("\n⚠️  Weak predictive power (50-55% accuracy)")
        else:
            print("\n❌ No predictive power (<50% accuracy, worse than random)")
        
        if results['sharpe_ratio'] > 1.0:
            print("✅ Excellent risk-adjusted returns (Sharpe > 1.0)")
        elif results['sharpe_ratio'] > 0.5:
            print("✓ Good risk-adjusted returns (Sharpe 0.5-1.0)")
        elif results['sharpe_ratio'] > 0:
            print("⚠️  Marginal risk-adjusted returns (Sharpe 0-0.5)")
        else:
            print("❌ Negative risk-adjusted returns (Sharpe < 0)")


def backtest_patterns(
    patterns: List[Dict[str, Any]],
    prices: np.ndarray,
    dates: np.ndarray,
    forward_days: int = 30,
    transaction_cost: float = 0.001
) -> Dict[str, Any]:
    """Backtest trading strategy based on pattern signals.
    
    Simple strategy: 
    - After Wave 5 completion, trade in reversal direction
    - Hold for forward_days
    - Calculate returns accounting for transaction costs
    
    Args:
        patterns: Detected patterns
        prices: Price array
        dates: Dates array
        forward_days: Holding period (days)
        transaction_cost: Transaction cost as fraction (default 0.1%)
        
    Returns:
        Backtest results dict
    """
    evaluator = PredictiveEvaluator(forward_horizon_days=forward_days)
    
    trades = []
    for pattern in patterns:
        waves = pattern.get('waves', [])
        if not waves:
            continue
        
        completion_idx = waves[-1].get('end_idx', 0)
        prediction = evaluator.predict_from_pattern(pattern)
        
        if prediction['predicted_direction'] == 'unknown':
            continue
        
        # Execute trade
        if completion_idx + forward_days >= len(prices):
            continue
        
        entry_price = prices[completion_idx]
        exit_price = prices[completion_idx + forward_days]
        
        # Account for direction
        if prediction['predicted_direction'] == 'up':
            gross_return = (exit_price - entry_price) / entry_price
        else:
            # Short position
            gross_return = (entry_price - exit_price) / entry_price
        
        # Subtract transaction costs
        net_return = gross_return - (2 * transaction_cost)  # Entry + exit
        
        trades.append({
            'entry_idx': completion_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': prediction['predicted_direction'],
            'gross_return': gross_return,
            'net_return': net_return,
            'pattern_confidence': prediction['confidence'],
        })
    
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'cumulative_return': 0.0,
            'sharpe_ratio': 0.0,
        }
    
    # Calculate metrics
    n = len(trades)
    net_returns = [t['net_return'] for t in trades]
    
    wins = sum(1 for r in net_returns if r > 0)
    win_rate = wins / n
    avg_return = np.mean(net_returns)
    std_return = np.std(net_returns) if n > 1 else 0.0
    cumulative_return = np.prod([1 + r for r in net_returns]) - 1
    sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0.0
    
    return {
        'total_trades': n,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'std_return': std_return,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'transaction_cost': transaction_cost,
        'holding_period_days': forward_days,
    }
