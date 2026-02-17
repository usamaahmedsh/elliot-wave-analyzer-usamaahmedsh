#!/usr/bin/env python3
"""
Elliott Wave Pattern Backtesting Framework

Simulates trading based on detected patterns and evaluates performance:
- Entry/exit signals from pattern detection
- Position sizing and risk management
- Performance metrics (returns, Sharpe, win rate, drawdown)
- Comparison against buy-and-hold baseline
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position_size: float
    pattern_type: str
    pattern_score: float
    return_pct: float
    profit_loss: float


@dataclass
class BacktestResults:
    """Backtesting performance results."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    avg_return: float
    median_return: float
    max_return: float
    min_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_profit: float
    total_loss: float
    avg_holding_days: float
    trades: List[Trade]


def load_results(json_path: str) -> Dict[str, Any]:
    """Load pattern detection results."""
    with open(json_path, 'r') as f:
        return json.load(f)


def fetch_price_data(symbol: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
    """Fetch historical price data for backtesting."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        return {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'close': df['Close'].tolist(),
            'high': df['High'].tolist(),
            'low': df['Low'].tolist(),
        }
    except Exception as e:
        print(f"Warning: Failed to fetch data for {symbol}: {e}")
        return None


def get_price_at_date(price_data: Dict[str, Any], target_date: str, price_type: str = 'close') -> Optional[float]:
    """Get price at a specific date."""
    # Normalize target_date to YYYY-MM-DD format
    if 'T' in target_date:
        target_date = target_date.split('T')[0]
    
    if target_date not in price_data['dates']:
        # Find nearest date
        dates = price_data['dates']
        if target_date < dates[0]:
            idx = 0
        elif target_date > dates[-1]:
            idx = len(dates) - 1
        else:
            # Find nearest
            for i, date in enumerate(dates):
                if date >= target_date:
                    idx = i
                    break
    else:
        idx = price_data['dates'].index(target_date)
    
    return price_data[price_type][idx]


def simulate_trade(
    symbol: str,
    pattern: Dict[str, Any],
    price_data: Dict[str, Any],
    holding_days: int = 20,
    position_size: float = 1000.0,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.10
) -> Optional[Trade]:
    """
    Simulate a trade based on a detected pattern.
    
    Strategy:
    - Enter at pattern end date
    - Exit after holding_days OR when stop-loss/take-profit hit
    - Impulsive patterns: Go long (expect continuation)
    - Corrective patterns: Skip or inverse (expect reversal)
    """
    pattern_type = pattern.get('best', {}).get('rule_name', 'unknown')
    
    # Skip corrective patterns for now (simple strategy)
    if pattern_type.lower() == 'corrective':
        return None
    
    entry_date = pattern.get('date_end')
    if not entry_date:
        return None
    
    # Normalize entry_date to YYYY-MM-DD format
    if isinstance(entry_date, str) and 'T' in entry_date:
        entry_date = entry_date.split('T')[0]
    elif hasattr(entry_date, 'strftime'):
        entry_date = entry_date.strftime('%Y-%m-%d')
    else:
        entry_date = str(entry_date)[:10]  # Take first 10 chars (YYYY-MM-DD)
    
    # Get entry price
    entry_price = get_price_at_date(price_data, entry_date)
    if entry_price is None:
        return None
    
    # Calculate exit date
    entry_idx = price_data['dates'].index(entry_date) if entry_date in price_data['dates'] else 0
    exit_idx = min(entry_idx + holding_days, len(price_data['dates']) - 1)
    
    # Simulate daily price movements
    best_exit_price = entry_price
    exit_reason = 'holding_period'
    actual_exit_idx = exit_idx
    
    for i in range(entry_idx + 1, exit_idx + 1):
        current_high = price_data['high'][i]
        current_low = price_data['low'][i]
        current_close = price_data['close'][i]
        
        # Check take profit
        if current_high >= entry_price * (1 + take_profit_pct):
            best_exit_price = entry_price * (1 + take_profit_pct)
            exit_reason = 'take_profit'
            actual_exit_idx = i
            break
        
        # Check stop loss
        if current_low <= entry_price * (1 - stop_loss_pct):
            best_exit_price = entry_price * (1 - stop_loss_pct)
            exit_reason = 'stop_loss'
            actual_exit_idx = i
            break
        
        # Update best exit (use close)
        best_exit_price = current_close
        actual_exit_idx = i
    
    exit_date = price_data['dates'][actual_exit_idx]
    exit_price = best_exit_price
    
    # Calculate returns
    return_pct = (exit_price - entry_price) / entry_price
    profit_loss = position_size * return_pct
    
    return Trade(
        symbol=symbol,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        position_size=position_size,
        pattern_type=pattern_type,
        pattern_score=pattern.get('ensemble_score', 0.0),
        return_pct=return_pct,
        profit_loss=profit_loss
    )


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns."""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(cumulative_returns: List[float]) -> float:
    """Calculate maximum drawdown."""
    if not cumulative_returns:
        return 0.0
    
    peak = cumulative_returns[0]
    max_dd = 0.0
    
    for value in cumulative_returns:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    
    return max_dd


def run_backtest(
    patterns: List[Dict[str, Any]],
    min_score: float = 0.70,
    holding_days: int = 20,
    position_size: float = 1000.0,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.10
) -> BacktestResults:
    """Run backtest on detected patterns."""
    
    # Filter patterns by score
    filtered_patterns = [p for p in patterns if p.get('ensemble_score', 0.0) >= min_score]
    
    print(f"\n🔍 Backtesting {len(filtered_patterns)} patterns (min_score={min_score})...")
    
    # Group by symbol to minimize data fetching
    by_symbol = {}
    for pattern in filtered_patterns:
        symbol = pattern.get('symbol', 'UNKNOWN')
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(pattern)
    
    # Simulate trades
    all_trades = []
    
    for symbol, symbol_patterns in by_symbol.items():
        print(f"  Processing {symbol}: {len(symbol_patterns)} patterns...")
        
        # Fetch price data (extended range for exits)
        min_date = min(p.get('date_start', '9999-12-31') for p in symbol_patterns)
        max_date = max(p.get('date_end', '1900-01-01') for p in symbol_patterns)
        
        # Normalize dates to YYYY-MM-DD format
        if isinstance(min_date, str) and 'T' in min_date:
            min_date = min_date.split('T')[0]
        if isinstance(max_date, str) and 'T' in max_date:
            max_date = max_date.split('T')[0]
        
        # Extend end date for holding period
        try:
            end_date_obj = datetime.strptime(max_date, '%Y-%m-%d') + timedelta(days=holding_days + 30)
            extended_end = end_date_obj.strftime('%Y-%m-%d')
        except:
            extended_end = '2026-12-31'
        
        price_data = fetch_price_data(symbol, min_date, extended_end)
        if not price_data:
            continue
        
        # Simulate each pattern
        for pattern in symbol_patterns:
            trade = simulate_trade(
                symbol, pattern, price_data,
                holding_days=holding_days,
                position_size=position_size,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
            if trade:
                all_trades.append(trade)
    
    if not all_trades:
        print("❌ No valid trades executed.")
        return BacktestResults(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_return=0.0, avg_return=0.0,
            median_return=0.0, max_return=0.0, min_return=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
            total_profit=0.0, total_loss=0.0, avg_holding_days=0.0,
            trades=[]
        )
    
    # Calculate metrics
    returns = [t.return_pct for t in all_trades]
    profits = [t.profit_loss for t in all_trades if t.profit_loss > 0]
    losses = [abs(t.profit_loss) for t in all_trades if t.profit_loss < 0]
    
    winning_trades = len([t for t in all_trades if t.profit_loss > 0])
    losing_trades = len([t for t in all_trades if t.profit_loss < 0])
    
    # Calculate holding days
    holding_days_list = []
    for t in all_trades:
        try:
            entry = datetime.strptime(t.entry_date, '%Y-%m-%d')
            exit = datetime.strptime(t.exit_date, '%Y-%m-%d')
            holding_days_list.append((exit - entry).days)
        except:
            pass
    
    # Cumulative returns for drawdown
    cumulative = [position_size]
    for t in all_trades:
        cumulative.append(cumulative[-1] + t.profit_loss)
    
    return BacktestResults(
        total_trades=len(all_trades),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / len(all_trades) if all_trades else 0.0,
        total_return=sum(returns),
        avg_return=np.mean(returns),
        median_return=np.median(returns),
        max_return=max(returns),
        min_return=min(returns),
        sharpe_ratio=calculate_sharpe_ratio(returns),
        max_drawdown=calculate_max_drawdown(cumulative),
        profit_factor=sum(profits) / sum(losses) if losses else float('inf'),
        total_profit=sum(profits),
        total_loss=sum(losses),
        avg_holding_days=np.mean(holding_days_list) if holding_days_list else 0.0,
        trades=all_trades
    )


def print_backtest_results(results: BacktestResults):
    """Pretty print backtest results."""
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    
    print(f"\n📊 Trade Statistics:")
    print(f"  Total Trades:     {results.total_trades}")
    print(f"  Winning Trades:   {results.winning_trades} ({results.win_rate*100:.1f}%)")
    print(f"  Losing Trades:    {results.losing_trades} ({(1-results.win_rate)*100:.1f}%)")
    print(f"  Avg Holding:      {results.avg_holding_days:.1f} days")
    
    print(f"\n💰 Returns:")
    print(f"  Total Return:     {results.total_return*100:.2f}%")
    print(f"  Average Return:   {results.avg_return*100:.2f}%")
    print(f"  Median Return:    {results.median_return*100:.2f}%")
    print(f"  Best Trade:       {results.max_return*100:.2f}%")
    print(f"  Worst Trade:      {results.min_return*100:.2f}%")
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Sharpe Ratio:     {results.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:     {results.max_drawdown*100:.2f}%")
    print(f"  Profit Factor:    {results.profit_factor:.2f}")
    print(f"  Total Profit:     ${results.total_profit:.2f}")
    print(f"  Total Loss:       ${results.total_loss:.2f}")
    print(f"  Net P/L:          ${results.total_profit - results.total_loss:.2f}")
    
    # Top 5 best trades
    if results.trades:
        print(f"\n🏆 Top 5 Best Trades:")
        sorted_trades = sorted(results.trades, key=lambda t: t.return_pct, reverse=True)[:5]
        for i, trade in enumerate(sorted_trades, 1):
            print(f"  {i}. {trade.symbol}: {trade.return_pct*100:+.2f}% "
                  f"(score={trade.pattern_score:.3f}, {trade.entry_date} → {trade.exit_date})")
        
        print(f"\n💩 Top 5 Worst Trades:")
        sorted_trades = sorted(results.trades, key=lambda t: t.return_pct)[:5]
        for i, trade in enumerate(sorted_trades, 1):
            print(f"  {i}. {trade.symbol}: {trade.return_pct*100:+.2f}% "
                  f"(score={trade.pattern_score:.3f}, {trade.entry_date} → {trade.exit_date})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest Elliott Wave patterns')
    parser.add_argument('--input', '-i', default='output/results.json',
                       help='Path to results JSON file')
    parser.add_argument('--min-score', type=float, default=0.70,
                       help='Minimum pattern score to trade (default: 0.70)')
    parser.add_argument('--symbol', '-s',
                       help='Backtest only this symbol (default: all symbols)')
    parser.add_argument('--pattern-type', choices=['impulse', 'corrective', 'all'], default='all',
                       help='Pattern type to backtest (default: all, but corrective currently skipped)')
    parser.add_argument('--holding-days', type=int, default=20,
                       help='Maximum holding period in days (default: 20)')
    parser.add_argument('--position-size', type=float, default=1000.0,
                       help='Position size per trade in $ (default: 1000)')
    parser.add_argument('--stop-loss', type=float, default=0.05,
                       help='Stop loss percentage (default: 0.05 = 5%%)')
    parser.add_argument('--take-profit', type=float, default=0.10,
                       help='Take profit percentage (default: 0.10 = 10%%)')
    parser.add_argument('--export-trades', '-e',
                       help='Export trade log to CSV file')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    try:
        results = load_results(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    patterns = results.get('patterns', [])
    
    if not patterns:
        print("No patterns found in results.")
        sys.exit(1)
    
    # Filter by symbol if specified
    if args.symbol:
        patterns = [p for p in patterns if p.get('symbol', '').upper() == args.symbol.upper()]
        if not patterns:
            print(f"No patterns found for symbol {args.symbol}")
            sys.exit(1)
        print(f"Filtered to {len(patterns)} patterns for {args.symbol}")
    
    # Filter by pattern type if specified
    if args.pattern_type != 'all':
        patterns = [p for p in patterns 
                   if p.get('best', {}).get('rule_name', '').lower() == args.pattern_type.lower()]
        if not patterns:
            print(f"No {args.pattern_type} patterns found")
            sys.exit(1)
        print(f"Filtered to {len(patterns)} {args.pattern_type} patterns")
    
    # Run backtest
    backtest_results = run_backtest(
        patterns,
        min_score=args.min_score,
        holding_days=args.holding_days,
        position_size=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit
    )
    
    # Print results
    print_backtest_results(backtest_results)
    
    # Export trades
    if args.export_trades and backtest_results.trades:
        import csv
        
        with open(args.export_trades, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Symbol', 'Entry Date', 'Exit Date', 'Entry Price', 'Exit Price',
                           'Return %', 'P/L $', 'Pattern Type', 'Pattern Score', 'Position Size'])
            
            for trade in backtest_results.trades:
                writer.writerow([
                    trade.symbol, trade.entry_date, trade.exit_date,
                    f"{trade.entry_price:.2f}", f"{trade.exit_price:.2f}",
                    f"{trade.return_pct*100:.2f}", f"{trade.profit_loss:.2f}",
                    trade.pattern_type, f"{trade.pattern_score:.3f}",
                    f"{trade.position_size:.2f}"
                ])
        
        print(f"\n✅ Exported {len(backtest_results.trades)} trades to {args.export_trades}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
