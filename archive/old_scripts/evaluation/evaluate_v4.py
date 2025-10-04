#!/usr/bin/env python3
"""Quick evaluation script for v4 model."""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester

# Load validation data
val_data = pd.read_csv('production/logs/training/val_data_v4_20251004_120629.csv',
                       index_col=0, parse_dates=True)

print("=" * 80)
print("v4 MODEL VALIDATION RESULTS (from training logs)")
print("=" * 80)

# From training logs
mean_return = 0.3650
std_return = 0.0000
mean_final_value = 136501.01
mean_reward = -7.8723

print(f"Mean Return:      {mean_return:.2%}")
print(f"Std Return:       {std_return:.2%}")
print(f"Mean Final Value: ₹{mean_final_value:,.0f}")
print(f"Mean Reward:      {mean_reward:.4f}")
print(f"Validation Period: {val_data.index[0].date()} to {val_data.index[-1].date()}")
print(f"Validation Samples: {len(val_data)}")
print("")

# Compare with baselines
print("Backtesting classical methods on validation set...")
backtester = PortfolioBacktester(val_data)

# Use equal weight portfolio as baseline
assets = ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS']
equal_weights = {asset: 0.2 for asset in assets}

eq_result = backtester.backtest_portfolio(
    weights=equal_weights,
    rebalance_freq='daily',
    transaction_cost=0.001
)

classical_results = {'equal_weight': eq_result}
print(f"  Equal Weight         | Return: {eq_result['total_return']:7.2%} | "
      f"Sharpe: {eq_result['sharpe_ratio']:6.3f}")

print("")
print("=" * 80)
print("VALIDATION COMPARISON")
print("=" * 80)
print(f"v4 (2007-2024):      {mean_return:.2%} return")
print(f"Equal Weight:        {classical_results['equal_weight']['total_return']:.2%} return")

print("")
print("COMPARISON WITH PREVIOUS MODELS:")
print(f"  v1 (Momentum):     14.62% validation")
print(f"  v2 (Defensive):    7.87% validation")
print(f"  v2.1 (Balanced):   11.83% validation")
print(f"  v4 (Historical):   {mean_return:.2%} validation")

# Calculate gaps
eq_gap = (mean_return - classical_results['equal_weight']['total_return']) * 100
print("")
if eq_gap > 0:
    print(f"✅ BEATS Equal Weight by {eq_gap:+.2f} pp")
else:
    print(f"❌ BELOW Equal Weight by {abs(eq_gap):.2f} pp")

# Success criteria
print("")
print("=" * 80)
print("v4 SUCCESS CRITERIA EVALUATION")
print("=" * 80)

go_criteria = []

# Criterion 1: Beat Equal Weight
if mean_return > classical_results['equal_weight']['total_return']:
    print("✅ Criterion 1: Beats Equal Weight baseline")
    go_criteria.append(True)
else:
    print(f"❌ Criterion 1: Below Equal Weight by {abs(eq_gap):.2f} pp")
    go_criteria.append(False)

# Criterion 2: Deterministic
if std_return < 0.02:
    print(f"✅ Criterion 2: Deterministic behavior (std={std_return:.2%} < 2%)")
    go_criteria.append(True)
else:
    print(f"⚠️  Criterion 2: High variance (std={std_return:.2%} >= 2%)")
    go_criteria.append(False)

# Criterion 3: Target return
if mean_return > 0.10:
    print(f"✅ Criterion 3: Target return met ({mean_return:.2%} > 10%)")
    go_criteria.append(True)
else:
    print(f"⚠️  Criterion 3: Below target ({mean_return:.2%} < 10%)")
    go_criteria.append(False)

print("")
passed = sum(go_criteria)
total = len(go_criteria)

if passed == total:
    print(f"🎯 DECISION: GO - All criteria met ({passed}/{total})")
    print("   ✅ Ready for comprehensive paper trading")
    print("   ✅ Expected: Superior bear market performance!")
elif passed >= 2:
    print(f"⚠️  DECISION: CONDITIONAL - Most criteria met ({passed}/{total})")
    print("   → Proceed to paper trading for detailed evaluation")
else:
    print(f"🛑 DECISION: NO-GO - Insufficient performance ({passed}/{total})")
    print("   → Consider further tuning")

# Save report
report = {
    'model_type': 'production_v4_historical_2007_2024',
    'validation_results': {
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'mean_final_value': float(mean_final_value),
        'mean_reward': float(mean_reward),
    },
    'classical_baselines': {
        name: {
            'total_return': float(result['total_return']),
            'final_value': float(result['final_value']),
            'sharpe_ratio': float(result['sharpe_ratio'])
        } for name, result in classical_results.items()
    },
    'success_criteria': {
        'beats_equal_weight': go_criteria[0],
        'deterministic': go_criteria[1],
        'target_return': go_criteria[2],
        'passed': passed,
        'total': total
    }
}

with open('production/models/v4_historical_2007_2024/validation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("")
print("Report saved to: production/models/v4_historical_2007_2024/validation_report.json")
