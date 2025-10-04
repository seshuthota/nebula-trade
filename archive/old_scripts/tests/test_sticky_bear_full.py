#!/usr/bin/env python3
"""
Test sticky BEAR on ALL 2025 periods

Tests:
- Q1 2025 (bull market) - ensure no degradation
- Q3 2025 (correction) - expect improvement
- Full 2025 YTD - expect overall improvement
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from astra.ensemble.backtest_ensemble import EnsembleBacktester

def test_period(config_path: str, description: str, start: str, end: str, period_name: str):
    """Test a specific period."""
    print(f"\n{'='*80}")
    print(f"{period_name}: {description}")
    print(f"{'='*80}\n")

    backtester = EnsembleBacktester(config_path=config_path)
    results = backtester.compare_with_baselines(
        start_date=start,
        end_date=end,
        initial_capital=100000
    )

    return results

def main():
    """Run comprehensive tests."""

    print("\n🎯 COMPREHENSIVE STICKY BEAR TEST")
    print("="*80)
    print("Testing 7-day sticky BEAR vs baseline on all 2025 periods")
    print("="*80)

    # Test periods
    periods = [
        ("2025-01-01", "2025-03-31", "Q1 2025 Bull Market", 62),
        ("2025-07-01", "2025-09-26", "Q3 2025 Correction", 62),
        ("2025-01-01", "2025-09-26", "2025 YTD Full", 186),
    ]

    configs = [
        ("config/ensemble.yaml", "Baseline (3-day)"),
        ("config/ensemble_sticky_7day.yaml", "7-day Sticky BEAR"),
    ]

    all_results = {}

    for start, end, period_name, days in periods:
        all_results[period_name] = {}

        for config_path, description in configs:
            results = test_period(config_path, description, start, end, f"{period_name} - {description}")
            all_results[period_name][description] = results

    # Final comparison
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON - ALL PERIODS")
    print("="*80)

    for period_name in all_results.keys():
        print(f"\n{period_name}:")
        print("-"*80)
        print(f"{'Config':<25} {'Return':>10} {'vs EW':>10} {'Switches':>10} {'v2 Usage':>12} {'Sharpe':>10}")
        print("-"*80)

        for config_name, results in all_results[period_name].items():
            ens = results['ensemble']
            perf = results['performance']
            stats = ens['stats']

            print(f"{config_name:<25} {ens['return']:>9.2%} {perf['return_gap_vs_ew']:>+9.2f}pp "
                  f"{stats['total_switches']:>9} {stats['model_usage']['v2']:>11.1%} {ens['sharpe']:>9.3f}")

        # Calculate improvement
        baseline = all_results[period_name]["Baseline (3-day)"]
        sticky = all_results[period_name]["7-day Sticky BEAR"]

        return_improvement = sticky['ensemble']['return'] - baseline['ensemble']['return']
        gap_improvement = sticky['performance']['return_gap_vs_ew'] - baseline['performance']['return_gap_vs_ew']

        status = "✅ IMPROVED" if gap_improvement > 0 else "⚠️ DEGRADED" if gap_improvement < -0.5 else "➡️ NEUTRAL"
        print(f"\n{status} Improvement: Return {return_improvement:+.2%}, Gap vs EW {gap_improvement:+.2f}pp")

    print("\n" + "="*80)
    print("🎯 FINAL VERDICT")
    print("="*80)

    # Calculate YTD improvement
    ytd_baseline = all_results["2025 YTD Full"]["Baseline (3-day)"]
    ytd_sticky = all_results["2025 YTD Full"]["7-day Sticky BEAR"]

    ytd_return_improvement = ytd_sticky['ensemble']['return'] - ytd_baseline['ensemble']['return']
    ytd_gap_improvement = ytd_sticky['performance']['return_gap_vs_ew'] - ytd_baseline['performance']['return_gap_vs_ew']

    print(f"\n2025 YTD Performance:")
    print(f"  Baseline:    {ytd_baseline['ensemble']['return']:.2%} (gap: {ytd_baseline['performance']['return_gap_vs_ew']:+.2f}pp)")
    print(f"  Sticky BEAR: {ytd_sticky['ensemble']['return']:.2%} (gap: {ytd_sticky['performance']['return_gap_vs_ew']:+.2f}pp)")
    print(f"  Improvement: {ytd_return_improvement:+.2%} (gap: {ytd_gap_improvement:+.2f}pp)")

    if ytd_sticky['ensemble']['return'] >= 0.125:  # 12.5% target
        print(f"\n✅ SUCCESS: Sticky BEAR achieves {ytd_sticky['ensemble']['return']:.2%} YTD (target: 12-13%)")
    else:
        print(f"\n⚠️  Below 12.5% target: {ytd_sticky['ensemble']['return']:.2%}")

    if ytd_gap_improvement > 0:
        print(f"✅ Sticky BEAR beats baseline by {ytd_gap_improvement:+.2f}pp")
    else:
        print(f"❌ Sticky BEAR underperforms baseline by {ytd_gap_improvement:.2f}pp")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
