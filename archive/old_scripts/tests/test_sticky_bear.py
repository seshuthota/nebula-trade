#!/usr/bin/env python3
"""
Test sticky BEAR configurations on Q3 2025

Compares:
- Baseline (3-day symmetric)
- 5-day sticky BEAR (bear_exit_hysteresis_days=5)
- 7-day sticky BEAR (bear_exit_hysteresis_days=7)
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from astra.ensemble.backtest_ensemble import EnsembleBacktester

def test_configuration(config_path: str, description: str):
    """Test a specific configuration."""
    print("\n" + "="*80)
    print(f"Testing: {description}")
    print(f"Config: {config_path}")
    print("="*80 + "\n")

    backtester = EnsembleBacktester(config_path=config_path)
    results = backtester.compare_with_baselines(
        start_date="2025-07-01",
        end_date="2025-09-26",
        initial_capital=100000
    )

    return results

def main():
    """Run all tests and compare."""

    print("\n" + "🔬 STICKY BEAR HYPOTHESIS TEST")
    print("="*80)
    print("Testing asymmetric hysteresis on Q3 2025 correction")
    print("Objective: Reduce whipsaw losses by increasing BEAR exit delay")
    print("="*80)

    # Test configurations
    configs = [
        ("config/ensemble.yaml", "Baseline (3-day symmetric)"),
        ("config/ensemble_sticky_5day.yaml", "5-day Sticky BEAR"),
        ("config/ensemble_sticky_7day.yaml", "7-day Sticky BEAR"),
    ]

    all_results = {}

    for config_path, description in configs:
        results = test_configuration(config_path, description)
        all_results[description] = results

    # Comparison summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY - Q3 2025 (Jul-Sep, 62 days)")
    print("="*80)
    print(f"{'Configuration':<30} {'Return':>10} {'vs EW':>10} {'Switches':>10} {'v2 Usage':>12}")
    print("-"*80)

    for desc, results in all_results.items():
        ens = results['ensemble']
        perf = results['performance']
        stats = ens['stats']

        print(f"{desc:<30} {ens['return']:>9.2%} {perf['return_gap_vs_ew']:>+9.2f}pp "
              f"{stats['total_switches']:>9} {stats['model_usage']['v2']:>11.1%}")

    print("="*80)

    # Analysis
    print("\n📈 ANALYSIS:")
    baseline = all_results["Baseline (3-day symmetric)"]
    baseline_gap = baseline['performance']['return_gap_vs_ew']

    for desc, results in all_results.items():
        if desc == "Baseline (3-day symmetric)":
            continue

        gap = results['performance']['return_gap_vs_ew']
        improvement = gap - baseline_gap

        status = "✅ IMPROVED" if improvement > 0 else "❌ WORSE"
        print(f"{status}: {desc} - Gap improvement: {improvement:+.2f}pp")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
