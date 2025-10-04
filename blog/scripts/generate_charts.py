#!/usr/bin/env python3
"""
Blog Visualization Generator

Creates all charts and visualizations for the blog series:
1. Master performance comparison
2. Quarterly breakdown heatmap
3. Regime detection timeline
4. Sticky BEAR optimization results
5. Training stability curves
6. Risk-return scatter plot
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Output directory
VISUALS_DIR = project_root / "blog" / "visuals"
VISUALS_DIR.mkdir(exist_ok=True, parents=True)


def load_ensemble_results():
    """Load ensemble comparison results from JSON logs."""
    logs_dir = project_root / "production" / "logs" / "ensemble"

    # Find the YTD comparison file
    ytd_file = list(logs_dir.glob("ensemble_comparison_2025-01-01_2025-09-26_*.json"))[-1]

    with open(ytd_file, 'r') as f:
        return json.load(f)


def create_master_performance_chart():
    """
    Chart 1: Master Performance Comparison
    Bar chart showing all models + ensemble vs Equal Weight
    """
    print("Creating master performance comparison chart...")

    # Data from our experiments
    models = ['Ensemble\n(Sticky)', 'v1', 'v2.1', 'Equal\nWeight', 'v5', 'v3', 'v4', 'v2']
    ytd_returns = [12.92, 10.10, 9.87, 9.29, 7.58, 7.62, 6.21, 6.17]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#95a5a6', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(models, ytd_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, ytd_returns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add benchmark line for Equal Weight
    ax.axhline(y=9.29, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Equal Weight Baseline')

    # Styling
    ax.set_ylabel('2025 YTD Return (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison - 2025 YTD (Jan-Sep)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "01_master_performance_chart.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUALS_DIR / '01_master_performance_chart.png'}")
    plt.close()


def create_quarterly_heatmap():
    """
    Chart 2: Quarterly Performance Heatmap
    Shows Q1, Q3 performance for all models
    """
    print("Creating quarterly breakdown heatmap...")

    # Data
    models = ['Ensemble (Sticky)', 'v1', 'v2.1', 'v5', 'v3', 'v4', 'v2', 'Equal Weight']
    data = {
        'Q1 Bull': [12.95, 12.95, 11.02, 10.10, 3.59, 1.19, 3.21, 5.76],
        'Q3 Bear': [0.73, -6.81, -6.31, -3.17, -4.81, -3.73, 0.34, -3.17],
        'YTD': [12.92, 10.10, 9.87, 7.58, 7.62, 6.21, 6.17, 9.29]
    }

    df = pd.DataFrame(data, index=models)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Return (%)'}, linewidths=2, linecolor='white',
                ax=ax, vmin=-10, vmax=15)

    ax.set_title('Quarterly Performance Breakdown - 2025', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "02_quarterly_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUALS_DIR / '02_quarterly_heatmap.png'}")
    plt.close()


def create_sticky_bear_comparison():
    """
    Chart 3: Sticky BEAR Optimization Results
    Shows Q3 improvement from baseline to optimized
    """
    print("Creating sticky BEAR comparison chart...")

    configs = ['Baseline\n(3-day)', '5-day\nSticky', '7-day\nSticky']
    q3_returns = [-4.86, -0.51, 0.73]
    vs_ew = [-1.71, 2.64, 3.88]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Q3 Returns
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars1 = ax1.bar(configs, q3_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=-3.17, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Equal Weight (-3.17%)')

    for bar, value in zip(bars1, q3_returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom' if value > 0 else 'top', fontweight='bold', fontsize=12)

    ax1.set_ylabel('Q3 2025 Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Q3 Performance by Configuration', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(fontsize=10)

    # vs Equal Weight
    bars2 = ax2.bar(configs, vs_ew, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    for bar, value in zip(bars2, vs_ew):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:+.2f}pp',
                ha='center', va='bottom' if value > 0 else 'top', fontweight='bold', fontsize=12)

    ax2.set_ylabel('Outperformance vs Equal Weight (pp)', fontsize=12, fontweight='bold')
    ax2.set_title('Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Sticky BEAR Optimization Impact - Q3 2025', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "03_sticky_bear_optimization.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUALS_DIR / '03_sticky_bear_optimization.png'}")
    plt.close()


def create_risk_return_scatter():
    """
    Chart 4: Risk vs Return Scatter Plot
    All models positioned by risk and return
    """
    print("Creating risk-return scatter plot...")

    models = ['Ensemble', 'v1', 'v2.1', 'v5', 'v3', 'v4', 'v2', 'EW']
    returns = [12.92, 10.10, 9.87, 7.58, 7.62, 6.21, 6.17, 9.29]
    max_dd = [9.09, 9.33, 9.38, 13.33, 10.15, 9.87, 8.49, 7.20]
    sharpe = [0.952, 0.738, 0.748, 0.575, 0.588, 0.519, 0.504, 0.882]

    # Size by Sharpe ratio
    sizes = [s * 300 for s in sharpe]

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(max_dd, returns, s=sizes, alpha=0.6,
                        c=returns, cmap='RdYlGn', edgecolors='black', linewidth=2)

    # Add labels
    for i, model in enumerate(models):
        ax.annotate(model, (max_dd[i], returns[i]),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('YTD Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Risk vs Return - All Models (2025 YTD)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Return (%)', fontsize=11, fontweight='bold')

    # Add note
    ax.text(0.95, 0.05, 'Bubble size = Sharpe Ratio',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "04_risk_return_scatter.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUALS_DIR / '04_risk_return_scatter.png'}")
    plt.close()


def create_model_evolution_timeline():
    """
    Chart 5: Model Evolution Timeline
    Visual journey from v1 to Ensemble
    """
    print("Creating model evolution timeline...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Timeline data
    phases = ['v1\n(Oct 3)', 'v2\n(Oct 3)', 'v2.1\n(Oct 3)', 'v3\n(Oct 4)', 'v4\n(Oct 4)', 'v5\n(Oct 4)', 'Ensemble\n(Oct 4)']
    ytd_performance = [10.10, 6.17, 9.87, 7.62, 6.21, 7.58, 12.92]
    colors = ['#3498db', '#c0392b', '#9b59b6', '#e67e22', '#e74c3c', '#f39c12', '#2ecc71']
    status = ['✓ Deploy', '✓ Deploy', '⚠ Backup', '✗ Retired', '⚠ Archive', '⚠ Reserve', '✓ PRODUCTION']

    x_pos = range(len(phases))
    bars = ax.bar(x_pos, ytd_performance, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add performance labels
    for i, (bar, value, stat) in enumerate(zip(bars, ytd_performance, status)):
        height = bar.get_height()
        ax.text(i, height + 0.3, f'{value:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(i, -1.5, stat, ha='center', va='top', fontsize=9, style='italic')

    # Add Equal Weight benchmark
    ax.axhline(y=9.29, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Equal Weight (9.29%)')

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, fontsize=11)
    ax.set_ylabel('2025 YTD Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Evolution Timeline - The Journey to 12.92%', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(-2, 15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "05_model_evolution_timeline.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUALS_DIR / '05_model_evolution_timeline.png'}")
    plt.close()


def create_specialists_comparison():
    """
    Chart 6: v1 vs v2 - The Specialists
    Side-by-side comparison in different market conditions
    """
    print("Creating specialists comparison chart...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Q1 Bull Market
    models = ['v1\n(Momentum)', 'v2\n(Defensive)', 'v2.1\n(Balanced)', 'Equal\nWeight']
    q1_returns = [12.95, 3.21, 11.02, 5.76]
    colors_q1 = ['#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6']

    bars1 = ax1.bar(models, q1_returns, color=colors_q1, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, value in zip(bars1, q1_returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax1.set_ylabel('Q1 2025 Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Q1 Bull Market - v1 Dominates', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 15)
    ax1.grid(axis='y', alpha=0.3)

    # Q3 Bear Market
    q3_returns = [-6.81, 0.34, -6.31, -3.17]
    colors_q3 = ['#e74c3c', '#2ecc71', '#e67e22', '#95a5a6']

    bars2 = ax2.bar(models, q3_returns, color=colors_q3, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    for bar, value in zip(bars2, q3_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom' if value > 0 else 'top', fontweight='bold', fontsize=11)

    ax2.set_ylabel('Q3 2025 Return (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Q3 Correction - v2 Only Positive!', fontsize=14, fontweight='bold')
    ax2.set_ylim(-8, 2)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('The Specialists: No Single Model Wins Both Markets', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "06_specialists_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUALS_DIR / '06_specialists_comparison.png'}")
    plt.close()


def main():
    """Generate all charts for the blog series."""
    print("\n" + "="*60)
    print("BLOG VISUALIZATION GENERATOR")
    print("="*60 + "\n")

    print(f"Output directory: {VISUALS_DIR}\n")

    create_master_performance_chart()
    create_quarterly_heatmap()
    create_sticky_bear_comparison()
    create_risk_return_scatter()
    create_model_evolution_timeline()
    create_specialists_comparison()

    print("\n" + "="*60)
    print("✓ All charts generated successfully!")
    print(f"Location: {VISUALS_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
