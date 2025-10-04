#!/usr/bin/env python3
"""
Compare Multiple Models on Paper Trading
Tests v1, v2, v2.1, v3, v4 on the same period for fair comparison.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from astra.rl_framework.environment import PortfolioEnvironment
from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare multiple trained models on same test period."""

    def __init__(self, data_path: str = "data/portfolio_data_processed.csv"):
        self.project_root = Path(__file__).resolve().parent
        self.data_path = self.project_root / data_path
        self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        # Define models to compare
        self.models = {
            'v1': 'production/models/v1_20251003_131020',
            'v2': 'production/models/v2_defensive_20251003_212109',
            'v2.1': 'production/models/v2.1_balanced_20251003_233318',
            'v3': 'production/models/v3_stage2_balanced_20251004_102729',
            'v4': 'production/models/v4_historical_2007_2024',
            'v5': 'production/models/v5_tuned_historical_2007_2024'
        }

        logger.info("Model Comparison Framework Initialized")
        logger.info(f"Data range: {self.data.index[0].date()} to {self.data.index[-1].date()}")

    def test_model(self, model_name: str, model_dir: str, test_data: pd.DataFrame,
                   initial_capital: float = 100000) -> Dict:
        """Test a single model on test data."""
        try:
            model_path = self.project_root / model_dir
            logger.info(f"Testing {model_name}...")

            # Load model
            env = PortfolioEnvironment(test_data, lookback_window=30)
            vec_env = DummyVecEnv([lambda: env])

            vec_norm_path = model_path / "vec_normalize.pkl"
            if vec_norm_path.exists():
                vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
                vec_env.training = False

            model = SAC.load(str(model_path / "final_model.zip"), env=vec_env)

            # Run simulation
            obs = vec_env.reset()
            done = False
            daily_values = []
            daily_returns = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                done = dones[0]
                info = infos[0]

                daily_values.append(info.get('portfolio_value', 0))
                if len(daily_values) > 1:
                    daily_return = (daily_values[-1] - daily_values[-2]) / daily_values[-2]
                    daily_returns.append(daily_return)

            # Calculate metrics
            final_value = daily_values[-1]
            total_return = (final_value - initial_capital) / initial_capital

            if len(daily_returns) > 1:
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0.0

            cumulative = np.array(daily_values)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            logger.info(f"  ✅ {model_name}: {total_return:.2%} return, Sharpe {sharpe:.3f}")

            return {
                'model': model_name,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'daily_values': daily_values,
                'daily_returns': daily_returns,
                'success': True
            }

        except Exception as e:
            logger.error(f"  ❌ {model_name} failed: {str(e)}")
            return {
                'model': model_name,
                'success': False,
                'error': str(e)
            }

    def run_comparison(self, start_date: str, end_date: str, initial_capital: float = 100000):
        """Run comparison across all models."""
        logger.info("="*80)
        logger.info("MODEL COMPARISON - PAPER TRADING")
        logger.info("="*80)

        # Get test data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if self.data.index.tz is not None:
            start = start.tz_localize(self.data.index.tz)
            end = end.tz_localize(self.data.index.tz)

        test_data = self.data[(self.data.index >= start) & (self.data.index <= end)]

        logger.info(f"Test Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
        logger.info(f"Duration: {len(test_data)} trading days")
        logger.info(f"Initial Capital: ₹{initial_capital:,.0f}")
        logger.info("")

        # Get classical baselines
        train_data = self.data[self.data.index < start]
        logger.info("Setting up classical baselines...")

        assets = []
        for col in train_data.columns:
            if col.endswith('_returns'):
                assets.append(col.replace('_returns', ''))

        returns_cols = [f"{asset}_returns" for asset in assets]
        returns_data = train_data[returns_cols].dropna()
        optimizer = ClassicalPortfolioOptimizer(returns_data)
        portfolios = optimizer.get_all_portfolios()

        backtester = PortfolioBacktester(test_data)
        baseline_results = {}

        for name, portfolio in portfolios.items():
            backtest = backtester.backtest_portfolio(
                weights=portfolio['weights'],
                rebalance_freq='daily',
                transaction_cost=0.001
            )
            baseline_results[name] = {
                'model': name,
                'final_value': backtest['final_value'],
                'total_return': backtest['total_return'],
                'sharpe_ratio': backtest['sharpe_ratio'],
                'max_drawdown': backtest.get('max_drawdown', 0.0)
            }

        logger.info("✅ Classical baselines ready")
        logger.info("")

        # Test all RL models
        logger.info("Testing RL models...")
        rl_results = {}
        for model_name, model_dir in self.models.items():
            result = self.test_model(model_name, model_dir, test_data, initial_capital)
            if result['success']:
                rl_results[model_name] = result

        logger.info("")

        # Print comparison table
        self._print_comparison(rl_results, baseline_results)

        # Save results
        self._save_comparison(rl_results, baseline_results, test_data, start_date, end_date)

    def _print_comparison(self, rl_results: Dict, baseline_results: Dict):
        """Print comparison table."""
        logger.info("="*80)
        logger.info("RESULTS COMPARISON")
        logger.info("="*80)
        logger.info("")
        logger.info(f"{'Model':<20} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'vs EW':>10}")
        logger.info("-"*80)

        # Sort by return
        eq_weight_return = baseline_results.get('equal_weight', {}).get('total_return', 0)

        all_results = []

        # RL models
        for name, result in rl_results.items():
            gap = (result['total_return'] - eq_weight_return) * 100
            all_results.append({
                'name': name,
                'return': result['total_return'],
                'sharpe': result['sharpe_ratio'],
                'max_dd': result['max_drawdown'],
                'gap': gap,
                'type': 'RL'
            })

        # Classical models
        for name, result in baseline_results.items():
            gap = (result['total_return'] - eq_weight_return) * 100
            all_results.append({
                'name': name,
                'return': result['total_return'],
                'sharpe': result['sharpe_ratio'],
                'max_dd': result['max_drawdown'],
                'gap': gap,
                'type': 'Classical'
            })

        # Sort by return
        all_results.sort(key=lambda x: x['return'], reverse=True)

        for r in all_results:
            marker = "🥇" if r == all_results[0] else "🥈" if r == all_results[1] else "🥉" if r == all_results[2] else "  "
            type_tag = f"[{r['type']}]"
            logger.info(f"{marker} {r['name']:<15} {type_tag:<12} "
                       f"{r['return']:>9.2%} {r['sharpe']:>10.3f} "
                       f"{r['max_dd']:>10.2%} {r['gap']:>9.2f} pp")

        logger.info("")
        logger.info("="*80)
        logger.info("KEY INSIGHTS")
        logger.info("="*80)

        # Best RL model
        best_rl = max(rl_results.values(), key=lambda x: x['total_return'])
        logger.info(f"🏆 Best RL Model: {best_rl['model']} ({best_rl['total_return']:.2%})")

        # Best overall
        best_overall = all_results[0]
        logger.info(f"🏆 Best Overall: {best_overall['name']} ({best_overall['return']:.2%})")

        # RL vs Classical
        avg_rl_return = np.mean([r['total_return'] for r in rl_results.values()])
        logger.info(f"📊 Average RL Return: {avg_rl_return:.2%}")
        logger.info(f"📊 Equal Weight Return: {eq_weight_return:.2%}")
        logger.info(f"📊 Gap: {(avg_rl_return - eq_weight_return)*100:+.2f} pp")

        logger.info("")

    def _save_comparison(self, rl_results: Dict, baseline_results: Dict,
                        test_data: pd.DataFrame, start_date: str, end_date: str):
        """Save comparison results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {
            'timestamp': timestamp,
            'test_period': {
                'start': start_date,
                'end': end_date,
                'days': len(test_data)
            },
            'rl_models': {
                name: {
                    'return': float(result['total_return']),
                    'sharpe': float(result['sharpe_ratio']),
                    'max_drawdown': float(result['max_drawdown']),
                    'final_value': float(result['final_value'])
                } for name, result in rl_results.items()
            },
            'classical_baselines': {
                name: {
                    'return': float(result['total_return']),
                    'sharpe': float(result['sharpe_ratio']),
                    'max_drawdown': float(result['max_drawdown']),
                    'final_value': float(result['final_value'])
                } for name, result in baseline_results.items()
            }
        }

        output_dir = self.project_root / "production" / "logs" / "model_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"comparison_{start_date}_{end_date}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✅ Comparison saved: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare all models on paper trading")
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')

    args = parser.parse_args()

    comparison = ModelComparison()
    comparison.run_comparison(args.start, args.end, args.capital)


if __name__ == "__main__":
    main()
