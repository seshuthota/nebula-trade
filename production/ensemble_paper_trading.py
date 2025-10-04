#!/usr/bin/env python3
"""
Ensemble Paper Trading

Production paper trading using v1 + v2 ensemble with regime switching.
Based on production/paper_trading.py but with EnsembleManager integration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from astra.rl_framework.environment import PortfolioEnvironment
from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester
from astra.ensemble.ensemble_manager import EnsembleManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsemblePaperTrader:
    """Paper trading simulator using ensemble."""

    def __init__(
        self,
        v1_model_dir: str = "production/models/v1_20251003_131020",
        v2_model_dir: str = "production/models/v2_defensive_20251003_212109",
        data_path: str = "data/portfolio_data_processed.csv",
        config_path: str = "config/ensemble.yaml"
    ):
        self.project_root = Path(__file__).resolve().parent.parent
        self.v1_model_dir = v1_model_dir
        self.v2_model_dir = v2_model_dir
        self.data_path = self.project_root / data_path
        self.config_path = self.project_root / config_path

        # Load data
        self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        # Setup logging
        self.log_dir = self.project_root / "production" / "logs" / "ensemble_paper_trading"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"ensemble_paper_trading_{timestamp}"
        self.log_file = self.log_dir / f"{self.run_id}.csv"

        logger.info(f"Ensemble Paper Trader initialized")
        logger.info(f"v1 Model: {v1_model_dir}")
        logger.info(f"v2 Model: {v2_model_dir}")
        logger.info(f"Logs: {self.log_file}")

    def load_ensemble(self):
        """Load ensemble manager."""
        logger.info("Loading ensemble...")

        self.ensemble = EnsembleManager(
            v1_model_path=self.v1_model_dir,
            v2_model_path=self.v2_model_dir,
            config_path=str(self.config_path),
            primary_model='v1'
        )

        logger.info("✅ Ensemble loaded successfully")

    def setup_baselines(self, train_data: pd.DataFrame):
        """Setup classical portfolio baselines."""
        logger.info("Setting up classical baselines...")

        # Get assets from data columns
        assets = []
        for col in train_data.columns:
            if col.endswith('_returns'):
                asset = col.replace('_returns', '')
                assets.append(asset)

        # Train classical portfolios
        returns_cols = [f"{asset}_returns" for asset in assets]
        returns_data = train_data[returns_cols].dropna()

        optimizer = ClassicalPortfolioOptimizer(returns_data)
        self.classical_portfolios = optimizer.get_all_portfolios()

        logger.info(f"✅ Classical portfolios ready: {list(self.classical_portfolios.keys())}")

    def run_paper_trading(
        self,
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 100000
    ):
        """
        Run ensemble paper trading simulation.

        Args:
            start_date: Start date (default: most recent 30 days)
            end_date: End date (default: last available date)
            initial_capital: Starting capital
        """
        logger.info("="*80)
        logger.info("ENSEMBLE PAPER TRADING SIMULATION")
        logger.info("="*80)

        # Determine date range
        if end_date is None:
            end_date = self.data.index[-1]
        else:
            end_date = pd.to_datetime(end_date)
            if self.data.index.tz is not None:
                end_date = end_date.tz_localize(self.data.index.tz)

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        else:
            start_date = pd.to_datetime(start_date)
            if self.data.index.tz is not None:
                start_date = start_date.tz_localize(self.data.index.tz)

        # Get test data
        test_data = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]

        if len(test_data) < 30:
            raise ValueError(f"Insufficient data: {len(test_data)} days. Need at least 30 days.")

        logger.info(f"Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
        logger.info(f"Duration: {len(test_data)} trading days")
        logger.info(f"Initial Capital: ₹{initial_capital:,.0f}")
        logger.info("")

        # Setup baselines
        train_data = self.data[self.data.index < start_date]
        self.setup_baselines(train_data)

        # Load ensemble
        self.load_ensemble()

        # Run ensemble simulation
        logger.info("Running ensemble simulation...")
        ensemble_results = self._simulate_ensemble(test_data, initial_capital)

        # Run baseline simulations
        logger.info("Running baseline simulations...")
        baseline_results = self._simulate_baselines(test_data, initial_capital)

        # Compare and save results
        self._save_results(ensemble_results, baseline_results, test_data, initial_capital)

        # Print summary
        self._print_summary(ensemble_results, baseline_results)

    def _simulate_ensemble(self, data: pd.DataFrame, initial_capital: float) -> Dict:
        """Simulate ensemble trading."""
        env = PortfolioEnvironment(data, lookback_window=30)

        # Wrap with VecNormalize (use v1's normalizer as primary)
        vec_env = DummyVecEnv([lambda: env])
        v1_vec_norm_path = self.ensemble.vec_normalizers['v1']
        vec_env = VecNormalize.load(str(v1_vec_norm_path), vec_env)
        vec_env.training = False

        # Reset
        obs = vec_env.reset()
        self.ensemble.reset()
        done = False

        # Tracking
        daily_values = []
        daily_returns = []
        daily_weights = []
        daily_regimes = []
        daily_models = []
        portfolio_returns_history = []

        while not done:
            # Get action from ensemble
            action, ensemble_info = self.ensemble.get_action(
                observation=obs,
                portfolio_returns=portfolio_returns_history,
                vec_env=vec_env,
                deterministic=True
            )

            # Step
            obs, reward, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]

            # Track
            daily_values.append(info.get('portfolio_value', 0))
            daily_weights.append(info.get('weights', {}))
            daily_regimes.append(ensemble_info['regime'])
            daily_models.append(ensemble_info['active_model'])

            if len(daily_values) > 1:
                daily_return = (daily_values[-1] - daily_values[-2]) / daily_values[-2]
                daily_returns.append(daily_return)
                portfolio_returns_history.append(daily_return)

        final_value = daily_values[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # Sharpe
        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.array(daily_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Ensemble stats
        ensemble_stats = self.ensemble.get_statistics()

        return {
            'method': 'Ensemble (v1+v2)',
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'daily_values': daily_values,
            'daily_returns': daily_returns,
            'daily_weights': daily_weights,
            'daily_regimes': daily_regimes,
            'daily_models': daily_models,
            'ensemble_stats': ensemble_stats
        }

    def _simulate_baselines(self, data: pd.DataFrame, initial_capital: float) -> Dict:
        """Simulate classical portfolio methods."""
        backtester = PortfolioBacktester(data)
        results = {}

        for name, portfolio in self.classical_portfolios.items():
            backtest = backtester.backtest_portfolio(
                weights=portfolio['weights'],
                rebalance_freq='daily',
                transaction_cost=0.0003
            )

            results[name] = {
                'method': name.replace('_', ' ').title(),
                'final_value': backtest['final_value'],
                'total_return': backtest['total_return'],
                'sharpe_ratio': backtest['sharpe_ratio'],
                'max_drawdown': backtest.get('max_drawdown', 0.0),
                'daily_values': backtest.get('portfolio_values', [])
            }

        return results

    def _save_results(
        self,
        ensemble_results: Dict,
        baseline_results: Dict,
        data: pd.DataFrame,
        initial_capital: float
    ):
        """Save results to log file."""
        # Daily log
        dates = data.index.tolist()
        log_data = []

        for i, date in enumerate(dates):
            if i < len(ensemble_results['daily_values']):
                row = {
                    'date': date,
                    'ensemble_value': ensemble_results['daily_values'][i],
                    'ensemble_return': ensemble_results['daily_returns'][i] if i < len(ensemble_results['daily_returns']) else 0.0,
                    'regime': ensemble_results['daily_regimes'][i] if i < len(ensemble_results['daily_regimes']) else '',
                    'active_model': ensemble_results['daily_models'][i] if i < len(ensemble_results['daily_models']) else ''
                }

                # Add baseline values
                for name, result in baseline_results.items():
                    if i < len(result['daily_values']):
                        row[f'{name}_value'] = result['daily_values'][i]

                log_data.append(row)

        # Save daily log
        df = pd.DataFrame(log_data)
        df.to_csv(self.log_file, index=False)
        logger.info(f"✅ Daily log saved: {self.log_file}")

        # Save summary
        summary = {
            'run_id': self.run_id,
            'period': {
                'start': str(data.index[0].date()),
                'end': str(data.index[-1].date()),
                'days': len(data)
            },
            'initial_capital': initial_capital,
            'ensemble_results': {
                'final_value': float(ensemble_results['final_value']),
                'total_return': float(ensemble_results['total_return']),
                'sharpe_ratio': float(ensemble_results['sharpe_ratio']),
                'max_drawdown': float(ensemble_results['max_drawdown']),
                'stats': ensemble_results['ensemble_stats']
            },
            'baseline_results': {
                name: {
                    'final_value': float(result['final_value']),
                    'total_return': float(result['total_return']),
                    'sharpe_ratio': float(result['sharpe_ratio']),
                    'max_drawdown': float(result['max_drawdown'])
                } for name, result in baseline_results.items()
            }
        }

        summary_file = self.log_dir / f"{self.run_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Summary saved: {summary_file}")

    def _print_summary(self, ensemble_results: Dict, baseline_results: Dict):
        """Print trading summary."""
        logger.info("")
        logger.info("="*80)
        logger.info("ENSEMBLE PAPER TRADING RESULTS")
        logger.info("="*80)

        # Ensemble results
        logger.info(f"Ensemble (v1+v2):")
        logger.info(f"  Final Value:   ₹{ensemble_results['final_value']:,.0f}")
        logger.info(f"  Total Return:  {ensemble_results['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio:  {ensemble_results['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown:  {ensemble_results['max_drawdown']:.2%}")

        # Ensemble stats
        stats = ensemble_results['ensemble_stats']
        regime_stats = stats['regime_stats']
        model_usage = stats['model_usage']

        logger.info(f"  Regimes: Bull={regime_stats.get('bull_pct', 0):.1%}, "
                   f"Bear={regime_stats.get('bear_pct', 0):.1%}, "
                   f"Neutral={regime_stats.get('neutral_pct', 0):.1%}")
        logger.info(f"  Model Usage: v1={model_usage['v1']:.1%}, v2={model_usage['v2']:.1%}")
        logger.info(f"  Total Switches: {stats['total_switches']}")
        logger.info("")

        # Baselines
        logger.info("Classical Baselines:")
        for name, result in baseline_results.items():
            gap = (ensemble_results['total_return'] - result['total_return']) * 100
            status = "✅" if gap > 0 else "❌"
            logger.info(f"  {status} {result['method']:20s} "
                       f"Return: {result['total_return']:7.2%}  "
                       f"Sharpe: {result['sharpe_ratio']:6.3f}  "
                       f"Gap: {gap:+.2f} pp")

        logger.info("")
        logger.info("="*80)
        logger.info("DECISION CRITERIA:")
        logger.info("="*80)

        # Check criteria
        equal_weight = baseline_results.get('equal_weight', {})
        eq_gap = (ensemble_results['total_return'] - equal_weight.get('total_return', 0)) * 100

        criteria = []

        # Criterion 1: Beat Equal Weight
        if eq_gap > 0:
            logger.info(f"✅ Beat Equal Weight: +{eq_gap:.2f} pp")
            criteria.append(True)
        else:
            logger.info(f"❌ Below Equal Weight: {eq_gap:.2f} pp")
            criteria.append(False)

        # Criterion 2: Sharpe > 0.8
        if ensemble_results['sharpe_ratio'] > 0.8:
            logger.info(f"✅ Sharpe Ratio: {ensemble_results['sharpe_ratio']:.3f} > 0.8")
            criteria.append(True)
        else:
            logger.info(f"⚠️  Sharpe Ratio: {ensemble_results['sharpe_ratio']:.3f} < 0.8")
            criteria.append(False)

        # Criterion 3: Max Drawdown < 15%
        if abs(ensemble_results['max_drawdown']) < 0.15:
            logger.info(f"✅ Max Drawdown: {ensemble_results['max_drawdown']:.2%} < 15%")
            criteria.append(True)
        else:
            logger.info(f"⚠️  Max Drawdown: {ensemble_results['max_drawdown']:.2%} > 15%")
            criteria.append(False)

        logger.info("")
        logger.info("="*80)

        # Decision
        if all(criteria):
            logger.info("🎯 DECISION: GO - Ensemble ready for production")
        elif criteria.count(True) >= 2:
            logger.info("⚠️  DECISION: CONDITIONAL - Review results closely")
        else:
            logger.info("🛑 DECISION: NO-GO - More testing needed")

        logger.info("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble paper trading simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on last 30 days
  python production/ensemble_paper_trading.py

  # Run on specific period
  python production/ensemble_paper_trading.py --start 2025-01-01 --end 2025-09-26

  # Custom capital
  python production/ensemble_paper_trading.py --capital 500000
        """
    )

    parser.add_argument('--v1-model', type=str,
                       default='production/models/v1_20251003_131020',
                       help='Path to v1 model directory')
    parser.add_argument('--v2-model', type=str,
                       default='production/models/v2_defensive_20251003_212109',
                       help='Path to v2 model directory')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date (YYYY-MM-DD), default: last 30 days')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD), default: last available')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')

    args = parser.parse_args()

    trader = EnsemblePaperTrader(
        v1_model_dir=args.v1_model,
        v2_model_dir=args.v2_model
    )
    trader.run_paper_trading(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )


if __name__ == "__main__":
    main()
