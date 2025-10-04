#!/usr/bin/env python3
"""
Simple Paper Trading Framework

Tests production model on live/recent data WITHOUT real money.
Tracks daily performance vs baselines.
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

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from astra.rl_framework.environment import PortfolioEnvironment
from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperTrader:
    """Simple paper trading simulator."""

    def __init__(self, model_dir: str, data_path: str = "data/portfolio_data_processed.csv"):
        self.project_root = Path(__file__).resolve().parent.parent
        self.model_dir = Path(model_dir)
        self.data_path = self.project_root / data_path

        # Load data
        self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        # Setup logging
        self.log_dir = self.project_root / "production" / "logs" / "paper_trading"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"paper_trading_{timestamp}"
        self.log_file = self.log_dir / f"{self.run_id}.csv"

        logger.info(f"Paper Trader initialized")
        logger.info(f"Model: {model_dir}")
        logger.info(f"Logs: {self.log_file}")

    def load_production_model(self):
        """Load production model with VecNormalize."""
        logger.info("Loading production model...")

        # Create dummy environment for loading
        dummy_data = self.data.iloc[-100:]  # Just for structure
        dummy_env_fn = lambda: PortfolioEnvironment(dummy_data, lookback_window=30)
        dummy_vec_env = DummyVecEnv([dummy_env_fn])

        # Load VecNormalize
        vec_norm_path = self.model_dir / "vec_normalize.pkl"
        if not vec_norm_path.exists():
            raise FileNotFoundError(f"VecNormalize not found: {vec_norm_path}")

        self.vec_env = VecNormalize.load(str(vec_norm_path), dummy_vec_env)
        self.vec_env.training = False  # Evaluation mode

        # Load model
        model_path = self.model_dir / "final_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = SAC.load(str(model_path), env=self.vec_env)
        logger.info("✅ Model loaded successfully")

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

    def run_paper_trading(self, start_date: str = None, end_date: str = None,
                         initial_capital: float = 100000):
        """
        Run paper trading simulation.

        Args:
            start_date: Start date (default: most recent 30 days)
            end_date: End date (default: last available date)
            initial_capital: Starting capital
        """
        logger.info("="*80)
        logger.info("PAPER TRADING SIMULATION")
        logger.info("="*80)

        # Determine date range
        if end_date is None:
            end_date = self.data.index[-1]
        else:
            end_date = pd.to_datetime(end_date)
            # Match timezone with data
            if self.data.index.tz is not None:
                end_date = end_date.tz_localize(self.data.index.tz)

        if start_date is None:
            # Use last 30 days by default
            start_date = end_date - timedelta(days=30)
        else:
            start_date = pd.to_datetime(start_date)
            # Match timezone with data
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

        # Setup baselines (train on data before test period)
        train_data = self.data[self.data.index < start_date]
        self.setup_baselines(train_data)

        # Load model
        self.load_production_model()

        # Run RL simulation
        logger.info("Running RL simulation...")
        rl_results = self._simulate_rl(test_data, initial_capital)

        # Run baseline simulations
        logger.info("Running baseline simulations...")
        baseline_results = self._simulate_baselines(test_data, initial_capital)

        # Compare and save results
        self._save_results(rl_results, baseline_results, test_data, initial_capital)

        # Print summary
        self._print_summary(rl_results, baseline_results)

    def _simulate_rl(self, data: pd.DataFrame, initial_capital: float) -> Dict:
        """Simulate RL agent trading."""
        env = PortfolioEnvironment(data, lookback_window=30)

        # Wrap with VecNormalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(
            str(self.model_dir / "vec_normalize.pkl"),
            vec_env
        )
        vec_env.training = False

        # Run episode
        obs = vec_env.reset()
        done = False

        daily_values = []
        daily_returns = []
        daily_weights = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]

            daily_values.append(info.get('portfolio_value', 0))
            daily_weights.append(info.get('weights', {}))

            if len(daily_values) > 1:
                daily_return = (daily_values[-1] - daily_values[-2]) / daily_values[-2]
                daily_returns.append(daily_return)

        final_value = daily_values[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # Calculate Sharpe
        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Calculate max drawdown
        cumulative = np.array(daily_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        return {
            'method': 'RL (SAC)',
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'daily_values': daily_values,
            'daily_returns': daily_returns,
            'daily_weights': daily_weights
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

    def _save_results(self, rl_results: Dict, baseline_results: Dict,
                     data: pd.DataFrame, initial_capital: float):
        """Save results to log file."""
        # Prepare daily log
        dates = data.index.tolist()

        log_data = []
        for i, date in enumerate(dates):
            if i < len(rl_results['daily_values']):
                row = {
                    'date': date,
                    'rl_value': rl_results['daily_values'][i],
                    'rl_return': rl_results['daily_returns'][i] if i < len(rl_results['daily_returns']) else 0.0,
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
            'rl_results': {
                'final_value': float(rl_results['final_value']),
                'total_return': float(rl_results['total_return']),
                'sharpe_ratio': float(rl_results['sharpe_ratio']),
                'max_drawdown': float(rl_results['max_drawdown'])
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

    def _print_summary(self, rl_results: Dict, baseline_results: Dict):
        """Print trading summary."""
        logger.info("")
        logger.info("="*80)
        logger.info("PAPER TRADING RESULTS")
        logger.info("="*80)

        # RL results
        logger.info(f"RL (SAC):")
        logger.info(f"  Final Value:   ₹{rl_results['final_value']:,.0f}")
        logger.info(f"  Total Return:  {rl_results['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio:  {rl_results['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown:  {rl_results['max_drawdown']:.2%}")
        logger.info("")

        # Baselines
        logger.info("Classical Baselines:")
        for name, result in baseline_results.items():
            gap = (rl_results['total_return'] - result['total_return']) * 100
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
        eq_gap = (rl_results['total_return'] - equal_weight.get('total_return', 0)) * 100

        criteria = []

        # Criterion 1: Beat Equal Weight
        if eq_gap > 0:
            logger.info(f"✅ Beat Equal Weight: +{eq_gap:.2f} pp")
            criteria.append(True)
        else:
            logger.info(f"❌ Below Equal Weight: {eq_gap:.2f} pp")
            criteria.append(False)

        # Criterion 2: Sharpe > 0.8
        if rl_results['sharpe_ratio'] > 0.8:
            logger.info(f"✅ Sharpe Ratio: {rl_results['sharpe_ratio']:.3f} > 0.8")
            criteria.append(True)
        else:
            logger.info(f"⚠️  Sharpe Ratio: {rl_results['sharpe_ratio']:.3f} < 0.8")
            criteria.append(False)

        # Criterion 3: Max Drawdown < 15%
        if abs(rl_results['max_drawdown']) < 0.15:
            logger.info(f"✅ Max Drawdown: {rl_results['max_drawdown']:.2%} < 15%")
            criteria.append(True)
        else:
            logger.info(f"⚠️  Max Drawdown: {rl_results['max_drawdown']:.2%} > 15%")
            criteria.append(False)

        logger.info("")
        logger.info("="*80)

        # Decision
        if all(criteria):
            logger.info("🎯 DECISION: GO - Ready for live trading")
        elif criteria.count(True) >= 2:
            logger.info("⚠️  DECISION: CONDITIONAL - Review results closely")
        else:
            logger.info("🛑 DECISION: NO-GO - More paper trading needed")

        logger.info("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Paper trading simulator for production model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on last 30 days
  python production/paper_trading.py

  # Run on specific period
  python production/paper_trading.py --start 2025-08-01 --end 2025-09-30

  # Custom capital
  python production/paper_trading.py --capital 500000

Note: Uses most recent production model automatically
        """
    )

    parser.add_argument('--model', type=str,
                       default='production/models/v1_20251003_131020',
                       help='Path to model directory')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date (YYYY-MM-DD), default: last 30 days')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD), default: last available')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')

    args = parser.parse_args()

    trader = PaperTrader(model_dir=args.model)
    trader.run_paper_trading(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )


if __name__ == "__main__":
    main()
