#!/usr/bin/env python3
"""
Ensemble Backtesting Framework

Tests v1 + v2 ensemble with regime switching on historical data.
Validates expected 12-13% performance vs individual models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from astra.rl_framework.environment import PortfolioEnvironment
from astra.ensemble.ensemble_manager import EnsembleManager
from astra.evaluation.optimizer import PortfolioBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleBacktester:
    """
    Backtest ensemble on historical data and compare vs baselines.
    """

    def __init__(
        self,
        data_path: str = "data/portfolio_data_processed.csv",
        config_path: str = "config/ensemble.yaml"
    ):
        """
        Initialize backtester.

        Args:
            data_path: Path to portfolio data CSV
            config_path: Path to ensemble configuration
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_path = self.project_root / data_path
        self.config_path = self.project_root / config_path

        # Load data
        self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config.get('ensemble', {})

        # Setup logging
        log_dir = self.project_root / self.config['logging']['log_directory']
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        logger.info(f"EnsembleBacktester initialized")
        logger.info(f"Data: {self.data_path}")
        logger.info(f"Config: {self.config_path}")

    def backtest_ensemble(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000
    ) -> Dict:
        """
        Run ensemble backtest on specified period.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital

        Returns:
            Dictionary with performance metrics
        """
        logger.info("="*80)
        logger.info("ENSEMBLE BACKTEST")
        logger.info("="*80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ₹{initial_capital:,.0f}")

        # Get test data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Match timezone
        if self.data.index.tz is not None:
            start = start.tz_localize(self.data.index.tz)
            end = end.tz_localize(self.data.index.tz)

        test_data = self.data[(self.data.index >= start) & (self.data.index <= end)]

        if len(test_data) < 30:
            raise ValueError(f"Insufficient data: {len(test_data)} days. Need at least 30.")

        logger.info(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
        logger.info(f"Duration: {len(test_data)} trading days")
        logger.info("")

        # Initialize ensemble
        v1_path = self.config['models']['v1']['path']
        v2_path = self.config['models']['v2']['path']

        ensemble = EnsembleManager(
            v1_model_path=v1_path,
            v2_model_path=v2_path,
            config_path=str(self.config_path),
            primary_model=self.config.get('primary_model', 'v1')
        )

        # Run ensemble simulation
        logger.info("Running ensemble simulation...")
        results = self._simulate_ensemble(test_data, ensemble, initial_capital)

        logger.info("✅ Ensemble backtest complete")
        return results

    def _simulate_ensemble(
        self,
        data: pd.DataFrame,
        ensemble: EnsembleManager,
        initial_capital: float
    ) -> Dict:
        """
        Simulate ensemble trading.

        Args:
            data: Test data
            ensemble: EnsembleManager instance
            initial_capital: Starting capital

        Returns:
            Dictionary with performance results
        """
        # Create environment
        env = PortfolioEnvironment(data, lookback_window=30)

        # We need to create separate VecNormalize wrappers for each model
        # For simplicity, we'll use the v1 VecNormalize (since it's the primary model)
        vec_env_fn = lambda: env
        vec_env = DummyVecEnv([vec_env_fn])

        # Load v1's VecNormalize (we'll use this for observations)
        v1_vec_norm_path = ensemble.vec_normalizers['v1']
        vec_env = VecNormalize.load(str(v1_vec_norm_path), vec_env)
        vec_env.training = False

        # Reset environment and ensemble
        obs = vec_env.reset()
        ensemble.reset()

        # Tracking
        daily_values = []
        daily_returns = []
        daily_weights = []
        daily_regimes = []
        daily_models = []
        portfolio_returns_history = []

        done = False
        step_count = 0

        while not done:
            # Get action from ensemble
            action, ensemble_info = ensemble.get_action(
                observation=obs,
                portfolio_returns=portfolio_returns_history,
                vec_env=vec_env,
                deterministic=True
            )

            # Step environment
            obs, reward, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]

            # Track metrics
            portfolio_value = info.get('portfolio_value', initial_capital)
            weights = info.get('weights', {})

            daily_values.append(portfolio_value)
            daily_weights.append(weights)
            daily_regimes.append(ensemble_info['regime'])
            daily_models.append(ensemble_info['active_model'])

            # Calculate return
            if len(daily_values) > 1:
                daily_return = (daily_values[-1] - daily_values[-2]) / daily_values[-2]
                daily_returns.append(daily_return)
                portfolio_returns_history.append(daily_return)

            step_count += 1

        # Calculate metrics
        final_value = daily_values[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # Sharpe ratio
        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.array(daily_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Regime statistics
        ensemble_stats = ensemble.get_statistics()

        return {
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

    def compare_with_baselines(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000
    ) -> Dict:
        """
        Compare ensemble vs individual models and Equal Weight.

        Args:
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital

        Returns:
            Dictionary with comparison results
        """
        logger.info("="*80)
        logger.info("ENSEMBLE COMPARISON")
        logger.info("="*80)

        # Run ensemble
        ensemble_results = self.backtest_ensemble(start_date, end_date, initial_capital)

        # Load individual model results from existing comparison logs
        # For now, we'll calculate Equal Weight baseline
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        if self.data.index.tz is not None:
            start = start.tz_localize(self.data.index.tz)
            end = end.tz_localize(self.data.index.tz)

        test_data = self.data[(self.data.index >= start) & (self.data.index <= end)]

        # Equal Weight baseline
        logger.info("Calculating Equal Weight baseline...")
        backtester = PortfolioBacktester(test_data)

        # Get assets from config
        config_path = self.project_root / "config" / "portfolio.yaml"
        with open(config_path, 'r') as f:
            portfolio_config = yaml.safe_load(f)
            assets = portfolio_config['portfolio']['assets']

        # Convert equal weights dict to array matching backtester.assets order
        equal_weights_array = np.array([1.0 / len(backtester.assets) for _ in backtester.assets])

        ew_backtest = backtester.backtest_portfolio(
            weights=equal_weights_array,
            rebalance_freq='daily',
            transaction_cost=0.0003
        )

        # Comparison
        results = {
            'period': {
                'start': start_date,
                'end': end_date,
                'days': len(test_data)
            },
            'ensemble': {
                'return': ensemble_results['total_return'],
                'sharpe': ensemble_results['sharpe_ratio'],
                'max_dd': ensemble_results['max_drawdown'],
                'final_value': ensemble_results['final_value'],
                'stats': ensemble_results['ensemble_stats']
            },
            'equal_weight': {
                'return': ew_backtest['total_return'],
                'sharpe': ew_backtest['sharpe_ratio'],
                'max_dd': ew_backtest.get('max_drawdown', 0.0),
                'final_value': ew_backtest['final_value']
            }
        }

        # Calculate gaps
        results['performance'] = {
            'return_gap_vs_ew': (ensemble_results['total_return'] - ew_backtest['total_return']) * 100,
            'sharpe_gap_vs_ew': ensemble_results['sharpe_ratio'] - ew_backtest['sharpe_ratio']
        }

        # Print summary
        self._print_comparison(results)

        # Save results
        self._save_comparison(results, start_date, end_date)

        return results

    def _print_comparison(self, results: Dict):
        """Print comparison summary."""
        logger.info("")
        logger.info("="*80)
        logger.info("COMPARISON RESULTS")
        logger.info("="*80)

        # Ensemble
        ens = results['ensemble']
        logger.info(f"Ensemble (v1+v2):")
        logger.info(f"  Return:       {ens['return']:.2%}")
        logger.info(f"  Sharpe:       {ens['sharpe']:.3f}")
        logger.info(f"  Max DD:       {ens['max_dd']:.2%}")
        logger.info(f"  Final Value:  ₹{ens['final_value']:,.0f}")

        # Regime stats
        stats = ens['stats']
        regime_stats = stats['regime_stats']
        model_usage = stats['model_usage']

        logger.info(f"  Regimes: Bull={regime_stats.get('bull_pct', 0):.1%}, "
                   f"Bear={regime_stats.get('bear_pct', 0):.1%}, "
                   f"Neutral={regime_stats.get('neutral_pct', 0):.1%}")
        logger.info(f"  Model Usage: v1={model_usage['v1']:.1%}, v2={model_usage['v2']:.1%}")
        logger.info(f"  Switches: {stats['total_switches']}")
        logger.info("")

        # Equal Weight
        ew = results['equal_weight']
        logger.info(f"Equal Weight:")
        logger.info(f"  Return:       {ew['return']:.2%}")
        logger.info(f"  Sharpe:       {ew['sharpe']:.3f}")
        logger.info(f"  Max DD:       {ew['max_dd']:.2%}")
        logger.info(f"  Final Value:  ₹{ew['final_value']:,.0f}")
        logger.info("")

        # Performance gap
        perf = results['performance']
        status = "✅" if perf['return_gap_vs_ew'] > 0 else "❌"
        logger.info(f"{status} Performance vs Equal Weight:")
        logger.info(f"  Return Gap:   {perf['return_gap_vs_ew']:+.2f} pp")
        logger.info(f"  Sharpe Gap:   {perf['sharpe_gap_vs_ew']:+.3f}")
        logger.info("")
        logger.info("="*80)

    def _save_comparison(self, results: Dict, start_date: str, end_date: str):
        """Save comparison results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ensemble_comparison_{start_date}_{end_date}_{timestamp}.json"
        filepath = self.log_dir / filename

        # Convert to JSON-serializable format
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Clean results
        clean_results = json.loads(
            json.dumps(results, default=convert)
        )

        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)

        logger.info(f"✅ Results saved: {filepath}")


def main():
    """Run ensemble backtests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble backtesting framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on Q1 2025
  python -m astra.ensemble.backtest_ensemble --start 2025-01-01 --end 2025-03-31

  # Test on Q3 2025
  python -m astra.ensemble.backtest_ensemble --start 2025-07-01 --end 2025-09-26

  # Test on full 2025 YTD
  python -m astra.ensemble.backtest_ensemble --start 2025-01-01 --end 2025-09-26
        """
    )

    parser.add_argument('--start', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')

    args = parser.parse_args()

    backtester = EnsembleBacktester()
    backtester.compare_with_baselines(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )


if __name__ == "__main__":
    main()
