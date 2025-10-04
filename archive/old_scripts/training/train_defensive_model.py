#!/usr/bin/env python3
"""
Defensive Production Model Training Script.

This script trains a defensive RL model designed to perform well in BOTH bull and bear markets.
Uses modified reward function with:
- Asymmetric loss aversion (losses hurt 2x more)
- Drawdown penalties
- Higher Sharpe ratio weight
- Volatility penalties

Based on Phase 1 Extended config (173 features) but with defensive reward engineering.
"""

import os
import json
import pandas as pd
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

from astra.rl_framework.trainer_optimized import PortfolioTrainerOptimized
from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester
from astra.data_pipeline.data_manager import PortfolioDataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefensiveModelTrainer:
    """Train defensive production model for all-weather performance."""

    def __init__(self, config_path: str = "config/portfolio.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create production directories (v2 for defensive)
        self.production_dir = self.project_root / "production"
        self.model_dir = self.production_dir / "models" / f"v2_defensive_{self.timestamp}"
        self.logs_dir = self.production_dir / "logs" / "training"

        for dir_path in [self.model_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("DEFENSIVE PRODUCTION MODEL TRAINING")
        logger.info("=" * 80)
        logger.info("Configuration: Defensive v1 (Phase 1 Extended + Defensive Rewards)")
        logger.info("  - Observation Space: 173 features")
        logger.info("  - Reward: Asymmetric loss aversion, drawdown penalties")
        logger.info("  - Training: 1M steps, 90/10 split")
        logger.info(f"Model will be saved to: {self.model_dir}")
        logger.info(f"Logs will be saved to: {self.logs_dir}")
        logger.info("")
        logger.info("DEFENSIVE REWARD COMPONENTS:")
        logger.info("  • Asymmetric Loss Aversion: Losses hurt 2x more than gains")
        logger.info("  • Drawdown Penalty: Heavy penalty for falling below peak")
        logger.info("  • Sharpe Ratio: 30% weight (vs 1% in original)")
        logger.info("  • Volatility Penalty: Discourage unstable portfolios")
        logger.info("  • Turnover Penalty: Reduced overtrading")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into train/validation sets (90/10 split)."""
        logger.info("=" * 80)
        logger.info("STEP 1: Data Preparation (PRODUCTION SPLIT)")
        logger.info("=" * 80)

        data_manager = PortfolioDataManager(config_path=str(self.config_path))
        data, _ = data_manager.process_and_initialize()

        # Production Split: 90% train, 10% validation (most recent data)
        n = len(data)
        train_end = int(n * 0.90)

        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:]

        logger.info(f"Total samples: {n}")
        logger.info(f"Train (90%): {len(train_data)} samples ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        logger.info(f"Validation (10%): {len(val_data)} samples ({val_data.index[0].date()} to {val_data.index[-1].date()})")
        logger.info("")
        logger.info("NOTE: Validation set uses MOST RECENT data for current market conditions")

        # Save splits
        train_data.to_csv(self.logs_dir / f"train_data_defensive_{self.timestamp}.csv")
        val_data.to_csv(self.logs_dir / f"val_data_defensive_{self.timestamp}.csv")

        return train_data, val_data

    def train_defensive_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                              total_timesteps: int = 1000000) -> PortfolioTrainerOptimized:
        """Train defensive model with modified reward function."""
        logger.info("=" * 80)
        logger.info("STEP 2: Training Defensive Model")
        logger.info("=" * 80)
        logger.info("Configuration: Defensive v1 (173 features, defensive rewards)")
        logger.info("  - Network: [256, 256]")
        logger.info("  - Learning Rate: 1e-4")
        logger.info("  - Training Steps: 1,000,000")
        logger.info("  - VecNormalize: Enabled")
        logger.info("")

        # Save data for trainer
        train_path = self.logs_dir / f"train_data_defensive_{self.timestamp}_processed.csv"
        train_data.to_csv(train_path)

        # Initialize optimized trainer
        logger.info("Initializing PortfolioTrainerOptimized...")
        trainer = PortfolioTrainerOptimized(
            config_path=str(self.config_path),
            data_path=str(train_path),
            n_envs=None,  # Auto-detect
            use_gpu=True
        )

        # Override train/test split since we're managing it externally
        trainer.train_data = train_data
        trainer.test_data = val_data

        logger.info(f"Training for {total_timesteps:,} timesteps...")
        logger.info("This will take approximately 2-3 hours depending on hardware")
        logger.info("")
        logger.info("The model is learning to:")
        logger.info("  1. Avoid losses (2x penalty)")
        logger.info("  2. Minimize drawdowns (heavy penalty)")
        logger.info("  3. Optimize risk-adjusted returns (30% Sharpe weight)")
        logger.info("  4. Maintain stable portfolios (volatility penalty)")
        logger.info("")

        # Train
        trainer.train(
            total_timesteps=total_timesteps,
            log_interval=10,
            save_path=str(self.model_dir)
        )

        logger.info("=" * 80)
        logger.info("DEFENSIVE MODEL TRAINING COMPLETED!")
        logger.info("=" * 80)

        return trainer

    def evaluate_defensive_model(self, trainer: PortfolioTrainerOptimized,
                                 val_data: pd.DataFrame,
                                 n_episodes: int = 10):
        """Evaluate defensive model on validation set."""
        logger.info("=" * 80)
        logger.info("STEP 3: Defensive Model Evaluation")
        logger.info("=" * 80)

        # Evaluate on validation set (most recent data)
        logger.info(f"Evaluating on validation set ({len(val_data)} samples, {n_episodes} episodes)...")
        results = trainer.evaluate(val_data, n_episodes=n_episodes)

        # Calculate return manually
        initial_capital = 100000
        mean_return = (results['mean_final_value'] - initial_capital) / initial_capital
        std_return = results['std_final_value'] / initial_capital

        logger.info("")
        logger.info("=" * 80)
        logger.info("DEFENSIVE MODEL VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Mean Return:      {mean_return:.2%}")
        logger.info(f"Std Return:       {std_return:.2%}")
        logger.info(f"Mean Final Value: ₹{results['mean_final_value']:,.0f}")
        logger.info(f"Mean Reward:      {results['mean_reward']:.4f}")
        logger.info("")

        # Compare with baselines
        logger.info("Backtesting classical methods on validation set...")
        backtester = PortfolioBacktester(val_data)

        # Get classical portfolios (trained on train data)
        returns_cols = [f"{asset}_returns" for asset in self.assets]
        returns_data = trainer.train_data[returns_cols].dropna()
        optimizer = ClassicalPortfolioOptimizer(returns_data)
        portfolios = optimizer.get_all_portfolios()

        classical_results = {}
        for name, portfolio in portfolios.items():
            backtest_result = backtester.backtest_portfolio(
                weights=portfolio['weights'],
                rebalance_freq='daily',
                transaction_cost=self.config['rebalance']['transaction_cost']
            )
            classical_results[name] = backtest_result
            logger.info(f"  {name:20s} | Return: {backtest_result['total_return']:7.2%} | "
                       f"Sharpe: {backtest_result['sharpe_ratio']:6.3f}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("VALIDATION COMPARISON")
        logger.info("=" * 80)
        logger.info(f"Defensive Model:     {mean_return:.2%} return")
        logger.info(f"Equal Weight:        {classical_results['equal_weight']['total_return']:.2%} return")
        logger.info(f"Min Volatility:      {classical_results['min_volatility']['total_return']:.2%} return")
        logger.info(f"Max Sharpe:          {classical_results['max_sharpe']['total_return']:.2%} return")

        # Calculate gaps
        eq_gap = (mean_return - classical_results['equal_weight']['total_return']) * 100
        logger.info("")
        if eq_gap > 0:
            logger.info(f"✅ BEATS Equal Weight by {eq_gap:+.2f} pp")
        else:
            logger.info(f"❌ BELOW Equal Weight by {abs(eq_gap):.2f} pp")

        # Save results
        report = {
            'timestamp': self.timestamp,
            'model_type': 'production_v2_defensive',
            'configuration': 'defensive_v1',
            'training_timesteps': 1000000,
            'observation_features': 173,
            'defensive_components': {
                'asymmetric_loss_aversion': '2x penalty for losses',
                'drawdown_penalty': '20% weight',
                'sharpe_ratio_weight': '30% (vs 1% original)',
                'volatility_penalty': '5% weight',
                'turnover_penalty': '5% weight'
            },
            'validation_results': {
                'mean_return': float(mean_return),
                'std_return': float(std_return),
                'mean_final_value': float(results['mean_final_value']),
                'mean_reward': float(results['mean_reward']),
                'n_episodes': n_episodes
            },
            'classical_baselines': {
                name: {
                    'total_return': float(result['total_return']),
                    'final_value': float(result['final_value']),
                    'sharpe_ratio': float(result['sharpe_ratio'])
                } for name, result in classical_results.items()
            },
            'validation_period': {
                'start': str(val_data.index[0].date()),
                'end': str(val_data.index[-1].date()),
                'n_samples': len(val_data)
            }
        }

        with open(self.model_dir / "validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("")
        logger.info(f"Validation report saved to: {self.model_dir}/validation_report.json")

        # Decision criteria for DEFENSIVE model
        logger.info("")
        logger.info("=" * 80)
        logger.info("GO/NO-GO DECISION FOR DEFENSIVE MODEL")
        logger.info("=" * 80)

        go_criteria = []

        # Criterion 1: Beat Equal Weight
        if mean_return > classical_results['equal_weight']['total_return']:
            logger.info("✅ Criterion 1: Beats Equal Weight baseline")
            go_criteria.append(True)
        else:
            logger.info(f"❌ Criterion 1: Below Equal Weight by {abs(eq_gap):.2f} pp")
            go_criteria.append(False)

        # Criterion 2: Deterministic (low variance)
        if std_return < 0.02:  # Less than 2% variance
            logger.info(f"✅ Criterion 2: Deterministic behavior (std={std_return:.2%} < 2%)")
            go_criteria.append(True)
        else:
            logger.info(f"⚠️  Criterion 2: High variance (std={std_return:.2%} >= 2%)")
            go_criteria.append(False)

        logger.info("")
        passed = sum(go_criteria)
        total = len(go_criteria)

        if passed == total:
            logger.info(f"🎯 DECISION: GO - All criteria met ({passed}/{total})")
            logger.info("   ✅ Ready for paper trading on multiple market periods")
        elif passed >= 1:
            logger.info(f"⚠️  DECISION: CONDITIONAL - Some criteria met ({passed}/{total})")
            logger.info("   → Proceed to paper trading with caution")
        else:
            logger.info(f"🛑 DECISION: NO-GO - Insufficient performance ({passed}/{total})")
            logger.info("   → Consider Stage 2: Training data rebalancing")

        logger.info("")
        logger.info("=" * 80)
        logger.info("NEXT STEPS")
        logger.info("=" * 80)
        logger.info("1. Test defensive model on paper trading periods:")
        logger.info("   - Bull market: Jan-Apr 2025")
        logger.info("   - Bear market: July-Sept 2025")
        logger.info("   - Mixed: Full Jan-Sept 2025")
        logger.info("")
        logger.info("2. Compare with original v1 model performance")
        logger.info("")
        logger.info("3. Expected improvement: Better bear market performance")

        return report


def main():
    """Main training pipeline."""
    try:
        # Initialize trainer
        trainer_obj = DefensiveModelTrainer()

        # Step 1: Prepare data
        train_data, val_data = trainer_obj.prepare_data()

        # Step 2: Train model
        trainer = trainer_obj.train_defensive_model(
            train_data=train_data,
            val_data=val_data,
            total_timesteps=1000000
        )

        # Step 3: Evaluate
        report = trainer_obj.evaluate_defensive_model(
            trainer=trainer,
            val_data=val_data,
            n_episodes=10
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ DEFENSIVE MODEL TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {trainer_obj.model_dir}")
        logger.info(f"Validation report: {trainer_obj.model_dir}/validation_report.json")
        logger.info("")
        logger.info("Next: Run paper trading tests")
        logger.info("  python production/paper_trading.py --model v2_defensive_TIMESTAMP --start 2025-01-01 --end 2025-09-30")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
