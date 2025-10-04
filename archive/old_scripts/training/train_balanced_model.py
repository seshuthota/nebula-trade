#!/usr/bin/env python3
"""
Balanced Defensive Model Training Script (v2.1)

This script trains a TUNED defensive RL model with balanced risk/return trade-offs.
Reward function tuned to:
- Capture more upside in bull markets (vs v2 over-defensive)
- Still protect downside in bear markets (vs v1 momentum)

Key Changes from v2 (too defensive):
- Loss aversion: 2x → 1.5x
- Return weight: 40% → 60%
- Sharpe weight: 30% → 20%
- Drawdown weight: 20% → 10%

Target Performance:
- Bull market: 12-14% (vs 7% for v2, 16% for v1)
- Bear market: +1-3% (vs +0.3% for v2, -7% for v1)
- Overall: Beat Equal Weight by +1-2 pp
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


class BalancedModelTrainer:
    """Train balanced defensive model for all-weather performance."""

    def __init__(self, config_path: str = "config/portfolio.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create production directories (v2.1 for balanced)
        self.production_dir = self.project_root / "production"
        self.model_dir = self.production_dir / "models" / f"v2.1_balanced_{self.timestamp}"
        self.logs_dir = self.production_dir / "logs" / "training"

        for dir_path in [self.model_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("BALANCED DEFENSIVE MODEL TRAINING (v2.1)")
        logger.info("=" * 80)
        logger.info("Configuration: Balanced v2.1 (Phase 1 Extended + TUNED Defensive Rewards)")
        logger.info("  - Observation Space: 173 features")
        logger.info("  - Reward: TUNED balance of returns and risk")
        logger.info("  - Training: 1M steps, 90/10 split")
        logger.info(f"Model will be saved to: {self.model_dir}")
        logger.info(f"Logs will be saved to: {self.logs_dir}")
        logger.info("")
        logger.info("TUNED REWARD COMPONENTS (from v2 → v2.1):")
        logger.info("  • Loss Aversion: 2.0x → 1.5x (less defensive)")
        logger.info("  • Return Weight: 40% → 60% (focus more on gains)")
        logger.info("  • Sharpe Weight: 30% → 20% (moderate risk adjustment)")
        logger.info("  • Drawdown Penalty: 20% → 10% (less fearful)")
        logger.info("  • Volatility & Turnover: 5% each (unchanged)")
        logger.info("")
        logger.info("EXPECTED PERFORMANCE:")
        logger.info("  • Bull markets: 12-14% (vs 7% v2, 16% v1)")
        logger.info("  • Bear markets: +1-3% (vs +0.3% v2, -7% v1)")
        logger.info("  • Overall: Beat Equal Weight by +1-2 pp")

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
        train_data.to_csv(self.logs_dir / f"train_data_balanced_{self.timestamp}.csv")
        val_data.to_csv(self.logs_dir / f"val_data_balanced_{self.timestamp}.csv")

        return train_data, val_data

    def train_balanced_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                             total_timesteps: int = 1000000) -> PortfolioTrainerOptimized:
        """Train balanced model with tuned reward function."""
        logger.info("=" * 80)
        logger.info("STEP 2: Training Balanced Defensive Model")
        logger.info("=" * 80)
        logger.info("Configuration: Balanced v2.1 (173 features, tuned rewards)")
        logger.info("  - Network: [256, 256]")
        logger.info("  - Learning Rate: 1e-4")
        logger.info("  - Training Steps: 1,000,000")
        logger.info("  - VecNormalize: Enabled")
        logger.info("")

        # Save data for trainer
        train_path = self.logs_dir / f"train_data_balanced_{self.timestamp}_processed.csv"
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
        logger.info("  1. Maximize returns (60% weight - UP from 40%)")
        logger.info("  2. Avoid severe losses (1.5x penalty - DOWN from 2x)")
        logger.info("  3. Optimize risk-adjusted returns (20% Sharpe - DOWN from 30%)")
        logger.info("  4. Limit drawdowns moderately (10% weight - DOWN from 20%)")
        logger.info("  5. Maintain reasonable stability (5% each)")
        logger.info("")

        # Train
        trainer.train(
            total_timesteps=total_timesteps,
            log_interval=10,
            save_path=str(self.model_dir)
        )

        logger.info("=" * 80)
        logger.info("BALANCED MODEL TRAINING COMPLETED!")
        logger.info("=" * 80)

        return trainer

    def evaluate_balanced_model(self, trainer: PortfolioTrainerOptimized,
                                val_data: pd.DataFrame,
                                n_episodes: int = 10):
        """Evaluate balanced model on validation set."""
        logger.info("=" * 80)
        logger.info("STEP 3: Balanced Model Evaluation")
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
        logger.info("BALANCED MODEL VALIDATION RESULTS")
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
        logger.info(f"Balanced v2.1:       {mean_return:.2%} return")
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
            'model_type': 'production_v2.1_balanced',
            'configuration': 'balanced_v2.1',
            'training_timesteps': 1000000,
            'observation_features': 173,
            'balanced_components': {
                'asymmetric_loss_aversion': '1.5x penalty for losses (tuned from 2x)',
                'return_weight': '60% (tuned from 40%)',
                'sharpe_ratio_weight': '20% (tuned from 30%)',
                'drawdown_penalty': '10% weight (tuned from 20%)',
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

        # Decision criteria for BALANCED model
        logger.info("")
        logger.info("=" * 80)
        logger.info("GO/NO-GO DECISION FOR BALANCED MODEL")
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

        # Criterion 3: Target return (>10%)
        if mean_return > 0.10:
            logger.info(f"✅ Criterion 3: Target return met ({mean_return:.2%} > 10%)")
            go_criteria.append(True)
        else:
            logger.info(f"⚠️  Criterion 3: Below target ({mean_return:.2%} < 10%)")
            go_criteria.append(False)

        logger.info("")
        passed = sum(go_criteria)
        total = len(go_criteria)

        if passed == total:
            logger.info(f"🎯 DECISION: GO - All criteria met ({passed}/{total})")
            logger.info("   ✅ Ready for comprehensive paper trading")
        elif passed >= 2:
            logger.info(f"⚠️  DECISION: CONDITIONAL - Most criteria met ({passed}/{total})")
            logger.info("   → Proceed to paper trading for detailed evaluation")
        else:
            logger.info(f"🛑 DECISION: NO-GO - Insufficient performance ({passed}/{total})")
            logger.info("   → Consider further reward tuning or ensemble approach")

        logger.info("")
        logger.info("=" * 80)
        logger.info("NEXT STEPS")
        logger.info("=" * 80)
        logger.info("1. Test balanced model on paper trading periods:")
        logger.info("   - Bull market: Jan-Apr 2025 (target: 12-14%)")
        logger.info("   - Bear market: July-Sept 2025 (target: +1-3%)")
        logger.info("   - Mixed: Full Jan-Sept 2025 (target: beat Equal Weight)")
        logger.info("")
        logger.info("2. Compare all three models:")
        logger.info("   - v1 (momentum): Bull +16%, Bear -7%")
        logger.info("   - v2 (defensive): Bull +7%, Bear +0.3%")
        logger.info("   - v2.1 (balanced): Bull 12-14%?, Bear +1-3%?")
        logger.info("")
        logger.info("3. Make final recommendation based on all results")

        return report


def main():
    """Main training pipeline."""
    try:
        # Initialize trainer
        trainer_obj = BalancedModelTrainer()

        # Step 1: Prepare data
        train_data, val_data = trainer_obj.prepare_data()

        # Step 2: Train model
        trainer = trainer_obj.train_balanced_model(
            train_data=train_data,
            val_data=val_data,
            total_timesteps=1000000
        )

        # Step 3: Evaluate
        report = trainer_obj.evaluate_balanced_model(
            trainer=trainer,
            val_data=val_data,
            n_episodes=10
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ BALANCED MODEL TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {trainer_obj.model_dir}")
        logger.info(f"Validation report: {trainer_obj.model_dir}/validation_report.json")
        logger.info("")
        logger.info("Next: Run comprehensive paper trading tests")
        logger.info("  # Bull market")
        logger.info(f"  python production/paper_trading.py --model {trainer_obj.model_dir} --start 2025-01-01 --end 2025-04-30")
        logger.info("")
        logger.info("  # Bear market")
        logger.info(f"  python production/paper_trading.py --model {trainer_obj.model_dir} --start 2025-07-01 --end 2025-09-30")
        logger.info("")
        logger.info("  # Mixed period")
        logger.info(f"  python production/paper_trading.py --model {trainer_obj.model_dir} --start 2025-01-01 --end 2025-09-30")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
