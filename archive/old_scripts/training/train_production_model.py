#!/usr/bin/env python3
"""
Production Model Training Script for Live Deployment.

This script trains the final production model using all available data (90/10 split).
Uses Phase 1 Extended configuration (173 features) that achieved 29.44% test return.

Key Differences from Research Training:
- 90% training, 10% validation (vs 70/15/15 research split)
- Uses most recent data for validation (current market conditions)
- Saves to production/ directory
- Optimized for deployment, not experimentation
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


class ProductionModelTrainer:
    """Train production model for live deployment."""

    def __init__(self, config_path: str = "config/portfolio.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create production directories
        self.production_dir = self.project_root / "production"
        self.model_dir = self.production_dir / "models" / f"v1_{self.timestamp}"
        self.logs_dir = self.production_dir / "logs" / "training"

        for dir_path in [self.model_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("PRODUCTION MODEL TRAINING")
        logger.info("=" * 80)
        logger.info(f"Configuration: Phase 1 Extended (173 features, 1M steps)")
        logger.info(f"Model will be saved to: {self.model_dir}")
        logger.info(f"Logs will be saved to: {self.logs_dir}")

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
        logger.info("      This is different from research split where test set was chronologically last")

        # Save splits
        train_data.to_csv(self.logs_dir / "train_data.csv")
        val_data.to_csv(self.logs_dir / "val_data.csv")

        return train_data, val_data

    def train_production_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                               total_timesteps: int = 1000000) -> PortfolioTrainerOptimized:
        """Train production model with Phase 1 Extended configuration."""
        logger.info("=" * 80)
        logger.info("STEP 2: Training Production Model")
        logger.info("=" * 80)
        logger.info("Configuration: Phase 1 Extended (PROVEN BEST)")
        logger.info("  - Observation Space: 173 features")
        logger.info("  - Network: [256, 256]")
        logger.info("  - Learning Rate: 1e-4")
        logger.info("  - Training Steps: 1,000,000")
        logger.info("  - VecNormalize: Enabled")
        logger.info("")

        # Save data for trainer
        train_path = self.logs_dir / "train_data_processed.csv"
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

        # Train
        trainer.train(
            total_timesteps=total_timesteps,
            log_interval=10,
            save_path=str(self.model_dir)
        )

        logger.info("=" * 80)
        logger.info("PRODUCTION MODEL TRAINING COMPLETED!")
        logger.info("=" * 80)

        return trainer

    def evaluate_production_model(self, trainer: PortfolioTrainerOptimized,
                                  val_data: pd.DataFrame,
                                  n_episodes: int = 10):
        """Evaluate production model on validation set."""
        logger.info("=" * 80)
        logger.info("STEP 3: Production Model Evaluation")
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
        logger.info("PRODUCTION MODEL VALIDATION RESULTS")
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
                       f"Final Value: ₹{backtest_result['final_value']:,.0f}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("VALIDATION COMPARISON")
        logger.info("=" * 80)
        logger.info(f"Production Model:    {mean_return:.2%} return, ₹{results['mean_final_value']:,.0f}")
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
            'model_type': 'production_v1',
            'configuration': 'phase_1_extended',
            'training_timesteps': 1000000,
            'observation_features': 173,
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

        # Decision criteria
        logger.info("")
        logger.info("=" * 80)
        logger.info("GO/NO-GO DECISION FOR PAPER TRADING")
        logger.info("=" * 80)

        go_criteria = []

        # Criterion 1: Beat Equal Weight
        if mean_return > classical_results['equal_weight']['total_return']:
            logger.info("✅ Criterion 1: Beats Equal Weight baseline")
            go_criteria.append(True)
        else:
            logger.info("❌ Criterion 1: Below Equal Weight baseline")
            go_criteria.append(False)

        # Criterion 2: Return > 20% (reasonable for recent data)
        if mean_return > 0.20:
            logger.info("✅ Criterion 2: Return > 20%")
            go_criteria.append(True)
        else:
            logger.info(f"⚠️  Criterion 2: Return {mean_return:.2%} < 20% target")
            go_criteria.append(False)

        # Criterion 3: Std return < 2% (deterministic)
        if std_return < 0.02:
            logger.info("✅ Criterion 3: Deterministic behavior (std < 2%)")
            go_criteria.append(True)
        else:
            logger.info(f"⚠️  Criterion 3: High variance (std = {std_return:.2%})")
            go_criteria.append(False)

        logger.info("")
        if all(go_criteria):
            logger.info("🎯 DECISION: GO - Proceed to paper trading")
            logger.info("   Model is ready for paper trading phase")
        elif go_criteria.count(True) >= 2:
            logger.info("⚠️  DECISION: CONDITIONAL GO - Review results")
            logger.info("   Model meets most criteria, manual review recommended")
        else:
            logger.info("🛑 DECISION: NO-GO - Retrain or adjust")
            logger.info("   Model does not meet success criteria")

        logger.info("=" * 80)

        return results

    def save_production_artifacts(self):
        """Save final production artifacts with proper naming."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: Saving Production Artifacts")
        logger.info("=" * 80)

        # Create metadata
        metadata = {
            'version': 'v1',
            'timestamp': self.timestamp,
            'configuration': 'phase_1_extended',
            'observation_features': 173,
            'network_architecture': [256, 256],
            'training_timesteps': 1000000,
            'assets': self.assets,
            'trained_on': {
                'start_date': None,  # Will be filled from data
                'end_date': None,
                'n_samples': None
            },
            'status': 'production_ready',
            'deployment_status': 'pending_paper_trading'
        }

        with open(self.model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Production model saved to: {self.model_dir}/final_model.zip")
        logger.info(f"✅ VecNormalize stats saved to: {self.model_dir}/vec_normalize.pkl")
        logger.info(f"✅ Metadata saved to: {self.model_dir}/metadata.json")
        logger.info(f"✅ Validation report saved to: {self.model_dir}/validation_report.json")
        logger.info("")
        logger.info("=" * 80)
        logger.info("PRODUCTION MODEL READY")
        logger.info("=" * 80)
        logger.info("Next Steps:")
        logger.info("1. Review validation results above")
        logger.info("2. If approved, proceed to paper trading:")
        logger.info("   python production/paper_trading.py --duration 30")
        logger.info("=" * 80)

    def run_production_training(self, timesteps: int = 1000000):
        """Run complete production model training pipeline."""
        # 1. Prepare data (90/10 split)
        train_data, val_data = self.prepare_data()

        # 2. Train production model
        trainer = self.train_production_model(train_data, val_data, total_timesteps=timesteps)

        # 3. Evaluate on validation set
        self.evaluate_production_model(trainer, val_data)

        # 4. Save production artifacts
        self.save_production_artifacts()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train production model for live deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train production model (default 1M steps)
  python train_production_model.py

  # Train with custom timesteps
  python train_production_model.py --timesteps 500000

  # Use custom config
  python train_production_model.py --config config/production.yaml

Note: This script uses 90/10 train/val split with most recent data for validation.
      This is different from research training which used 70/15/15 split.
        """
    )

    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Number of training timesteps (default: 1000000)')
    parser.add_argument('--config', type=str, default='config/portfolio.yaml',
                       help='Path to config file (default: config/portfolio.yaml)')

    args = parser.parse_args()

    trainer = ProductionModelTrainer(config_path=args.config)
    trainer.run_production_training(timesteps=args.timesteps)


if __name__ == "__main__":
    main()
