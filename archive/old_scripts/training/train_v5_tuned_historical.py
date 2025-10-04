#!/usr/bin/env python3
"""
v5 Training: Tuned Historical 2007-2024 with Optimized 70/30 Balance

This script trains the v5 model - a refined version of v4 with better bull/bear balance.

Problem with v4:
- 50/50 bull/bear split (12x weighting) made model TOO defensive
- 2025 YTD: 6.21% (worst performer, -3.08 pp vs EW)
- Q1 2025 bull: 1.19% (worst, -4.57 pp vs EW)
- Over-trained on GFC/COVID extreme scenarios

v5 Solution:
- 70/30 bull/bear split (5.0x weighting) - more realistic
- Still 4x more bear exposure than v1/v3 (30% vs 7.9%)
- Retains GFC/COVID lessons without being dominated by them
- Spends more time learning normal bull markets

Key Configuration:
- 2007-2024 data (4,623 samples, 12 bear periods)
- 329 bear samples (7.9%) × 5.0x = 30% effective bear
- 3,831 bull samples (92.1%) × 1.0x = 70% effective bull
- v2.1 balanced defensive reward function
- 1M training steps

Expected Performance:
- Q1 2025 bull: 10-12% (vs v4's 1.19%) - huge improvement
- Q3 2025 correction: -2% to +1% (vs v4's -3.73%) - maintains protection
- 2025 YTD: 10-12% (vs v4's 6.21%) - best single model
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from astra.rl_framework.trainer_optimized import PortfolioTrainerOptimized
from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester
from astra.data_pipeline.data_manager import PortfolioDataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V5TunedHistoricalTrainer:
    """Train v5 model with 2007-2024 historical data and optimized 70/30 balance."""

    def __init__(self, config_path: str = "config/portfolio.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create production directories for v5
        self.production_dir = self.project_root / "production"
        self.model_dir = self.production_dir / "models" / f"v5_tuned_historical_2007_2024"
        self.logs_dir = self.production_dir / "logs" / "training"

        for dir_path in [self.model_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("v5 TRAINING: TUNED HISTORICAL 2007-2024 WITH 70/30 BALANCE")
        logger.info("=" * 80)
        logger.info("Configuration: v5 (2007-2024 Historical + 5.0x Weighted Sampling)")
        logger.info(f"  - Data Period: 2007-2024 (4,623 samples)")
        logger.info(f"  - Bear Periods: 12 (including 2008 GFC, COVID, corrections)")
        logger.info(f"  - Bear Samples: 329 (7.9% of data)")
        logger.info(f"  - Balance: 70/30 bull/bear with 5.0x weighting")
        logger.info(f"  - Observation Space: 173 features")
        logger.info(f"  - Reward: v2.1 balanced defensive function")
        logger.info(f"  - Training: 1M steps")
        logger.info(f"Model will be saved to: {self.model_dir}")
        logger.info(f"Logs will be saved to: {self.logs_dir}")
        logger.info("")
        logger.info("PROBLEM WITH v4 (50/50 balance):")
        logger.info("  ✗ Too defensive - over-trained on GFC/COVID")
        logger.info("  ✗ 2025 YTD: 6.21% (worst performer)")
        logger.info("  ✗ Q1 2025 bull: 1.19% (worst, -4.57 pp vs EW)")
        logger.info("  ✗ Sacrificed gains for crash protection not needed")
        logger.info("")
        logger.info("v5 SOLUTION (70/30 balance):")
        logger.info("  ✓ More realistic balance: 70% bull / 30% bear")
        logger.info("  ✓ Still 4x more bear exposure than v1/v3")
        logger.info("  ✓ Learns from GFC/COVID without being dominated")
        logger.info("  ✓ More training on normal bull markets")
        logger.info("  ✓ Expected: Best single all-weather model")

    def load_balanced_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Load pre-balanced 2007-2024 data with v5 sample weights."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 1: Loading Pre-Balanced 2007-2024 Data (v5 Weights)")
        logger.info("=" * 80)

        # Load full 2007-2024 data
        data_path = self.project_root / "notebooks" / "data" / "portfolio_data_2007_2024.csv"
        logger.info(f"Loading data from: {data_path}")

        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(data)} samples")
        logger.info(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

        # Load v5 sample weights (5.0x for 70/30 balance)
        weights_path = self.project_root / "notebooks" / "data" / "sample_weights_v5.npy"
        logger.info(f"Loading v5 sample weights from: {weights_path}")

        sample_weights = np.load(weights_path)
        logger.info(f"Loaded {len(sample_weights)} sample weights")

        # Validation split (last 10% chronologically)
        split_idx = int(len(data) * 0.9)
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
        train_weights = sample_weights[:split_idx]

        logger.info("")
        logger.info("Data Split:")
        logger.info(f"  Training: {len(train_data)} samples ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        logger.info(f"  Validation: {len(val_data)} samples ({val_data.index[0].date()} to {val_data.index[-1].date()})")
        logger.info("")
        logger.info("v5 Sample Weights Statistics (5.0x):")
        logger.info(f"  Min weight: {train_weights.min():.1f}x")
        logger.info(f"  Max weight: {train_weights.max():.1f}x")
        logger.info(f"  Mean weight: {train_weights.mean():.2f}x")
        logger.info(f"  Bear samples (5.0x): {(train_weights == 5.0).sum()} ({(train_weights == 5.0).sum() / len(train_weights):.1%})")
        logger.info(f"  Bull samples (1.0x): {(train_weights == 1.0).sum()} ({(train_weights == 1.0).sum() / len(train_weights):.1%})")

        # Calculate effective distribution
        bear_eff = (train_weights == 5.0).sum() * 5.0
        bull_eff = (train_weights == 1.0).sum() * 1.0
        total_eff = bear_eff + bull_eff
        logger.info(f"  Effective balance: {bear_eff/total_eff:.1%} bear / {bull_eff/total_eff:.1%} bull ✅")
        logger.info("")
        logger.info("v5 vs v4 Comparison:")
        logger.info(f"  v4 (12.0x): 50.8% bear / 49.2% bull - TOO DEFENSIVE")
        logger.info(f"  v5 ( 5.0x): {bear_eff/total_eff:.1%} bear / {bull_eff/total_eff:.1%} bull - BALANCED ✅")
        logger.info("")

        # Save training artifacts
        train_data.to_csv(self.logs_dir / f"train_data_v5_{self.timestamp}.csv")
        val_data.to_csv(self.logs_dir / f"val_data_v5_{self.timestamp}.csv")
        np.save(self.logs_dir / f"sample_weights_v5_{self.timestamp}.npy", train_weights)

        logger.info("✅ Balanced data loaded successfully!")
        logger.info("")

        return train_data, train_weights, val_data

    def train_v5_model(
        self,
        train_data: pd.DataFrame,
        sample_weights: np.ndarray,
        val_data: pd.DataFrame,
        total_timesteps: int = 1000000
    ) -> PortfolioTrainerOptimized:
        """Train v5 model with 2007-2024 data and 70/30 weighted sampling."""
        logger.info("=" * 80)
        logger.info("STEP 2: Training v5 Model with Tuned 70/30 Balance")
        logger.info("=" * 80)
        logger.info("Configuration: v5 (173 features, 5.0x weighted sampling, v2.1 rewards)")
        logger.info("  - Network: [256, 256]")
        logger.info("  - Learning Rate: 1e-4")
        logger.info("  - Training Steps: 1,000,000")
        logger.info("  - VecNormalize: Enabled")
        logger.info("  - Weighted Sampling: YES (70/30 bull/bear with 5.0x weight)")
        logger.info("")

        # Save training data for trainer
        train_path = self.logs_dir / f"train_data_v5_{self.timestamp}_processed.csv"
        train_data.to_csv(train_path)

        # Save sample weights
        weights_path = self.logs_dir / f"sample_weights_v5_{self.timestamp}.npy"
        np.save(weights_path, sample_weights)

        # Initialize optimized trainer
        logger.info("Initializing PortfolioTrainerOptimized with v5 weighted sampling...")
        trainer = PortfolioTrainerOptimized(
            config_path=str(self.config_path),
            data_path=str(train_path),
            n_envs=None,  # Auto-detect
            use_gpu=True
        )

        # Override train/test split
        trainer.train_data = train_data
        trainer.test_data = val_data
        trainer.sample_weights = sample_weights

        logger.info(f"Training for {total_timesteps:,} timesteps...")
        logger.info("This will take approximately 2-3 hours depending on hardware")
        logger.info("")
        logger.info("v5 Learning Strategy (70/30 balance):")
        logger.info("  • 70% training on NORMAL bull markets → capture upside")
        logger.info("  • 30% training on BEAR markets (GFC, COVID, corrections) → downside protection")
        logger.info("  • Result: All-weather model that isn't over-defensive")
        logger.info("")
        logger.info("Expected Improvements over v4:")
        logger.info("  ✓ Q1 2025 bull: 10-12% (vs v4's 1.19%) - HUGE gain")
        logger.info("  ✓ Q3 2025 correction: -2 to +1% (vs v4's -3.73%) - maintains protection")
        logger.info("  ✓ 2025 YTD: 10-12% (vs v4's 6.21%) - best single model")
        logger.info("")

        # Train
        trainer.train(
            total_timesteps=total_timesteps,
            log_interval=10,
            save_path=str(self.model_dir)
        )

        logger.info("=" * 80)
        logger.info("v5 MODEL TRAINING COMPLETED!")
        logger.info("=" * 80)

        return trainer

    def evaluate_v5_model(
        self,
        trainer: PortfolioTrainerOptimized,
        val_data: pd.DataFrame,
        n_episodes: int = 10
    ):
        """Evaluate v5 model on chronological validation set."""
        logger.info("=" * 80)
        logger.info("STEP 3: v5 Model Evaluation")
        logger.info("=" * 80)

        # Evaluate on validation set (chronological, NO weighting)
        logger.info(f"Evaluating on validation set ({len(val_data)} samples, {n_episodes} episodes)...")
        logger.info("IMPORTANT: Validation uses chronological data with NO weighting")
        logger.info("           This ensures fair, unbiased performance measurement")
        logger.info("")

        results = trainer.evaluate(val_data, n_episodes=n_episodes)

        # Calculate return
        initial_capital = 100000
        mean_return = (results['mean_final_value'] - initial_capital) / initial_capital
        std_return = results['std_final_value'] / initial_capital

        logger.info("")
        logger.info("=" * 80)
        logger.info("v5 MODEL VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Mean Return:      {mean_return:.2%}")
        logger.info(f"Std Return:       {std_return:.2%}")
        logger.info(f"Mean Final Value: ₹{results['mean_final_value']:,.0f}")
        logger.info(f"Mean Reward:      {results['mean_reward']:.4f}")
        logger.info("")

        # Save results
        report = {
            'timestamp': self.timestamp,
            'model_type': 'production_v5_tuned_historical_2007_2024',
            'configuration': 'v5_tuned_70_30',
            'training_timesteps': 1000000,
            'observation_features': 173,
            'v5_strategy': {
                'description': '70/30 bull/bear balance - optimized from v4',
                'data_samples': 4623,
                'data_period': '2007-2024',
                'bear_periods': 12,
                'bear_samples': 329,
                'bear_ratio': '7.9%',
                'bear_weight': 5.0,
                'bull_weight': 1.0,
                'effective_balance': '30% bear / 70% bull',
                'reward_function': 'v2.1 balanced defensive',
                'improvement_over_v4': 'Reduced from 50/50 to 70/30 for better bull performance'
            },
            'validation_results': {
                'mean_return': float(mean_return),
                'std_return': float(std_return),
                'mean_final_value': float(results['mean_final_value']),
                'mean_reward': float(results['mean_reward']),
                'n_episodes': n_episodes
            },
            'validation_period': {
                'start': str(val_data.index[0].date()),
                'end': str(val_data.index[-1].date()),
                'n_samples': len(val_data)
            }
        }

        with open(self.model_dir / "validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to: {self.model_dir}/validation_report.json")

        # Success criteria
        logger.info("")
        logger.info("=" * 80)
        logger.info("v5 SUCCESS CRITERIA EVALUATION")
        logger.info("=" * 80)

        go_criteria = []

        # Criterion 1: Target return
        if mean_return > 0.10:
            logger.info(f"✅ Criterion 1: Target return met ({mean_return:.2%} > 10%)")
            go_criteria.append(True)
        else:
            logger.info(f"⚠️  Criterion 1: Below target ({mean_return:.2%} < 10%)")
            go_criteria.append(False)

        # Criterion 2: Deterministic
        if std_return < 0.02:
            logger.info(f"✅ Criterion 2: Deterministic behavior (std={std_return:.2%} < 2%)")
            go_criteria.append(True)
        else:
            logger.info(f"⚠️  Criterion 2: High variance (std={std_return:.2%} >= 2%)")
            go_criteria.append(False)

        # Criterion 3: Beat v4
        v4_val_return = 0.3650  # From v4 results
        if mean_return > v4_val_return:
            logger.info(f"✅ Criterion 3: Beats v4 ({mean_return:.2%} > {v4_val_return:.2%})")
            go_criteria.append(True)
        else:
            logger.info(f"❌ Criterion 3: Below v4 ({mean_return:.2%} < {v4_val_return:.2%})")
            go_criteria.append(False)

        logger.info("")
        passed = sum(go_criteria)
        total = len(go_criteria)

        if passed == total:
            logger.info(f"🎯 DECISION: GO - All criteria met ({passed}/{total})")
            logger.info("   ✅ Ready for comprehensive paper trading!")
        elif passed >= 2:
            logger.info(f"⚠️  DECISION: CONDITIONAL - Most criteria met ({passed}/{total})")
            logger.info("   → Proceed to paper trading for detailed evaluation")
        else:
            logger.info(f"🛑 DECISION: NO-GO - Insufficient performance ({passed}/{total})")

        logger.info("")
        logger.info("=" * 80)
        logger.info("NEXT STEPS")
        logger.info("=" * 80)
        logger.info("1. Paper trading tests on 2025:")
        logger.info("   - Full YTD (Jan-Sep 2025)")
        logger.info("   - Q1 2025 (bull market)")
        logger.info("   - Q3 2025 (correction)")
        logger.info("")
        logger.info("2. Compare v5 vs all models (v1, v2, v2.1, v3, v4):")
        logger.info("   - Expected: v5 ranks #1 or #2 in 2025 YTD")
        logger.info("   - Key test: Q1 bull performance (v4 failed)")
        logger.info("")
        logger.info("3. Final production decision")

        return report


def main():
    """Main v5 training pipeline."""
    try:
        # Initialize trainer
        trainer_obj = V5TunedHistoricalTrainer()

        # Step 1: Load pre-balanced data with v5 weights
        train_data, sample_weights, val_data = trainer_obj.load_balanced_data()

        # Step 2: Train model
        trainer = trainer_obj.train_v5_model(
            train_data=train_data,
            sample_weights=sample_weights,
            val_data=val_data,
            total_timesteps=1000000
        )

        # Step 3: Evaluate
        report = trainer_obj.evaluate_v5_model(
            trainer=trainer,
            val_data=val_data,
            n_episodes=10
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ v5 TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {trainer_obj.model_dir}")
        logger.info(f"Validation report: {trainer_obj.model_dir}/validation_report.json")
        logger.info("")
        logger.info("v5 Strategy: 70/30 bull/bear balance (tuned from v4's 50/50)")
        logger.info("  • Retains GFC/COVID lessons without over-defensive bias")
        logger.info("  • More realistic balance for normal market conditions")
        logger.info("  • Expected: Best single all-weather model")
        logger.info("")
        logger.info("Next: Run comprehensive paper trading and compare with all models")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
