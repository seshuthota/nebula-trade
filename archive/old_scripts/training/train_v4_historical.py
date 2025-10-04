#!/usr/bin/env python3
"""
v4 Training: Historical 2007-2024 with Balanced Bear Market Coverage

This script trains the v4 model using expanded 2007-2024 data with proper bear market representation.

Key Improvements from v3:
- 2007-2024 data (4,623 samples vs 2,654 for 2015-2024) - 75% more data!
- 12 bear periods (vs 3 in 2015-2024) - 4x more bear coverage!
- 329 bear samples (7.9%) vs 64 samples (2.7%) - 5x more bear examples!
- Includes 2008 GFC (-38% drawdown), COVID crash, multiple corrections
- 12x weight (vs 35x) for more natural balance
- v2.1 balanced reward function (proven defensive)

Expected Performance:
- Better generalization from diverse bear markets (fast crashes, slow grinds, corrections)
- More robust defensive behavior learned from 2008 GFC, COVID, etc.
- Target: 10-12% overall with <5% bear market drawdowns
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


class V4HistoricalTrainer:
    """Train v4 model with 2007-2024 historical data and balanced bear coverage."""

    def __init__(self, config_path: str = "config/portfolio.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create production directories for v4
        self.production_dir = self.project_root / "production"
        self.model_dir = self.production_dir / "models" / f"v4_historical_2007_2024"
        self.logs_dir = self.production_dir / "logs" / "training"

        for dir_path in [self.model_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("v4 TRAINING: HISTORICAL 2007-2024 WITH BALANCED BEAR COVERAGE")
        logger.info("=" * 80)
        logger.info("Configuration: v4 (2007-2024 Historical + 12x Weighted Sampling)")
        logger.info(f"  - Data Period: 2007-2024 (4,623 samples)")
        logger.info(f"  - Bear Periods: 12 (including 2008 GFC, COVID, corrections)")
        logger.info(f"  - Bear Samples: 329 (7.9% of data)")
        logger.info(f"  - Balance: 50/50 bull/bear with 12x weighting")
        logger.info(f"  - Observation Space: 173 features")
        logger.info(f"  - Reward: v2.1 balanced defensive function")
        logger.info(f"  - Training: 1M steps")
        logger.info(f"Model will be saved to: {self.model_dir}")
        logger.info(f"Logs will be saved to: {self.logs_dir}")
        logger.info("")
        logger.info("MAJOR BEAR MARKETS INCLUDED:")
        logger.info("  • 2008 GFC: -38% drawdown (Lehman collapse)")
        logger.info("  • 2008 Subprime: -32% drawdown")
        logger.info("  • 2020 COVID: -43% drawdown")
        logger.info("  • 2015 China correction: -18%")
        logger.info("  • Multiple 2010-2013 corrections")
        logger.info("")
        logger.info("v4 ADVANTAGES:")
        logger.info("  ✓ 75% more training data vs v3")
        logger.info("  ✓ 4x more bear periods (12 vs 3)")
        logger.info("  ✓ 5x more bear samples (329 vs 64)")
        logger.info("  ✓ Includes severe crashes (GFC, COVID)")
        logger.info("  ✓ More natural 12x weight (vs 35x for limited data)")
        logger.info("  ✓ Diverse bear types: crashes, grinds, corrections, recoveries")

    def load_balanced_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Load pre-balanced 2007-2024 data with sample weights."""
        logger.info("=" * 80)
        logger.info("STEP 1: Loading Pre-Balanced 2007-2024 Data")
        logger.info("=" * 80)

        # Load full 2007-2024 data
        data_path = self.project_root / "notebooks" / "data" / "portfolio_data_2007_2024.csv"
        logger.info(f"Loading data from: {data_path}")

        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(data)} samples")
        logger.info(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

        # Load pre-computed sample weights
        weights_path = self.project_root / "notebooks" / "data" / "sample_weights_v4.npy"
        logger.info(f"Loading sample weights from: {weights_path}")

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
        logger.info("Sample Weights Statistics:")
        logger.info(f"  Min weight: {train_weights.min():.1f}x")
        logger.info(f"  Max weight: {train_weights.max():.1f}x")
        logger.info(f"  Mean weight: {train_weights.mean():.2f}x")
        logger.info(f"  Bear samples (12x): {(train_weights == 12.0).sum()} ({(train_weights == 12.0).sum() / len(train_weights):.1%})")
        logger.info(f"  Bull samples (1x): {(train_weights == 1.0).sum()} ({(train_weights == 1.0).sum() / len(train_weights):.1%})")

        # Calculate effective distribution
        bear_eff = (train_weights == 12.0).sum() * 12.0
        bull_eff = (train_weights == 1.0).sum() * 1.0
        total_eff = bear_eff + bull_eff
        logger.info(f"  Effective balance: {bear_eff/total_eff:.1%} bear / {bull_eff/total_eff:.1%} bull")
        logger.info("")

        # Save training artifacts
        train_data.to_csv(self.logs_dir / f"train_data_v4_{self.timestamp}.csv")
        val_data.to_csv(self.logs_dir / f"val_data_v4_{self.timestamp}.csv")
        np.save(self.logs_dir / f"sample_weights_v4_{self.timestamp}.npy", train_weights)

        logger.info("✅ Balanced data loaded successfully!")
        logger.info("")

        return train_data, train_weights, val_data

    def train_v4_model(
        self,
        train_data: pd.DataFrame,
        sample_weights: np.ndarray,
        val_data: pd.DataFrame,
        total_timesteps: int = 1000000
    ) -> PortfolioTrainerOptimized:
        """Train v4 model with 2007-2024 data and weighted sampling."""
        logger.info("=" * 80)
        logger.info("STEP 2: Training v4 Model with Historical 2007-2024 Data")
        logger.info("=" * 80)
        logger.info("Configuration: v4 (173 features, 12x weighted sampling, v2.1 rewards)")
        logger.info("  - Network: [256, 256]")
        logger.info("  - Learning Rate: 1e-4")
        logger.info("  - Training Steps: 1,000,000")
        logger.info("  - VecNormalize: Enabled")
        logger.info("  - Weighted Sampling: YES (50/50 bull/bear with 12x weight)")
        logger.info("")

        # Save training data for trainer
        train_path = self.logs_dir / f"train_data_v4_{self.timestamp}_processed.csv"
        train_data.to_csv(train_path)

        # Save sample weights
        weights_path = self.logs_dir / f"sample_weights_v4_{self.timestamp}.npy"
        np.save(weights_path, sample_weights)

        # Initialize optimized trainer
        logger.info("Initializing PortfolioTrainerOptimized with weighted sampling...")
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
        logger.info("Model is learning from 17+ YEARS of market history:")
        logger.info("  • 2007-2009: Global Financial Crisis era")
        logger.info("  • 2010-2014: Recovery and corrections")
        logger.info("  • 2015-2019: China shock, multiple corrections")
        logger.info("  • 2020-2024: COVID crash and recovery")
        logger.info("")
        logger.info("Expected learning outcomes:")
        logger.info("  ✓ Recognize early warning signs of crashes")
        logger.info("  ✓ Take defensive positions during corrections")
        logger.info("  ✓ Recover quickly after drawdowns")
        logger.info("  ✓ Balance growth and protection across regimes")
        logger.info("")

        # Train
        trainer.train(
            total_timesteps=total_timesteps,
            log_interval=10,
            save_path=str(self.model_dir)
        )

        logger.info("=" * 80)
        logger.info("v4 MODEL TRAINING COMPLETED!")
        logger.info("=" * 80)

        return trainer

    def evaluate_v4_model(
        self,
        trainer: PortfolioTrainerOptimized,
        val_data: pd.DataFrame,
        n_episodes: int = 10
    ):
        """Evaluate v4 model on chronological validation set."""
        logger.info("=" * 80)
        logger.info("STEP 3: v4 Model Evaluation")
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
        logger.info("v4 MODEL VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Mean Return:      {mean_return:.2%}")
        logger.info(f"Std Return:       {std_return:.2%}")
        logger.info(f"Mean Final Value: ₹{results['mean_final_value']:,.0f}")
        logger.info(f"Mean Reward:      {results['mean_reward']:.4f}")
        logger.info("")

        # Compare with baselines
        logger.info("Backtesting classical methods on validation set...")
        backtester = PortfolioBacktester(val_data)

        # Load full data to get returns for classical optimizer
        full_data = pd.read_csv(self.project_root / "notebooks" / "data" / "portfolio_data_2007_2024.csv",
                                index_col=0, parse_dates=True)
        split_idx = int(len(full_data) * 0.9)
        train_full = full_data.iloc[:split_idx]

        returns_cols = [f"{asset}_returns" for asset in self.assets]
        returns_data = train_full[returns_cols].dropna()
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
        logger.info(f"v4 (2007-2024):      {mean_return:.2%} return")
        logger.info(f"Equal Weight:        {classical_results['equal_weight']['total_return']:.2%} return")
        logger.info(f"Min Volatility:      {classical_results['min_volatility']['total_return']:.2%} return")
        logger.info(f"Max Sharpe:          {classical_results['max_sharpe']['total_return']:.2%} return")

        # Compare with previous models
        logger.info("")
        logger.info("COMPARISON WITH PREVIOUS MODELS:")
        logger.info(f"  v1 (Momentum):     14.62% validation")
        logger.info(f"  v2 (Defensive):    7.87% validation")
        logger.info(f"  v2.1 (Balanced):   11.83% validation")
        logger.info(f"  v3 (Stage 2):      TBD validation")
        logger.info(f"  v4 (Historical):   {mean_return:.2%} validation")

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
            'model_type': 'production_v4_historical_2007_2024',
            'configuration': 'v4_historical',
            'training_timesteps': 1000000,
            'observation_features': 173,
            'v4_advantages': {
                'description': '2007-2024 historical data with 12 bear periods',
                'data_samples': 4623,
                'data_period': '2007-2024',
                'bear_periods': 12,
                'bear_samples': 329,
                'bear_ratio': '7.9%',
                'bear_weight': 12.0,
                'bull_weight': 1.0,
                'effective_balance': '50.8% bear / 49.2% bull',
                'reward_function': 'v2.1 balanced defensive',
                'major_crises': ['2008 GFC (-38%)', '2020 COVID (-43%)', '2015 China (-18%)']
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

        # Success criteria
        logger.info("")
        logger.info("=" * 80)
        logger.info("v4 SUCCESS CRITERIA EVALUATION")
        logger.info("=" * 80)

        go_criteria = []

        # Criterion 1: Beat Equal Weight
        if mean_return > classical_results['equal_weight']['total_return']:
            logger.info("✅ Criterion 1: Beats Equal Weight baseline")
            go_criteria.append(True)
        else:
            logger.info(f"❌ Criterion 1: Below Equal Weight by {abs(eq_gap):.2f} pp")
            go_criteria.append(False)

        # Criterion 2: Deterministic
        if std_return < 0.02:
            logger.info(f"✅ Criterion 2: Deterministic behavior (std={std_return:.2%} < 2%)")
            go_criteria.append(True)
        else:
            logger.info(f"⚠️  Criterion 2: High variance (std={std_return:.2%} >= 2%)")
            go_criteria.append(False)

        # Criterion 3: Target return
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
            logger.info("   ✅ Expected: Superior bear market performance from historical learning!")
        elif passed >= 2:
            logger.info(f"⚠️  DECISION: CONDITIONAL - Most criteria met ({passed}/{total})")
            logger.info("   → Proceed to paper trading for detailed evaluation")
        else:
            logger.info(f"🛑 DECISION: NO-GO - Insufficient performance ({passed}/{total})")
            logger.info("   → Consider further tuning")

        logger.info("")
        logger.info("=" * 80)
        logger.info("NEXT STEPS")
        logger.info("=" * 80)
        logger.info("1. Paper trading tests:")
        logger.info("   - Recent period: 2023-2024 (includes recent volatility)")
        logger.info("   - Bull period test")
        logger.info("   - Bear period test")
        logger.info("   - Mixed regime test")
        logger.info("")
        logger.info("2. Compare v4 vs v3 vs v2.1:")
        logger.info("   - Bear market performance (key metric)")
        logger.info("   - Overall returns")
        logger.info("   - Defensive behavior analysis")
        logger.info("")
        logger.info("3. Production decision based on comprehensive results")

        return report


def main():
    """Main v4 training pipeline."""
    try:
        # Initialize trainer
        trainer_obj = V4HistoricalTrainer()

        # Step 1: Load pre-balanced data
        train_data, sample_weights, val_data = trainer_obj.load_balanced_data()

        # Step 2: Train model
        trainer = trainer_obj.train_v4_model(
            train_data=train_data,
            sample_weights=sample_weights,
            val_data=val_data,
            total_timesteps=1000000
        )

        # Step 3: Evaluate
        report = trainer_obj.evaluate_v4_model(
            trainer=trainer,
            val_data=val_data,
            n_episodes=10
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ v4 TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {trainer_obj.model_dir}")
        logger.info(f"Validation report: {trainer_obj.model_dir}/validation_report.json")
        logger.info("")
        logger.info("v4 trained on 17+ years of market history including:")
        logger.info("  • 2008 Global Financial Crisis")
        logger.info("  • 2020 COVID-19 crash")
        logger.info("  • Multiple corrections and recoveries")
        logger.info("")
        logger.info("Next: Run comprehensive paper trading tests and compare with v3")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
