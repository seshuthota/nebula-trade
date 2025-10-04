#!/usr/bin/env python3
"""Quick evaluation of resumed training model."""

from astra.rl_framework.trainer_optimized import PortfolioTrainerOptimized
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading data...")
data = pd.read_csv("data/portfolio_data_processed.csv", index_col=0, parse_dates=True)

# Split: 70% train, 15% val, 15% test
n = len(data)
val_end = int(n * 0.85)
test_data = data.iloc[val_end:]

logger.info(f"Test data: {len(test_data)} samples from {test_data.index[0]} to {test_data.index[-1]}")

# Create trainer
logger.info("Creating trainer...")
trainer = PortfolioTrainerOptimized(
    config_path="config/portfolio.yaml",
    data_path="data/portfolio_data_processed.csv",
    n_envs=1,
    use_gpu=False
)

# Load the Phase 4 Extended model (2M steps with features)
logger.info("Loading Phase 4 Extended model (2M steps with features)...")
vec_normalize_path = "models/20251003_000711_resumed/vec_normalize.pkl"
if trainer.env.__class__.__name__ == 'VecNormalize':
    base_env = trainer.env.venv
    trainer.env = VecNormalize.load(vec_normalize_path, base_env)

trainer.agent = SAC.load(
    "models/20251003_000711_resumed/final_model.zip",
    env=trainer.env
)

# Evaluate on test set
logger.info("Evaluating on test set (10 episodes)...")
results = trainer.evaluate(test_data, n_episodes=10)

# Calculate return manually
initial_capital = 100000
mean_return = (results['mean_final_value'] - initial_capital) / initial_capital
std_return = results['std_final_value'] / initial_capital

print("\n" + "="*80)
print("EVALUATION RESULTS - PHASE 4 EXTENDED (2M steps total)")
print("FEATURE ENGINEERING: Added momentum, volatility, regime features")
print("="*80)
print(f"Mean Return: {mean_return:.2%}")
print(f"Std Return:  {std_return:.2%}")
print(f"Mean Final Value: ₹{results['mean_final_value']:,.0f}")
print(f"Mean Reward: {results['mean_reward']:.4f}")
print("="*80)

print("\nPHASE COMPARISON:")
print("Phase 1 (600k steps):              24.20% return, ₹123,897")
print("Phase 1 Extended (1M steps):       29.44% return, ₹129,436  ⭐ Baseline")
print("Phase 2 (1.6M, aggressive reward): 22.42% return, ₹122,415  ❌ Failed")
print("Phase 3 (1.6M, continued):         29.39% return, ₹129,389  🔄 Plateau")
print("Phase 4 (1M, with features):       24.27% return, ₹123,969  ❌ Regression")
print(f"Phase 4 Extended (2M, features):   {mean_return:.2%} return, ₹{results['mean_final_value']:,.0f}")

improvement_from_phase1_ext = (mean_return - 0.2944) * 100
improvement_from_phase1 = (mean_return - 0.2420) * 100
print(f"\nImprovement from Phase 1 Extended: {improvement_from_phase1_ext:+.2f} pp")
print(f"Total improvement from Phase 1: {improvement_from_phase1:+.2f} pp")

# Compare with classical methods
print("\n" + "="*80)
print("COMPARISON WITH CLASSICAL METHODS (Test Set):")
print("="*80)
print(f"RL Phase 4 Ext (2M): {mean_return:.2%} return, ₹{results['mean_final_value']:,.0f}")
print(f"Max Sharpe:          32.87% return, ₹132,874  (Gap: {(0.3287 - mean_return)*100:+.2f} pp)")
print(f"Min Volatility:      31.13% return, ₹131,134  (Gap: {(0.3113 - mean_return)*100:+.2f} pp)")
print(f"Equal Weight:        23.53% return, ₹123,530  (Gap: {(mean_return - 0.2353)*100:+.2f} pp)")
print("="*80)

print("\n🎯 ACHIEVEMENT TRACKER:")
if mean_return > 0.2353:
    print("✅ BEAT Equal Weight baseline!")
if mean_return > 0.2420:
    print("✅ IMPROVED over Phase 1 (600k)!")
if mean_return > 0.2944:
    print("✅ IMPROVED over Phase 1 Extended (1M)!")
if mean_return > 0.30:
    print("🎯 COMPETITIVE with classical methods!")
if mean_return > 0.3113:
    print("🏆 BEAT Min Volatility!")
if mean_return > 0.3287:
    print("🏆🏆 BEAT ALL CLASSICAL METHODS INCLUDING MAX SHARPE!")
print("="*80)
