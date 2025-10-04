"""
Test script for defensive reward function in PortfolioEnvironment.
Verifies all reward components work correctly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astra.rl_framework.environment import PortfolioEnvironment

def test_defensive_rewards():
    """Test that defensive reward components are working."""

    print("=" * 80)
    print("TESTING DEFENSIVE REWARD FUNCTION")
    print("=" * 80)

    # Load data
    data_path = Path("notebooks/data/portfolio_data_processed.csv")
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return False

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✅ Loaded data: {len(data)} samples")

    # Create environment
    config_path = Path("config/portfolio.yaml")
    env = PortfolioEnvironment(data, config_path=str(config_path))
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")

    # Reset environment
    obs, info = env.reset()
    print(f"\n✅ Environment reset at {info['date']}")
    print(f"   Initial portfolio value: ₹{env.portfolio_value:,.2f}")

    # Run 50 steps and collect reward statistics
    print("\n" + "=" * 80)
    print("RUNNING 50 STEPS TO TEST REWARD COMPONENTS")
    print("=" * 80)

    rewards = []
    positive_rewards = []
    negative_rewards = []
    returns = []

    for step in range(50):
        # Random action (normalized to sum to 1)
        action = np.random.random(env.n_assets)
        action = action / action.sum()

        obs, reward, done, truncated, info = env.step(action)

        # Calculate return for this step
        if step == 0:
            prev_value = 100000
        else:
            prev_value = returns[-1]['portfolio_value']

        daily_return = (info['portfolio_value'] - prev_value) / prev_value

        rewards.append(reward)
        if reward > 0:
            positive_rewards.append(reward)
        else:
            negative_rewards.append(reward)

        returns.append({
            'step': step + 1,
            'date': info['date'],
            'portfolio_value': info['portfolio_value'],
            'daily_return': daily_return,
            'reward': reward,
            'turnover': info['turnover']
        })

        if done:
            print(f"\n⚠️  Episode terminated early at step {step + 1}")
            break

    # Display results
    print(f"\n" + "=" * 80)
    print("REWARD STATISTICS")
    print("=" * 80)
    print(f"Total steps: {len(rewards)}")
    print(f"Mean reward: {np.mean(rewards):.4f}")
    print(f"Std reward: {np.std(rewards):.4f}")
    print(f"Min reward: {np.min(rewards):.4f}")
    print(f"Max reward: {np.max(rewards):.4f}")
    print(f"\nPositive rewards: {len(positive_rewards)} ({len(positive_rewards)/len(rewards)*100:.1f}%)")
    print(f"  Mean: {np.mean(positive_rewards) if positive_rewards else 0:.4f}")
    print(f"Negative rewards: {len(negative_rewards)} ({len(negative_rewards)/len(rewards)*100:.1f}%)")
    print(f"  Mean: {np.mean(negative_rewards) if negative_rewards else 0:.4f}")

    # Check asymmetric loss aversion
    print(f"\n" + "=" * 80)
    print("ASYMMETRIC LOSS AVERSION CHECK")
    print("=" * 80)
    if positive_rewards and negative_rewards:
        avg_positive = np.mean(positive_rewards)
        avg_negative = np.mean(negative_rewards)
        ratio = abs(avg_negative) / avg_positive if avg_positive > 0 else 0
        print(f"Average positive reward: {avg_positive:.4f}")
        print(f"Average negative reward: {avg_negative:.4f}")
        print(f"Ratio (|negative|/positive): {ratio:.2f}")
        if ratio > 1.2:
            print("✅ Asymmetric loss aversion working (losses penalized more)")
        else:
            print("⚠️  Loss aversion may be weak (ratio < 1.2)")
    else:
        print("⚠️  Need both positive and negative rewards to test asymmetry")

    # Display sample of steps
    print(f"\n" + "=" * 80)
    print("SAMPLE STEPS (First 10)")
    print("=" * 80)
    print(f"{'Step':<6} {'Date':<12} {'Value':<12} {'Return':<10} {'Reward':<10} {'Turnover':<10}")
    print("-" * 80)
    for r in returns[:10]:
        print(f"{r['step']:<6} {str(r['date'])[:10]:<12} ₹{r['portfolio_value']:<11,.0f} {r['daily_return']:>9.2%} {r['reward']:>9.2f} {r['turnover']:>9.1%}")

    # Final portfolio value
    final_value = returns[-1]['portfolio_value']
    total_return = (final_value - 100000) / 100000
    print(f"\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Initial value: ₹100,000")
    print(f"Final value: ₹{final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Total reward: {sum(rewards):.2f}")

    print(f"\n✅ Defensive environment test completed successfully!")

    return True

if __name__ == "__main__":
    test_defensive_rewards()
