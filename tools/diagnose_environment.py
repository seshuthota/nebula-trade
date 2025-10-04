#!/usr/bin/env python3
"""
Diagnostic script to trace environment behavior step-by-step.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from astra.rl_framework.environment import PortfolioEnvironment

def trace_episode(model_path, data_path, num_steps=20):
    """Trace an episode step by step to find the bug."""
    
    print("=" * 80)
    print("ENVIRONMENT DIAGNOSTIC TRACE")
    print("=" * 80)
    
    # Load model and data
    agent = SAC.load(model_path)
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    print(f"\nData period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total days: {len(data)}")
    
    # Create environment
    env = PortfolioEnvironment(data, lookback_window=30)
    obs, info = env.reset()
    
    print(f"\nInitial State:")
    print(f"  Portfolio Value: ₹{env.portfolio_value:,.2f}")
    print(f"  Date: {info.get('date', 'N/A')}")
    
    # Trace first N steps
    print(f"\nTracing first {num_steps} steps:")
    print("=" * 80)
    
    cumulative_reward = 0
    portfolio_values = [env.portfolio_value]
    
    for step in range(num_steps):
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        cumulative_reward += reward
        portfolio_values.append(info['portfolio_value'])
        
        # Print detailed info
        print(f"\nStep {step + 1}:")
        print(f"  Date: {info['date'].date()}")
        print(f"  Action (weights): {action}")
        print(f"  Reward: {reward:.6f}")
        print(f"  Cumulative Reward: {cumulative_reward:.6f}")
        print(f"  Portfolio Value: ₹{info['portfolio_value']:,.2f}")
        print(f"  Turnover: {info.get('turnover', 0):.4f}")
        print(f"  Transaction Cost: ₹{info.get('transaction_cost', 0):.2f}")
        
        # Show weights
        weights = info['weights']
        print(f"  Current Weights:")
        for asset, weight in weights.items():
            if weight > 0.001:  # Only show significant weights
                print(f"    {asset}: {weight:.4f}")
        
        # Calculate step return
        step_return = (portfolio_values[-1] / portfolio_values[-2] - 1) * 100
        print(f"  Step Return: {step_return:.4f}%")
        
        if terminated or truncated:
            print("\n  Episode terminated!")
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    print(f"\nInitial Value: ₹{portfolio_values[0]:,.2f}")
    print(f"Final Value: ₹{portfolio_values[-1]:,.2f}")
    print(f"Total Return ({len(portfolio_values)-1} steps): {total_return:.2f}%")
    print(f"Cumulative Reward: {cumulative_reward:.6f}")
    
    # Check if reward matches return
    print(f"\nDIAGNOSTIC CHECKS:")
    print(f"  Reward sum: {cumulative_reward:.6f}")
    print(f"  Actual return: {total_return/100:.6f}")
    
    if abs(cumulative_reward - total_return/100) > 0.01:
        print(f"  ⚠️  MISMATCH: Reward doesn't match actual return!")
    else:
        print(f"  ✓ Reward matches return")
    
    # Check for unrealistic growth
    avg_daily_return = total_return / (len(portfolio_values) - 1)
    print(f"\n  Average daily return: {avg_daily_return:.4f}%")
    
    if avg_daily_return > 1.0:
        print(f"  ⚠️  UNREALISTIC: {avg_daily_return:.2f}% daily return is impossible!")
    
    # Project annualized
    if len(portfolio_values) > 1:
        annualized = ((portfolio_values[-1] / portfolio_values[0]) ** (252 / (len(portfolio_values)-1)) - 1) * 100
        print(f"\n  Projected annualized return: {annualized:.2f}%")
        
        if annualized > 200:
            print(f"  ⚠️  UNREALISTIC: {annualized:.0f}% annual return suggests a bug!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose environment behavior")
    parser.add_argument('--model', type=str, 
                       default='models/20250930_001718/final_model.zip',
                       help='Path to trained model')
    parser.add_argument('--data', type=str,
                       default='results/20250930_001718/test_data.csv',
                       help='Path to test data')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of steps to trace')
    
    args = parser.parse_args()
    
    trace_episode(args.model, args.data, num_steps=args.steps)

if __name__ == "__main__":
    main()
