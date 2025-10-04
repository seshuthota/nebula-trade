#!/usr/bin/env python3
"""
Complete pipeline runner for Project Astra Portfolio Optimization.
Downloads data, processes it, initializes portfolio, and demonstrates classical optimization.
"""

import pandas as pd
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("="*60)
    print("Project Astra: Portfolio Optimization Pipeline")
    print("="*60)

    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "portfolio.yaml"

    # 1. Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    assets = config['portfolio']['assets']
    print(f"Portfolio: {', '.join(assets)}")
    print(f"Initialization: {config['portfolio']['initial_weights']['mode']}")
    print()

    # 2. Download data (if needed)
    data_file = project_root / 'data' / 'portfolio_data.csv'
    if not Path(data_file).exists():
        print("Step 1: Downloading data...")
        from astra.data_pipeline.downloader import PortfolioDataDownloader
        downloader = PortfolioDataDownloader(config_path=config_path)
        data = downloader.download_portfolio_data()
        downloader.save_data(data)
    else:
        print("Step 1: Data already downloaded ✓")

    # 3. Process data (if needed)
    processed_file = project_root / 'data' / 'portfolio_data_processed.csv'
    if not Path(processed_file).exists():
        print("Step 2: Preprocessing data...")
        from astra.data_pipeline.preprocessor import PortfolioDataPreprocessor
        preprocessor = PortfolioDataPreprocessor(config_path=config_path)
        processed_data = preprocessor.process_all(str(data_file))
    else:
        print("Step 2: Data already processed ✓")

    # 4. Initialize portfolio
    print("Step 3: Initializing portfolio...")
    from astra.data_pipeline.data_manager import PortfolioDataManager
    manager = PortfolioDataManager(config_path=config_path)
    data, initial_state = manager.process_and_initialize()

    print("✓ Portfolio initialized successfully!")
    print(f"  • Capital: ₹{initial_state['capital']:,.0f}")
    print(f"  • Assets: {', '.join(initial_state['positions'].keys())}")
    asset_weights = {asset: initial_state['weights'].get(asset, 0) for asset in assets}
    print(f"  • Asset Weights: {', '.join([f'{k}: {v:.2%}' for k, v in asset_weights.items()])}")
    print(f"  • Cash Position: ₹{initial_state['cash']:,.0f} ({initial_state['weights']['cash']:.1%})")
    print(f"  • Data Points: {len(data)} (from {data.index[0].date()} to {data.index[-1].date()})")
    print()

    # 5. Classical optimization demo
    print("Step 4: Running classical portfolio optimization...")
    returns_cols = [f"{asset}_returns" for asset in assets]
    returns_data = data[returns_cols].dropna()

    from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester
    optimizer = ClassicalPortfolioOptimizer(returns_data)
    classical_portfolios = optimizer.get_all_portfolios()

    print("Classical Optimization Results:")
    print("-" * 40)
    for name, portfolio in classical_portfolios.items():
        print(f"{name:15s} | Sharpe: {portfolio['sharpe_ratio']:5.2f} | Ret: {portfolio['expected_return']:5.2f}")
    print()

    # 6. Backtest best portfolio
    print("Step 5: Backtesting Max Sharpe portfolio...")
    max_sharpe_weights = classical_portfolios['max_sharpe']['weights']
    backtester = PortfolioBacktester(data)
    backtest_result = backtester.backtest_portfolio(max_sharpe_weights)

    print("Backtest Results for Max Sharpe Portfolio:")
    print("-" * 40)
    print(f"{'Final Value':15s}: ₹{backtest_result['final_value']:,.0f}")
    print(f"{'Total Return':15s}: {backtest_result['total_return']:.2%}")
    print(f"{'Sharpe Ratio':15s}: {backtest_result['sharpe_ratio']:.2f}")
    print()

    # 7. Test RL environment
    print("Step 6: Testing RL environment...")
    from astra.rl_framework.environment import PortfolioEnvironment
    env = PortfolioEnvironment(data[:100], config_path=config_path, lookback_window=30)  # Small sample for demo
    obs, _ = env.reset()

    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 3 == 0:
            print(f"Step {step}: Portfolio Value: ₹{info['portfolio_value']:,.0f}")
        if terminated or truncated:
            break

    print(f"Demo completed. Total reward over 10 steps: {total_reward:.6f}")
    print()

    # 7. RL trainer smoke test
    print("Step 7: RL trainer environment smoke test...")
    from astra.rl_framework.trainer import PortfolioTrainer

    trainer = PortfolioTrainer(config_path=config_path)
    rl_env = trainer.env
    obs, info = rl_env.reset()
    total_reward = 0
    for step in range(10):
        action = rl_env.action_space.sample()
        obs, reward, terminated, truncated, info = rl_env.step(action)
        total_reward += reward
        if step % 5 == 0:
            print(f"  Trainer Env Step {step}: Value ₹{info['portfolio_value']:,.2f}, Reward {reward:.6f}")
        if terminated or truncated:
            break

    print(f"Trainer env demo reward (10 steps): {total_reward:.6f}")
    print()

    print("="*60)
    print("✓ Pipeline execution complete!")
    print("Next steps:")
    print("  • Launch notebooks/03_portfolio_optimization.ipynb for full analysis")
    print("  • Run astra.rl_framework.trainer for RL training")
    print("  • Compare classical vs RL performance")
    print("="*60)

if __name__ == "__main__":
    main()
