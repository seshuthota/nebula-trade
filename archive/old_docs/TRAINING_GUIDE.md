# Portfolio Optimization Training Guide

This guide explains how to train and evaluate the portfolio optimization RL agent.

## Overview

The training pipeline consists of:
1. **Data Preparation**: Downloads and processes historical stock data
2. **Classical Baselines**: Trains classical portfolio optimization methods (Markowitz, etc.)
3. **RL Training**: Trains SAC (Soft Actor-Critic) agent
4. **Evaluation**: Evaluates both RL and classical methods on validation and test sets
5. **Visualization**: Generates comparison plots and performance reports

## Prerequisites

Make sure you have all dependencies installed:
```bash
pip install stable-baselines3 pypfopt matplotlib seaborn pandas numpy pyyaml yfinance
```

## Quick Start

### 1. Verify Data Setup

First, run the demo pipeline to ensure data is downloaded and processed:
```bash
python run_pipeline.py
```

This will:
- Download stock data if not present
- Process and prepare the data
- Run a quick smoke test of the environment

### 2. Train the Complete Model

Run the complete training pipeline:
```bash
python train_portfolio.py --timesteps 100000
```

**Options:**
- `--timesteps`: Number of training timesteps (default: 100000)
  - For quick test: `--timesteps 10000`
  - For better performance: `--timesteps 500000` or more
- `--config`: Path to config file (default: config/portfolio.yaml)

**Example:**
```bash
# Quick training (10K timesteps, ~5-10 minutes)
python train_portfolio.py --timesteps 10000

# Standard training (100K timesteps, ~30-60 minutes)
python train_portfolio.py --timesteps 100000

# Extended training (500K timesteps, several hours)
python train_portfolio.py --timesteps 500000
```

### 3. Evaluate a Trained Model

After training, evaluate the saved model:
```bash
python evaluate_portfolio.py models/TIMESTAMP/final_model.zip --episodes 20
```

**Options:**
- `model_path`: Path to the trained model (required)
- `--episodes`: Number of evaluation episodes (default: 20)
- `--test-data`: Path to custom test data CSV (optional)
- `--config`: Path to config file (default: config/portfolio.yaml)

**Example:**
```bash
# Evaluate with default settings
python evaluate_portfolio.py models/20240101_120000/final_model.zip

# Evaluate with more episodes
python evaluate_portfolio.py models/20240101_120000/final_model.zip --episodes 50

# Evaluate on custom data
python evaluate_portfolio.py models/20240101_120000/final_model.zip --test-data data/custom_test.csv
```

## Output Structure

After training, you'll get:

```
results/TIMESTAMP/
├── REPORT.md                      # Human-readable report
├── training_report.json           # Detailed results in JSON
├── classical_portfolios.json      # Classical optimization results
├── comparison_validation.csv      # Validation set comparison
├── comparison_test.csv            # Test set comparison
├── train_data.csv                 # Training data used
├── val_data.csv                   # Validation data used
├── test_data.csv                  # Test data used
└── plots/
    ├── portfolio_trajectories.png # Value over time
    ├── returns_comparison.png     # Returns bar chart
    └── rl_evaluation_curves.png   # Learning curves

models/TIMESTAMP/
├── final_model.zip                # Trained RL model
└── portfolio_model_*.zip          # Checkpoints during training
```

## Understanding the Results

### Key Metrics

1. **Total Return**: Overall profit/loss percentage
2. **Sharpe Ratio**: Risk-adjusted return (higher is better)
3. **Max Drawdown**: Largest peak-to-trough decline
4. **Final Portfolio Value**: End value in rupees

### Comparing RL vs Classical

The pipeline automatically compares:
- **RL (SAC)**: Reinforcement learning agent
- **Max Sharpe**: Classical portfolio maximizing Sharpe ratio
- **Min Volatility**: Classical portfolio minimizing risk
- **Equal Weight**: Naive 1/N benchmark
- **Efficient Return**: Classical portfolio at target return

Check `REPORT.md` for the full comparison.

## Training Pipeline Details

### Data Split

The pipeline automatically splits data:
- **Training**: 70% (trains RL agent and classical methods)
- **Validation**: 15% (model selection and hyperparameter tuning)
- **Test**: 15% (final unbiased evaluation)

### Classical Optimization

Classical methods use Modern Portfolio Theory:
- Markowitz mean-variance optimization
- Efficient frontier calculation
- Risk-return tradeoff analysis

### RL Training

The RL agent uses SAC (Soft Actor-Critic):
- **Algorithm**: SAC (continuous action space)
- **Policy**: MLP (256x256 hidden layers)
- **Action Space**: Continuous weights for each asset
- **Observation**: Price history, returns, portfolio state
- **Reward**: Portfolio value change

## Configuration

Edit `config/portfolio.yaml` to customize:

```yaml
portfolio:
  assets:
    - "HDFCBANK.NS"
    - "ICICIBANK.NS"
    - "SBIN.NS"
  
  initial_weights:
    mode: "equal"
    cash_buffer: 0.1
  
  constraints:
    min_weight: 0.0  # No shorting
    max_weight: 1.0
  
  rebalance:
    frequency: "daily"
    transaction_cost: 0.001  # 0.1% per trade
  
  training:
    test_split_ratio: 0.2
```

## Advanced Usage

### Walk-Forward Optimization

For more robust evaluation, you can implement walk-forward validation:
1. Split data into multiple windows
2. Train on each window
3. Test on the next period
4. Average results across windows

### Hyperparameter Tuning

Modify training parameters in the code:
```python
agent = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # Learning rate
    buffer_size=100000,      # Replay buffer size
    batch_size=256,          # Batch size
    gamma=0.99,              # Discount factor
    tau=0.005,               # Soft update coefficient
)
```

### Custom Rewards

Modify the environment in `astra/rl_framework/environment.py` to:
- Add Sharpe ratio to reward
- Penalize drawdowns
- Encourage diversification
- Add transaction cost penalties

## Troubleshooting

### Out of Memory

If training uses too much memory:
- Reduce `buffer_size` in trainer
- Use fewer timesteps
- Process data in smaller batches

### Slow Training

To speed up training:
- Use fewer timesteps for testing
- Reduce network size
- Use GPU if available (install `stable-baselines3[extra]`)

### Poor Performance

If RL underperforms classical methods:
- Train for more timesteps (try 500K-1M)
- Tune reward function
- Adjust learning rate
- Try different architectures
- Check data quality

## Example Workflow

```bash
# 1. Initial setup and data verification
python run_pipeline.py

# 2. Quick training test (10K timesteps, ~10 min)
python train_portfolio.py --timesteps 10000

# 3. Check results
cat results/*/REPORT.md

# 4. If promising, train longer (500K timesteps)
python train_portfolio.py --timesteps 500000

# 5. Evaluate best model
python evaluate_portfolio.py models/20240101_120000/final_model.zip --episodes 50

# 6. Compare multiple models
python evaluate_portfolio.py models/20240101_120000/final_model.zip
python evaluate_portfolio.py models/20240101_150000/final_model.zip
```

## Next Steps

1. **Experiment with Assets**: Change assets in `config/portfolio.yaml`
2. **Try Different Algorithms**: Implement PPO, TD3, or other RL algorithms
3. **Add Features**: Include technical indicators, market sentiment, etc.
4. **Deploy**: Use best model for live trading (with proper risk management!)

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)
- Modern Portfolio Theory (Markowitz, 1952)
- Soft Actor-Critic (Haarnoja et al., 2018)

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review error messages carefully
3. Verify data quality and configuration
4. Ensure all dependencies are installed correctly
