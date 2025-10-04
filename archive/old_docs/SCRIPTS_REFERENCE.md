# Scripts Reference Guide

Quick reference for all portfolio optimization scripts in the project.

## Script Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_pipeline.py` | Demo/Verification | Initial setup, smoke tests |
| `train_portfolio.py` | Complete Training | Train new models from scratch |
| `evaluate_portfolio.py` | Model Evaluation | Evaluate trained models |

---

## 1. run_pipeline.py (Demo)

### Purpose
Quick verification that everything is set up correctly.

### What It Does
- Downloads data (if needed)
- Processes data (if needed)
- Initializes portfolio
- Runs classical optimization demo
- Tests RL environment (10 steps)
- Does smoke test of trainer

### When to Use
- ✅ First time setup
- ✅ Verifying data pipeline
- ✅ Quick system check
- ✅ Testing environment setup
- ❌ NOT for actual training
- ❌ NOT for production

### Usage
```bash
python run_pipeline.py
```

### Output
- Console logs only
- Basic verification results
- Data files created

### Time
~2-5 minutes

---

## 2. train_portfolio.py (Complete Training)

### Purpose
Train RL agents with proper methodology and comprehensive evaluation.

### What It Does
1. **Data Preparation**
   - Loads/processes data
   - Splits into train/val/test (70/15/15)
   - Saves splits for reproducibility

2. **Classical Training**
   - Trains Max Sharpe portfolio
   - Trains Min Volatility portfolio
   - Trains Efficient Return portfolio
   - Trains Equal Weight benchmark

3. **RL Training**
   - Trains SAC agent
   - Regular checkpointing
   - Saves final model

4. **Validation Evaluation**
   - Multi-episode RL evaluation
   - Backtests classical methods
   - Compares performance

5. **Test Evaluation**
   - Final unbiased evaluation
   - Statistical analysis
   - Performance comparison

6. **Reporting**
   - Generates plots
   - Creates JSON/CSV results
   - Writes markdown report

### When to Use
- ✅ Training new models
- ✅ Experimenting with hyperparameters
- ✅ Comparing RL vs classical
- ✅ Production model training
- ✅ Research and analysis

### Usage
```bash
# Quick test (10K timesteps, ~10 min)
python train_portfolio.py --timesteps 10000

# Standard (100K timesteps, ~30-60 min)
python train_portfolio.py --timesteps 100000

# Extended (500K timesteps, several hours)
python train_portfolio.py --timesteps 500000

# Custom config
python train_portfolio.py --timesteps 100000 --config custom_portfolio.yaml
```

### Options
- `--timesteps INT`: Training duration (default: 100000)
- `--config PATH`: Config file path (default: config/portfolio.yaml)

### Output
```
results/TIMESTAMP/
├── REPORT.md
├── training_report.json
├── classical_portfolios.json
├── comparison_validation.csv
├── comparison_test.csv
├── train_data.csv
├── val_data.csv
├── test_data.csv
└── plots/
    ├── portfolio_trajectories.png
    ├── returns_comparison.png
    └── rl_evaluation_curves.png

models/TIMESTAMP/
├── final_model.zip
└── portfolio_model_*.zip (checkpoints)
```

### Time
- 10K steps: ~10 minutes
- 100K steps: ~30-60 minutes
- 500K steps: ~2-4 hours

---

## 3. evaluate_portfolio.py (Evaluation)

### Purpose
Evaluate previously trained models without retraining.

### What It Does
1. **Model Loading**
   - Loads saved RL model
   - Initializes environment

2. **RL Evaluation**
   - Runs N evaluation episodes
   - Calculates statistics
   - Tracks trajectories

3. **Classical Comparison**
   - Backtests classical methods
   - Compares performance

4. **Visualization**
   - Performance comparison plots
   - Weight allocation over time
   - Returns distribution

5. **Reporting**
   - Saves detailed results
   - Generates evaluation report

### When to Use
- ✅ Evaluating trained models
- ✅ Testing on new data
- ✅ Comparing multiple models
- ✅ Final model validation
- ✅ Performance analysis
- ❌ NOT for training (use train_portfolio.py)

### Usage
```bash
# Basic evaluation
python evaluate_portfolio.py models/TIMESTAMP/final_model.zip

# More episodes for better statistics
python evaluate_portfolio.py models/TIMESTAMP/final_model.zip --episodes 50

# Custom test data
python evaluate_portfolio.py models/TIMESTAMP/final_model.zip --test-data data/custom.csv

# Full options
python evaluate_portfolio.py \
    models/20240101_120000/final_model.zip \
    --episodes 30 \
    --test-data data/test_2024.csv \
    --config config/portfolio.yaml
```

### Options
- `model_path`: Path to trained model (required)
- `--episodes INT`: Number of evaluation episodes (default: 20)
- `--test-data PATH`: Custom test data CSV (optional)
- `--config PATH`: Config file (default: config/portfolio.yaml)

### Output
```
evaluation_results/TIMESTAMP/
├── EVALUATION_REPORT.md
├── evaluation_summary.json
├── rl_episodes.csv
└── plots/
    ├── evaluation_comparison.png
    └── weight_allocation.png
```

### Time
~5-15 minutes (depending on episodes)

---

## Workflow Recommendations

### Initial Setup
```bash
# 1. Verify everything works
python run_pipeline.py

# 2. Quick training test
python train_portfolio.py --timesteps 10000

# 3. Check results
cat results/*/REPORT.md
```

### Development Cycle
```bash
# 1. Train with moderate timesteps
python train_portfolio.py --timesteps 50000

# 2. Review results
cat results/TIMESTAMP/REPORT.md

# 3. If promising, train longer
python train_portfolio.py --timesteps 500000

# 4. Evaluate best model
python evaluate_portfolio.py models/BEST_TIMESTAMP/final_model.zip --episodes 50
```

### Model Comparison
```bash
# Train multiple models
python train_portfolio.py --timesteps 100000  # Model A
python train_portfolio.py --timesteps 200000  # Model B

# Evaluate each
python evaluate_portfolio.py models/TIMESTAMP_A/final_model.zip
python evaluate_portfolio.py models/TIMESTAMP_B/final_model.zip

# Compare results
diff evaluation_results/*/EVALUATION_REPORT.md
```

### Production Workflow
```bash
# 1. Final training with long horizon
python train_portfolio.py --timesteps 1000000

# 2. Thorough evaluation
python evaluate_portfolio.py models/TIMESTAMP/final_model.zip --episodes 100

# 3. Test on out-of-sample data
python evaluate_portfolio.py \
    models/TIMESTAMP/final_model.zip \
    --test-data data/latest_market_data.csv \
    --episodes 50
```

---

## Choosing the Right Script

### Use `run_pipeline.py` when:
- ⚡ You're setting up for the first time
- ⚡ You want to verify data pipeline
- ⚡ You need a quick system check
- ⚡ Testing environment works

### Use `train_portfolio.py` when:
- 🎯 You want to train a new model
- 🎯 You're experimenting with configurations
- 🎯 You need comprehensive evaluation
- 🎯 You want to compare RL vs classical
- 🎯 You're doing research/analysis

### Use `evaluate_portfolio.py` when:
- 📊 You have a trained model to evaluate
- 📊 You want to test on new data
- 📊 You're comparing multiple models
- 📊 You need detailed performance analysis
- 📊 You want to validate before deployment

---

## Common Patterns

### Pattern 1: Quick Iteration
```bash
# Train fast, check results, iterate
python train_portfolio.py --timesteps 10000
cat results/*/REPORT.md
# (Adjust config/hyperparameters)
python train_portfolio.py --timesteps 10000
```

### Pattern 2: Deep Training
```bash
# One long training run
python train_portfolio.py --timesteps 1000000
# Review comprehensive results
cat results/*/REPORT.md
ls -lh results/*/plots/
```

### Pattern 3: Model Selection
```bash
# Train multiple candidates
for steps in 50000 100000 200000; do
    python train_portfolio.py --timesteps $steps
done

# Evaluate each thoroughly
for model in models/*/final_model.zip; do
    python evaluate_portfolio.py $model --episodes 50
done
```

### Pattern 4: Walk-Forward Testing
```bash
# Train on period 1
python train_portfolio.py --timesteps 100000

# Evaluate on period 2
python evaluate_portfolio.py \
    models/TIMESTAMP/final_model.zip \
    --test-data data/period2.csv

# Evaluate on period 3
python evaluate_portfolio.py \
    models/TIMESTAMP/final_model.zip \
    --test-data data/period3.csv
```

---

## Tips

### For Faster Experimentation
- Use `--timesteps 10000` for quick tests
- Start with smaller portfolios (5-10 assets)
- Use checkpoints to resume training

### For Better Performance
- Train longer (500K-1M timesteps)
- Tune hyperparameters systematically
- Use validation set for model selection

### For Production
- Train on maximum available data
- Evaluate thoroughly (50+ episodes)
- Test on multiple out-of-sample periods
- Monitor performance over time

### For Debugging
- Check `run_pipeline.py` output first
- Review logs in console
- Inspect saved data CSVs
- Verify plots make sense

---

## Quick Reference Table

| Task | Script | Command |
|------|--------|---------|
| Setup verification | run_pipeline.py | `python run_pipeline.py` |
| Quick training test | train_portfolio.py | `python train_portfolio.py --timesteps 10000` |
| Full training | train_portfolio.py | `python train_portfolio.py --timesteps 100000` |
| Extended training | train_portfolio.py | `python train_portfolio.py --timesteps 500000` |
| Evaluate model | evaluate_portfolio.py | `python evaluate_portfolio.py models/*/final_model.zip` |
| Thorough evaluation | evaluate_portfolio.py | `python evaluate_portfolio.py models/*/final_model.zip --episodes 50` |
| Custom data test | evaluate_portfolio.py | `python evaluate_portfolio.py models/*/final_model.zip --test-data data/custom.csv` |

---

## Help Commands

```bash
# Get help for train_portfolio.py
python train_portfolio.py --help

# Get help for evaluate_portfolio.py
python evaluate_portfolio.py --help
```

---

## Summary

- **run_pipeline.py**: Quick demo and verification
- **train_portfolio.py**: Complete training pipeline
- **evaluate_portfolio.py**: Standalone evaluation

Start with `run_pipeline.py`, move to `train_portfolio.py` for real work, use `evaluate_portfolio.py` for model validation.
