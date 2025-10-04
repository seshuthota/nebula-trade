# Quick Reference - Nebula Trade Refactored

## Common Commands

### Training

```bash
# Train a model with default config
python scripts/train.py --model v1_momentum

# Train with production config (more checkpoints)
python scripts/train.py --model v5_tuned --training production

# Quick test (10k steps)
python scripts/train.py --model v1_momentum --training quick_test

# Override timesteps
python scripts/train.py --model v1_momentum --timesteps 50000

# Disable WandB logging
python scripts/train.py --model v1_momentum --no-wandb

# Force CPU training
python scripts/train.py --model v1_momentum --no-gpu
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py --model models/production/v1_momentum_20251004_120000

# Evaluate with more episodes
python scripts/evaluate.py --model models/production/v1_momentum_20251004_120000 --episodes 50

# Compare multiple models
python scripts/evaluate.py --compare v1_momentum,v2_defensive,v5_tuned

# Evaluate on specific period
python scripts/evaluate.py --model [path] --period 2025_ytd
python scripts/evaluate.py --model [path] --period q1_2025
```

### Information

```bash
# List available models
python scripts/train.py --list-models

# List training configs
python scripts/train.py --list-configs

# Show model details
python scripts/train.py --model v5_tuned --info

# List trained models
python scripts/evaluate.py --list
```

### Using CLI

```bash
# Train
python cli/main.py train --model v1_momentum

# Evaluate
python cli/main.py evaluate --model models/production/v1_momentum_[timestamp]

# Compare
python cli/main.py compare --models v1_momentum,v2_defensive,v5_tuned

# List
python cli/main.py list models
python cli/main.py list configs

# Info
python cli/main.py info --model v5_tuned
```

## Model Configurations

Available models in `config/models/`:

| Model | Strategy | Status | Use Case |
|-------|----------|--------|----------|
| `v1_momentum` | Momentum-focused | ✅ Active | Bull markets |
| `v2_defensive` | Risk-averse | ✅ Active | Bear markets |
| `v2.1_balanced` | Balanced | ⚠️ Backup | All-weather backup |
| `v3_stage2` | Oversampled | ❌ Retired | Experimental |
| `v4_historical` | Historical 50/50 | 🔒 Archived | Too defensive |
| `v5_tuned` | Historical 70/30 | 🔒 Reserve | Historical learning |

## Training Configurations

Available in `config/training/`:

- **`default`** - Standard training (1M steps, 6 parallel envs)
- **`production`** - Production training (more checkpoints, longer eval)
- **`quick_test`** - Fast testing (10k steps, 2 parallel envs, no WandB)

## Configuration Files

### Model Config Structure

```yaml
# config/models/my_model.yaml
model:
  reward:
    type: "balanced"  # momentum, defensive, balanced, standard
    components:
      return_weight: 0.60
      sharpe_weight: 0.20
      drawdown_penalty: 0.10
      volatility_penalty: 0.05
      turnover_penalty: 0.05
    loss_aversion: 1.5
  
  data:
    period: "2015-2024"
    split:
      train: 0.90
      val: 0.10
    balancing:
      enabled: false

metadata:
  version: "v6"
  description: "Custom model"
  strategy: "custom"
  tags: ["experimental"]
```

## Environment Variables

Override configuration via environment:

```bash
# Override timesteps
NEBULA_TIMESTEPS=50000 python scripts/train.py --model v1_momentum

# Disable WandB
NEBULA_USE_WANDB=false python scripts/train.py --model v1_momentum

# Force CPU
NEBULA_GPU=false python scripts/train.py --model v1_momentum
```

## Directory Structure

```
nebula-trade/
├── config/
│   ├── models/        # Model configurations
│   ├── training/      # Training configurations
│   └── ensemble/      # Ensemble configurations
│
├── scripts/
│   ├── train.py       # Unified training
│   ├── evaluate.py    # Unified evaluation
│   └── *.py          # Data scripts
│
├── cli/
│   └── main.py        # CLI interface
│
├── astra/
│   ├── core/          # Core abstractions
│   ├── training/      # Training modules
│   ├── rl_framework/  # RL environment
│   ├── data_pipeline/ # Data processing
│   ├── evaluation/    # Evaluation tools
│   └── ensemble/      # Ensemble system
│
├── models/
│   └── production/    # Trained models
│
└── tools/             # Utility scripts
```

## Creating a New Model

1. **Copy existing config:**
   ```bash
   cp config/models/v5_tuned.yaml config/models/my_model.yaml
   ```

2. **Edit the config** (change reward params, data settings, etc.)

3. **Train:**
   ```bash
   python scripts/train.py --model my_model
   ```

4. **Evaluate:**
   ```bash
   python scripts/evaluate.py --model models/production/my_model_[timestamp]
   ```

## Migrating from Old Scripts

| Old Script | New Command |
|------------|-------------|
| `python train_v4_historical.py` | `python scripts/train.py --model v4_historical` |
| `python train_v5_tuned_historical.py` | `python scripts/train.py --model v5_tuned` |
| `python train_balanced_model.py` | `python scripts/train.py --model v2.1_balanced` |
| `python train_defensive_model.py` | `python scripts/train.py --model v2_defensive` |
| `python evaluate_v4.py` | `python scripts/evaluate.py --model [path]` |
| `python evaluate_portfolio.py` | `python scripts/evaluate.py --model [path]` |

## Tips

1. **Start with quick test:**
   ```bash
   python scripts/train.py --model v1_momentum --training quick_test
   ```

2. **Check model info before training:**
   ```bash
   python scripts/train.py --model v5_tuned --info
   ```

3. **Use production config for real training:**
   ```bash
   python scripts/train.py --model v5_tuned --training production
   ```

4. **Compare before deploying:**
   ```bash
   python scripts/evaluate.py --compare v1_momentum,v2_defensive,v5_tuned
   ```

## Documentation

- `REFACTORING_GUIDE.md` - Complete refactoring documentation
- `docs/guides/` - Training and evaluation guides
- `docs/architecture/` - System architecture
- `config/models/` - Model configuration examples

## Support

For issues or questions:
1. Check `REFACTORING_GUIDE.md`
2. Look at example configs in `config/models/`
3. Use `--info` flag to get model details
4. Use `--list-models` to see available options
