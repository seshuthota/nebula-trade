# Nebula Trade - Portfolio Optimization (Refactored)

**Status:** ✅ **Production Ready** | **Version:** 2.0 (Refactored) | **Date:** October 2024

---

## 🎉 What's New?

This codebase has been **completely refactored** to be more maintainable, scalable, and easier to use!

### Key Improvements

- ✅ **Configuration-driven training** - No more hard-coded parameters
- ✅ **Unified interface** - One script for all models
- ✅ **Zero code duplication** - Eliminated ~3000 lines of duplicate code
- ✅ **Easy experimentation** - Create new models with YAML configs
- ✅ **Better organization** - Clear structure and separation of concerns
- ✅ **Comprehensive docs** - Complete guides and examples

### Quick Start

```bash
# List available models
python scripts/train.py --list-models

# Get model information
python scripts/train.py --model v1_momentum --info

# Train a model
python scripts/train.py --model v1_momentum --training quick_test

# Evaluate a model
python scripts/evaluate.py --list
```

---

## 📖 Documentation

- **[REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)** - Complete refactoring documentation
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference
- **[DEPRECATED_SCRIPTS.md](DEPRECATED_SCRIPTS.md)** - Migration from old scripts
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Summary of changes

---

## 🚀 Quick Examples

### Training Models

```bash
# Train with default configuration
python scripts/train.py --model v1_momentum

# Train with production configuration (more checkpoints)
python scripts/train.py --model v5_tuned --training production

# Quick test (10k steps, no WandB)
python scripts/train.py --model v1_momentum --training quick_test

# Override parameters
python scripts/train.py --model v1_momentum --timesteps 50000 --no-wandb
```

### Evaluating Models

```bash
# Evaluate a specific model
python scripts/evaluate.py --model models/production/v1_momentum_20251004_120000

# Compare multiple models
python scripts/evaluate.py --compare v1_momentum,v2_defensive,v5_tuned

# Evaluate on specific period
python scripts/evaluate.py --model [path] --period 2025_ytd
python scripts/evaluate.py --model [path] --period q1_2025
```

### Using the CLI

```bash
# More user-friendly commands
python cli/main.py list models
python cli/main.py info --model v5_tuned
python cli/main.py train --model v1_momentum
python cli/main.py evaluate --model [path]
python cli/main.py compare --models v1_momentum,v2_defensive
```

---

## 📁 New Directory Structure

```
nebula-trade/
├── config/
│   ├── models/           # Model configurations (NEW)
│   │   ├── v1_momentum.yaml
│   │   ├── v2_defensive.yaml
│   │   ├── v5_tuned.yaml
│   │   └── ...
│   ├── training/         # Training configurations (NEW)
│   │   ├── default.yaml
│   │   ├── production.yaml
│   │   └── quick_test.yaml
│   └── ensemble/         # Ensemble configurations
│
├── astra/
│   ├── core/             # Core abstractions (NEW)
│   │   ├── config_manager.py
│   │   └── model_registry.py
│   ├── training/         # Unified training (NEW)
│   │   ├── unified_trainer.py
│   │   └── reward_functions.py
│   ├── rl_framework/     # RL environment
│   ├── data_pipeline/    # Data processing
│   ├── evaluation/       # Evaluation tools
│   └── ensemble/         # Ensemble system
│
├── scripts/              # Main scripts (NEW)
│   ├── train.py         # Unified training
│   ├── evaluate.py      # Unified evaluation
│   └── *.py            # Data scripts
│
├── cli/                  # CLI interface (NEW)
│   └── main.py
│
├── tools/                # Utilities (NEW)
│   ├── benchmark_performance.py
│   └── diagnose_environment.py
│
├── models/
│   └── production/      # Trained models
│
├── docs/                # Documentation
│   ├── architecture/
│   ├── guides/
│   └── experiments/
│
└── [Old training scripts]  # Deprecated, kept for reference
```

---

## 🎯 Available Models

| Model | Strategy | Status | Use Case |
|-------|----------|--------|----------|
| `v1_momentum` | Momentum-focused | ✅ Active | Bull markets (Primary) |
| `v2_defensive` | Risk-averse | ✅ Active | Bear markets (Shield) |
| `v2.1_balanced` | Balanced | ⚠️ Backup | All-weather backup |
| `v3_stage2` | Oversampled | ❌ Retired | Experimental |
| `v4_historical` | Historical 50/50 | 🔒 Archived | Too defensive |
| `v5_tuned` | Historical 70/30 | 🔒 Reserve | Historical learning |

### Model Performance (2025 YTD)

| Model | Q1 Bull | Q3 Bear | Overall |
|-------|---------|---------|---------|
| v1 (momentum) | **+12.95%** 🥇 | -6.81% ❌ | **+10.10%** 🥇 |
| v2 (defensive) | +3.21% | **+0.34%** 🥇 | +6.17% |
| v2.1 (balanced) | +11.02% 🥈 | -6.31% | +9.87% 🥈 |
| v5 (tuned) | +10.10% 🥉 | -3.17% | +7.58% |

**Recommended:** Use **v1 + v2 Ensemble** with sticky BEAR switching for best results (12.92% YTD)

---

## 🔧 Creating Custom Models

### 1. Create Configuration File

```bash
# Copy an existing config
cp config/models/v5_tuned.yaml config/models/my_model.yaml
```

### 2. Edit Configuration

```yaml
# config/models/my_model.yaml
model:
  reward:
    type: "balanced"  # or "momentum", "defensive"
    components:
      return_weight: 0.70    # Adjust weights
      sharpe_weight: 0.15
      drawdown_penalty: 0.10
    loss_aversion: 1.5

  data:
    period: "2015-2024"
    split:
      train: 0.90
      val: 0.10

metadata:
  version: "v6"
  description: "My custom model"
  strategy: "custom"
  tags: ["experimental", "custom"]
```

### 3. Train Your Model

```bash
python scripts/train.py --model my_model
```

That's it! No code changes needed.

---

## 🔄 Migrating from Old Scripts

### Old Way (Deprecated)

```bash
python train_v5_tuned_historical.py    # 417 lines of code
python evaluate_v4.py                   # Separate script
```

### New Way

```bash
python scripts/train.py --model v5_tuned      # Use config
python scripts/evaluate.py --model [path]     # One script for all
```

See [DEPRECATED_SCRIPTS.md](DEPRECATED_SCRIPTS.md) for complete migration guide.

---

## 📊 Training Configurations

### Default (`config/training/default.yaml`)
- 1M timesteps
- 6 parallel environments
- Standard checkpointing
- WandB logging enabled

### Production (`config/training/production.yaml`)
- 1M timesteps
- More frequent checkpoints (50k intervals)
- More frequent evaluation (5k intervals)
- Extended monitoring

### Quick Test (`config/training/quick_test.yaml`)
- 10k timesteps (for testing)
- 2 parallel environments
- Minimal checkpointing
- WandB disabled

### Usage

```bash
python scripts/train.py --model v1_momentum --training production
python scripts/train.py --model v1_momentum --training quick_test
```

---

## 🌐 Environment Variables

Override configuration via environment variables:

```bash
# Override training steps
NEBULA_TIMESTEPS=50000 python scripts/train.py --model v1_momentum

# Disable WandB logging
NEBULA_USE_WANDB=false python scripts/train.py --model v1_momentum

# Force CPU training
NEBULA_GPU=false python scripts/train.py --model v1_momentum
```

---

## 🏗️ Architecture

### Core Components

1. **ConfigManager** - Loads and validates configurations
2. **ModelRegistry** - Tracks trained models and versions
3. **UnifiedTrainer** - Configuration-driven training
4. **RewardFunctions** - Centralized reward logic

### Training Flow

```
Config Files → ConfigManager → UnifiedTrainer → Model
     ↓              ↓               ↓             ↓
  YAML files    Validation      Training    Save & Register
```

### Evaluation Flow

```
Model Path → ModelEvaluator → Results
              ↓                  ↓
         Load & Test        Metrics & Comparison
```

---

## 🧪 Testing

### Quick Functionality Test

```bash
# 1. List models
python scripts/train.py --list-models

# 2. Get model info
python scripts/train.py --model v1_momentum --info

# 3. Quick training test
python scripts/train.py --model v1_momentum --training quick_test

# 4. Verify training works
ls models/production/v1_momentum_*
```

### Full System Test

```bash
# 1. Train a model
python scripts/train.py --model v1_momentum --training quick_test

# 2. List trained models
python scripts/evaluate.py --list

# 3. Evaluate the model
python scripts/evaluate.py --model models/production/v1_momentum_[timestamp]
```

---

## 📚 Further Documentation

### In-Depth Guides

- **[REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)** - Complete refactoring documentation
  - Detailed architecture
  - Configuration system
  - Creating custom models
  - Reward functions
  - Troubleshooting

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command quick reference
  - All commands with examples
  - Common workflows
  - Tips and tricks

- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - What changed
  - Before/after comparison
  - Files created
  - Impact assessment

### Original Documentation

- `docs/` - Original experiment documentation
- `blog/` - Blog posts about the project
- Production models documentation in `production/`

---

## 🤝 Contributing

### Adding New Models

1. Create config in `config/models/`
2. Document expected performance in metadata
3. Train and evaluate
4. Update documentation

### Adding New Reward Functions

1. Add class to `astra/training/reward_functions.py`
2. Register in `get_reward_function()`
3. Document in config examples

### Improving Documentation

- All documentation in Markdown
- Keep examples up-to-date
- Add troubleshooting tips

---

## ❓ FAQ

**Q: Can I still use the old training scripts?**  
A: Yes, they're still in the repository but deprecated. Use the new system instead.

**Q: Will my existing trained models work?**  
A: Yes! All existing models are fully compatible with the new evaluation script.

**Q: How do I create a new model?**  
A: Just create a YAML config file in `config/models/` and run `python scripts/train.py --model your_model`

**Q: Can I use the new system in production?**  
A: Yes! It's production-ready and already powers the ensemble system.

**Q: Where do I report issues?**  
A: Check `REFACTORING_GUIDE.md` troubleshooting section first, then create an issue.

---

## 📈 Project Status

- **Refactoring:** ✅ Complete
- **Testing:** ✅ Manual verification complete
- **Documentation:** ✅ Comprehensive
- **Production:** ✅ Ready to deploy

---

## 🎓 Learning Resources

1. **Start here:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **Deep dive:** [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)
3. **Migration:** [DEPRECATED_SCRIPTS.md](DEPRECATED_SCRIPTS.md)
4. **Configs:** Look at examples in `config/models/`

---

## 🚦 Getting Started

```bash
# 1. Explore available models
python scripts/train.py --list-models
python scripts/train.py --model v1_momentum --info

# 2. Quick training test
python scripts/train.py --model v1_momentum --training quick_test

# 3. Evaluate
python scripts/evaluate.py --list
python scripts/evaluate.py --model models/production/v1_momentum_[timestamp]

# 4. Create your own model
cp config/models/v1_momentum.yaml config/models/my_model.yaml
# Edit my_model.yaml
python scripts/train.py --model my_model
```

---

## 📞 Support

- **Documentation:** See `docs/` and `*.md` files
- **Examples:** Check `config/models/` for configuration examples
- **Troubleshooting:** See [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)

---

**Happy Training! 🚀**

*The codebase is now maintainable, scalable, and ready for your next experiment!*
