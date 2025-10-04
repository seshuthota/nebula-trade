# Deprecated Training Scripts

The following scripts have been replaced by the unified configuration-driven training system. They are kept for reference but **should not be used** for new training runs.

## Deprecated Training Scripts

### Individual Model Training Scripts

❌ **`train_v4_historical.py`**
- **Replaced by:** `python scripts/train.py --model v4_historical`
- **Config:** `config/models/v4_historical.yaml`

❌ **`train_v5_tuned_historical.py`**
- **Replaced by:** `python scripts/train.py --model v5_tuned`
- **Config:** `config/models/v5_tuned.yaml`

❌ **`train_balanced_model.py`**
- **Replaced by:** `python scripts/train.py --model v2.1_balanced`
- **Config:** `config/models/v2.1_balanced.yaml`

❌ **`train_defensive_model.py`**
- **Replaced by:** `python scripts/train.py --model v2_defensive`
- **Config:** `config/models/v2_defensive.yaml`

❌ **`train_production_model.py`**
- **Replaced by:** `python scripts/train.py --model v1_momentum`
- **Config:** `config/models/v1_momentum.yaml`

❌ **`train_stage2_balanced.py`**
- **Replaced by:** `python scripts/train.py --model v3_stage2`
- **Config:** `config/models/v3_stage2.yaml`

❌ **`train_portfolio.py`**
- **Replaced by:** `python scripts/train.py --model [any_model]`
- **Note:** Generic training pipeline, now unified in `scripts/train.py`

### Deprecated Evaluation Scripts

❌ **`evaluate_v4.py`**
- **Replaced by:** `python scripts/evaluate.py --model [path]`

❌ **`evaluate_portfolio.py`**
- **Replaced by:** `python scripts/evaluate.py --model [path]`

❌ **`evaluate_resumed_model.py`**
- **Replaced by:** `python scripts/evaluate.py --model [path]`

## Why These Scripts Were Deprecated

1. **Code Duplication:** Each script contained ~400 lines with 80%+ duplication
2. **Hard-coded Parameters:** Model configs were embedded in code
3. **Difficult to Maintain:** Bug fixes needed to be applied to 7 different files
4. **Hard to Experiment:** Creating a new model required copying and modifying 400+ lines

## Benefits of the New System

✅ **Single Training Script:** All models use `scripts/train.py`
✅ **Configuration-Driven:** Model parameters in YAML files
✅ **Easy to Maintain:** Fix bugs once, not 7 times
✅ **Easy to Experiment:** Create config file instead of copying code
✅ **Better Organization:** Clear separation of config and code

## Migration Examples

### Example 1: Training v5 Model

**Old Way:**
```bash
python train_v5_tuned_historical.py
```

**New Way:**
```bash
python scripts/train.py --model v5_tuned
```

### Example 2: Training with Custom Parameters

**Old Way:**
```python
# Edit train_v5_tuned_historical.py
# Find line 58: class V5TunedHistoricalTrainer:
# Modify hard-coded parameters
# Save and run
python train_v5_tuned_historical.py
```

**New Way:**
```bash
# Edit config/models/v5_tuned.yaml
# Change parameters in YAML
# Run with new config
python scripts/train.py --model v5_tuned
```

### Example 3: Creating a New Model Variant

**Old Way:**
```bash
# Copy entire script
cp train_v5_tuned_historical.py train_v6_experimental.py
# Edit 417 lines
# Debug new script
python train_v6_experimental.py
```

**New Way:**
```bash
# Copy config file
cp config/models/v5_tuned.yaml config/models/v6_experimental.yaml
# Edit 50 lines of YAML
# Run
python scripts/train.py --model v6_experimental
```

## Timeline for Removal

- **Phase 1 (Current):** Scripts marked as deprecated, kept for reference
- **Phase 2 (Next release):** Scripts moved to `archive/deprecated_scripts/`
- **Phase 3 (Future):** Scripts may be removed entirely

## What to Do If You Need the Old Scripts

1. **For reference:** They're still in the repository root
2. **For the same functionality:** Use the new unified system
3. **If you really need them:** They can be found in git history

## Questions?

- See `REFACTORING_GUIDE.md` for complete migration guide
- See `QUICK_REFERENCE.md` for command examples
- Use `python scripts/train.py --help` for command help

## Summary

**Don't use these deprecated scripts!**

Instead, use:
```bash
# Training
python scripts/train.py --model [model_name]

# Evaluation  
python scripts/evaluate.py --model [model_path]

# Or use the CLI
python cli/main.py train --model [model_name]
python cli/main.py evaluate --model [model_path]
```

All the functionality is preserved and improved in the new system!
