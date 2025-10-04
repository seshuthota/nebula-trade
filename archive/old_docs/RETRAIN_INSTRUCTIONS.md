# Retrain Instructions After Bug Fix

## The Fix is Verified ✅

The environment bug has been fixed and verified. Daily returns are now realistic (0-3% range instead of 8-10%).

## What to Expect Now

### Realistic Performance Targets

With the fixed environment, expect these results:

**Test Period (Feb 2024 - Sep 2025, ~19 months):**
- Individual assets: 6-38% returns
- Classical methods: 23-33% returns
- RL agent (good performance): 30-50% returns
- RL agent (excellent): 40-60% returns

**NOT 3,800%!** That was the bug.

### Training Recommendations

**Quick Test (Recommended First):**
```bash
# 50K timesteps, ~20-30 minutes
python train_portfolio.py --timesteps 50000
```

Expected results:
- RL: 25-45% return
- Classical best: 23-33% return
- RL should be comparable or slightly better

**Standard Training:**
```bash
# 100K timesteps, ~45-60 minutes
python train_portfolio.py --timesteps 100000
```

Expected results:
- RL: 30-50% return
- Better than quick test
- More stable performance

**Extended Training:**
```bash
# 500K timesteps, ~4-6 hours
python train_portfolio.py --timesteps 500000
```

Expected results:
- RL: 35-60% return
- Best performance
- Most stable

## Interpreting New Results

### Good Results Look Like:

**Validation Set:**
```
RL (SAC)         : 42.5% return, Sharpe 1.25
Max Sharpe       : 32.7% return, Sharpe 1.16
Equal Weight     : 26.1% return, Sharpe 1.11
```

**Test Set:**
```
RL (SAC)         : 38.2% return, Sharpe 1.18
Max Sharpe       : 32.7% return, Sharpe 1.16
Equal Weight     : 23.3% return, Sharpe 0.89
```

### What to Look For:

✅ **RL outperforms by 1.2-1.5x**: Good learning
✅ **Sharpe ratio > 1.0**: Risk-adjusted performance
✅ **Test ≈ Validation**: No overfitting
✅ **Returns match asset range**: 25-50% is realistic for 19 months

⚠️ **Warning Signs:**
- RL much worse than classical → training issue
- Test << Validation → overfitting
- Returns > 100% → still has a bug
- Sharpe < 0.5 → not learning well

## Training Strategy

### Phase 1: Quick Verification (Now)
```bash
python train_portfolio.py --timesteps 50000
```
- Verify fix is working
- Check results are reasonable
- Time: ~30 minutes

### Phase 2: Standard Training
```bash
python train_portfolio.py --timesteps 100000
```
- Get decent model
- Compare RL vs classical
- Time: ~1 hour

### Phase 3: Hyperparameter Tuning (Optional)
If Phase 2 results are good, try:
- Different learning rates
- Network architectures
- Reward functions
- Longer training (500K)

## Expected Training Output

### During Training:
```
Episode 100: reward_mean=0.002, reward_std=0.015
Episode 200: reward_mean=0.003, reward_std=0.014
Episode 300: reward_mean=0.004, reward_std=0.013
...
```

Mean rewards around 0.002-0.005 are good (0.2-0.5% average daily return).

### Final Results:
```
Validation Results
  RL Agent (SAC)
  - Mean Return: 38.24%
  - Std Return: 2.15%
  - Mean Reward: 0.0032

  Classical Methods
    Max Sharpe: 32.70% return, Sharpe 1.163
    Equal Weight: 26.14% return, Sharpe 1.110

Test Results
  RL Agent (SAC)
  - Mean Return: 35.67%
  - Std Return: 1.89%

  Classical Methods
    Max Sharpe: 32.70% return, Sharpe 1.163
    Equal Weight: 23.31% return, Sharpe 0.895
```

## Troubleshooting

### If RL is Much Worse Than Classical:

1. **Train Longer**: Try 200K-500K timesteps
2. **Check Learning**: Review training logs for learning progress
3. **Adjust Hyperparameters**: Learning rate, network size
4. **Check Data Quality**: Ensure no NaN or extreme values

### If RL is Only Slightly Better:

This is **normal and good**! RL should be 1.1-1.5x better than classical, not 100x.

### If Returns Are Still Unrealistic (>100%):

1. Run diagnostic: `python diagnose_environment.py --steps 50`
2. Check for any remaining bugs
3. Verify data quality

## What Success Looks Like

**Realistic Success:**
- RL: 35-55% over 19 months
- Outperforms classical by 5-15 percentage points
- Sharpe ratio 1.0-1.3
- Stable across validation and test

**Why This is Good:**
- Beats buy-and-hold by 10-20%
- Better risk-adjusted returns
- Outperforms naive strategies
- Reproducible and deployable

## After Training

Once you have good results:

1. **Evaluate Thoroughly:**
```bash
python evaluate_portfolio.py models/TIMESTAMP/final_model.zip --episodes 50
```

2. **Test on New Data:**
- Download latest market data
- Test model on unseen period

3. **Deploy (Carefully):**
- Start with paper trading
- Monitor performance
- Use proper risk management

## Key Insight

The goal is NOT to achieve 1000% returns (that's a bug). The goal is to **consistently beat classical methods by 10-30%** with good risk-adjusted returns. That's what real portfolio optimization looks like!
