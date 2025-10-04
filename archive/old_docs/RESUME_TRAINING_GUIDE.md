# Resume Training Guide

## Your Situation

You accidentally stopped training at **400,000 steps** (out of 1,000,000).

**Good news:** Checkpoints were automatically saved! ✅

## Available Checkpoints

Latest run: `models/20250930_225350/`

Checkpoints saved:
- ✅ **portfolio_model_400000_steps.zip** (your latest)
- portfolio_model_320000_steps.zip
- portfolio_model_240000_steps.zip
- portfolio_model_160000_steps.zip
- portfolio_model_80000_steps.zip
- best_model.zip (best validation performance)

## How to Resume

### Option 1: Resume from 400K (Recommended)

```bash
python resume_training.py \
    models/20250930_225350/portfolio_model_400000_steps.zip \
    --steps 600000
```

This will:
- Load the 400K checkpoint
- Train for 600K MORE steps
- Total: 1,000,000 steps (as planned)
- Time: ~2 hours

### Option 2: Use Best Model

If you want to resume from the best performing checkpoint:

```bash
python resume_training.py \
    models/20250930_225350/best_model.zip \
    --steps 600000
```

### Option 3: Quick Resume (Test)

To test that resume works:

```bash
python resume_training.py \
    models/20250930_225350/portfolio_model_400000_steps.zip \
    --steps 10000
```

## What Happens When You Resume

1. **Loads checkpoint** - Restores model weights and training state
2. **Loads same data** - Uses original training data for consistency
3. **Continues training** - Picks up where it left off
4. **Saves to new directory** - Won't overwrite original
5. **Creates checkpoints** - New checkpoints every 10K steps

## Expected Timeline

| Action | Duration | Total Steps |
|--------|----------|-------------|
| Already trained | ✅ Done | 400,000 |
| Resume training | ~2 hours | +600,000 |
| **Total** | **~2 hours** | **1,000,000** |

## Output Structure

```
models/20251001_HHMMSS_resumed/
├── RESUME_INFO.txt          # Info about resume
├── final_model.zip          # Final 1M step model
├── portfolio_model_*_steps.zip  # New checkpoints
└── best_model.zip          # Best from resumed training
```

## Advantages of Resuming

✅ **Don't lose progress** - 400K steps already done  
✅ **Save time** - Only 2 hours instead of 3  
✅ **Same trajectory** - Continues learning from where it stopped  
✅ **Checkpoints preserved** - Original checkpoints untouched  

## Important Notes

### 1. GPU Usage
The resume script automatically uses GPU (same as original training).

Monitor with:
```bash
watch -n 1 nvidia-smi
```

### 2. TensorBoard
New logs will be in `tensorboard_logs/`. View with:
```bash
tensorboard --logdir ./tensorboard_logs/
```

### 3. Data Consistency
The script tries to use the same training data as the original run:
- Looks for: `results/20250930_225350/train_data_processed.csv`
- Falls back to: `data/portfolio_data_processed.csv`

This ensures consistent learning.

### 4. Checkpoint Frequency
New checkpoints saved every **10,000 steps** during resumed training.

## Comparison: Resume vs Restart

| Aspect | Resume (Recommended) | Restart from Scratch |
|--------|---------------------|---------------------|
| Time | ~2 hours | ~3 hours |
| Progress lost | None | 400K steps |
| Learning continuity | Smooth | Restart learning |
| Total steps | 1M | 1M |
| Recommendation | ✅ YES | ❌ Unnecessary |

## Troubleshooting

### Checkpoint not found
```bash
# List available checkpoints
ls -lh models/20250930_225350/*.zip
```

### Out of memory
If you get OOM errors:
```bash
# Use fewer parallel environments (edit resume_training.py)
# Change: n_envs=None to n_envs=4
```

### Different data
If you want to use different data:
```bash
# Edit resume_training.py
# Change: data_path = "your_new_data.csv"
```

## Verification

After resuming, check that it's working:

1. **GPU Usage**: Should be 80-90%
   ```bash
   nvidia-smi
   ```

2. **Training Progress**: Should see increasing steps
   ```bash
   # Watch console output
   INFO: Steps: 410000 | Steps/sec: 90
   INFO: Steps: 420000 | Steps/sec: 89
   ```

3. **Checkpoints**: New files should appear
   ```bash
   ls -lh models/*_resumed/
   ```

## Quick Commands

### Resume from 400K
```bash
python resume_training.py models/20250930_225350/portfolio_model_400000_steps.zip --steps 600000
```

### Resume from best
```bash
python resume_training.py models/20250930_225350/best_model.zip --steps 600000
```

### Check available checkpoints
```bash
ls -lh models/20250930_225350/
```

### Monitor during training
```bash
# Terminal 1: Run training
python resume_training.py models/20250930_225350/portfolio_model_400000_steps.zip --steps 600000

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: TensorBoard
tensorboard --logdir ./tensorboard_logs/
```

## What to Expect

After resuming and completing to 1M steps:

**Expected Performance:**
- Validation: 15-22%
- Test: 20-28%
- Better than 500K results (10%)
- Competitive with classical (23-33%)

**If results are good:**
- You're done! Use the final model
- Compare with classical methods
- Deploy or further tune

**If results are still poor:**
- Try training to 1.5M or 2M
- Adjust Sharpe weight (increase from 0.01 to 0.05)
- Try different reward functions

## Example Session

```bash
$ python resume_training.py models/20250930_225350/portfolio_model_400000_steps.zip --steps 600000

INFO: ================================================================================
INFO: RESUMING TRAINING FROM CHECKPOINT
INFO: ================================================================================
INFO: Checkpoint: models/20250930_225350/portfolio_model_400000_steps.zip
INFO: Additional steps: 600,000

INFO: Initializing trainer environment...
INFO: Hardware Configuration:
INFO:   CPU cores: 12
INFO:   Parallel environments: 8
INFO:   Device: cuda
INFO:   GPU: NVIDIA GeForce GTX 1660 SUPER

INFO: Loading checkpoint: portfolio_model_400000_steps.zip
INFO: ✓ Checkpoint loaded successfully!

INFO: ================================================================================
INFO: CONTINUING TRAINING
INFO: ================================================================================

[Training progress bar appears]
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 600,000/600,000  [1:54:32 < 0:00:00, 87 it/s]

INFO: ================================================================================
INFO: TRAINING COMPLETED!
INFO: ================================================================================
INFO: Models saved to: models/20251001_003045_resumed
```

## Summary

You have everything needed to resume:

1. ✅ **Checkpoint exists** at 400K steps
2. ✅ **Resume script ready** (`resume_training.py`)
3. ✅ **Simple command** to continue
4. ⏱️ **2 hours** to complete remaining 600K steps

**Run this now:**

```bash
python resume_training.py \
    models/20250930_225350/portfolio_model_400000_steps.zip \
    --steps 600000
```

You'll reach 1M steps in ~2 hours! 🚀
