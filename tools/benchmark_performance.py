#!/usr/bin/env python3
"""
Performance benchmark: Standard vs Optimized trainer.
Tests both configurations and reports speedup.
"""

import time
import torch
from multiprocessing import cpu_count
from pathlib import Path

print("=" * 80)
print("PERFORMANCE BENCHMARK: Standard vs Optimized Trainer")
print("=" * 80)

# System info
print("\n🖥️  HARDWARE CONFIGURATION")
print(f"  CPU cores: {cpu_count()}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    print("  ⚠️  No GPU detected - optimized trainer will still use multiprocessing")

print("\n" + "=" * 80)

# Check if data is ready
data_file = Path("data/portfolio_data_processed.csv")
if not data_file.exists():
    print("\n⚠️  Data file not found. Run this first:")
    print("  python run_pipeline.py")
    exit(1)

# Quick benchmark parameters
BENCHMARK_STEPS = 5000
N_PARALLEL_ENVS = 6

print(f"\n📊 BENCHMARK CONFIGURATION")
print(f"  Training steps: {BENCHMARK_STEPS:,}")
print(f"  Parallel environments (optimized): {N_PARALLEL_ENVS}")
print(f"  Standard trainer envs: 1")

print("\n" + "=" * 80)

# Import trainers
try:
    from astra.rl_framework.trainer import PortfolioTrainer
    from astra.rl_framework.trainer_optimized import PortfolioTrainerOptimized
    
    print("\n✅ Both trainers available")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    exit(1)

# Test 1: Standard Trainer
print("\n" + "=" * 80)
print("TEST 1: STANDARD TRAINER (Single CPU, No GPU)")
print("=" * 80)

try:
    print("\nInitializing standard trainer...")
    start_time = time.time()
    
    trainer_std = PortfolioTrainer(
        config_path="config/portfolio.yaml",
        data_path=str(data_file)
    )
    
    init_time_std = time.time() - start_time
    print(f"  Initialization: {init_time_std:.2f}s")
    
    print(f"\nTraining for {BENCHMARK_STEPS:,} steps...")
    train_start = time.time()
    
    trainer_std.train(
        total_timesteps=BENCHMARK_STEPS,
        log_interval=100,
        save_path="benchmark_models/standard"
    )
    
    train_time_std = time.time() - train_start
    total_time_std = init_time_std + train_time_std
    steps_per_sec_std = BENCHMARK_STEPS / train_time_std
    
    print(f"\n📈 STANDARD TRAINER RESULTS:")
    print(f"  Training time: {train_time_std:.2f}s")
    print(f"  Total time: {total_time_std:.2f}s")
    print(f"  Steps/second: {steps_per_sec_std:.1f}")
    
except Exception as e:
    print(f"\n❌ Standard trainer failed: {e}")
    train_time_std = None

# Test 2: Optimized Trainer
print("\n" + "=" * 80)
print("TEST 2: OPTIMIZED TRAINER (Multi-CPU + GPU)")
print("=" * 80)

try:
    print("\nInitializing optimized trainer...")
    start_time = time.time()
    
    trainer_opt = PortfolioTrainerOptimized(
        config_path="config/portfolio.yaml",
        data_path=str(data_file),
        n_envs=N_PARALLEL_ENVS,
        use_gpu=True
    )
    
    init_time_opt = time.time() - start_time
    print(f"  Initialization: {init_time_opt:.2f}s")
    
    print(f"\nTraining for {BENCHMARK_STEPS:,} steps...")
    train_start = time.time()
    
    trainer_opt.train(
        total_timesteps=BENCHMARK_STEPS,
        log_interval=100,
        save_path="benchmark_models/optimized",
        eval_freq=0  # Disable eval for fair comparison
    )
    
    train_time_opt = time.time() - train_start
    total_time_opt = init_time_opt + train_time_opt
    steps_per_sec_opt = BENCHMARK_STEPS / train_time_opt
    
    print(f"\n📈 OPTIMIZED TRAINER RESULTS:")
    print(f"  Training time: {train_time_opt:.2f}s")
    print(f"  Total time: {total_time_opt:.2f}s")
    print(f"  Steps/second: {steps_per_sec_opt:.1f}")
    
except Exception as e:
    print(f"\n❌ Optimized trainer failed: {e}")
    train_time_opt = None

# Comparison
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

if train_time_std and train_time_opt:
    speedup = train_time_std / train_time_opt
    steps_speedup = steps_per_sec_opt / steps_per_sec_std
    
    print(f"\n⚡ SPEEDUP METRICS:")
    print(f"  Training time speedup: {speedup:.2f}x")
    print(f"  Steps/second speedup: {steps_speedup:.2f}x")
    
    time_saved = train_time_std - train_time_opt
    print(f"  Time saved: {time_saved:.1f}s ({time_saved/60:.1f} min)")
    
    # Projection for larger training
    print(f"\n📊 PROJECTED TIME FOR 100K STEPS:")
    proj_std = (train_time_std / BENCHMARK_STEPS) * 100000
    proj_opt = (train_time_opt / BENCHMARK_STEPS) * 100000
    proj_saved = proj_std - proj_opt
    
    print(f"  Standard trainer: {proj_std/60:.1f} min")
    print(f"  Optimized trainer: {proj_opt/60:.1f} min")
    print(f"  Time saved: {proj_saved/60:.1f} min ({speedup:.1f}x faster)")
    
    print(f"\n📊 PROJECTED TIME FOR 500K STEPS:")
    proj_std_500k = (train_time_std / BENCHMARK_STEPS) * 500000
    proj_opt_500k = (train_time_opt / BENCHMARK_STEPS) * 500000
    proj_saved_500k = proj_std_500k - proj_opt_500k
    
    print(f"  Standard trainer: {proj_std_500k/3600:.1f} hours")
    print(f"  Optimized trainer: {proj_opt_500k/60:.1f} min")
    print(f"  Time saved: {proj_saved_500k/60:.1f} min ({speedup:.1f}x faster)")
    
    # Visual comparison
    print(f"\n📊 VISUAL COMPARISON:")
    bar_std = "█" * int(speedup * 10)
    bar_opt = "█" * 10
    
    print(f"  Standard:  {bar_std}  {train_time_std:.1f}s")
    print(f"  Optimized: {bar_opt}  {train_time_opt:.1f}s")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    if speedup > 3:
        print(f"  ✅ Excellent speedup! Always use optimized trainer.")
        print(f"  You're saving {speedup:.1f}x time on every training run!")
    elif speedup > 1.5:
        print(f"  ✅ Good speedup! Use optimized trainer for longer runs.")
    else:
        print(f"  ⚠️  Limited speedup. Check GPU utilization.")
else:
    print("\n⚠️  Could not complete comparison due to errors.")

print("\n" + "=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)

print("\n📝 Next steps:")
print("  1. Use optimized trainer: python train_portfolio.py --timesteps 100000")
print("  2. Monitor GPU: watch -n 1 nvidia-smi")
print("  3. View TensorBoard: tensorboard --logdir ./tensorboard_logs/")
print("\n" + "=" * 80)
