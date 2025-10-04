# Post 2: The Research Phase - When Everything Breaks (And How We Fixed It)

**Series:** Building a Production RL Trading System
**Part 2 of 6**

---

## Day 1: Complete Disaster

It's September 30, 2025, early morning. I've been working on this RL trading system for weeks. The theory is solid. The code compiles. Time to train the model.

I kick off training and grab coffee. Come back 30 minutes later to check on progress.

**Critic Loss: 7.05e+12**

Wait, what?

That's 7 *trillion*. For context, a stable critic loss should be around 0.001. My critic loss was literally exploding into astronomical numbers.

```python
# Training metrics from the disaster
Timesteps: 1,000,000
Critic Loss: 7.05e+12  ← Exploded!
Actor Loss: -8.57e+06   ← Also exploded!
Entropy: 8650           ← Massive randomness
Test Return: 0.24%      ← Basically broken
Equal Weight: 23.53%
Gap: -23.29 pp          ← Catastrophic underperformance
```

The model wasn't just bad. It was completely broken. Let me show you what went wrong.

## The Four Deadly Sins of My Initial Implementation

### Sin #1: Unbounded Observations

```python
# What I did (wrong):
observation = np.concatenate([
    prices,              # Raw prices: [100, 150, 200...]
    returns,             # Raw returns: [0.001, -0.002...]
    moving_averages,     # Unbounded values
    technical_indicators # All over the place
])
# Fed directly to neural network → NaN gradients
```

Neural networks hate unbounded inputs. When you feed raw prices (100, 150, 200) alongside raw returns (0.001, 0.002), the network can't learn stable features. Gradients explode. Training diverges.

### Sin #2: Tiny, Unscaled Rewards

```python
# What I did (wrong):
daily_return = (new_value - old_value) / old_value
reward = daily_return  # ≈ 0.001 magnitude

# What the network sees:
# Action 1: reward = 0.0012
# Action 2: reward = 0.0009
# Network: "These look identical, I'll ignore both"
```

Daily returns are typically 0.1% to 0.5% (0.001 to 0.005 magnitude). RL algorithms struggle with such tiny signals. The network couldn't distinguish between good and bad actions.

### Sin #3: No Normalization

Without VecNormalize:
- Observations had wildly different scales (prices in hundreds, returns in thousandths)
- Rewards had no running statistics for stability
- Network weights updated chaotically
- Training never converged

### Sin #4: Oversized Network

```python
# What I did (wrong):
network_architecture = [512, 512, 256]  # 3 hidden layers, large
# Without proper regularization → overfitting
# With exploding gradients → disaster
```

Big networks are great... when you have stable training. With all the above issues, the large network just amplified the chaos.

## The Breakthrough: VecNormalize

After a day of debugging, I found the solution buried in Stable Baselines3 documentation: **VecNormalize**.

```python
# The fix that changed everything:
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(
    env,
    norm_obs=True,          # Normalize observations to zero mean, unit variance
    norm_reward=True,       # Normalize rewards using running statistics
    clip_obs=10.0,          # Clip observations to [-10, 10] range
    clip_reward=10.0,       # Clip rewards to [-10, 10] range
    gamma=0.99
)
```

**What VecNormalize does:**

1. **Observation Normalization:**
   - Maintains running mean/std for each observation dimension
   - Transforms: `obs_normalized = (obs - running_mean) / (running_std + epsilon)`
   - Result: All observations in similar scale (mean=0, std=1)

2. **Reward Normalization:**
   - Tracks running statistics of rewards
   - Normalizes rewards to consistent scale
   - Helps network learn from reward signals

3. **Clipping:**
   - Prevents outliers from destabilizing training
   - Keeps values in safe [-10, 10] range
   - No more gradient explosions

## Day 2: First Success (Phase 1)

With VecNormalize in place, I made three more critical changes:

### Change 1: Reward Scaling ×100

```python
# Scale rewards into learnable range
def calculate_reward(self, daily_return):
    scaled_return = daily_return * 100  # 0.001 → 0.1

    # Add Sharpe component
    if len(self.returns_history) > 1:
        mean_ret = np.mean(self.returns_history)
        std_ret = np.std(self.returns_history) + 1e-8
        sharpe_component = (mean_ret / std_ret) * 0.1
    else:
        sharpe_component = 0

    # Penalize excessive trading
    turnover_penalty = 0
    if self.turnover > 0.5:
        turnover_penalty = -0.1 * (self.turnover - 0.5)

    reward = scaled_return + sharpe_component + turnover_penalty
    return np.clip(reward, -20, 20)
```

Now rewards were in the 0.01 to 1.0 range—much easier for the network to learn from.

### Change 2: Smaller, More Stable Network

```python
# Phase 1 config (the one that worked):
policy_kwargs = dict(
    net_arch=[256, 256],  # Smaller than before
    activation_fn=nn.ReLU
)

model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,      # Conservative
    batch_size=256,
    gradient_steps=1,        # Stable updates
    buffer_size=200_000,
    verbose=1
)
```

### Change 3: Conservative Hyperparameters

| Parameter | Initial (Failed) | Phase 1 (Success) | Rationale |
|-----------|-----------------|-------------------|-----------|
| Learning Rate | 3e-4 | **1e-4** | Slower, more stable |
| Gradient Steps | 10 | **1** | Fewer updates per step |
| Network Size | [512,512,256] | **[256,256]** | Less overfitting |
| Timesteps | 1M | **600k first** | Test convergence |

## The Results: Night and Day

**Initial Attempt (Disaster):**
```
Test Return: 0.24%
Critic Loss: 7.05e+12
Actor Loss: -8.57e+06
Entropy: 8650
Status: Completely broken
```

**Phase 1 (Success):**
```
Test Return: 24.20%
Test Sharpe: 1.481
Critic Loss: 0.000178  ← Stable!
Actor Loss: -0.777     ← Stable!
Entropy: 0.000421      ← Converged!
vs Equal Weight: +0.67 pp ✅
```

Let me show you what stable training looks like:

### Training Curves - Before VecNormalize:
```
Step 100k:  Critic Loss = 8.2e+8   (exploding)
Step 200k:  Critic Loss = 2.1e+10  (still exploding)
Step 300k:  Critic Loss = 7.0e+12  (completely diverged)
Result: Model unusable
```

### Training Curves - After VecNormalize:
```
Step 100k:  Critic Loss = 0.012    (decreasing)
Step 200k:  Critic Loss = 0.003    (converging)
Step 400k:  Critic Loss = 0.0006   (stable)
Step 600k:  Critic Loss = 0.0002   (fully converged)
Result: Model achieved 24.20% test return
```

The difference is stark. Without normalization, losses exploded exponentially. With normalization, they smoothly decreased and converged.

> **📊 Visual Note:** A before/after training curve chart would powerfully illustrate this transformation—showing the failed run's critic loss shooting to 10^12 versus the successful run's smooth convergence to 0.0002. This is the single most dramatic "debugging win" visual in the series.

## Day 3: Peak Performance (Phase 1 Extended)

Here's the question I asked myself after Phase 1: "Has the model plateaued at 600k steps, or can we do better?"

Looking at the training curves, entropy was still decreasing slightly. The critic loss was stable but not completely flat. Time to test the hypothesis.

**Decision:** Resume training from 600k checkpoint, run to 1M total steps.

```bash
# Resume from checkpoint
python resume_training.py \
  --model models/20251002_175708 \
  --additional_steps 400000 \
  --total_steps 1000000
```

**Phase 1 Extended Results (1M steps):**

```
Test Return: 29.44%
Test Sharpe: 1.847
Mean Reward: 0.394
Entropy: 0.000302  ← Fully converged
Std Return: 0%     ← Deterministic policy

vs Equal Weight: +5.91 pp ✅
vs Min Volatility: -1.69 pp
vs Max Sharpe: -3.43 pp

Status: BEST RESEARCH MODEL ⭐
```

**Performance improvement from 600k → 1M steps:**
- Return: 24.20% → 29.44% (+5.24 pp)
- Sharpe: 1.481 → 1.847 (+0.366)
- Entropy: 0.000421 → 0.000302 (more deterministic)

The extra 400k steps added 5.24 percentage points of return. The model hadn't plateaued at all—it just needed more time to fully converge.

## What About Going Further?

**Q: Why not train to 2M steps?**

Look at the entropy progression:

```
600k steps:  Entropy = 0.000421
1M steps:    Entropy = 0.000302
Improvement: -28% (diminishing)
```

Entropy measures exploration vs. exploitation. At 0.000302, the model is 99.97% deterministic. It's fully exploiting its learned policy. More training would likely:
1. Overfit to the training data
2. Show minimal improvement
3. Waste compute time

We hit the convergence sweet spot at 1M steps.

## The Network Architecture Experiments

Before settling on [256, 256], we tried several configurations:

| Architecture | Test Return | Critic Stability | Decision |
|--------------|-------------|------------------|----------|
| [512, 512, 256] | Diverged | ❌ Unstable | Too large |
| [128, 128] | 21.3% | ✅ Stable | Too small |
| **[256, 256]** | **29.44%** | ✅ **Stable** | **Optimal** ✅ |
| [256, 256, 256] | 27.8% | ⚠️ Slower | Diminishing returns |

**Key insight:** Bigger isn't always better. [256, 256] hit the sweet spot between capacity and stability.

## The Complete Phase 1 Configuration

Here's the exact config that achieved 29.44% test return:

```python
# Environment setup
env = VecNormalize(
    DummyVecEnv([lambda: PortfolioEnv(data, features)]),
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99
)

# Model configuration
model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=1e-4,
    batch_size=256,
    gradient_steps=1,
    tau=0.005,
    gamma=0.99,
    buffer_size=200_000,
    train_freq=1,
    learning_starts=1000,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

# Reward function
def calculate_reward(self, daily_return, turnover):
    scaled_return = daily_return * 100

    sharpe_component = 0
    if len(self.returns_history) > 1:
        sharpe_component = (
            np.mean(self.returns_history) /
            (np.std(self.returns_history) + 1e-8)
        ) * 0.1

    turnover_penalty = 0
    if turnover > 0.5:
        turnover_penalty = -0.1 * (turnover - 0.5)

    reward = scaled_return + sharpe_component + turnover_penalty
    return np.clip(reward, -20, 20)

# Training
model.learn(total_timesteps=1_000_000)
```

## Key Learnings from the Research Phase

### 1. **VecNormalize is Non-Negotiable**

Without it:
- Critic Loss: 10^12 (diverged)
- Test Return: 0.24% (broken)

With it:
- Critic Loss: 0.0002 (stable)
- Test Return: 29.44% (success)

Don't skip normalization. Ever.

### 2. **Reward Scaling Matters**

```python
# Bad: Tiny signals
reward = daily_return  # ≈ 0.001

# Good: Learnable signals
reward = daily_return * 100  # ≈ 0.1
```

RL algorithms need rewards in the 0.01 to 1.0 range to learn effectively.

### 3. **Observation Clipping Prevents Disasters**

```python
clip_obs=10.0  # Critical!
```

A single outlier observation (e.g., a sudden price spike) can destabilize an entire training run. Clipping to [-10, 10] after normalization keeps everything in bounds.

### 4. **Conservative Hyperparameters for Stability**

| Setting | Aggressive | Conservative | Result |
|---------|-----------|--------------|--------|
| LR | 3e-4 | **1e-4** | More stable |
| Grad Steps | 10 | **1** | Less variance |
| Network | [512,512,256] | **[256,256]** | Less overfit |

Start conservative. You can always speed up later. You can't recover from divergence.

### 5. **Convergence Takes Time**

```
600k steps:  24.20% return (good)
1M steps:    29.44% return (great) ← +5.24 pp!
```

Don't stop training early just because loss is stable. Check if entropy is still decreasing. If yes, keep going.

### 6. **Entropy Tells You When to Stop**

```
High entropy (>1.0):    Still exploring, keep training
Medium entropy (0.1-1): Converging, almost there
Low entropy (<0.01):    Fully converged, diminishing returns
```

At 0.000302 entropy, our model was 99.97% deterministic. No point training further.

## Comparing to Classical Baselines

How did our best RL model (29.44%) stack up?

| Strategy | Return | Sharpe | Gap to RL |
|----------|--------|--------|-----------|
| RL Model (Phase 1 Extended) | **29.44%** | **1.847** | — |
| Min Volatility | 31.13% | 1.653 | +1.69 pp |
| Max Sharpe | 32.87% | 1.892 | +3.43 pp |
| Equal Weight | 23.53% | 1.418 | **-5.91 pp** |

Our RL model:
- ✅ Beat Equal Weight by 5.91 pp
- ✅ Came close to Min Vol (only -1.69 pp gap)
- ⚠️ Still behind Max Sharpe by 3.43 pp

**Important context:** Max Sharpe uses the *full test period* to optimize—it knows the future. RL models can only use training data. Closing within 3.43 pp is actually impressive.

## From Research to Production

Phase 1 Extended proved that stable RL trading was possible. The question became: "How do we make this production-ready?"

**Research model limitations:**
1. Trained on 2015-2024, tested on historical data (not real 2025 validation)
2. Single reward function (doesn't adapt to market conditions)
3. No regime awareness (momentum strategy always on)

**Next step:** Retrain on most recent data, validate on real 2025 markets, and test different reward functions for different conditions.

That's where v1, v2, and v2.1 came in—but that's a story for the next post.

## Key Takeaways from Post 2

**What We Learned:**

1. ✅ **VecNormalize is critical** - Prevents gradient explosions, enables stable training
2. ✅ **Reward scaling ×100** - Makes tiny daily returns learnable
3. ✅ **Observation clipping** - Prevents outliers from breaking training
4. ✅ **[256, 256] network** - Sweet spot between capacity and stability
5. ✅ **1M timesteps optimal** - Convergence at 0.0003 entropy, diminishing returns beyond
6. ✅ **Conservative hyperparameters** - Start stable, optimize later
7. ❌ **Don't over-engineer rewards** - Simple worked better than complex

**The Numbers:**

- Day 1 (broken): 0.24% return, 10^12 critic loss
- Day 2 (stable): 24.20% return, 0.0002 critic loss
- Day 3 (optimized): 29.44% return, +5.91 pp vs Equal Weight

From complete disaster to beating benchmarks in 72 hours. All thanks to proper normalization and patient debugging.

---

**Next Post:** The specialists (v1 vs v2) - How we discovered that training data bias matters more than reward function tuning, and why v2 was the only model positive in Q3 2025's correction.

*Continue to Post 3: [v1 vs v2: The Specialists That Changed Everything](#)*

---

**Total words: ~2,950** (target achieved)
