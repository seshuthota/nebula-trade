# Production Journey - Defensive Model Development

**Complete timeline of production deployment experiments and defensive model development**

---

## Executive Summary

**Timeline:** October 3-4, 2025 (2 days intensive experiments)
**Models Trained:** 3 production models (v1, v2, v2.1)
**Result:** Identified training data bias as root cause of bear market failure
**Recommendation:** Stage 2 - Training Data Rebalancing required
**Status:** Production experiments complete ✅, awaiting Stage 2 implementation

---

## Background: Transition to Production

**Research Phase Complete:**
- Best model: Phase 1 Extended (29.44% on historical test set)
- Decision: Retrain on recent data for production deployment
- Goal: Beat Equal Weight baseline on current market conditions

**Production Strategy:**
- 90/10 split (vs 70/15/15 research) - use all available data
- Validate on most recent period (Sept 2024 - Sept 2025)
- Test on multiple market conditions (bull, bear, mixed)

---

## Model v1: Momentum (Baseline Production Model)

### Date: October 3, 2025

### Configuration
```yaml
Model Type: SAC
Network: [256, 256]
Features: 173 (Phase 1 config)
Training Steps: 1,000,000
Data Split: 90/10 (2,363 train / 263 validation)
Training Period: Feb 2015 - Sept 2024
Validation Period: Sept 2024 - Sept 2025 (most recent)
VecNormalize: Yes
```

### Reward Function
```python
# Simple returns-focused (Phase 1 original)
scaled_return = daily_return * 100
sharpe_component = (mean_ret / std_ret) * 0.1  # Small weight
reward = scaled_return + sharpe_component
reward = np.clip(reward, -20, 20)

# Turnover penalty
if turnover > 0.5:
    reward += -0.1 * (turnover - 0.5)
```

**Characteristics:** Momentum-following, return-focused

### Results

**Validation (Sept 2024 - Sept 2025):**
```
Return: 14.62%
vs Equal Weight: +4.40 pp ✅
vs Min Volatility: +0.55 pp ✅
vs Max Sharpe: +0.15 pp ✅
Mean Reward: 0.413
Status: PASS (beats all baselines)
```

**Paper Trading - Bull Market (Jan-Apr 2025, 81 days):**
```
Return: +16.11%
vs Equal Weight: +5.47 pp ✅
Sharpe Ratio: 2.141
Max Drawdown: -7.11%
Decision: GO ✅
```

**Paper Trading - Bear Market (July-Sept 2025, 63 days):**
```
Return: -6.81%
vs Equal Weight: -3.66 pp ❌
Sharpe Ratio: -1.654
Max Drawdown: -9.71%
Decision: NO-GO ❌
```

**Paper Trading - Mixed (Jan-Sept 2025, 186 days):**
```
Return: +10.10%
vs Equal Weight: +0.73 pp ✅
Sharpe Ratio: 0.738
Max Drawdown: -9.33%
Decision: CONDITIONAL ⚠️
```

### Analysis

**Strengths:**
- Excellent bull market performance (+16.11%)
- Strong validation results (beats all baselines)
- High Sharpe in up markets (2.141)
- Captures momentum effectively

**Weaknesses:**
- Poor bear market protection (-6.81%)
- Loses to Equal Weight in downturns (-3.66 pp)
- Amplifies losses due to momentum following
- Overall edge marginal (+0.73 pp)

**Model Location:** `production/models/v1_20251003_131020/`

**Status:** Marginal production viability ⚠️

---

## Model v2: Defensive (Over-Tuned Protection)

### Date: October 3, 2025

### Motivation
v1 failed in bear markets (-6.81%). Need defensive model that:
- Protects downside in bear markets
- Reduces drawdowns
- Maintains reasonable upside

### Configuration
```yaml
Model Type: SAC
Network: [256, 256]
Features: 173 (same as v1)
Training Steps: 1,000,000
Data Split: 90/10 (same as v1)
Training Period: Feb 2015 - Sept 2024
Validation Period: Sept 2024 - Sept 2025
VecNormalize: Yes
```

### Reward Function (Defensive)
```python
# COMPONENT 1: Asymmetric Loss Aversion
if daily_return >= 0:
    return_component = daily_return * 100
else:
    return_component = daily_return * 200  # Losses hurt 2x more

# COMPONENT 2: Sharpe Ratio (high weight)
sharpe_component = (mean_ret / std_ret) * 5.0
sharpe_component = np.clip(sharpe_component, -10, 10)

# COMPONENT 3: Drawdown Penalty (heavy)
drawdown_pct = (peak_value - current_value) / peak_value
drawdown_penalty = -200 * drawdown_pct  # -20 points for 10% drawdown

# COMPONENT 4: Volatility Penalty
if volatility > 0.02:
    volatility_penalty = -100 * (volatility - 0.02)

# COMPONENT 5: Turnover Penalty (increased)
if turnover > 0.5:
    turnover_penalty = -5.0 * (turnover - 0.5)

# COMBINE
reward = (
    return_component * 0.4 +     # 40% returns
    sharpe_component * 0.3 +     # 30% Sharpe
    drawdown_penalty * 0.2 +     # 20% drawdown protection
    volatility_penalty * 0.05 +  # 5% stability
    turnover_penalty * 0.05      # 5% turnover
)
reward = np.clip(reward, -40, 25)
```

**Characteristics:** Heavily defensive, loss-averse

### Results

**Validation (Sept 2024 - Sept 2025):**
```
Return: 7.87%
vs Equal Weight: -2.35 pp ❌
vs Min Volatility: -6.20 pp ❌
vs Max Sharpe: -6.60 pp ❌
Mean Reward: -5.34 (negative = heavy penalties)
Status: FAIL (below all baselines)
```

**Paper Trading - Bull Market (Jan-Apr 2025):**
```
Return: +7.36%
vs Equal Weight: -3.28 pp ❌
Sharpe Ratio: 1.070
Max Drawdown: -8.49%
Decision: NO-GO ❌
```

**Paper Trading - Bear Market (July-Sept 2025):**
```
Return: +0.34% ✅ (ONLY POSITIVE MODEL!)
vs Equal Weight: +3.49 pp ✅
Sharpe Ratio: 0.376
Max Drawdown: -4.71% (best!)
Decision: CONDITIONAL ⚠️
```

**Paper Trading - Mixed (Jan-Sept 2025):**
```
Return: +6.17%
vs Equal Weight: -3.20 pp ❌
Sharpe Ratio: 0.504
Max Drawdown: -8.49%
Decision: NO-GO ❌
```

### Analysis

**Strengths:**
- **EXCELLENT bear market protection** (+0.34% vs -6.81% for v1)
- Only model that stayed positive in downturn
- Smallest drawdowns across all periods (-4.71%)
- Effective downside risk management

**Weaknesses:**
- **TOO CONSERVATIVE in bull markets** (+7.36% vs +16.11% for v1)
- Sacrificed 8.75 pp upside for bear protection
- Below Equal Weight overall (-3.20 pp)
- Negative mean reward (-5.34) indicates over-penalization

**Key Insight:**
- Defensive penalties work BUT are too strong
- Model learned: "avoid losses" > "maximize returns"
- Trade-off unfavorable: Lose 9pp in bull to save 7pp in bear

**Model Location:** `production/models/v2_defensive_20251003_212109/`

**Status:** Not production viable (too conservative) ❌

---

## Model v2.1: Balanced (Tuned Defense)

### Date: October 3-4, 2025

### Motivation
v2 proved defensive penalties work but are over-tuned. Goal:
- Recover bull market performance (target: 12-14%)
- Maintain some bear market protection (target: +1-3%)
- Beat Equal Weight overall

### Configuration
```yaml
Model Type: SAC
Network: [256, 256]
Features: 173 (same as v1, v2)
Training Steps: 1,000,000
Data Split: 90/10 (same as v1, v2)
Training Period: Feb 2015 - Sept 2024
Validation Period: Sept 2024 - Sept 2025
VecNormalize: Yes
```

### Reward Function (Tuned - Balanced)
```python
# COMPONENT 1: Asymmetric Loss Aversion (REDUCED)
if daily_return >= 0:
    return_component = daily_return * 100
else:
    return_component = daily_return * 150  # 1.5x (vs 2x in v2)

# COMPONENT 2: Sharpe Ratio (REDUCED)
sharpe_component = (mean_ret / std_ret) * 4.0  # 4.0 vs 5.0
sharpe_component = np.clip(sharpe_component, -8, 8)

# COMPONENT 3: Drawdown Penalty (REDUCED)
drawdown_pct = (peak_value - current_value) / peak_value
drawdown_penalty = -100 * drawdown_pct  # -10 for 10% DD (vs -20)

# COMPONENT 4 & 5: Unchanged
volatility_penalty = ...  # Same as v2
turnover_penalty = ...    # Same as v2

# COMBINE (REBALANCED WEIGHTS)
reward = (
    return_component * 0.6 +     # 60% returns (UP from 40%)
    sharpe_component * 0.2 +     # 20% Sharpe (DOWN from 30%)
    drawdown_penalty * 0.1 +     # 10% drawdown (DOWN from 20%)
    volatility_penalty * 0.05 +  # 5% (same)
    turnover_penalty * 0.05      # 5% (same)
)
reward = np.clip(reward, -30, 20)  # Less restrictive bounds
```

**Key Changes from v2:**
- Loss aversion: 2x → 1.5x
- Return weight: 40% → 60%
- Sharpe weight: 30% → 20%
- Drawdown weight: 20% → 10%
- Reward bounds: (-40,25) → (-30,20)

**Characteristics:** Balanced return and risk focus

### Results

**Validation (Sept 2024 - Sept 2025):**
```
Return: 11.83%
vs Equal Weight: +1.61 pp ✅
vs Min Volatility: -2.24 pp
vs Max Sharpe: -2.64 pp
Mean Reward: -5.73 (still negative but less)
Status: PASS (beats Equal Weight)
```

**Paper Trading - Bull Market (Jan-Apr 2025):**
```
Return: +15.13% ✅ (target: 12-14%, exceeded!)
vs Equal Weight: +4.49 pp ✅
Sharpe Ratio: 2.142
Max Drawdown: -7.04%
Decision: GO ✅
```

**Paper Trading - Bear Market (July-Sept 2025):**
```
Return: -6.31% ❌ (target: +1-3%, FAILED)
vs Equal Weight: -3.16 pp ❌
Sharpe Ratio: -1.760
Max Drawdown: -9.37%
Decision: NO-GO ❌
```

**Paper Trading - Mixed (Jan-Sept 2025):**
```
Return: +9.87%
vs Equal Weight: +0.49 pp ✅
Sharpe Ratio: 0.748
Max Drawdown: -9.38%
Decision: CONDITIONAL ⚠️
```

### Analysis

**Strengths:**
- Successfully recovered bull market performance (+15.13%, nearly matches v1's +16.11%)
- Beats Equal Weight on validation (+1.61 pp)
- Strong bull/mixed period performance
- Balanced reward weights work in up markets

**Weaknesses:**
- **FAILED to maintain bear market protection** (-6.31%, similar to v1's -6.81%)
- Lost defensive benefits from v2
- Reward tuning couldn't overcome training data bias
- Bear market performance essentially identical to v1

**Key Insight:**
- Reward tuning can recover upside OR maintain downside protection, **NOT BOTH**
- Reducing penalties → recovered bull performance ✅
- But also → lost bear protection ❌
- **Root cause is training data, not reward function**

**Model Location:** `production/models/v2.1_balanced_20251003_233318/`

**Status:** Marginal production viability ⚠️ (same as v1)

---

## Complete Model Comparison

### Validation Results (Sept 2024 - Sept 2025)

| Model | Return | vs Equal Weight | Decision |
|-------|--------|-----------------|----------|
| **v1 (Momentum)** | 14.62% | +4.40 pp | ✅ PASS |
| **v2 (Defensive)** | 7.87% | -2.35 pp | ❌ FAIL |
| **v2.1 (Balanced)** | 11.83% | +1.61 pp | ✅ PASS |
| Equal Weight | 10.22% | — | Baseline |

### Paper Trading - Bull Market (Jan-Apr 2025)

| Model | Return | vs Equal Weight | Sharpe | Drawdown |
|-------|--------|-----------------|--------|----------|
| **v1** | **+16.11%** | +5.47 pp ✅ | 2.141 | -7.11% |
| v2 | +7.36% | -3.28 pp ❌ | 1.070 | -8.49% |
| **v2.1** | **+15.13%** | +4.49 pp ✅ | 2.142 | -7.04% |
| Equal Weight | +10.64% | — | 1.817 | — |

**Winner:** v1 (+16.11%) and v2.1 (+15.13%) nearly tied

### Paper Trading - Bear Market (July-Sept 2025)

| Model | Return | vs Equal Weight | Sharpe | Drawdown |
|-------|--------|-----------------|--------|----------|
| v1 | -6.81% | -3.66 pp ❌ | -1.654 | -9.71% |
| **v2** | **+0.34%** | +3.49 pp ✅ | 0.376 | **-4.71%** |
| v2.1 | -6.31% | -3.16 pp ❌ | -1.760 | -9.37% |
| Equal Weight | -3.15% | — | -1.208 | — |

**Winner:** v2 (+0.34%) - ONLY model that stayed positive!

### Paper Trading - Mixed (Jan-Sept 2025)

| Model | Return | vs Equal Weight | Sharpe | Drawdown |
|-------|--------|-----------------|--------|----------|
| **v1** | **+10.10%** | +0.73 pp ✅ | 0.738 | -9.33% |
| v2 | +6.17% | -3.20 pp ❌ | 0.504 | -8.49% |
| **v2.1** | **+9.87%** | +0.49 pp ✅ | 0.748 | -9.38% |
| Equal Weight | +9.38% | — | 0.889 | — |

**Winner:** v1 (+10.10%) slightly ahead of v2.1 (+9.87%)

---

## Root Cause Analysis

### The Fundamental Problem: Training Data Bias

**Training Period:** Feb 2015 - Sept 2024 (10 years)

**Market Composition Analysis:**
```
Bull/Neutral markets: ~80% of training data
  - 2015-2018: Recovery and growth
  - 2019: Strong year
  - 2020-2021: Post-COVID rally
  - 2022-2023: Mixed but mostly up
  - 2024: Growth

Bear markets: ~20% of training data
  - 2020 Q1: COVID crash (brief)
  - 2022 Q2-Q3: Inflation fears (moderate)
  - Few other significant downturns
```

### What Models Learned

**All Three Models (v1, v2, v2.1) Learned:**
1. **"Follow momentum when rising"** (dominant 80% pattern)
2. **"Reduce risk slightly when falling"** (insufficient 20% data)
3. **Net behavior:** Momentum-following with weak defense

**Evidence:**
- v1: -6.81% in bear market (momentum amplifies losses)
- v2: +0.34% in bear market (heavy defense works!)
- v2.1: -6.31% in bear market (tuned defense → reverted to momentum)

### Why Reward Tuning Failed

**v2 (Heavy Defense):**
```
Penalties so strong → model avoids all risk
Bull: Misses opportunities → +7.36% (poor)
Bear: Protects well → +0.34% (excellent)
Overall: Too conservative → +6.17% (poor)
```

**v2.1 (Balanced):**
```
Penalties reduced → model takes more risk
Bull: Captures opportunities → +15.13% (excellent)
Bear: Insufficient defense → -6.31% (poor, same as v1!)
Overall: Marginal edge → +9.87% (okay)
```

**Conclusion:**
- **Can't have both** with reward tuning alone
- Training data (80/20) dominates learned behavior
- Defensive penalties only work if strong enough (v2)
- But strong enough = too conservative overall

### Trade-off Mathematics

**v2 (Defensive) Trade-off:**
```
Bull market sacrifice: 16.11% - 7.36% = -8.75 pp
Bear market gain: 0.34% - (-6.81%) = +7.15 pp
Net (assuming 70% bull, 30% bear):
  0.7 × (-8.75) + 0.3 × (+7.15) = -3.98 pp
Actual observed overall impact: -3.93 pp ✓ (matches!)
```

**v2.1 (Balanced) Result:**
```
Bull recovery: 15.13% - 7.36% = +7.77 pp ✅
Bear loss: -6.31% - 0.34% = -6.65 pp ❌
Net: Model reverted to v1-like behavior in bear markets
```

**Insight:** Reward function cannot overcome 4:1 data imbalance

---

## Key Technical Learnings

### What Worked

1. **Asymmetric Loss Aversion**
   - 2x penalty (v2): Effective but too strong
   - 1.5x penalty (v2.1): More balanced but insufficient for bears
   - **Conclusion:** Works in principle, needs data support

2. **Drawdown Penalties**
   - 20% weight (v2): Heavy protection, limited drawdowns (-4.71%)
   - 10% weight (v2.1): More upside, but lost protection
   - **Conclusion:** Effective defensive mechanism

3. **Sharpe Ratio Component**
   - 30% weight (v2): Risk-focused, too conservative
   - 20% weight (v2.1): Better balanced
   - **Conclusion:** Important for risk-adjusted returns

### What Didn't Work

1. **Reward Tuning to Fix Data Bias**
   - Cannot overcome 80/20 bull/bear imbalance
   - Either too defensive (v2) or insufficient defense (v2.1)
   - **Conclusion:** Data rebalancing required

2. **One Model for All Conditions**
   - v1: Good bull, bad bear
   - v2: Bad bull, good bear
   - v2.1: Good bull, bad bear (same as v1!)
   - **Conclusion:** Ensemble or Stage 2 needed

### Critical Insight

**The Real Problem:**
```
Training Data:
  2015-2024: 80% bull/neutral, 20% bear
  Model learns: "Momentum works most of the time"

Bear Market Test:
  July-Sept 2025: Strong bear period
  Model fails: Only saw 20% bear examples

Solution Options:
  1. Oversample bear periods → balanced 50/50 data
  2. Use ensemble with regime detection
  3. Accept limitation and use circuit breakers
```

---

## Model Artifacts

### v1 (Momentum)
```
production/models/v1_20251003_131020/
├── final_model.zip
├── vec_normalize.pkl
├── validation_report.json
└── metadata.json
```

### v2 (Defensive)
```
production/models/v2_defensive_20251003_212109/
├── final_model.zip
├── vec_normalize.pkl
├── validation_report.json
└── metadata.json
```

### v2.1 (Balanced)
```
production/models/v2.1_balanced_20251003_233318/
├── final_model.zip
├── vec_normalize.pkl
├── validation_report.json
└── metadata.json
```

### Paper Trading Logs
```
production/logs/paper_trading/
├── paper_trading_20251003_*.csv      # Daily performance
├── paper_trading_*_summary.json      # Summary stats
└── [Multiple test runs for each model]
```

---

## Training Scripts

### v1 (Momentum)
```bash
python train_production_model.py
```

**File:** `train_production_model.py`
- 90/10 split on all data
- Phase 1 reward function
- Saves to `production/models/v1_{timestamp}/`

### v2 (Defensive)
```bash
python train_defensive_model.py
```

**File:** `train_defensive_model.py`
- Same data split as v1
- Heavy defensive reward penalties
- Saves to `production/models/v2_defensive_{timestamp}/`

### v2.1 (Balanced)
```bash
python train_balanced_model.py
```

**File:** `train_balanced_model.py`
- Same data split as v1, v2
- Tuned defensive rewards
- Saves to `production/models/v2.1_balanced_{timestamp}/`

---

## Paper Trading Framework

### Implementation
**File:** `production/paper_trading.py`

**Features:**
- Loads production model with VecNormalize
- Simulates trading on selected period
- Compares with classical baselines
- Generates GO/NO-GO decisions
- Logs to CSV and JSON

### Usage
```bash
# Bull market test
python production/paper_trading.py --model production/models/MODEL_DIR --start 2025-01-01 --end 2025-04-30

# Bear market test
python production/paper_trading.py --model production/models/MODEL_DIR --start 2025-07-01 --end 2025-09-30

# Mixed period test
python production/paper_trading.py --model production/models/MODEL_DIR --start 2025-01-01 --end 2025-09-30
```

### Decision Criteria
```python
# GO criteria:
- Beat Equal Weight: ✅
- Sharpe Ratio > 0.8: ✅
- Max Drawdown < 15%: ✅

# All must pass for GO decision
```

---

## Conclusions

### Production Status: Not Ready ⚠️

**Current Models:**
- v1: Marginal (+10.10% overall, fails in bear)
- v2: Too conservative (+6.17% overall, good in bear only)
- v2.1: Marginal (+9.87% overall, fails in bear)

**None are production-ready for all-weather trading**

### Root Cause Identified ✅

**Training Data Bias:**
- 80% bull markets, 20% bear markets
- Models learn momentum-following behavior
- Insufficient bear market examples for proper defense
- Reward engineering cannot fix fundamental data imbalance

### Solutions Available

**Short-Term (1 week):**
- Deploy v1 with circuit breakers
- Implement risk limits (5% daily loss, 15% max drawdown)
- Auto-switch to Equal Weight in bear conditions

**Long-Term (2-3 weeks) ⭐ RECOMMENDED:**
- **Stage 2: Training Data Rebalancing**
- Oversample bear periods 3-4x
- Create balanced 50/50 bull/bear dataset
- Retrain with v2.1 reward function
- Expected: Bull +13-15%, Bear +1-3%, Overall +10-12%

**Advanced (3-4 weeks):**
- Ensemble approach with regime detection
- Use v1 (bull specialist) + v2 (bear specialist)
- Switch models based on market conditions
- Expected: Best of both worlds

---

**Last Updated:** October 4, 2025
**Status:** Production Experiments Complete ✅
**Recommendation:** Proceed to Stage 2 (Training Data Rebalancing)
**Timeline:** 2-3 weeks to all-weather production model
