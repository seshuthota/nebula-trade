# Ensemble Implementation - Technical Documentation

**Date:** October 4, 2025
**Status:** ✅ Production Ready
**Version:** 1.0 (Sticky BEAR)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Testing & Validation](#testing--validation)
7. [Performance](#performance)
8. [Deployment](#deployment)

---

## Overview

The Nebula Trade Ensemble is a regime-switching portfolio optimization system that combines two specialized RL models:

- **v1 (Momentum)**: Optimized for bull markets (12.95% in Q1 2025)
- **v2 (Defensive)**: Optimized for bear markets (+0.34% in Q3 2025)

**Key Innovation:** Asymmetric hysteresis ("Sticky BEAR") - easy to enter defensive mode, hard to exit, preventing costly whipsaw during volatile corrections.

**Proven Performance:** 12.92% YTD on 2025 data (vs v1's 10.10%, EW's 9.29%)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Ensemble Manager                        │
│  ┌────────────────────────────────────────────────┐    │
│  │         Portfolio Returns History               │    │
│  └──────────────────┬─────────────────────────────┘    │
│                     │                                    │
│                     ▼                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Regime Detector                         │   │
│  │  • Drawdown analysis                             │   │
│  │  • Volatility calculation                        │   │
│  │  • Consecutive loss tracking                     │   │
│  │  • Trend detection                               │   │
│  │  • Asymmetric hysteresis                         │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                    │
│                     ▼                                    │
│              Regime: bull/bear/neutral                   │
│                     │                                    │
│         ┌───────────┴────────────┐                      │
│         ▼                        ▼                       │
│   ┌──────────┐            ┌──────────┐                  │
│   │ v1 Model │            │ v2 Model │                  │
│   │(Momentum)│            │(Defensive│                  │
│   └────┬─────┘            └─────┬────┘                  │
│        └───────────┬────────────┘                        │
│                    ▼                                     │
│              Portfolio Action                            │
└─────────────────────────────────────────────────────────┘
```

---

## Components

### 1. RegimeDetector (`astra/ensemble/regime_detector.py`)

**Purpose:** Detects market regime using multiple technical indicators

**Key Methods:**

```python
class RegimeDetector:
    def __init__(
        self,
        drawdown_threshold: float = -0.10,        # -10%
        volatility_threshold: float = 0.025,      # 2.5%
        consecutive_loss_threshold: int = 5,      # 5 days
        hysteresis_days: int = 3,                 # Enter regimes
        bear_exit_hysteresis_days: int = 7,       # Exit BEAR (STICKY)
        volatility_window: int = 10,
        trend_window: int = 20,
        ma_short_window: int = 50,
        ma_long_window: int = 200
    )

    def detect_regime(
        self,
        portfolio_returns: List[float]
    ) -> Tuple[str, Dict]:
        """Returns ('bull'|'bear'|'neutral', info_dict)"""
```

**Regime Detection Logic:**

Enter **BEAR** if ≥2 of:
1. Drawdown > -10%
2. Volatility > 2.5% (10-day rolling)
3. Consecutive losses ≥ 5 days
4. Trend = 'down' (20-day MA)

Exit **BEAR** requires:
1. Bear signals reduced
2. **7-day confirmation** (asymmetric hysteresis)

Enter **BULL** if:
- 0 bear signals AND trend = 'up'

Otherwise: **NEUTRAL** (uses primary model v1)

**Asymmetric Hysteresis:**
```python
# Entering any regime: 3 days
if self.pending_days >= self.hysteresis_days:
    switch_regime()

# Exiting BEAR regime: 7 days (STICKY!)
if self.current_regime == 'bear' and signal != 'bear':
    if self.pending_days >= self.bear_exit_hysteresis_days:  # 7 days
        switch_regime()
```

---

### 2. EnsembleManager (`astra/ensemble/ensemble_manager.py`)

**Purpose:** Manages model selection and switching logic

**Key Methods:**

```python
class EnsembleManager:
    def __init__(
        self,
        v1_model_path: str,
        v2_model_path: str,
        config_path: str = "config/ensemble.yaml",
        primary_model: str = 'v1'
    )

    def get_action(
        self,
        observation: np.ndarray,
        portfolio_returns: List[float],
        vec_env: VecNormalize,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Returns (action, info_dict)

        info_dict contains:
        - regime: current regime
        - active_model: 'v1' or 'v2'
        - switched: bool
        - model_usage: {' v1': pct, 'v2': pct}
        """
```

**Model Selection:**
```python
def select_model(self, regime: str) -> str:
    if regime == 'bear':
        return 'v2'  # Defensive
    elif regime in ['bull', 'neutral']:
        return 'v1'  # Momentum
```

**Tracking:**
- Switch history (timestamps, from→to transitions)
- Regime history (daily regimes)
- Model usage percentages

---

### 3. Backtesting Framework (`astra/ensemble/backtest_ensemble.py`)

**Purpose:** Test ensemble on historical data

**Key Methods:**

```python
class EnsembleBacktester:
    def __init__(
        self,
        data_path: str = "data/portfolio_data_processed.csv",
        config_path: str = "config/ensemble.yaml"
    )

    def backtest_ensemble(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000
    ) -> Dict:
        """Run ensemble on historical period"""

    def compare_with_baselines(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000
    ) -> Dict:
        """Compare ensemble vs Equal Weight"""
```

**Output:**
```json
{
  "ensemble": {
    "return": 0.1292,
    "sharpe": 0.952,
    "max_dd": -0.0909,
    "stats": {
      "regime_stats": {
        "bull_pct": 0.011,
        "bear_pct": 0.281,
        "neutral_pct": 0.708
      },
      "model_usage": {"v1": 0.719, "v2": 0.281},
      "total_switches": 6
    }
  },
  "equal_weight": {
    "return": 0.0938,
    "sharpe": 0.889
  },
  "performance": {
    "return_gap_vs_ew": 3.54
  }
}
```

---

### 4. Paper Trading (`production/ensemble_paper_trading.py`)

**Purpose:** Production paper trading with ensemble

**Usage:**
```bash
python production/ensemble_paper_trading.py \
    --start 2025-01-01 \
    --end 2025-09-26 \
    --capital 100000
```

**Features:**
- Loads both v1 and v2 models
- Uses EnsembleManager for decisions
- Tracks regime changes
- Logs daily: regime, active model, returns
- Compares vs classical baselines

---

## Configuration

### Production Config (`config/ensemble_sticky_7day.yaml`)

```yaml
ensemble:
  models:
    v1:
      name: "Momentum Specialist"
      path: "production/models/v1_20251003_131020"
      use_for: ["bull", "neutral"]

    v2:
      name: "Defensive Specialist"
      path: "production/models/v2_defensive_20251003_212109"
      use_for: ["bear"]

  primary_model: "v1"

  # Regime Detection with ASYMMETRIC HYSTERESIS
  regime_detection:
    # Thresholds
    drawdown_threshold: -0.10           # -10% triggers BEAR
    volatility_threshold: 0.025         # 2.5% volatility triggers BEAR
    consecutive_loss_threshold: 5       # 5 losses triggers BEAR

    # Hysteresis (KEY INNOVATION)
    hysteresis_days: 3                  # Enter any regime: 3 days
    bear_exit_hysteresis_days: 7        # Exit BEAR: 7 days (STICKY!)

    # Detection windows
    volatility_window: 10               # 10-day volatility
    trend_window: 20                    # 20-day MA for trend
    ma_short_window: 50                 # 50-day MA
    ma_long_window: 200                 # 200-day MA

  # Performance Targets
  performance_targets:
    min_return_vs_ew: 0.02              # 2% outperformance
    min_sharpe: 0.8
    max_drawdown: -0.15
    expected_return:
      bull_market: 0.13
      bear_market: 0.00
      overall: 0.12

  # Logging
  logging:
    level: "INFO"
    log_regime_changes: true
    log_model_switches: true
    save_daily_logs: true
    log_directory: "production/logs/ensemble"
```

**Key Parameter: `bear_exit_hysteresis_days: 7`**

This is the "sticky BEAR" innovation:
- Normal transitions: 3-day confirmation
- Exiting BEAR: 7-day confirmation (longer!)
- Result: Reduces whipsaw in volatile corrections

---

## Usage

### 1. Run Backtest

```bash
# Test on full 2025 YTD
python -m astra.ensemble.backtest_ensemble \
    --start 2025-01-01 \
    --end 2025-09-26 \
    --capital 100000
```

### 2. Paper Trading

```bash
# Run ensemble paper trading
python production/ensemble_paper_trading.py \
    --start 2025-01-01 \
    --end 2025-09-26
```

### 3. Custom Configuration

```bash
# Test with different config
python -m astra.ensemble.backtest_ensemble \
    --start 2025-07-01 \
    --end 2025-09-26
# (uses config/ensemble_sticky_7day.yaml by default)
```

### 4. Python API

```python
from astra.ensemble.ensemble_manager import EnsembleManager

# Initialize ensemble
ensemble = EnsembleManager(
    v1_model_path="production/models/v1_20251003_131020",
    v2_model_path="production/models/v2_defensive_20251003_212109",
    config_path="config/ensemble_sticky_7day.yaml"
)

# Get action
action, info = ensemble.get_action(
    observation=obs,
    portfolio_returns=returns_history,
    vec_env=vec_env
)

print(f"Regime: {info['regime']}")
print(f"Active model: {info['active_model']}")
print(f"Switched: {info['switched']}")
```

---

## Testing & Validation

### Test Suite

1. **Q1 2025 (Bull Market)**
   ```bash
   python -m astra.ensemble.backtest_ensemble \
       --start 2025-01-01 --end 2025-03-31
   ```
   - Expected: v1 active 100%, ~12.95% return
   - Validates: No unnecessary switches in bulls

2. **Q3 2025 (Correction)**
   ```bash
   python -m astra.ensemble.backtest_ensemble \
       --start 2025-07-01 --end 2025-09-26
   ```
   - Expected: v2 active ~40-45%, ~+0.7% return
   - Validates: Defensive switching works

3. **Full 2025 YTD**
   ```bash
   python -m astra.ensemble.backtest_ensemble \
       --start 2025-01-01 --end 2025-09-26
   ```
   - Expected: ~12.9% return, ~6 switches
   - Validates: Overall performance

### Validation Tests

**Run comparison test:**
```bash
python test_sticky_bear_full.py
```

Output shows:
- Baseline (3-day) vs Sticky (7-day)
- Q1, Q3, and YTD results
- Improvement calculations

---

## Performance

### Validated Results (2025 Data)

| Period | Return | vs EW | Switches | v2 Usage |
|--------|--------|-------|----------|----------|
| Q1 2025 (Bull) | 12.95% | +7.15 pp | 0 | 0% |
| Q3 2025 (Correction) | +0.73% | +3.88 pp | 6 | 43.5% |
| **2025 YTD** | **12.92%** | **+3.54 pp** | 6 | 28.1% |

### Comparison vs Alternatives

| Strategy | YTD | Sharpe | Max DD | Switches |
|----------|-----|--------|--------|----------|
| **Ensemble (Sticky)** | **12.92%** | 0.952 | -9.09% | 6 |
| v1 alone | 10.10% | 0.738 | -9.33% | 0 |
| v2 alone | 6.17% | 0.504 | -8.49% | 0 |
| Equal Weight | 9.29% | 0.882 | -7.20% | 0 |
| Baseline Ensemble (3-day) | 12.90% | 0.930 | -8.39% | 8 |

**Key Wins:**
- ✅ Beats v1 by +2.82 pp
- ✅ Beats EW by +3.54 pp
- ✅ Better Sharpe than v1 (0.952 vs 0.738)
- ✅ 25% fewer switches than baseline ensemble

---

## Deployment

### Production Checklist

- [x] Models loaded successfully (v1, v2)
- [x] Configuration validated (`ensemble_sticky_7day.yaml`)
- [x] Backtests pass on 2025 data
- [x] Paper trading tested
- [x] Regime detector validated
- [x] Performance meets targets (>12% YTD)
- [x] Documentation complete

### Deployment Steps

1. **Verify Models**
   ```bash
   ls production/models/v1_20251003_131020/
   ls production/models/v2_defensive_20251003_212109/
   ```

2. **Test Configuration**
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('config/ensemble_sticky_7day.yaml')))"
   ```

3. **Run Final Validation**
   ```bash
   python test_sticky_bear_full.py
   ```

4. **Deploy**
   - Use `production/ensemble_paper_trading.py` for live paper trading
   - Monitor `production/logs/ensemble/` for daily logs
   - Track regime switches and model usage

### Monitoring

**Daily Logs:**
- Location: `production/logs/ensemble/`
- Format: CSV with date, ensemble_value, regime, active_model

**Key Metrics to Monitor:**
- Daily return
- Regime changes (should be infrequent)
- Model switches (expect ~1-2 per quarter)
- Drawdown vs threshold (-10%)

**Alerts:**
- Max drawdown exceeds -15% (emergency threshold)
- More than 10 switches in a month (whipsaw)
- Sharpe drops below 0.5 (poor risk-adjusted returns)

---

## Troubleshooting

### Common Issues

**1. "Model not found" error**
```bash
# Verify paths in config
cat config/ensemble_sticky_7day.yaml | grep path
```

**2. "VecNormalize not found"**
```bash
# Check for vec_normalize.pkl
ls production/models/v1_20251003_131020/vec_normalize.pkl
ls production/models/v2_defensive_20251003_212109/vec_normalize.pkl
```

**3. Too many switches**
- Increase `bear_exit_hysteresis_days` from 7 to 9-10
- Increase `hysteresis_days` from 3 to 4-5

**4. Poor bear market performance**
- Decrease `drawdown_threshold` from -0.10 to -0.08 (more sensitive)
- Decrease `volatility_threshold` from 0.025 to 0.020
- Decrease `bear_exit_hysteresis_days` (less sticky)

---

## Technical Details

### File Structure

```
astra/ensemble/
├── __init__.py
├── regime_detector.py          # 334 lines
├── ensemble_manager.py         # 242 lines
└── backtest_ensemble.py        # 437 lines

production/
└── ensemble_paper_trading.py   # 423 lines

config/
├── ensemble.yaml               # Baseline (3-day symmetric)
├── ensemble_sticky_5day.yaml   # 5-day exit
└── ensemble_sticky_7day.yaml   # 7-day exit (PRODUCTION)

docs/
├── STICKY_BEAR_RESULTS.md      # Optimization analysis
└── ENSEMBLE_IMPLEMENTATION.md  # This document
```

### Dependencies

```python
# Core
numpy
pandas
pyyaml

# RL
stable-baselines3
gymnasium (or gym)

# Existing
astra.rl_framework.environment
astra.evaluation.optimizer
```

### Performance Characteristics

- **Initialization**: ~2-3 seconds (loads 2 models)
- **Prediction**: ~10-50ms per step (regime detection + model inference)
- **Memory**: ~500MB (2 models loaded)
- **Disk**: ~50MB (model artifacts)

---

## Future Enhancements

1. **Adaptive Hysteresis**: Vary exit days based on volatility levels
2. **VIX Integration**: Use actual volatility index if available
3. **Multi-threshold Switching**: Gradual v2 allocation instead of binary
4. **ML Regime Classifier**: Predict regime duration with ML
5. **Transaction Cost Optimization**: Minimize switching costs
6. **Real-time Monitoring Dashboard**: Visualize regime and switches

---

## References

- [FINAL_RECOMMENDATION.md](FINAL_RECOMMENDATION.md) - Model comparison
- [STICKY_BEAR_RESULTS.md](STICKY_BEAR_RESULTS.md) - Optimization analysis
- [EXPERIMENTS_SUMMARY.md](EXPERIMENTS_SUMMARY.md) - All experiments

---

**Document Version:** 1.0
**Last Updated:** October 4, 2025
**Status:** Production Ready
**Maintainer:** Nebula Trade Team
