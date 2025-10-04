# Production Directory

**Production deployment infrastructure for live trading**

---

## Directory Structure

```
production/
├── models/              # Production model versions
│   └── v1_{timestamp}/  # Version 1 model
│       ├── final_model.zip
│       ├── vec_normalize.pkl
│       ├── metadata.json
│       └── validation_report.json
├── data/                # Production data (latest market data)
│   └── latest_market_data.csv
├── logs/                # Production logs
│   ├── training/        # Training logs
│   ├── paper_trading/   # Paper trading logs
│   └── live_trading/    # Live trading logs
├── config/              # Production configuration
│   ├── production.yaml  # Model & training config
│   └── risk_limits.yaml # Risk management rules
└── README.md           # This file
```

---

## Configuration Files

### production.yaml
**Production model and training configuration**

Key settings:
- **Data Split:** 90% train, 10% validation (vs 70/15/15 research)
- **Model:** Phase 1 Extended config (173 features, 1M steps)
- **Network:** [256, 256]
- **Hyperparameters:** Same as Phase 1 Extended (proven best)

### risk_limits.yaml
**Risk management rules for live trading**

Critical limits:
- **Position Limits:** Max 40% per stock, min 5% per stock
- **Circuit Breakers:** Pause on >5% daily loss, >15% drawdown
- **Execution Controls:** Slippage monitoring, order validation
- **Recovery:** Manual approval required after circuit breaker

⚠️ **IMPORTANT:** Review and adjust limits based on your risk tolerance

---

## Usage

### Step 1: Train Production Model

**Update data first:**
```bash
python scripts/update_market_data.py
```

**Train production model:**
```bash
python train_production_model.py --timesteps 1000000
```

**What it does:**
- Uses 90/10 train/val split (all available data)
- Validation on most recent data (current market conditions)
- Saves to `production/models/v1_{timestamp}/`
- Generates validation report

**Success criteria:**
- Validation return >20%
- Beats Equal Weight baseline
- Deterministic behavior (std <2%)

---

### Step 2: Paper Trading (2-4 weeks)

**Coming soon:** `paper_trading.py`

**What it will do:**
- Test model on live data WITHOUT real money
- Track daily performance vs baselines
- Monitor for anomalies
- Generate go/no-go decision for live trading

**Success criteria:**
- Returns > Equal Weight
- Sharpe ratio >0.8
- Max drawdown <15%
- No anomalous behavior

---

### Step 3: Live Deployment (After Paper Trading Success)

**Coming soon:** `live_trading.py`

**What it will do:**
- Execute real trades based on model decisions
- Apply risk limits from `risk_limits.yaml`
- Monitor performance real-time
- Send alerts on anomalies

**Capital allocation:**
- Start: ₹100k-500k (test amount)
- Scale: After 1-2 months of success
- Max: Define based on risk tolerance

---

## Model Versioning

### Version 1 (Current)
**Configuration:** Phase 1 Extended (173 features)
**Status:** In development
**Expected:** Oct 2025

**Files:**
- `final_model.zip` - Trained SAC agent
- `vec_normalize.pkl` - Normalization statistics (REQUIRED)
- `metadata.json` - Model metadata and config
- `validation_report.json` - Validation results

### Future Versions
**Version 2:** After first retraining (monthly)
**Version 3:** After major improvements
**etc.**

---

## Safety Features

### Circuit Breakers ⚠️
Automatic trading pause on:
- Daily loss >5%
- Weekly loss >10%
- Drawdown >15%
- Single stock loss >8%
- Portfolio value <80% of initial

### Position Limits 📊
Hard constraints:
- Max 40% in single stock
- Min 5% in each stock (diversification)
- Max 30% daily turnover
- Min 5% cash reserve
- No leverage

### Monitoring & Alerts 🔔
Real-time tracking:
- Daily P&L
- Slippage vs target
- Transaction costs
- Position changes
- Market anomalies

---

## Data Management

### Latest Market Data
**Location:** `production/data/latest_market_data.csv`
**Update frequency:** Daily (before market open)
**Update command:**
```bash
python scripts/update_market_data.py
```

### Data Quality Checks
- No missing values
- No infinite values
- No extreme outliers (>50% daily return)
- Volume validation

---

## Logging

### Training Logs
**Location:** `production/logs/training/`
**Contents:**
- Training progress
- Validation results
- Model checkpoints
- Performance metrics

### Paper Trading Logs
**Location:** `production/logs/paper_trading/`
**Contents:**
- Daily decisions
- Simulated trades
- Performance tracking
- Anomaly detection

### Live Trading Logs
**Location:** `production/logs/live_trading/`
**Contents:**
- All trade executions
- Model decisions
- Slippage tracking
- Circuit breaker triggers
- Daily P&L

---

## Deployment Checklist

### Before Training
- [ ] Data updated to current date
- [ ] Environment reverted to Phase 1 config (173 features)
- [ ] Configuration reviewed (`production.yaml`)
- [ ] Hardware ready (GPU recommended)

### After Training
- [ ] Validation results reviewed
- [ ] Beats Equal Weight baseline
- [ ] Return >20% on validation
- [ ] Deterministic behavior confirmed
- [ ] Model artifacts saved correctly

### Before Paper Trading
- [ ] Paper trading infrastructure built
- [ ] Real-time data feed tested
- [ ] Logging configured
- [ ] Monitoring dashboard ready

### Before Live Trading
- [ ] Paper trading successful (2-4 weeks)
- [ ] Risk limits configured (`risk_limits.yaml`)
- [ ] Broker API integrated
- [ ] Emergency stop tested
- [ ] Capital allocated
- [ ] Final approval obtained

---

## Risk Disclaimer

⚠️ **IMPORTANT DISCLAIMER:**

1. **No Guarantee:** Past performance does not guarantee future results
2. **Market Risk:** All trading involves risk of loss
3. **Start Small:** Begin with capital you can afford to lose
4. **Monitor Closely:** Especially during first weeks of live trading
5. **Have Exit Plan:** Know when to stop (circuit breakers, max loss)

**Test thoroughly in paper trading before risking real capital.**

---

## Support

**For production issues:**
- Review logs in `production/logs/`
- Check configuration in `production/config/`
- Verify model artifacts in `production/models/`
- Consult deployment plan in `docs/LIVE_DEPLOYMENT_PLAN.md`

**Documentation:**
- Training progress: `docs/TRAINING_PROGRESS.md`
- Code cleanup: `docs/CODE_CLEANUP_PLAN.md`
- Live deployment plan: `docs/LIVE_DEPLOYMENT_PLAN.md`

---

**Last Updated:** October 3, 2025
**Status:** Development - Production model training pending
**Next:** Train production model → Paper trading → Live deployment
