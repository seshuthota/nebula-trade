# Portfolio Environment Bug Fix

## The Bug

The environment in `astra/rl_framework/environment.py` has a critical bug in the `step()` function that causes unrealistic portfolio value growth.

### Symptoms
- RL agent shows 3,800% returns vs 30% for classical methods
- Daily returns of 8-10% (impossible)
- Portfolio grows 40x in a few months

### Root Cause

The `step()` function incorrectly calculates portfolio value changes:

```python
# BUGGY CODE (current):
old_value = sum(self.positions[asset] * current_prices[i] for i, asset in enumerate(self.assets)) + self.cash
new_value = sum(self.positions[asset] * next_prices[i] for i, asset in enumerate(self.assets)) + self.cash

# Update positions according to new weights
total_value_after_cost = new_value - transaction_cost
for i, asset in enumerate(self.assets):
    new_position = (new_weights[i] * total_value_after_cost) / next_prices[i]
    self.positions[asset] = new_position

new_portfolio_value = sum(self.positions[asset] * next_prices[i] for i, asset in enumerate(self.assets)) + self.cash
```

**Problem:**
1. `new_value` is calculated using OLD positions at NEW prices
2. Then positions are immediately updated based on `new_weights`
3. `new_portfolio_value` is calculated with NEW positions at NEW prices
4. But the cash isn't properly updated to reflect the rebalancing
5. This creates a mismatch where gains are counted incorrectly

### The Correct Logic

Portfolio value updates should follow this sequence:

1. **At end of day T**: Portfolio has positions P_T, prices are S_T
2. **Move to day T+1**: Prices change to S_{T+1}, positions stay P_T
3. **Calculate natural appreciation**: Value = sum(P_T * S_{T+1}) + cash
4. **Agent chooses new weights**: w_{T+1}
5. **Rebalance**: 
   - Calculate target positions: P_{T+1} based on w_{T+1}
   - Calculate turnover: sum(|P_{T+1} - P_T| * S_{T+1})
   - Apply transaction costs
   - Update cash to reflect trading
6. **Final value**: sum(P_{T+1} * S_{T+1}) + cash_after_costs

## The Fix

Replace the step() function logic with:

```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """Execute one step in the environment."""
    
    # Normalize action to valid weights
    weights_sum = np.sum(action)
    if weights_sum > 0:
        normalized_weights = action / weights_sum
    else:
        normalized_weights = np.ones(self.n_assets) / self.n_assets
    
    # Clip and renormalize
    normalized_weights = np.clip(normalized_weights, self.min_weight, self.max_weight)
    weights_sum = np.sum(normalized_weights)
    if weights_sum > 0:
        new_weights = normalized_weights / weights_sum
    else:
        new_weights = np.ones(self.n_assets) / self.n_assets
    
    # Move to next time step
    next_idx = self.current_idx + 1
    
    if next_idx >= len(self.data):
        return self._get_observation(), 0.0, True, False, {
            'date': self.current_date,
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'turnover': 0.0,
            'transaction_cost': 0.0
        }
    
    # Get current and next prices
    current_prices = np.array([self.data.iloc[self.current_idx, self.data.columns.get_loc(f"{asset}_close")] 
                               for asset in self.assets])
    next_prices = np.array([self.data.iloc[next_idx, self.data.columns.get_loc(f"{asset}_close")] 
                           for asset in self.assets])
    
    # Step 1: Calculate portfolio value at start of day (before rebalancing)
    # This is the natural price appreciation from yesterday's positions
    portfolio_value_before_rebalance = sum(
        self.positions[asset] * next_prices[i] for i, asset in enumerate(self.assets)
    ) + self.cash
    
    # Step 2: Calculate turnover for proposed rebalancing
    old_weights_array = np.array([self.weights.get(asset, 0.0) for asset in self.assets])
    turnover = np.sum(np.abs(new_weights - old_weights_array))
    
    # Step 3: Apply transaction costs if turnover exceeds threshold
    if turnover > self.turnover_threshold:
        transaction_cost = turnover * self.transaction_cost * portfolio_value_before_rebalance
    else:
        transaction_cost = 0.0
    
    # Step 4: Calculate total value available for rebalancing (after costs)
    total_value_after_cost = portfolio_value_before_rebalance - transaction_cost
    
    # Step 5: Rebalance portfolio according to new weights
    new_positions = {}
    cash_needed = 0.0
    
    for i, asset in enumerate(self.assets):
        # Calculate target value for this asset
        target_value = new_weights[i] * total_value_after_cost
        # Calculate shares needed
        target_shares = target_value / next_prices[i]
        new_positions[asset] = target_shares
        cash_needed += target_value
    
    # Update cash (total value - invested amount)
    new_cash = total_value_after_cost - cash_needed
    
    # Step 6: Calculate final portfolio value (should equal total_value_after_cost)
    final_portfolio_value = sum(
        new_positions[asset] * next_prices[i] for i, asset in enumerate(self.assets)
    ) + new_cash
    
    # Step 7: Calculate reward (daily return after costs)
    old_value = self.portfolio_value
    daily_return = (final_portfolio_value - old_value) / old_value if old_value > 0 else 0.0
    reward = daily_return
    
    # Step 8: Update state
    self.positions = new_positions
    self.cash = new_cash
    self.portfolio_value = final_portfolio_value
    self.current_idx = next_idx
    self.current_date = self.data.index[next_idx]
    self.portfolio_returns.append(daily_return)
    
    # Update weight tracking
    for i, asset in enumerate(self.assets):
        self.weights[asset] = (self.positions[asset] * next_prices[i]) / final_portfolio_value if final_portfolio_value > 0 else 0.0
    self.weights['cash'] = new_cash / final_portfolio_value if final_portfolio_value > 0 else 0.0
    
    # Check termination
    terminated = (next_idx >= len(self.data) - 1)
    
    # Prepare info
    info = {
        'date': self.current_date,
        'portfolio_value': self.portfolio_value,
        'weights': self.weights.copy(),
        'turnover': turnover,
        'transaction_cost': transaction_cost,
        'current_step': self.current_idx
    }
    
    return self._get_observation(), reward, terminated, False, info
```

## Key Changes

1. **Proper sequencing**: First calculate natural appreciation, then rebalance
2. **Consistent cash tracking**: Cash is properly updated to reflect rebalancing
3. **Clear value flow**: 
   - Start value (after price change)
   - Subtract transaction costs
   - Rebalance to new weights
   - Final value should equal start minus costs
4. **Reward = actual return**: Daily return is based on true portfolio value change

## Expected Results After Fix

- Daily returns should be realistic (0-3% on good days, can be negative)
- RL performance should be comparable to classical methods (maybe 1.2-2x better)
- Returns over 1.5 years should be 30-50% for good performance, not 3,800%

## Testing the Fix

After applying the fix, run:

```bash
# Test with diagnostic
python diagnose_environment.py --steps 50

# Should see:
# - Step returns < 5% (usually < 2%)
# - Portfolio value grows gradually
# - Cumulative return over 50 days should be < 20%

# Then retrain
python train_portfolio.py --timesteps 50000

# Expected results:
# - RL: 30-60% return
# - Classical: 25-35% return
# - RL should be better but not 100x better
```

## Additional Improvements

Consider also:

1. **Add constraints**: Ensure portfolio_value never becomes negative
2. **Add logging**: Log value calculations for debugging
3. **Validate**: Assert that final_value ≈ start_value - costs after rebalancing
4. **Add limits**: Cap daily returns at some reasonable threshold (e.g., 20%) as a safety check

## Impact

This bug fix will:
- ✅ Make returns realistic
- ✅ Enable proper comparison of RL vs classical methods
- ✅ Allow meaningful model selection
- ✅ Make the system usable for real trading

Without this fix, the agent is learning to exploit a bug, not learning actual portfolio optimization.
