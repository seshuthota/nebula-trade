import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import yaml
import logging
from pathlib import Path

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for older installations
    import gym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioEnvironment(gym.Env):
    """Continuous portfolio optimization environment for RL."""

    def __init__(self, data: pd.DataFrame, config_path: str = None, lookback_window: int = 30):
        super(PortfolioEnvironment, self).__init__()

        # Load config
        project_root = Path(__file__).resolve().parents[2]

        candidates = []
        if config_path is None:
            candidates.append(project_root / "config/portfolio.yaml")
        else:
            cfg_path = Path(config_path)
            if cfg_path.is_absolute():
                candidates.append(cfg_path)
            else:
                candidates.extend([
                    Path.cwd() / cfg_path,
                    project_root / cfg_path,
                    project_root / "config/portfolio.yaml"
                ])

        for candidate in candidates:
            if candidate.exists():
                config_path = candidate.resolve()
                break
        else:
            raise FileNotFoundError(f"Config file not found. Tried: {[str(c) for c in candidates]}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        self.transaction_cost = self.config['rebalance']['transaction_cost']
        self.min_weight = self.config['constraints']['min_weight']
        self.max_weight = self.config['constraints']['max_weight']
        self.turnover_threshold = self.config['constraints']['turnover_threshold']

        self.data = data
        self.lookback_window = lookback_window
        self.n_assets = len(self.assets)

        # Action space: continuous weights for each asset (will be softmax normalized to sum to 1)
        self.action_space = gym.spaces.Box(low=self.min_weight,
                                          high=self.max_weight,
                                          shape=(self.n_assets,),
                                          dtype=np.float32)

        # Observation space: flattened state representation
        # [current_prices, current_weights, portfolio_value, cash, sharpe, correlations, returns_history]
        n_price_features = self.n_assets
        n_weight_features = self.n_assets
        n_other_features = 3  # portfolio_value, cash, sharpe
        n_correlation_features = self.n_assets * (self.n_assets - 1) // 2  # upper triangle
        n_returns_history = self.lookback_window * self.n_assets

        n_obs = n_price_features + n_weight_features + n_other_features + n_correlation_features + n_returns_history

        # Define bounded observation space based on clipping ranges
        # This helps the neural network by providing explicit bounds
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(n_obs,), dtype=np.float32)

        # Environment state
        self.reset()

    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state with proper normalization."""
        current_prices = np.array([self.data.loc[self.current_date, f"{asset}_close"] for asset in self.assets])

        # Normalize prices (divide by portfolio value) and clip
        if self.portfolio_value > 0:
            normalized_prices = current_prices / self.portfolio_value
            normalized_prices = np.clip(normalized_prices, 0, 10)  # Clip extreme values
        else:
            normalized_prices = np.zeros(self.n_assets)

        # Current weights (assets only, not including cash)
        current_weights = np.array([self.weights.get(asset, 0.0) for asset in self.assets])
        current_weights = np.clip(current_weights, 0, 1)  # Already normalized to [0, 1]

        # Portfolio metrics - normalized
        portfolio_value_norm = np.clip(self.portfolio_value / self.initial_capital, 0, 5)  # Clip to reasonable range
        cash_norm = np.clip(self.cash / self.initial_capital, 0, 1)

        # Sharpe ratio (30-day rolling) - clipped to reasonable range
        if len(self.portfolio_returns) >= 10:
            recent_returns = self.portfolio_returns[-30:] if len(self.portfolio_returns) >= 30 else self.portfolio_returns[-10:]
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns)
            sharpe = mean_ret / (std_ret + 1e-8) if std_ret > 1e-8 else 0.0
            sharpe = np.clip(sharpe, -3, 3)  # Clip to [-3, 3] range
        else:
            sharpe = 0.0

        # Correlations between assets - already in [-1, 1]
        returns_cols = [f"{asset}_returns" for asset in self.assets if f"{asset}_returns" in self.data.columns]
        correlations = []
        if self.lookback_window <= len(self.data) and self.current_idx >= self.lookback_window:
            window_data = self.data.iloc[self.current_idx-self.lookback_window:self.current_idx+1]
            for i in range(len(returns_cols)):
                for j in range(i+1, len(returns_cols)):
                    corr = window_data[returns_cols[i]].corr(window_data[returns_cols[j]])
                    correlations.append(np.clip(corr if not np.isnan(corr) else 0.0, -1, 1))

        correlations = np.array(correlations) if correlations else np.zeros(self.n_assets * (self.n_assets - 1) // 2)

        # Recent returns history - clip to reasonable range
        returns_history = []
        if self.current_idx >= self.lookback_window:
            for asset in self.assets:
                returns_col = f"{asset}_returns"
                if returns_col in self.data.columns:
                    hist = self.data[returns_col].iloc[self.current_idx-self.lookback_window+1:self.current_idx+1].values
                    # Clip daily returns to [-0.5, 0.5] (extreme 50% moves)
                    hist_clipped = np.clip(hist, -0.5, 0.5)
                    returns_history.extend(hist_clipped)
                else:
                    returns_history.extend([0.0] * self.lookback_window)
        else:
            returns_history = [0.0] * (self.lookback_window * self.n_assets)

        # Flatten and combine
        observation = np.concatenate([
            normalized_prices,
            current_weights,
            [portfolio_value_norm, cash_norm, sharpe],
            correlations,
            returns_history
        ]).astype(np.float32)

        # Final safety check - replace any NaN/Inf with 0
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)

        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)

        self.current_idx = 0
        self.current_date = self.data.index[0]
        self.initial_capital = 100000  # Can make configurable

        # Initialize with equal weights from config
        from ..data_pipeline.data_manager import PortfolioDataManager
        manager = PortfolioDataManager()
        initial_state = manager.get_initial_portfolio_state(self.data, self.initial_capital)

        self.weights = initial_state['weights']
        self.portfolio_value = initial_state['portfolio_value']
        self.cash = initial_state['cash']
        self.positions = initial_state['positions']
        self.portfolio_returns = []

        logger.info(f"Environment reset at {self.current_date}, portfolio value: {self.portfolio_value}")

        return self._get_observation(), {'date': self.current_date}

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

        # Step 7: Calculate reward with DEFENSIVE, risk-adjusted components
        old_value = self.portfolio_value
        daily_return = (final_portfolio_value - old_value) / old_value if old_value > 0 else 0.0

        # === COMPONENT 1: Asymmetric Loss Aversion ===
        # Losses hurt 1.5x more than equivalent gains (TUNED from 2x)
        if daily_return >= 0:
            return_component = daily_return * 100  # Gains scaled normally
        else:
            return_component = daily_return * 150  # Losses hurt 1.5x more (reduced from 2x)

        return_component = np.clip(return_component, -25, 15)  # Asymmetric clipping (less negative bound)

        # === COMPONENT 2: Sharpe Ratio (BALANCED WEIGHT) ===
        # Risk-adjusted performance - TUNED from 30% to 20% weight
        sharpe_component = 0.0
        if len(self.portfolio_returns) >= 10:  # Need at least 10 days
            recent_returns = self.portfolio_returns[-30:] if len(self.portfolio_returns) >= 30 else self.portfolio_returns[-10:]
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns)
            if std_ret > 1e-8:
                sharpe_component = (mean_ret / std_ret) * 4.0  # Reduced from 5.0 to 4.0
                sharpe_component = np.clip(sharpe_component, -8, 8)

        # === COMPONENT 3: Drawdown Penalty ===
        # Penalize portfolio value falling below recent peak (TUNED - less aggressive)
        drawdown_penalty = 0.0
        if len(self.portfolio_returns) >= 5:
            # Track peak value over last 30 days
            recent_values = [old_value * (1 + sum(self.portfolio_returns[-i:])) for i in range(1, min(31, len(self.portfolio_returns)+1))]
            peak_value = max(recent_values) if recent_values else old_value

            if final_portfolio_value < peak_value:
                drawdown_pct = (peak_value - final_portfolio_value) / peak_value
                # Moderate penalty for drawdowns: -10 points for 10% drawdown (reduced from -20)
                drawdown_penalty = -100 * drawdown_pct  # Reduced from -200
                drawdown_penalty = np.clip(drawdown_penalty, -15, 0)  # Less severe bound

        # === COMPONENT 4: Volatility Penalty ===
        # Penalize unstable portfolios (discourage wild swings)
        volatility_penalty = 0.0
        if len(self.portfolio_returns) >= 10:
            recent_returns = self.portfolio_returns[-30:] if len(self.portfolio_returns) >= 30 else self.portfolio_returns[-10:]
            volatility = np.std(recent_returns)
            # Penalize high volatility: -5 points for 5% daily std
            if volatility > 0.02:  # Above 2% daily std
                volatility_penalty = -100 * (volatility - 0.02)
                volatility_penalty = np.clip(volatility_penalty, -10, 0)

        # === COMPONENT 5: Turnover Penalty ===
        # Discourage excessive trading
        turnover_penalty = 0.0
        if turnover > 0.5:  # If turnover > 50%
            turnover_penalty = -5.0 * (turnover - 0.5)  # Increased from -0.1
            turnover_penalty = np.clip(turnover_penalty, -10, 0)

        # === COMBINE ALL COMPONENTS ===
        # TUNED: More balanced between returns and defense
        reward = (
            return_component * 0.6 +      # 60% weight on asymmetric returns (increased from 40%)
            sharpe_component * 0.2 +      # 20% weight on Sharpe ratio (reduced from 30%)
            drawdown_penalty * 0.1 +      # 10% weight on drawdown protection (reduced from 20%)
            volatility_penalty * 0.05 +   # 5% weight on stability
            turnover_penalty * 0.05       # 5% weight on turnover control
        )

        # Final clipping to prevent extreme rewards (less restrictive)
        reward = np.clip(reward, -30, 20)  # Reduced bounds for more flexibility

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

# For Gymnasium compatibility
from gymnasium import Env
from gymnasium.spaces import Box

class PortfolioEnvironmentGymnasium(Env):
    """Gymnasium version of portfolio environment."""

    def __init__(self, data: pd.DataFrame, config_path: str = "../config/portfolio.yaml"):
        super().__init__()
        # Wrap the old environment
        self.env = PortfolioEnvironment(data, config_path)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return next_obs, reward, terminated, truncated, info
