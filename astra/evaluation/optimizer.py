import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
except ImportError:
    print("PyPortfolioOpt not available. Classical optimization features disabled.")
    EfficientFrontier = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassicalPortfolioOptimizer:
    """Classical portfolio optimization using Markowitz and other methods."""

    def __init__(self, returns_data: pd.DataFrame):
        """
        Initialize with historical returns data.
        Expected shape: (n_periods, n_assets)
        """
        self.returns = returns_data
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)

        # Calculate expected returns and covariance
        if self.returns is not None and not self.returns.empty:
            self.mu = expected_returns.mean_historical_return(self.returns, returns_data=True)
            self.S = risk_models.sample_cov(self.returns, returns_data=True)
        else:
            logger.warning("No returns data provided")
            self.mu = None
            self.S = None

    def _solve_with_fallback(self, ef, method: str, *args, **kwargs):
        """Run optimizer method and surface detailed errors."""
        try:
            return getattr(ef, method)(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("%s optimization failed: %s", method, exc)
            raise

    def max_sharpe_portfolio(self, risk_free_rate: float = 0.02) -> Dict[str, np.ndarray]:
        """Optimize for maximum Sharpe ratio (tangency portfolio)."""
        if EfficientFrontier is None:
            return self._fallback_equal_weights()

        ef = EfficientFrontier(self.mu, self.S)
        self._solve_with_fallback(ef, 'max_sharpe', risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()

        performance = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)

        return {
            'weights': np.array(list(cleaned_weights.values())),
            'expected_return': performance[0],
            'expected_volatility': performance[1],
            'sharpe_ratio': performance[2],
            'method': 'max_sharpe'
        }

    def min_volatility_portfolio(self) -> Dict[str, np.ndarray]:
        """Optimize for minimum volatility portfolio."""
        if EfficientFrontier is None:
            return self._fallback_equal_weights()

        ef = EfficientFrontier(self.mu, self.S)
        self._solve_with_fallback(ef, 'min_volatility')
        cleaned_weights = ef.clean_weights()

        performance = ef.portfolio_performance(verbose=False)

        return {
            'weights': np.array(list(cleaned_weights.values())),
            'expected_return': performance[0],
            'expected_volatility': performance[1],
            'sharpe_ratio': performance[2],
            'method': 'min_volatility'
        }

    def efficient_return(self, target_return: float) -> Dict[str, np.ndarray]:
        """Efficient portfolio with target return."""
        if EfficientFrontier is None:
            return self._fallback_equal_weights()

        ef = EfficientFrontier(self.mu, self.S)
        self._solve_with_fallback(ef, 'efficient_return', target_return)
        cleaned_weights = ef.clean_weights()

        performance = ef.portfolio_performance(verbose=False)

        return {
            'weights': np.array(list(cleaned_weights.values())),
            'expected_return': performance[0],
            'expected_volatility': performance[1],
            'sharpe_ratio': performance[2],
            'method': 'efficient_return'
        }

    def equal_weight_portfolio(self) -> Dict[str, np.ndarray]:
        """Equal weight portfolio (naive benchmark)."""
        weights = np.ones(self.n_assets) / self.n_assets

        # Calculate performance metrics safely
        try:
            if self.mu is not None and not np.any(np.isnan(self.mu)) and not np.any(np.isinf(self.mu)):
                if self.S is not None and not np.any(np.isnan(self.S)) and not np.any(np.isinf(self.S)):
                    expected_return = np.dot(weights, self.mu)
                    expected_vol = np.sqrt(np.dot(weights.T, np.dot(self.S, weights)))
                    sharpe = expected_return / expected_vol if expected_vol > 0 else 0.0
                else:
                    # Fallback to historical means
                    portfolio_returns = self.returns.mean(axis=1).dropna()
                    expected_return = portfolio_returns.mean()
                    expected_vol = portfolio_returns.std()
                    sharpe = expected_return / expected_vol if expected_vol > 0 else 0.0
            else:
                # Direct calculation from returns
                portfolio_returns = self.returns.mean(axis=1).dropna()
                expected_return = portfolio_returns.mean()
                expected_vol = portfolio_returns.std()
                sharpe = expected_return / expected_vol if expected_vol > 0 else 0.0
        except Exception:
            # Ultimate fallback
            expected_return = expected_vol = sharpe = 0.0

        return {
            'weights': weights,
            'expected_return': expected_return,
            'expected_volatility': expected_vol,
            'sharpe_ratio': sharpe,
            'method': 'equal_weight'
        }

    def _fallback_equal_weights(self, method_name: str = 'fallback_equal') -> Dict[str, np.ndarray]:
        """Fallback when optimization fails."""
        weights = np.ones(self.n_assets) / self.n_assets

        # Try to calculate basic metrics from historical data
        if self.mu is not None and self.S is not None:
            try:
                expected_return = np.dot(weights, self.mu)
                expected_vol = np.sqrt(np.dot(weights.T, np.dot(self.S, weights)))
                sharpe = expected_return / expected_vol if expected_vol > 0 else 0.0
            except:
                expected_return = expected_vol = sharpe = 0.0
        else:
            # From returns data directly
            try:
                portfolio_returns = self.returns.mean(axis=1).dropna()
                expected_return = portfolio_returns.mean()
                expected_vol = portfolio_returns.std()
                sharpe = expected_return / expected_vol if expected_vol > 0 else 0.0
            except:
                expected_return = expected_vol = sharpe = 0.0

        return {
            'weights': weights,
            'expected_return': expected_return,
            'expected_volatility': expected_vol,
            'sharpe_ratio': sharpe,
            'method': method_name
        }

    def get_all_portfolios(self, risk_free_rate: float = 0.02) -> Dict[str, Dict]:
        """Get all portfolio optimization results."""
        portfolios = {}

        try:
            portfolios['max_sharpe'] = self.max_sharpe_portfolio(risk_free_rate)
        except Exception as e:
            logger.error(f"Error in max_sharpe: {e}")
            portfolios['max_sharpe'] = self._fallback_equal_weights()

        try:
            portfolios['min_volatility'] = self.min_volatility_portfolio()
        except Exception as e:
            logger.error(f"Error in min_volatility: {e}")
            portfolios['min_volatility'] = self._fallback_equal_weights()

        # Efficient return at average expected return
        if self.mu is not None:
            avg_return = np.mean(self.mu)
            try:
                portfolios['efficient_return'] = self.efficient_return(avg_return)
            except Exception as e:
                logger.error(f"Error in efficient_return: {e}")
                portfolios['efficient_return'] = self._fallback_equal_weights()
        else:
            portfolios['efficient_return'] = self._fallback_equal_weights()

        portfolios['equal_weight'] = self.equal_weight_portfolio()

        return portfolios

class PortfolioBacktester:
    """Backtester for portfolio strategies."""

    def __init__(self, price_data: pd.DataFrame, initial_capital: float = 100000):
        """
        Initialize with price data.
        price_data: DataFrame with OHLCV data for each asset
        """
        self.price_data = price_data
        self.initial_capital = initial_capital
        self.assets = [col.replace('_close', '') for col in price_data.columns if '_close' in col]

    def backtest_portfolio(self, weights: np.ndarray, rebalance_freq: str = 'daily',
                          transaction_cost: float = 0.001) -> Dict:
        """Backtest a fixed-weight portfolio."""
        # For simplicity, assume daily rebalancing

        portfolio_values = []
        current_weights = weights.copy()
        positions = {asset: (weights[i] * self.initial_capital) / self.price_data.iloc[0][f"{asset}_close"]
                    for i, asset in enumerate(self.assets)}
        cash = 0.0  # Assume no extra cash for now

        for idx, row in self.price_data.iterrows():
            # Calculate current portfolio value
            portfolio_value = sum(positions[asset] * row[f"{asset}_close"] for asset in self.assets) + cash
            portfolio_values.append(portfolio_value)

            # Skip if last day
            if idx == self.price_data.index[-1]:
                break

            # Rebalance if needed (for now, daily)
            if rebalance_freq == 'daily':
                total_value = portfolio_value

                # Apply transaction costs on rebalancing
                new_positions = {}
                total_cost = 0

                for i, asset in enumerate(self.assets):
                    target_value = weights[i] * total_value
                    target_shares = target_value / row[f"{asset}_close"]
                    current_value = positions[asset] * row[f"{asset}_close"]
                    shares_to_trade = target_shares - positions[asset]

                    # Transaction cost
                    trade_cost = abs(shares_to_trade) * row[f"{asset}_close"] * transaction_cost
                    total_cost += trade_cost

                    new_positions[asset] = target_shares

                # Update positions
                positions = new_positions
                cash = total_value - sum(positions[asset] * row[f"{asset}_close"] for asset in self.assets) - total_cost

        # Calculate returns
        returns = pd.Series(portfolio_values).pct_change().fillna(0)

        results = {
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),  # Annualized
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'volatility': np.sqrt(252) * returns.std(),  # Annualized
            'portfolio_values': portfolio_values,
            'returns': returns.tolist()
        }

        return results

    def _calculate_max_drawdown(self, values: list) -> float:
        """Calculate maximum drawdown."""
        peak = values[0]
        max_drawdown = 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'HDFCBANK_close': 100 + np.random.randn(100).cumsum(),
        'ICICIBANK_close': 100 + np.random.randn(100).cumsum(),
        'returns_HDFCBANK': np.random.randn(100) * 0.02,
        'returns_ICICIBANK': np.random.randn(100) * 0.02,
    }, index=dates)

    returns_data = data[['returns_HDFCBANK', 'returns_ICICIBANK']]

    optimizer = ClassicalPortfolioOptimizer(returns_data)
    portfolios = optimizer.get_all_portfolios()

    print("Classical Portfolios:")
    for name, port in portfolios.items():
        print("2d")
