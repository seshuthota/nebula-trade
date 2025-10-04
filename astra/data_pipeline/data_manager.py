import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from .preprocessor import PortfolioDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioDataManager:
    def __init__(self, config_path: str = None):
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
                self.config_path = candidate.resolve()
                break
        else:
            raise FileNotFoundError(f"Config file not found. Tried: {[str(c) for c in candidates]}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        self.initial_weights_config = self.config['initial_weights']

    def load_processed_data(self, filepath: str = "data/portfolio_data_processed.csv") -> pd.DataFrame:
        """Load processed portfolio data."""
        return pd.read_csv(filepath, index_col=0, parse_dates=True)

    def generate_equal_weights(self, n_assets: int, cash_buffer: float) -> np.ndarray:
        """Generate equal weights with cash buffer."""
        n_assets_with_cash = n_assets + 1 if cash_buffer > 0 else n_assets
        equal_weight = (1 - cash_buffer) / n_assets
        weights = np.full(n_assets, equal_weight)
        if cash_buffer > 0:
            weights = np.append(weights, cash_buffer)
        return weights

    def generate_market_cap_weights(self, data: pd.DataFrame) -> np.ndarray:
        """Generate market cap based weights from current prices."""
        # Simple approximation: use recent market caps
        # In practice, you'd fetch actual market caps
        weights = np.ones(len(self.assets)) / len(self.assets)
        logger.warning("Market cap weights approximated as equal - implement actual MCAP fetch")
        return weights

    def generate_initialized_weights(self) -> np.ndarray:
        """Generate initial portfolio weights based on config."""
        mode = self.initial_weights_config['mode']
        cash_buffer = self.initial_weights_config['cash_buffer']

        if mode == 'equal':
            weights = self.generate_equal_weights(len(self.assets), cash_buffer)
        elif mode == 'market_cap':
            # This would need data - placeholder for now
            weights = self.generate_market_cap_weights(None)
            if cash_buffer > 0:
                weights = weights * (1 - cash_buffer)
                weights = np.append(weights, cash_buffer)
        elif mode == 'custom':
            # Custom weights from config
            weights = np.array(self.initial_weights_config.get('weights', []))
            if not weights.size:
                raise ValueError("Custom mode requires 'weights' list in config")
            if cash_buffer > 0:
                weights = weights * (1 - cash_buffer)
                weights = np.append(weights, cash_buffer)
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        # Validate weights sum to 1
        if not np.isclose(weights.sum(), 1.0):
            logger.warning(f"Weights don't sum to 1: {weights.sum()}, normalizing")
            weights = weights / weights.sum()

        return weights

    def get_initial_portfolio_state(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Get initial portfolio state for RL environment."""
        # Start from first available date
        start_date = data.index[0]

        # Current prices
        current_prices = {}
        for asset in self.assets:
            current_prices[asset] = data.loc[start_date, f"{asset}_close"]

        # Initial weights
        asset_weights = self.generate_initialized_weights()
        weights = asset_weights[:-1] if len(asset_weights) > len(self.assets) else asset_weights
        cash_weight = asset_weights[-1] if len(asset_weights) > len(self.assets) else 0.0

        # Initial positions
        positions = {}
        for i, asset in enumerate(self.assets):
            positions[asset] = weights[i] * initial_capital / current_prices[asset]

        cash_position = cash_weight * initial_capital

        initial_state = {
            'date': start_date,
            'asset_prices': current_prices,
            'portfolio_value': initial_capital,
            'positions': positions,
            'cash': cash_position,
            'weights': dict(zip(self.assets + ['cash'], asset_weights)),
            'capital': initial_capital
        }

        logger.info(f"Initial portfolio state created: {initial_capital} capital, {len(self.assets)} assets")
        return initial_state

    def process_and_initialize(self,
                              input_file: str = "data/portfolio_data.csv",
                              output_file: str = "data/portfolio_data_processed.csv") -> Tuple[pd.DataFrame, Dict]:
        """Complete pipeline: process data and initialize portfolio."""
        preprocessor = PortfolioDataPreprocessor(self.config_path)

        # Process raw data
        processed_data = preprocessor.process_all(input_file)

        # Save processed data
        processed_data.to_csv(output_file)
        logger.info(f"Saved processed data to {output_file}")

        # Get initial state
        initial_state = self.get_initial_portfolio_state(processed_data)

        return processed_data, initial_state

if __name__ == "__main__":
    manager = PortfolioDataManager()
    data, initial_state = manager.process_and_initialize()
    print("Initial state:", initial_state)
