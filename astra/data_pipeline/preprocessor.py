import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioDataPreprocessor:
    def __init__(self, config_path: str = "../config/portfolio.yaml"):
        project_root = Path(__file__).resolve().parents[2]

        cfg_path = Path(config_path)
        candidates = []
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
                cfg_path = candidate.resolve()
                break
        else:
            raise FileNotFoundError(f"Config file not found. Tried: {[str(c) for c in candidates]}")

        with open(cfg_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']

    def load_data(self, filepath: str = "data/portfolio_data.csv") -> pd.DataFrame:
        """Load portfolio data from CSV."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index.name = 'date'
        return df

    def normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize close prices for each asset to start at 1."""
        normalized_df = df.copy()

        for asset in self.assets:
            close_col = f"{asset}_close"
            if close_col in normalized_df.columns:
                first_price = normalized_df[close_col].iloc[0]
                normalized_df[f"{asset}_normalized"] = normalized_df[close_col] / first_price

        return normalized_df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns for each asset."""
        returns_df = df.copy()

        for asset in self.assets:
            close_col = f"{asset}_close"
            if close_col in returns_df.columns:
                returns_df[f"{asset}_returns"] = returns_df[close_col].pct_change()
                # Clean returns: clip to reasonable range and remove inf
                returns_df[f"{asset}_returns"] = np.clip(returns_df[f"{asset}_returns"], -1.0, 1.0)  # +/-100% daily
                returns_df[f"{asset}_returns"] = returns_df[f"{asset}_returns"].replace([np.inf, -np.inf], np.nan)

                # Skip log returns for now due to data quality issues
                # returns_df[f"{asset}_log_returns"] = np.log(returns_df[f"{asset}_returns"] + 1)  # approximate

        return returns_df

    def add_technical_indicators(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add technical indicators (SMA, EMA, volatility)."""
        tech_df = df.copy()

        for asset in self.assets:
            close_col = f"{asset}_close"
            if close_col in tech_df.columns:
                # Simple moving average
                tech_df[f"{asset}_sma_{window}"] = tech_df[close_col].rolling(window).mean()
                # Exponential moving average
                tech_df[f"{asset}_ema_{window}"] = tech_df[close_col].ewm(span=window).mean()
                # Rolling volatility (30-day)
                returns_col = f"{asset}_returns"
                if returns_col in tech_df.columns:
                    tech_df[f"{asset}_volatility"] = tech_df[returns_col].rolling(30).std()

        return tech_df

    def add_portfolio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add portfolio-level features like correlations, beta to market."""
        portfolio_df = df.copy()

        # Extract returns matrix for correlation
        returns_cols = [f"{asset}_returns" for asset in self.assets if f"{asset}_returns" in portfolio_df.columns]
        if returns_cols:
            returns_df = portfolio_df[returns_cols]
            # Rolling correlation matrix (flatten for features)
            for i, asset1 in enumerate(self.assets):
                if f"{asset1}_returns" not in portfolio_df.columns:
                    continue
                for j, asset2 in enumerate(self.assets):
                    if i < j and f"{asset2}_returns" in portfolio_df.columns:
                        corr_col = f"corr_{asset1}_{asset2}"
                        portfolio_df[corr_col] = portfolio_df[f"{asset1}_returns"].rolling(30).corr(portfolio_df[f"{asset2}_returns"])

            # Portfolio volatility (simple average)
            portfolio_df['portfolio_volatility'] = portfolio_df[returns_cols].std(axis=1)

        return portfolio_df

    def clean_data(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Clean data: drop NaN values, handle outliers."""
        cleaned_df = df.copy()

        if drop_na:
            # Drop initial rows with NaN
            cleaned_df = cleaned_df.dropna()

        logger.info(f"Data cleaned: {len(cleaned_df)} rows remaining")
        return cleaned_df

    def process_all(self, filepath: str = "data/portfolio_data.csv") -> pd.DataFrame:
        """Run full preprocessing pipeline."""
        logger.info("Starting data preprocessing...")

        # Load
        df = self.load_data(filepath)

        # Normalize prices
        df = self.normalize_prices(df)

        # Add returns
        df = self.add_returns(df)

        # Add technical indicators
        df = self.add_technical_indicators(df)

        # Add portfolio features
        df = self.add_portfolio_features(df)

        # Clean data
        df = self.clean_data(df)

        logger.info(f"Preprocessing complete: {df.shape} shape")
        return df

if __name__ == "__main__":
    preprocessor = PortfolioDataPreprocessor()
    processed_data = preprocessor.process_all()
    print(processed_data.head())
