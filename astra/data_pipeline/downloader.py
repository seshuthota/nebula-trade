import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioDataDownloader:
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
                config_path = candidate.resolve()
                break
        else:
            raise FileNotFoundError(f"Config file not found. Tried: {[str(c) for c in candidates]}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        self.interval = self._map_interval(self.config['data']['interval'])
        self.lookback_days = self.config['data']['lookback_days']
        self.start_date = self.config['data']['start_date']
        self.end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    def _map_interval(self, interval_str):
        """Map config interval to yfinance format."""
        mapping = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
        return mapping.get(interval_str, '1d')

    def download_asset_data(self, ticker: str) -> pd.DataFrame:
        """Download historical data for a single asset."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date,
                              end=self.end_date,
                              interval=self.interval)

            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()

            # Rename columns to standard OHLCV
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'date'

            logger.info(f"Downloaded {len(df)} bars for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return pd.DataFrame()

    def download_portfolio_data(self) -> pd.DataFrame:
        """Download data for all assets in portfolio."""
        portfolio_data = {}

        for asset in self.assets:
            logger.info(f"Downloading data for {asset}")
            df = self.download_asset_data(asset)
            if not df.empty:
                portfolio_data[asset] = df

        if not portfolio_data:
            raise ValueError("Failed to download data for any assets")

        # Combine all assets into wide format
        combined_data = pd.concat(portfolio_data, axis=1, keys=portfolio_data.keys())

        # Flatten column names
        combined_data.columns = [f"{asset}_{col}" for asset, col in combined_data.columns]

        # Ensure common date index
        combined_data = combined_data.sort_index().ffill()

        return combined_data

    def save_data(self, data: pd.DataFrame, output_path: str = "data/portfolio_data.csv"):
        """Save portfolio data to CSV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path)
        logger.info(f"Saved portfolio data to {output_path}")

if __name__ == "__main__":
    downloader = PortfolioDataDownloader()
    data = downloader.download_portfolio_data()
    downloader.save_data(data)
