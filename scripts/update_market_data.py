#!/usr/bin/env python3
"""
Market Data Update Script for Production Deployment.

Fetches latest data for 5 Indian bank stocks and updates the dataset.
Validates data quality before updating production dataset.
"""

import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
from datetime import datetime, timedelta
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketDataUpdater:
    """Fetch and update market data for production training."""

    def __init__(self, config_path: str = "config/portfolio.yaml"):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / config_path
        self.data_path = self.project_root / "data" / "portfolio_data_processed.csv"

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']

        self.assets = self.config['assets']
        logger.info(f"Initialized MarketDataUpdater for {len(self.assets)} assets")

    def load_current_data(self) -> pd.DataFrame:
        """Load currently saved dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded current data: {len(data)} samples")
        logger.info(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

        return data

    def fetch_latest_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch latest market data from Yahoo Finance."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching data from {start_date} to {end_date}")

        all_data = []

        for asset in self.assets:
            logger.info(f"Fetching {asset}...")

            try:
                # Fetch data from Yahoo Finance
                ticker = yf.Ticker(asset)
                df = ticker.history(start=start_date, end=end_date)

                if df.empty:
                    logger.warning(f"No data fetched for {asset}")
                    continue

                # Rename columns to match our format
                df_renamed = pd.DataFrame({
                    f'{asset}_open': df['Open'],
                    f'{asset}_high': df['High'],
                    f'{asset}_low': df['Low'],
                    f'{asset}_close': df['Close'],
                    f'{asset}_volume': df['Volume']
                })

                # Calculate returns
                df_renamed[f'{asset}_returns'] = df_renamed[f'{asset}_close'].pct_change()

                all_data.append(df_renamed)
                logger.info(f"  Fetched {len(df)} samples for {asset}")

            except Exception as e:
                logger.error(f"Error fetching {asset}: {str(e)}")
                raise

        if not all_data:
            raise ValueError("No data fetched for any asset")

        # Combine all asset data
        combined_data = pd.concat(all_data, axis=1)

        # Handle timezone
        if combined_data.index.tz is None:
            combined_data.index = pd.to_datetime(combined_data.index).tz_localize('Asia/Kolkata')
        else:
            combined_data.index = pd.to_datetime(combined_data.index).tz_convert('Asia/Kolkata')

        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Date range: {combined_data.index[0].date()} to {combined_data.index[-1].date()}")

        return combined_data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality."""
        logger.info("Validating data quality...")

        issues = []

        # Check for missing values
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found {null_counts.sum()} null values")
            for col in null_counts[null_counts > 0].index:
                logger.warning(f"  {col}: {null_counts[col]} nulls")
            issues.append("null_values")

        # Check for infinite values
        inf_counts = data.isin([float('inf'), float('-inf')]).sum()
        if inf_counts.sum() > 0:
            logger.warning(f"Found {inf_counts.sum()} infinite values")
            issues.append("infinite_values")

        # Check for extreme values (returns > 50%)
        returns_cols = [col for col in data.columns if col.endswith('_returns')]
        for col in returns_cols:
            extreme = data[col].abs() > 0.5
            if extreme.sum() > 0:
                logger.warning(f"{col}: {extreme.sum()} extreme values (>50%)")
                logger.warning(f"  Max: {data[col].max():.4f}, Min: {data[col].min():.4f}")

        # Check for zero volume
        volume_cols = [col for col in data.columns if col.endswith('_volume')]
        for col in volume_cols:
            zero_vol = (data[col] == 0).sum()
            if zero_vol > 0:
                logger.warning(f"{col}: {zero_vol} days with zero volume")

        if issues:
            logger.warning(f"Data validation found issues: {issues}")
            return False
        else:
            logger.info("✅ Data validation passed")
            return True

    def merge_and_save(self, current_data: pd.DataFrame, new_data: pd.DataFrame,
                      force: bool = False):
        """Merge new data with existing and save."""
        logger.info("Merging new data with existing dataset...")

        # Find overlap
        last_current_date = current_data.index[-1]
        first_new_date = new_data.index[0]

        logger.info(f"Current data ends: {last_current_date.date()}")
        logger.info(f"New data starts: {first_new_date.date()}")

        # Filter new data to only dates after last current date
        new_data_filtered = new_data[new_data.index > last_current_date]

        if len(new_data_filtered) == 0:
            logger.info("No new data to add (dataset is already up to date)")
            return False

        logger.info(f"Adding {len(new_data_filtered)} new samples")

        # Concatenate
        combined = pd.concat([current_data, new_data_filtered])
        combined = combined.sort_index()

        # Validate combined dataset
        if not self.validate_data(combined):
            if not force:
                logger.error("Validation failed on combined dataset")
                raise ValueError("Data validation failed. Use --force to override")
            else:
                logger.warning("Validation failed but continuing due to --force flag")

        # Create backup
        backup_path = self.data_path.parent / f"portfolio_data_processed_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        logger.info(f"Creating backup: {backup_path}")
        current_data.to_csv(backup_path)

        # Save updated dataset
        logger.info(f"Saving updated dataset to: {self.data_path}")
        combined.to_csv(self.data_path)

        # Save update log
        log_path = self.data_path.parent / "data_updates.log"
        with open(log_path, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: Added {len(new_data_filtered)} samples "
                   f"from {new_data_filtered.index[0].date()} to {new_data_filtered.index[-1].date()}\n")

        logger.info("=" * 80)
        logger.info("DATA UPDATE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Previous samples: {len(current_data)}")
        logger.info(f"New samples added: {len(new_data_filtered)}")
        logger.info(f"Total samples: {len(combined)}")
        logger.info(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
        logger.info("=" * 80)

        return True

    def run_update(self, force: bool = False):
        """Run complete data update pipeline."""
        logger.info("=" * 80)
        logger.info("MARKET DATA UPDATE")
        logger.info("=" * 80)

        try:
            # 1. Load current data
            current_data = self.load_current_data()

            # 2. Determine date range for fetch
            last_date = current_data.index[-1]
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            # Check if update is needed
            days_behind = (datetime.now().date() - last_date.date()).days
            logger.info(f"Current data is {days_behind} days behind")

            if days_behind == 0:
                logger.info("✅ Data is already up to date!")
                return

            # 3. Fetch latest data
            new_data = self.fetch_latest_data(start_date, end_date)

            # 4. Validate new data
            if not self.validate_data(new_data):
                if not force:
                    logger.error("New data validation failed")
                    raise ValueError("Data validation failed. Use --force to override")
                else:
                    logger.warning("Validation failed but continuing due to --force flag")

            # 5. Merge and save
            updated = self.merge_and_save(current_data, new_data, force=force)

            if updated:
                logger.info("✅ Data update successful!")
            else:
                logger.info("ℹ️  No updates were needed")

        except Exception as e:
            logger.error(f"Data update failed: {str(e)}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Update market data for production training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update data with latest available
  python scripts/update_market_data.py

  # Force update even if validation fails
  python scripts/update_market_data.py --force

  # Use custom config
  python scripts/update_market_data.py --config config/production.yaml

Note: This script:
- Fetches data from Yahoo Finance
- Validates data quality (checks for nulls, inf, extreme values)
- Creates backup before updating
- Logs all updates to data/data_updates.log
        """
    )

    parser.add_argument('--config', type=str, default='config/portfolio.yaml',
                       help='Path to config file (default: config/portfolio.yaml)')
    parser.add_argument('--force', action='store_true',
                       help='Force update even if validation fails')

    args = parser.parse_args()

    updater = MarketDataUpdater(config_path=args.config)
    updater.run_update(force=args.force)


if __name__ == "__main__":
    main()
