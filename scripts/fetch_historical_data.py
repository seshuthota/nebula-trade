#!/usr/bin/env python3
"""
Fetch Historical Data (2007-2014) for v4 Model Training

This script fetches 2007-2014 data from Yahoo Finance to expand our dataset
with rich bear market periods (2008 GFC, 2011 European crisis, etc.)

Goal: Create 2007-2024 dataset with 12-15% bear market exposure (vs 2.7% in 2015-2024)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_historical_data(
    tickers: list,
    start_date: str = "2007-01-01",
    end_date: str = "2014-12-31"
) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance for specified period.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data for all tickers
    """
    logger.info("=" * 80)
    logger.info("FETCHING HISTORICAL DATA FROM YAHOO FINANCE")
    logger.info("=" * 80)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Tickers: {tickers}")
    logger.info("")

    all_data = {}

    for ticker in tickers:
        logger.info(f"Fetching {ticker}...")

        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=True)

            if hist.empty:
                logger.error(f"  ❌ No data returned for {ticker}")
                continue

            # Rename columns to match our format
            hist = hist.rename(columns={
                'Open': f'{ticker}_open',
                'High': f'{ticker}_high',
                'Low': f'{ticker}_low',
                'Close': f'{ticker}_close',
                'Volume': f'{ticker}_volume'
            })

            # Keep only OHLCV columns
            cols = [f'{ticker}_open', f'{ticker}_high', f'{ticker}_low',
                   f'{ticker}_close', f'{ticker}_volume']
            hist = hist[cols]

            all_data[ticker] = hist

            logger.info(f"  ✅ Fetched {len(hist)} samples")
            logger.info(f"     Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
            logger.info(f"     Sample close prices: {hist[f'{ticker}_close'].iloc[:3].values}")

        except Exception as e:
            logger.error(f"  ❌ Error fetching {ticker}: {str(e)}")
            continue

    if not all_data:
        raise ValueError("No data fetched for any ticker!")

    # Combine all ticker data
    logger.info("")
    logger.info("Combining data from all tickers...")

    combined = pd.concat(all_data.values(), axis=1)
    combined.index.name = 'date'

    # Ensure timezone awareness (IST)
    if combined.index.tz is None:
        combined.index = pd.to_datetime(combined.index).tz_localize('Asia/Kolkata')
    else:
        combined.index = pd.to_datetime(combined.index).tz_convert('Asia/Kolkata')

    logger.info(f"✅ Combined dataset: {combined.shape}")
    logger.info(f"   Columns: {len(combined.columns)}")
    logger.info(f"   Rows: {len(combined)}")
    logger.info(f"   Date range: {combined.index[0].date()} to {combined.index[-1].date()}")

    return combined


def validate_data_quality(data: pd.DataFrame, tickers: list) -> dict:
    """
    Validate data quality and check for issues.

    Returns:
        Dict with validation metrics
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("=" * 80)

    metrics = {
        'total_rows': len(data),
        'total_cols': len(data.columns),
        'date_range': (data.index[0].date(), data.index[-1].date()),
        'missing_values': {},
        'price_ranges': {},
        'suspicious_values': []
    }

    # Check for missing values
    logger.info("Checking for missing values...")
    for col in data.columns:
        missing = data[col].isna().sum()
        if missing > 0:
            pct = missing / len(data) * 100
            metrics['missing_values'][col] = f"{missing} ({pct:.1f}%)"
            logger.warning(f"  {col}: {missing} missing ({pct:.1f}%)")

    if not metrics['missing_values']:
        logger.info("  ✅ No missing values found")

    # Check price ranges
    logger.info("")
    logger.info("Validating price ranges...")
    for ticker in tickers:
        close_col = f'{ticker}_close'
        if close_col in data.columns:
            prices = data[close_col].dropna()
            min_price = prices.min()
            max_price = prices.max()
            median_price = prices.median()

            metrics['price_ranges'][ticker] = {
                'min': float(min_price),
                'max': float(max_price),
                'median': float(median_price)
            }

            logger.info(f"  {ticker}: Min={min_price:.2f}, Max={max_price:.2f}, Median={median_price:.2f}")

            # Check for suspicious values (e.g., prices near 0 or huge spikes)
            if min_price < 1.0:
                metrics['suspicious_values'].append(f"{ticker}: Very low prices (min={min_price:.2f})")
                logger.warning(f"    ⚠️  Very low minimum price: {min_price:.2f}")

            # Check for huge daily changes (>50% in a day - likely split not adjusted)
            daily_returns = prices.pct_change().abs()
            max_daily_change = daily_returns.max()
            if max_daily_change > 0.5:
                metrics['suspicious_values'].append(f"{ticker}: Large daily change ({max_daily_change:.1%})")
                logger.warning(f"    ⚠️  Suspicious daily change: {max_daily_change:.1%}")

    # Check date continuity
    logger.info("")
    logger.info("Checking date continuity...")
    date_diffs = pd.Series(data.index).diff().dt.days.dropna()
    gaps = date_diffs[date_diffs > 7]  # Gaps longer than a week (excluding weekends)

    if len(gaps) > 0:
        logger.warning(f"  Found {len(gaps)} date gaps > 7 days")
        for idx, gap in gaps.items():
            if idx > 0:
                logger.warning(f"    {data.index[idx-1].date()} -> {data.index[idx].date()}: {gap} days")
    else:
        logger.info("  ✅ No significant date gaps found")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"VALIDATION SUMMARY: {'✅ PASS' if len(metrics['suspicious_values']) == 0 else '⚠️  WARNINGS'}")
    logger.info("=" * 80)

    return metrics


def merge_with_existing_data(
    historical_data: pd.DataFrame,
    existing_data_path: str
) -> pd.DataFrame:
    """
    Merge historical data (2007-2014) with existing data (2015-2024).

    Returns:
        Combined dataset (2007-2024)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("MERGING WITH EXISTING DATA")
    logger.info("=" * 80)

    # Load existing data
    logger.info(f"Loading existing data from: {existing_data_path}")
    existing_data = pd.read_csv(existing_data_path, index_col=0, parse_dates=True)

    logger.info(f"Existing data: {existing_data.shape}")
    logger.info(f"  Date range: {existing_data.index[0].date()} to {existing_data.index[-1].date()}")

    logger.info(f"Historical data: {historical_data.shape}")
    logger.info(f"  Date range: {historical_data.index[0].date()} to {historical_data.index[-1].date()}")

    # Ensure both have same timezone
    if existing_data.index.tz is None and historical_data.index.tz is not None:
        existing_data.index = pd.to_datetime(existing_data.index).tz_localize(historical_data.index.tz)
    elif existing_data.index.tz is not None and historical_data.index.tz is None:
        historical_data.index = pd.to_datetime(historical_data.index).tz_localize(existing_data.index.tz)

    # Align columns (historical data has only OHLCV, existing has more features)
    # We only keep OHLCV columns for merging
    common_cols = [col for col in historical_data.columns if col in existing_data.columns]

    logger.info(f"Common columns: {len(common_cols)}")
    logger.info(f"  {common_cols[:5]}...")  # Show first 5

    # Merge datasets
    merged = pd.concat([
        historical_data[common_cols],
        existing_data[common_cols]
    ], axis=0)

    # Sort by date
    merged = merged.sort_index()

    # Remove duplicates (if any overlap)
    duplicates = merged.index.duplicated()
    if duplicates.any():
        logger.warning(f"Removing {duplicates.sum()} duplicate dates")
        merged = merged[~duplicates]

    logger.info("")
    logger.info(f"✅ Merged dataset: {merged.shape}")
    logger.info(f"   Date range: {merged.index[0].date()} to {merged.index[-1].date()}")
    logger.info(f"   Total samples: {len(merged)}")

    # Validate no gaps at merge boundary
    merge_point = existing_data.index[0]
    historical_end = historical_data.index[-1]
    gap_days = (merge_point - historical_end).days

    if gap_days > 7:
        logger.warning(f"  ⚠️  Gap at merge boundary: {gap_days} days")
    else:
        logger.info(f"  ✅ Clean merge (gap: {gap_days} days)")

    return merged


def main():
    """Main execution pipeline."""

    # Configuration
    tickers = [
        'HDFCBANK.NS',
        'ICICIBANK.NS',
        'SBIN.NS',
        'AXISBANK.NS',
        'KOTAKBANK.NS'
    ]

    start_date = "2007-01-01"
    end_date = "2014-12-31"

    existing_data_path = "notebooks/data/portfolio_data.csv"
    output_path_historical = "notebooks/data/portfolio_data_2007_2014.csv"
    output_path_combined = "notebooks/data/portfolio_data_2007_2024.csv"

    try:
        # Step 1: Fetch historical data
        logger.info("STEP 1: Fetching historical data (2007-2014)...")
        historical_data = fetch_historical_data(tickers, start_date, end_date)

        # Step 2: Validate data quality
        logger.info("\nSTEP 2: Validating data quality...")
        validation_metrics = validate_data_quality(historical_data, tickers)

        # Step 3: Save historical data
        logger.info(f"\nSTEP 3: Saving historical data to {output_path_historical}...")
        historical_data.to_csv(output_path_historical)
        logger.info(f"✅ Saved {len(historical_data)} samples")

        # Step 4: Merge with existing data
        logger.info(f"\nSTEP 4: Merging with existing data...")
        combined_data = merge_with_existing_data(historical_data, existing_data_path)

        # Step 5: Save combined dataset
        logger.info(f"\nSTEP 5: Saving combined data to {output_path_combined}...")
        combined_data.to_csv(output_path_combined)
        logger.info(f"✅ Saved {len(combined_data)} samples")

        # Final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ DATA FETCH COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Historical data (2007-2014): {len(historical_data)} samples")
        logger.info(f"  Saved to: {output_path_historical}")
        logger.info(f"Combined data (2007-2024): {len(combined_data)} samples")
        logger.info(f"  Saved to: {output_path_combined}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run bear period detection on 2007-2024 data")
        logger.info("2. Expect 8-12 bear periods (vs 3 in 2015-2024)")
        logger.info("3. Calculate new weights (~6-8x vs 35x)")
        logger.info("4. Train v4 model with rich bear market data")

        # Save validation metrics
        import json
        metrics_path = "notebooks/data/validation_metrics_2007_2014.json"
        with open(metrics_path, 'w') as f:
            # Convert non-serializable values
            metrics_to_save = validation_metrics.copy()
            metrics_to_save['date_range'] = [str(d) for d in metrics_to_save['date_range']]
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Validation metrics saved to: {metrics_path}")

    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
