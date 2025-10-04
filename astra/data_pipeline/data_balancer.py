"""
Data Balancer for Stage 2: Training Data Rebalancing

This module provides tools to:
1. Detect bear market periods in historical data
2. Create balanced datasets with equal bull/bear representation
3. Generate sample weights for weighted random sampling

Goal: Train RL models on balanced bull/bear data for all-weather performance
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_bear_periods(
    data: pd.DataFrame,
    assets: List[str],
    lookback_window: int = 30,
    drawdown_threshold: float = 0.10,
    volatility_threshold: float = 0.02,
    min_duration: int = 10
) -> List[Tuple[str, str, Dict]]:
    """
    Detect bear market periods in portfolio data.

    Criteria for bear market:
    1. Portfolio drops >10% from recent peak (rolling lookback)
    2. High volatility (daily std > 2%)
    3. Duration > 10 days
    4. Negative trend across multiple assets

    Args:
        data: Portfolio data with price columns
        assets: List of asset tickers
        lookback_window: Window for peak calculation (default 30 days)
        drawdown_threshold: Min drawdown to qualify as bear (default 10%)
        volatility_threshold: Min daily volatility (default 2%)
        min_duration: Minimum bear period length (default 10 days)

    Returns:
        List of (start_date, end_date, metrics_dict) for each bear period
    """
    logger.info("=" * 80)
    logger.info("BEAR PERIOD DETECTION")
    logger.info("=" * 80)
    logger.info(f"Analyzing data from {data.index[0].date()} to {data.index[-1].date()}")
    logger.info(f"Lookback window: {lookback_window} days")
    logger.info(f"Drawdown threshold: {drawdown_threshold:.1%}")
    logger.info(f"Volatility threshold: {volatility_threshold:.1%}")
    logger.info(f"Min duration: {min_duration} days")

    # Calculate equal-weight portfolio value
    close_cols = [f"{asset}_close" for asset in assets]
    portfolio_value = data[close_cols].mean(axis=1)

    # Calculate returns
    returns = portfolio_value.pct_change()

    # Calculate rolling peak
    rolling_peak = portfolio_value.rolling(window=lookback_window, min_periods=1).max()

    # Calculate drawdown from peak
    drawdown = (portfolio_value - rolling_peak) / rolling_peak

    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=20, min_periods=10).std()

    # Identify bear periods
    is_bear = (
        (drawdown < -drawdown_threshold) &  # Significant drawdown
        (rolling_vol > volatility_threshold)  # High volatility
    )

    # Find continuous bear periods
    bear_periods = []
    in_bear = False
    bear_start = None
    bear_metrics = {
        'max_drawdown': 0,
        'avg_volatility': 0,
        'cumulative_return': 0,
        'num_down_days': 0
    }

    for i, (date, is_bear_day) in enumerate(is_bear.items()):
        if is_bear_day and not in_bear:
            # Start of bear period
            in_bear = True
            bear_start = date
            bear_start_idx = i
            bear_metrics = {
                'max_drawdown': drawdown.iloc[i],
                'avg_volatility': rolling_vol.iloc[i],
                'cumulative_return': 0,
                'num_down_days': 0
            }

        elif in_bear and is_bear_day:
            # Continue bear period, update metrics
            bear_metrics['max_drawdown'] = min(bear_metrics['max_drawdown'], drawdown.iloc[i])
            if i < len(returns) and returns.iloc[i] < 0:
                bear_metrics['num_down_days'] += 1

        elif in_bear and not is_bear_day:
            # End of bear period
            in_bear = False
            bear_end = data.index[i-1]
            bear_end_idx = i - 1
            duration = bear_end_idx - bear_start_idx + 1

            # Check minimum duration
            if duration >= min_duration:
                # Calculate final metrics
                period_returns = returns.iloc[bear_start_idx:bear_end_idx+1]
                period_vol = rolling_vol.iloc[bear_start_idx:bear_end_idx+1]

                bear_metrics['avg_volatility'] = period_vol.mean()
                bear_metrics['cumulative_return'] = (portfolio_value.iloc[bear_end_idx] /
                                                     portfolio_value.iloc[bear_start_idx] - 1)
                bear_metrics['duration'] = duration

                bear_periods.append((
                    str(bear_start.date()),
                    str(bear_end.date()),
                    bear_metrics.copy()
                ))

                logger.info(f"Bear period found: {bear_start.date()} to {bear_end.date()}")
                logger.info(f"  Duration: {duration} days")
                logger.info(f"  Max drawdown: {bear_metrics['max_drawdown']:.2%}")
                logger.info(f"  Avg volatility: {bear_metrics['avg_volatility']:.2%}")
                logger.info(f"  Cumulative return: {bear_metrics['cumulative_return']:.2%}")
                logger.info(f"  Down days: {bear_metrics['num_down_days']}/{duration}")

    # Handle if still in bear at end of data
    if in_bear:
        bear_end = data.index[-1]
        bear_end_idx = len(data) - 1
        duration = bear_end_idx - bear_start_idx + 1

        if duration >= min_duration:
            period_returns = returns.iloc[bear_start_idx:]
            period_vol = rolling_vol.iloc[bear_start_idx:]

            bear_metrics['avg_volatility'] = period_vol.mean()
            bear_metrics['cumulative_return'] = (portfolio_value.iloc[-1] /
                                                 portfolio_value.iloc[bear_start_idx] - 1)
            bear_metrics['duration'] = duration

            bear_periods.append((
                str(bear_start.date()),
                str(bear_end.date()),
                bear_metrics.copy()
            ))

            logger.info(f"Bear period found (ongoing): {bear_start.date()} to {bear_end.date()}")
            logger.info(f"  Duration: {duration} days")
            logger.info(f"  Max drawdown: {bear_metrics['max_drawdown']:.2%}")

    logger.info("")
    logger.info(f"Total bear periods detected: {len(bear_periods)}")
    logger.info("=" * 80)

    return bear_periods


def create_sample_weights(
    data: pd.DataFrame,
    bear_periods: List[Tuple[str, str, Dict]],
    bear_weight: float = 4.0,
    bull_weight: float = 1.0
) -> np.ndarray:
    """
    Create sample weights for weighted random sampling.

    Assigns high weights to bear period samples, low weights to bull period samples.
    This creates a balanced 50/50 effective distribution during training.

    Args:
        data: Portfolio data with DatetimeIndex
        bear_periods: List of (start_date, end_date, metrics) tuples
        bear_weight: Weight for bear period samples (default 4.0)
        bull_weight: Weight for bull period samples (default 1.0)

    Returns:
        Array of sample weights, same length as data
    """
    logger.info("Creating sample weights for balanced training...")

    # Initialize all weights as bull (default)
    weights = np.full(len(data), bull_weight, dtype=np.float32)

    # Set bear period weights (handle timezone-aware index)
    bear_sample_count = 0
    for start_date, end_date, _ in bear_periods:
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        # Match timezone if data index is timezone-aware
        if data.index.tz is not None:
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize(data.index.tz)
            if end_ts.tz is None:
                end_ts = end_ts.tz_localize(data.index.tz)

        mask = (data.index >= start_ts) & (data.index <= end_ts)
        weights[mask] = bear_weight
        bear_sample_count += mask.sum()

    # Calculate effective distribution
    total_weight = weights.sum()
    bear_total_weight = bear_sample_count * bear_weight
    bull_total_weight = (len(data) - bear_sample_count) * bull_weight

    bear_ratio = bear_total_weight / total_weight
    bull_ratio = bull_total_weight / total_weight

    logger.info(f"Sample counts:")
    logger.info(f"  Total samples: {len(data)}")
    logger.info(f"  Bear samples: {bear_sample_count} ({bear_sample_count/len(data):.1%})")
    logger.info(f"  Bull samples: {len(data) - bear_sample_count} ({(len(data)-bear_sample_count)/len(data):.1%})")
    logger.info(f"")
    logger.info(f"Effective distribution (with weights {bear_weight}:{bull_weight}):")
    logger.info(f"  Bear ratio: {bear_ratio:.1%}")
    logger.info(f"  Bull ratio: {bull_ratio:.1%}")
    logger.info(f"  Balance achieved: {'YES ✅' if 0.45 <= bear_ratio <= 0.55 else 'NO ❌'}")

    return weights


def create_balanced_dataset(
    original_data: pd.DataFrame,
    assets: List[str],
    approach: str = 'weighted',
    val_split: float = 0.1,
    bear_weight: float = 4.0,
    bull_weight: float = 1.0,
    **bear_detection_kwargs
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, List[Tuple]]:
    """
    Create balanced training set with chronological validation.

    CRITICAL LOGIC:
    1. Take original chronological dataset
    2. Split off LAST 10% for validation (set aside, untouched)
    3. Take FIRST 90% for training
    4. Perform bear detection on training 90% ONLY
    5. Create weights for training set ONLY
    6. Validation remains chronological with NO weighting

    Args:
        original_data: Complete dataset (chronological)
        assets: List of asset tickers
        approach: 'weighted' (recommended) or 'duplicate'
        val_split: Fraction for validation (default 0.1 = 10%)
        bear_weight: Weight for bear samples in weighted approach
        bull_weight: Weight for bull samples in weighted approach
        **bear_detection_kwargs: Additional args for detect_bear_periods()

    Returns:
        (train_data, sample_weights, val_data, bear_periods)
        - train_data: Training split (90%)
        - sample_weights: Weights for training samples (None if duplicate approach)
        - val_data: Validation split (10%, chronological, no weighting)
        - bear_periods: List of detected bear periods
    """
    logger.info("=" * 80)
    logger.info("BALANCED DATASET CREATION")
    logger.info("=" * 80)
    logger.info(f"Original data: {len(original_data)} samples")
    logger.info(f"Date range: {original_data.index[0].date()} to {original_data.index[-1].date()}")
    logger.info(f"Approach: {approach}")
    logger.info(f"Validation split: {val_split:.1%}")
    logger.info("")

    # Step 1: Split train/validation chronologically
    n = len(original_data)
    train_end_idx = int(n * (1 - val_split))

    train_data = original_data.iloc[:train_end_idx].copy()
    val_data = original_data.iloc[train_end_idx:].copy()

    logger.info(f"Train set: {len(train_data)} samples ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    logger.info(f"Val set: {len(val_data)} samples ({val_data.index[0].date()} to {val_data.index[-1].date()})")
    logger.info("")

    # Step 2: Detect bear periods in TRAINING set only
    logger.info("Detecting bear periods in training data only...")
    bear_periods = detect_bear_periods(train_data, assets, **bear_detection_kwargs)

    # Step 3: Create balanced training data
    sample_weights = None

    if approach == 'weighted':
        logger.info("")
        logger.info("Creating weighted sampler (recommended approach)...")
        sample_weights = create_sample_weights(
            train_data,
            bear_periods,
            bear_weight=bear_weight,
            bull_weight=bull_weight
        )

    elif approach == 'duplicate':
        logger.info("")
        logger.info("Creating duplicated dataset (alternative approach)...")
        # Calculate duplication factor
        bear_count = sum((train_data.index >= pd.to_datetime(start)) &
                        (train_data.index <= pd.to_datetime(end))
                        for start, end, _ in bear_periods).sum()
        bull_count = len(train_data) - bear_count

        duplication_factor = int(bull_count / bear_count) if bear_count > 0 else 1
        logger.info(f"Duplication factor: {duplication_factor}x")

        # Duplicate bear periods
        duplicated_segments = []
        for start, end, _ in bear_periods:
            mask = (train_data.index >= pd.to_datetime(start)) & (train_data.index <= pd.to_datetime(end))
            bear_segment = train_data[mask]
            for _ in range(duplication_factor - 1):
                duplicated_segments.append(bear_segment)

        if duplicated_segments:
            train_data = pd.concat([train_data] + duplicated_segments, ignore_index=False)
            train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            logger.info(f"Balanced train set: {len(train_data)} samples (after duplication)")

    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'weighted' or 'duplicate'")

    logger.info("")
    logger.info("=" * 80)
    logger.info("BALANCED DATASET CREATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)} (chronological, no weighting)")
    logger.info(f"Bear periods found: {len(bear_periods)}")
    logger.info(f"Sample weights: {'Created' if sample_weights is not None else 'Not used (duplication approach)'}")
    logger.info("")

    return train_data, sample_weights, val_data, bear_periods


def validate_balance(
    data: pd.DataFrame,
    sample_weights: np.ndarray,
    bear_periods: List[Tuple[str, str, Dict]]
) -> Dict[str, float]:
    """
    Validate bull/bear ratio in effective training distribution.

    Args:
        data: Training data
        sample_weights: Sample weights array
        bear_periods: List of detected bear periods

    Returns:
        Dictionary with balance metrics
    """
    # Count samples (handle timezone-aware index)
    bear_mask = np.zeros(len(data), dtype=bool)
    for start, end, _ in bear_periods:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        # Match timezone if data index is timezone-aware
        if data.index.tz is not None:
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize(data.index.tz)
            if end_ts.tz is None:
                end_ts = end_ts.tz_localize(data.index.tz)

        mask = (data.index >= start_ts) & (data.index <= end_ts)
        bear_mask |= mask

    bear_count = bear_mask.sum()
    bull_count = len(data) - bear_count

    # Calculate effective distribution with weights
    if sample_weights is not None:
        bear_total_weight = sample_weights[bear_mask].sum()
        bull_total_weight = sample_weights[~bear_mask].sum()
        total_weight = sample_weights.sum()

        bear_ratio = bear_total_weight / total_weight
        bull_ratio = bull_total_weight / total_weight
    else:
        bear_ratio = bear_count / len(data)
        bull_ratio = bull_count / len(data)

    metrics = {
        'total_samples': len(data),
        'bear_samples': int(bear_count),
        'bull_samples': int(bull_count),
        'bear_ratio_actual': bear_count / len(data),
        'bull_ratio_actual': bull_count / len(data),
        'bear_ratio_effective': bear_ratio,
        'bull_ratio_effective': bull_ratio,
        'is_balanced': 0.45 <= bear_ratio <= 0.55
    }

    logger.info("Balance validation:")
    logger.info(f"  Total samples: {metrics['total_samples']}")
    logger.info(f"  Bear samples: {metrics['bear_samples']} ({metrics['bear_ratio_actual']:.1%} actual)")
    logger.info(f"  Bull samples: {metrics['bull_samples']} ({metrics['bull_ratio_actual']:.1%} actual)")
    if sample_weights is not None:
        logger.info(f"  Effective distribution:")
        logger.info(f"    Bear: {metrics['bear_ratio_effective']:.1%}")
        logger.info(f"    Bull: {metrics['bull_ratio_effective']:.1%}")
        logger.info(f"  Balanced: {'YES ✅' if metrics['is_balanced'] else 'NO ❌'}")

    return metrics


def save_bear_periods(bear_periods: List[Tuple], output_path: str):
    """Save detected bear periods to JSON file."""
    bear_periods_dict = {
        'bear_periods': [
            {
                'start_date': start,
                'end_date': end,
                'metrics': metrics
            }
            for start, end, metrics in bear_periods
        ],
        'total_count': len(bear_periods)
    }

    with open(output_path, 'w') as f:
        json.dump(bear_periods_dict, f, indent=2)

    logger.info(f"Bear periods saved to: {output_path}")


if __name__ == "__main__":
    # Test bear period detection
    from astra.data_pipeline.data_manager import PortfolioDataManager

    logger.info("Testing bear period detection...")

    # Load data
    data_manager = PortfolioDataManager()
    data, _ = data_manager.process_and_initialize()

    # Create balanced dataset
    # Since bear periods are only 2.7%, we need weight ~35x to achieve 50/50
    # Formula: bear_weight / (bear_weight + bull_weight) = 0.5
    # With 2.7% bear: need bear_weight = 35.0 to get close to 50/50
    train_data, sample_weights, val_data, bear_periods = create_balanced_dataset(
        original_data=data,
        assets=['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS'],
        approach='weighted',
        val_split=0.1,
        bear_weight=35.0,  # High weight to balance 2.7% bear samples
        bull_weight=1.0,
        lookback_window=30,
        drawdown_threshold=0.10,
        volatility_threshold=0.02,
        min_duration=10
    )

    # Validate balance
    metrics = validate_balance(train_data, sample_weights, bear_periods)

    # Save bear periods
    save_bear_periods(bear_periods, "notebooks/data/bear_periods_detected.json")

    logger.info("Test complete!")
