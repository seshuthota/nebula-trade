#!/usr/bin/env python3
"""
Regime Detection System for Ensemble Trading

Detects market regimes (bull/bear/neutral) using multiple indicators:
- Drawdown analysis
- Volatility analysis
- Consecutive loss streaks
- Market trends
- Hysteresis to prevent rapid switching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regime using multiple technical indicators.

    Regimes:
    - 'bull': Market trending up, low volatility, use v1 (momentum)
    - 'bear': Market correction, high volatility, use v2 (defensive)
    - 'neutral': Mixed signals, default to primary model
    """

    def __init__(
        self,
        drawdown_threshold: float = -0.10,  # -10% triggers bear
        volatility_threshold: float = 0.025,  # 2.5% daily vol triggers bear
        consecutive_loss_threshold: int = 5,  # 5 losing days triggers bear
        hysteresis_days: int = 3,  # Require 3-day confirmation for ENTRY
        bear_exit_hysteresis_days: int = None,  # Days to exit BEAR (default: same as entry)
        volatility_window: int = 10,  # Rolling window for volatility
        trend_window: int = 20,  # MA window for trend detection
        ma_short_window: int = 50,  # Short MA for crossover
        ma_long_window: int = 200,  # Long MA for crossover
    ):
        """
        Initialize regime detector with asymmetric hysteresis.

        Args:
            drawdown_threshold: Max drawdown before switching to bear (-0.10 = -10%)
            volatility_threshold: Volatility threshold for bear regime (0.025 = 2.5%)
            consecutive_loss_threshold: Number of consecutive losses to trigger bear
            hysteresis_days: Days to wait before ENTERING a new regime
            bear_exit_hysteresis_days: Days to wait before EXITING bear regime (asymmetric)
                                       If None, uses same as hysteresis_days
            volatility_window: Days for rolling volatility calculation
            trend_window: Days for trend moving average
            ma_short_window: Short MA period for crossover signals
            ma_long_window: Long MA period for crossover signals
        """
        self.drawdown_threshold = drawdown_threshold
        self.volatility_threshold = volatility_threshold
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.hysteresis_days = hysteresis_days
        self.bear_exit_hysteresis_days = bear_exit_hysteresis_days if bear_exit_hysteresis_days is not None else hysteresis_days
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.ma_short_window = ma_short_window
        self.ma_long_window = ma_long_window

        # State tracking
        self.current_regime = 'bull'  # Start optimistic
        self.regime_history = []
        self.days_in_current_regime = 0
        self.pending_regime = None
        self.pending_days = 0

        logger.info(f"RegimeDetector initialized:")
        logger.info(f"  Drawdown threshold: {drawdown_threshold:.1%}")
        logger.info(f"  Volatility threshold: {volatility_threshold:.2%}")
        logger.info(f"  Consecutive loss threshold: {consecutive_loss_threshold} days")
        logger.info(f"  Entry hysteresis: {hysteresis_days} days")
        logger.info(f"  BEAR exit hysteresis: {self.bear_exit_hysteresis_days} days (ASYMMETRIC)")
        logger.info(f"  MA windows: {ma_short_window}/{ma_long_window} days")

    def calculate_drawdown(self, portfolio_returns: List[float]) -> float:
        """
        Calculate current drawdown from peak.

        Args:
            portfolio_returns: List of portfolio returns

        Returns:
            Current drawdown as negative percentage (e.g., -0.15 for -15%)
        """
        if len(portfolio_returns) < 2:
            return 0.0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + np.array(portfolio_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return float(drawdown[-1])

    def calculate_volatility(self, portfolio_returns: List[float]) -> float:
        """
        Calculate rolling volatility.

        Args:
            portfolio_returns: List of portfolio returns

        Returns:
            Annualized volatility
        """
        if len(portfolio_returns) < self.volatility_window:
            return 0.0

        recent_returns = portfolio_returns[-self.volatility_window:]
        volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized

        return float(volatility)

    def count_consecutive_losses(self, portfolio_returns: List[float]) -> int:
        """
        Count consecutive losing days.

        Args:
            portfolio_returns: List of portfolio returns

        Returns:
            Number of consecutive losses (from end)
        """
        if len(portfolio_returns) == 0:
            return 0

        count = 0
        for ret in reversed(portfolio_returns):
            if ret < 0:
                count += 1
            else:
                break

        return count

    def calculate_trend(self, portfolio_returns: List[float]) -> str:
        """
        Calculate market trend using moving average crossover.

        Args:
            portfolio_returns: List of portfolio returns

        Returns:
            'up', 'down', or 'neutral'
        """
        if len(portfolio_returns) < self.trend_window:
            return 'neutral'

        # Calculate cumulative value
        cumulative = np.cumprod(1 + np.array(portfolio_returns))

        # Short-term trend (last 5 days)
        short_term = np.mean(cumulative[-5:])

        # Long-term trend (MA)
        long_term = np.mean(cumulative[-self.trend_window:])

        if short_term > long_term * 1.01:  # 1% above MA
            return 'up'
        elif short_term < long_term * 0.99:  # 1% below MA
            return 'down'
        else:
            return 'neutral'

    def detect_regime_signal(self, portfolio_returns: List[float]) -> str:
        """
        Detect regime based on multiple indicators (without hysteresis).

        Args:
            portfolio_returns: List of portfolio returns

        Returns:
            'bull', 'bear', or 'neutral'
        """
        if len(portfolio_returns) < 10:
            return 'neutral'  # Not enough data

        # Calculate indicators
        drawdown = self.calculate_drawdown(portfolio_returns)
        volatility = self.calculate_volatility(portfolio_returns)
        consecutive_losses = self.count_consecutive_losses(portfolio_returns)
        trend = self.calculate_trend(portfolio_returns)

        # Count bear signals
        bear_signals = 0
        signal_details = []

        # 1. Drawdown check
        if drawdown <= self.drawdown_threshold:
            bear_signals += 2  # Heavy weight
            signal_details.append(f"Drawdown: {drawdown:.2%} (threshold: {self.drawdown_threshold:.2%})")

        # 2. Volatility check
        if volatility >= self.volatility_threshold:
            bear_signals += 1
            signal_details.append(f"Volatility: {volatility:.2%} (threshold: {self.volatility_threshold:.2%})")

        # 3. Consecutive losses
        if consecutive_losses >= self.consecutive_loss_threshold:
            bear_signals += 1
            signal_details.append(f"Consecutive losses: {consecutive_losses} (threshold: {self.consecutive_loss_threshold})")

        # 4. Trend
        if trend == 'down':
            bear_signals += 1
            signal_details.append(f"Trend: {trend}")

        # Decision logic
        if bear_signals >= 2:  # At least 2 bear signals
            regime = 'bear'
        elif bear_signals == 0 and trend == 'up':
            regime = 'bull'
        else:
            regime = 'neutral'

        # Log signals if detected
        if signal_details:
            logger.debug(f"Bear signals detected ({bear_signals}/4): {'; '.join(signal_details)}")

        return regime

    def detect_regime(
        self,
        portfolio_returns: List[float],
        force_update: bool = False
    ) -> Tuple[str, Dict]:
        """
        Detect regime with hysteresis to prevent rapid switching.

        Args:
            portfolio_returns: List of portfolio returns
            force_update: Skip hysteresis and update immediately

        Returns:
            Tuple of (regime, info_dict)
            - regime: 'bull', 'bear', or 'neutral'
            - info_dict: Diagnostic information
        """
        # Get raw signal
        signal = self.detect_regime_signal(portfolio_returns)

        # Calculate indicators for logging
        drawdown = self.calculate_drawdown(portfolio_returns)
        volatility = self.calculate_volatility(portfolio_returns)
        consecutive_losses = self.count_consecutive_losses(portfolio_returns)
        trend = self.calculate_trend(portfolio_returns)

        # Apply hysteresis
        if force_update:
            # Immediate switch
            self.current_regime = signal
            self.days_in_current_regime = 0
            self.pending_regime = None
            self.pending_days = 0
            switched = True
        else:
            switched = False

            if signal != self.current_regime:
                # New signal different from current regime
                if self.pending_regime == signal:
                    # Same pending signal, increment counter
                    self.pending_days += 1

                    # Determine required hysteresis based on transition type
                    # ASYMMETRIC: Exiting BEAR requires longer confirmation
                    if self.current_regime == 'bear' and signal != 'bear':
                        required_days = self.bear_exit_hysteresis_days
                    else:
                        required_days = self.hysteresis_days

                    if self.pending_days >= required_days:
                        # Confirmed switch
                        logger.info(f"Regime switch confirmed: {self.current_regime} → {signal} (after {self.pending_days} days)")
                        logger.info(f"  Drawdown: {drawdown:.2%}, Volatility: {volatility:.2%}, "
                                  f"Consecutive losses: {consecutive_losses}, Trend: {trend}")
                        self.current_regime = signal
                        self.days_in_current_regime = 0
                        self.pending_regime = None
                        self.pending_days = 0
                        switched = True
                else:
                    # New pending signal
                    self.pending_regime = signal
                    self.pending_days = 1

                    # Determine required days for logging
                    if self.current_regime == 'bear' and signal != 'bear':
                        required_days = self.bear_exit_hysteresis_days
                    else:
                        required_days = self.hysteresis_days

                    logger.debug(f"Pending regime change: {self.current_regime} → {signal} (1/{required_days})")
            else:
                # Signal matches current regime, reset pending
                self.pending_regime = None
                self.pending_days = 0
                self.days_in_current_regime += 1

        # Track history
        self.regime_history.append(self.current_regime)

        # Build info dict
        info = {
            'regime': self.current_regime,
            'signal': signal,
            'switched': switched,
            'days_in_regime': self.days_in_current_regime,
            'pending_regime': self.pending_regime,
            'pending_days': self.pending_days,
            'indicators': {
                'drawdown': drawdown,
                'volatility': volatility,
                'consecutive_losses': consecutive_losses,
                'trend': trend
            }
        }

        return self.current_regime, info

    def reset(self):
        """Reset detector state."""
        self.current_regime = 'bull'
        self.regime_history = []
        self.days_in_current_regime = 0
        self.pending_regime = None
        self.pending_days = 0
        logger.info("RegimeDetector reset to initial state")

    def get_statistics(self) -> Dict:
        """
        Get regime statistics from history.

        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {}

        total_days = len(self.regime_history)
        bull_days = self.regime_history.count('bull')
        bear_days = self.regime_history.count('bear')
        neutral_days = self.regime_history.count('neutral')

        # Count switches
        switches = 0
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i] != self.regime_history[i-1]:
                switches += 1

        return {
            'total_days': total_days,
            'bull_days': bull_days,
            'bear_days': bear_days,
            'neutral_days': neutral_days,
            'bull_pct': bull_days / total_days if total_days > 0 else 0,
            'bear_pct': bear_days / total_days if total_days > 0 else 0,
            'neutral_pct': neutral_days / total_days if total_days > 0 else 0,
            'switches': switches
        }
