"""
Reward functions for portfolio optimization.
Centralized reward logic for different training strategies.
"""

import numpy as np
from typing import Dict, Any


class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reward function.
        
        Args:
            config: Reward configuration from model config
        """
        self.config = config
        self.components = config.get('components', {})
        self.loss_aversion = config.get('loss_aversion', 1.0)
    
    def calculate(
        self,
        portfolio_return: float,
        sharpe_ratio: float,
        drawdown: float,
        volatility: float,
        turnover: float
    ) -> float:
        """Calculate reward based on portfolio metrics.
        
        Args:
            portfolio_return: Portfolio return for the period
            sharpe_ratio: Sharpe ratio
            drawdown: Current drawdown from peak
            volatility: Portfolio volatility
            turnover: Portfolio turnover
            
        Returns:
            Computed reward
        """
        raise NotImplementedError
    
    def _apply_loss_aversion(self, return_value: float) -> float:
        """Apply asymmetric loss aversion to returns.
        
        Args:
            return_value: Raw return value
            
        Returns:
            Loss-adjusted return
        """
        if return_value < 0:
            return return_value * self.loss_aversion
        return return_value


class StandardReward(RewardFunction):
    """Standard reward function - simple return-based."""
    
    def calculate(
        self,
        portfolio_return: float,
        sharpe_ratio: float,
        drawdown: float,
        volatility: float,
        turnover: float
    ) -> float:
        """Calculate standard reward (return + small Sharpe bonus)."""
        return_weight = self.components.get('return_weight', 1.0)
        sharpe_weight = self.components.get('sharpe_weight', 0.01)
        
        # Apply loss aversion to returns
        adjusted_return = self._apply_loss_aversion(portfolio_return)
        
        reward = (
            adjusted_return * return_weight +
            sharpe_ratio * sharpe_weight
        )
        
        return reward


class DefensiveReward(RewardFunction):
    """Defensive reward function with heavy risk penalties."""
    
    def calculate(
        self,
        portfolio_return: float,
        sharpe_ratio: float,
        drawdown: float,
        volatility: float,
        turnover: float
    ) -> float:
        """Calculate defensive reward with drawdown penalties."""
        # Get weights
        return_weight = self.components.get('return_weight', 0.40)
        sharpe_weight = self.components.get('sharpe_weight', 0.30)
        drawdown_penalty = self.components.get('drawdown_penalty', 0.20)
        volatility_penalty = self.components.get('volatility_penalty', 0.05)
        turnover_penalty = self.components.get('turnover_penalty', 0.05)
        
        # Apply loss aversion to returns
        adjusted_return = self._apply_loss_aversion(portfolio_return)
        
        # Calculate reward components
        reward = (
            adjusted_return * return_weight +
            sharpe_ratio * sharpe_weight -
            abs(drawdown) * drawdown_penalty -
            volatility * volatility_penalty -
            turnover * turnover_penalty
        )
        
        return reward


class BalancedReward(RewardFunction):
    """Balanced reward function - compromise between returns and risk."""
    
    def calculate(
        self,
        portfolio_return: float,
        sharpe_ratio: float,
        drawdown: float,
        volatility: float,
        turnover: float
    ) -> float:
        """Calculate balanced reward."""
        # Get weights
        return_weight = self.components.get('return_weight', 0.60)
        sharpe_weight = self.components.get('sharpe_weight', 0.20)
        drawdown_penalty = self.components.get('drawdown_penalty', 0.10)
        volatility_penalty = self.components.get('volatility_penalty', 0.05)
        turnover_penalty = self.components.get('turnover_penalty', 0.05)
        
        # Apply loss aversion to returns
        adjusted_return = self._apply_loss_aversion(portfolio_return)
        
        # Calculate reward components
        reward = (
            adjusted_return * return_weight +
            sharpe_ratio * sharpe_weight -
            abs(drawdown) * drawdown_penalty -
            volatility * volatility_penalty -
            turnover * turnover_penalty
        )
        
        return reward


class MomentumReward(RewardFunction):
    """Momentum reward function - focus on returns."""
    
    def calculate(
        self,
        portfolio_return: float,
        sharpe_ratio: float,
        drawdown: float,
        volatility: float,
        turnover: float
    ) -> float:
        """Calculate momentum reward (mostly just returns)."""
        return_weight = self.components.get('return_weight', 1.0)
        sharpe_weight = self.components.get('sharpe_weight', 0.01)
        
        # Apply loss aversion (usually 1.0 for momentum)
        adjusted_return = self._apply_loss_aversion(portfolio_return)
        
        reward = (
            adjusted_return * return_weight +
            sharpe_ratio * sharpe_weight
        )
        
        return reward


def get_reward_function(config: Dict[str, Any]) -> RewardFunction:
    """Factory function to get appropriate reward function.
    
    Args:
        config: Reward configuration with 'type' field
        
    Returns:
        Initialized reward function
        
    Raises:
        ValueError: If reward type is unknown
    """
    reward_type = config.get('type', 'standard')
    
    reward_classes = {
        'standard': StandardReward,
        'momentum': MomentumReward,
        'defensive': DefensiveReward,
        'balanced': BalancedReward,
    }
    
    if reward_type not in reward_classes:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available types: {list(reward_classes.keys())}"
        )
    
    return reward_classes[reward_type](config)


# Helper function to get reward for environment
def calculate_reward_from_config(
    config: Dict[str, Any],
    portfolio_return: float,
    sharpe_ratio: float = 0.0,
    drawdown: float = 0.0,
    volatility: float = 0.0,
    turnover: float = 0.0
) -> float:
    """Convenience function to calculate reward from config.
    
    Args:
        config: Full model config with 'reward' section
        portfolio_return: Portfolio return
        sharpe_ratio: Sharpe ratio
        drawdown: Current drawdown
        volatility: Portfolio volatility
        turnover: Portfolio turnover
        
    Returns:
        Calculated reward
    """
    reward_config = config.get('reward', {'type': 'standard'})
    reward_fn = get_reward_function(reward_config)
    
    return reward_fn.calculate(
        portfolio_return=portfolio_return,
        sharpe_ratio=sharpe_ratio,
        drawdown=drawdown,
        volatility=volatility,
        turnover=turnover
    )
