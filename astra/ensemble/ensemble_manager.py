#!/usr/bin/env python3
"""
Ensemble Manager for v1 + v2 Model Switching

Manages multiple RL models and switches between them based on detected market regime.
Combines v1 (momentum/bull specialist) with v2 (defensive/bear specialist).
"""

import sys
from pathlib import Path
import numpy as np
import yaml
import logging
from typing import Dict, Tuple, Optional

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from astra.rl_framework.environment import PortfolioEnvironment
from astra.ensemble.regime_detector import RegimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleManager:
    """
    Manages ensemble of RL models with regime-based switching.

    Models:
    - v1: Momentum specialist (bull markets)
    - v2: Defensive specialist (bear markets)

    Switching Logic:
    - Bull regime → v1 (aggressive, capture upside)
    - Bear regime → v2 (defensive, protect capital)
    - Neutral regime → Use primary model (default: v1)
    """

    def __init__(
        self,
        v1_model_path: str,
        v2_model_path: str,
        config_path: str = None,
        primary_model: str = 'v1'
    ):
        """
        Initialize ensemble manager.

        Args:
            v1_model_path: Path to v1 model directory
            v2_model_path: Path to v2 model directory
            config_path: Path to ensemble configuration (optional)
            primary_model: Default model for neutral regime ('v1' or 'v2')
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.v1_path = self.project_root / v1_model_path
        self.v2_path = self.project_root / v2_model_path
        self.primary_model = primary_model

        # Load configuration
        self.config = self._load_config(config_path)

        # Load models
        self.models = {}
        self.vec_normalizers = {}
        self._load_models()

        # Initialize regime detector
        regime_config = self.config.get('regime_detection', {})
        self.regime_detector = RegimeDetector(
            drawdown_threshold=regime_config.get('drawdown_threshold', -0.10),
            volatility_threshold=regime_config.get('volatility_threshold', 0.025),
            consecutive_loss_threshold=regime_config.get('consecutive_loss_threshold', 5),
            hysteresis_days=regime_config.get('hysteresis_days', 3),
            bear_exit_hysteresis_days=regime_config.get('bear_exit_hysteresis_days', None),
            volatility_window=regime_config.get('volatility_window', 10),
            trend_window=regime_config.get('trend_window', 20),
            ma_short_window=regime_config.get('ma_short_window', 50),
            ma_long_window=regime_config.get('ma_long_window', 200)
        )

        # Tracking
        self.active_model = primary_model
        self.switch_history = []
        self.regime_history = []

        logger.info("="*80)
        logger.info("Ensemble Manager Initialized")
        logger.info("="*80)
        logger.info(f"v1 (momentum): {v1_model_path}")
        logger.info(f"v2 (defensive): {v2_model_path}")
        logger.info(f"Primary model: {primary_model}")
        logger.info("="*80)

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load ensemble configuration."""
        if config_path is None:
            config_path = self.project_root / "config" / "ensemble.yaml"
        else:
            config_path = Path(config_path)

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Loaded config: {config_path}")
                return config.get('ensemble', {})
        else:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {}

    def _load_models(self):
        """Load both v1 and v2 models with their VecNormalize wrappers."""
        logger.info("Loading models...")

        # Load v1 (momentum)
        logger.info("Loading v1 (momentum)...")
        self.models['v1'], self.vec_normalizers['v1'] = self._load_single_model(
            self.v1_path, 'v1'
        )

        # Load v2 (defensive)
        logger.info("Loading v2 (defensive)...")
        self.models['v2'], self.vec_normalizers['v2'] = self._load_single_model(
            self.v2_path, 'v2'
        )

        logger.info("✅ All models loaded successfully")

    def _load_single_model(
        self,
        model_path: Path,
        model_name: str
    ) -> Tuple[SAC, VecNormalize]:
        """
        Load a single model with its VecNormalize wrapper.

        Args:
            model_path: Path to model directory
            model_name: Model identifier

        Returns:
            Tuple of (model, vec_normalize)
        """
        # Check model exists
        model_zip = model_path / "final_model.zip"
        vec_norm_pkl = model_path / "vec_normalize.pkl"

        if not model_zip.exists():
            raise FileNotFoundError(f"Model not found: {model_zip}")
        if not vec_norm_pkl.exists():
            raise FileNotFoundError(f"VecNormalize not found: {vec_norm_pkl}")

        # Load model
        model = SAC.load(str(model_zip))

        # Load VecNormalize (we'll create env wrappers as needed during prediction)
        # For now, just return the model
        logger.info(f"  ✅ {model_name} loaded from {model_path}")

        return model, vec_norm_pkl

    def select_model(self, regime: str) -> str:
        """
        Select appropriate model based on regime.

        Args:
            regime: Current market regime ('bull', 'bear', 'neutral')

        Returns:
            Model name to use ('v1' or 'v2')
        """
        model_selection = self.config.get('models', {})

        # Default mapping if config not found
        default_mapping = {
            'bull': 'v1',
            'bear': 'v2',
            'neutral': self.primary_model
        }

        # Use config if available
        if model_selection:
            v1_regimes = model_selection.get('v1', {}).get('use_for', ['bull', 'neutral'])
            v2_regimes = model_selection.get('v2', {}).get('use_for', ['bear'])

            if regime in v2_regimes:
                return 'v2'
            elif regime in v1_regimes:
                return 'v1'
            else:
                return self.primary_model
        else:
            return default_mapping.get(regime, self.primary_model)

    def get_action(
        self,
        observation: np.ndarray,
        portfolio_returns: list,
        vec_env: VecNormalize,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get action from ensemble based on current regime.

        Args:
            observation: Current environment observation
            portfolio_returns: Historical portfolio returns for regime detection
            vec_env: VecNormalize wrapper for current environment
            deterministic: Use deterministic action

        Returns:
            Tuple of (action, info_dict)
            - action: Portfolio weights
            - info_dict: Information about regime and model selection
        """
        # Detect regime
        regime, regime_info = self.regime_detector.detect_regime(portfolio_returns)

        # Select model
        selected_model = self.select_model(regime)

        # Check if we switched models
        switched = (selected_model != self.active_model)
        if switched:
            logger.info(f"🔄 Model switch: {self.active_model} → {selected_model} (regime: {regime})")
            self.switch_history.append({
                'day': len(portfolio_returns),
                'from': self.active_model,
                'to': selected_model,
                'regime': regime
            })
            self.active_model = selected_model

        # Track regime
        self.regime_history.append(regime)

        # Get action from selected model
        model = self.models[selected_model]
        action, _ = model.predict(observation, deterministic=deterministic)

        # Build info dict
        info = {
            'regime': regime,
            'regime_info': regime_info,
            'active_model': selected_model,
            'switched': switched,
            'model_usage': self._get_model_usage()
        }

        return action, info

    def _get_model_usage(self) -> Dict:
        """Get statistics on model usage."""
        if not self.regime_history:
            return {'v1': 0.0, 'v2': 0.0}

        total = len(self.regime_history)
        v1_count = sum(1 for r in self.regime_history if self.select_model(r) == 'v1')
        v2_count = sum(1 for r in self.regime_history if self.select_model(r) == 'v2')

        return {
            'v1': v1_count / total if total > 0 else 0,
            'v2': v2_count / total if total > 0 else 0
        }

    def get_statistics(self) -> Dict:
        """
        Get comprehensive ensemble statistics.

        Returns:
            Dictionary with ensemble performance stats
        """
        regime_stats = self.regime_detector.get_statistics()
        model_usage = self._get_model_usage()

        return {
            'regime_stats': regime_stats,
            'model_usage': model_usage,
            'total_switches': len(self.switch_history),
            'switch_history': self.switch_history,
            'current_model': self.active_model
        }

    def reset(self):
        """Reset ensemble state."""
        self.regime_detector.reset()
        self.active_model = self.primary_model
        self.switch_history = []
        self.regime_history = []
        logger.info("Ensemble Manager reset to initial state")

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"EnsembleManager(\n"
            f"  Active Model: {self.active_model}\n"
            f"  Model Usage: v1={stats['model_usage']['v1']:.1%}, v2={stats['model_usage']['v2']:.1%}\n"
            f"  Total Switches: {stats['total_switches']}\n"
            f"  Regime: {stats['regime_stats']}\n"
            f")"
        )
