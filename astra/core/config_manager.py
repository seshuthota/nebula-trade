"""
Configuration Manager for nebula-trade.
Handles loading, merging, and validating configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy


class ConfigManager:
    """Central configuration manager for models and training."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize config manager.
        
        Args:
            project_root: Root directory of the project. If None, auto-detect.
        """
        if project_root is None:
            # Auto-detect project root (assumes this file is in astra/core/)
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.config_dir = self.project_root / "config"
        self.models_dir = self.config_dir / "models"
        self.training_dir = self.config_dir / "training"
        self.ensemble_dir = self.config_dir / "ensemble"
        
        # Validate directories exist
        if not self.config_dir.exists():
            raise ValueError(f"Config directory not found: {self.config_dir}")
    
    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Configuration to merge on top
            
        Returns:
            Merged configuration
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """Load configuration for a specific model.
        
        Args:
            model_name: Model identifier (e.g., 'v1_momentum', 'v5_tuned')
            
        Returns:
            Complete model configuration (defaults + model-specific)
        """
        # Load defaults
        defaults_path = self.models_dir / "model_defaults.yaml"
        if not defaults_path.exists():
            raise FileNotFoundError(f"Model defaults not found: {defaults_path}")
        
        defaults = self.load_yaml(defaults_path)
        
        # Load model-specific config
        model_path = self.models_dir / f"{model_name}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")
        
        model_config = self.load_yaml(model_path)
        
        # Merge configs (model-specific overrides defaults)
        config = self.merge_configs(defaults, model_config)
        
        return config
    
    def load_training_config(self, training_name: str = "default") -> Dict[str, Any]:
        """Load training configuration.
        
        Args:
            training_name: Training config name (e.g., 'default', 'production')
            
        Returns:
            Training configuration
        """
        training_path = self.training_dir / f"{training_name}.yaml"
        if not training_path.exists():
            raise FileNotFoundError(f"Training config not found: {training_path}")
        
        return self.load_yaml(training_path)
    
    def load_ensemble_config(self, ensemble_name: str = "ensemble") -> Dict[str, Any]:
        """Load ensemble configuration.
        
        Args:
            ensemble_name: Ensemble config name (e.g., 'ensemble', 'ensemble_sticky_7day')
            
        Returns:
            Ensemble configuration
        """
        ensemble_path = self.ensemble_dir / f"{ensemble_name}.yaml"
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble config not found: {ensemble_path}")
        
        return self.load_yaml(ensemble_path)
    
    def load_portfolio_config(self) -> Dict[str, Any]:
        """Load base portfolio configuration.
        
        Returns:
            Portfolio configuration
        """
        portfolio_path = self.config_dir / "portfolio.yaml"
        if not portfolio_path.exists():
            raise FileNotFoundError(f"Portfolio config not found: {portfolio_path}")
        
        return self.load_yaml(portfolio_path)
    
    def load_complete_config(
        self,
        model_name: str,
        training_name: str = "default",
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load complete configuration for training.
        
        Args:
            model_name: Model to train
            training_name: Training configuration to use
            overrides: Additional overrides (e.g., from CLI)
            
        Returns:
            Complete merged configuration
        """
        # Load all configs
        portfolio = self.load_portfolio_config()
        model = self.load_model_config(model_name)
        training = self.load_training_config(training_name)
        
        # Build complete config
        config = {
            'portfolio': portfolio['portfolio'],
            'model': model['model'],
            'metadata': model.get('metadata', {}),
            'training': training['training'],
            'resume': training.get('resume', {}),
        }
        
        # Apply overrides if provided
        if overrides:
            config = self.merge_configs(config, overrides)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration overrides from environment variables.
        
        Environment variables:
            NEBULA_TIMESTEPS: Override total_timesteps
            NEBULA_USE_WANDB: Override use_wandb (true/false)
            NEBULA_GPU: Override use_gpu (true/false)
            
        Args:
            config: Configuration to modify
            
        Returns:
            Modified configuration
        """
        result = deepcopy(config)
        
        # Timesteps override
        if 'NEBULA_TIMESTEPS' in os.environ:
            timesteps = int(os.environ['NEBULA_TIMESTEPS'])
            result['training']['total_timesteps'] = timesteps
        
        # WandB override
        if 'NEBULA_USE_WANDB' in os.environ:
            use_wandb = os.environ['NEBULA_USE_WANDB'].lower() == 'true'
            result['training']['logging']['use_wandb'] = use_wandb
        
        # GPU override
        if 'NEBULA_GPU' in os.environ:
            use_gpu = os.environ['NEBULA_GPU'].lower() == 'true'
            result['training']['optimization']['use_gpu'] = use_gpu
        
        return result
    
    def list_available_models(self) -> list[str]:
        """List all available model configurations.
        
        Returns:
            List of model names
        """
        if not self.models_dir.exists():
            return []
        
        models = []
        for file in self.models_dir.glob("*.yaml"):
            if file.stem != "model_defaults":
                models.append(file.stem)
        
        return sorted(models)
    
    def list_available_training_configs(self) -> list[str]:
        """List all available training configurations.
        
        Returns:
            List of training config names
        """
        if not self.training_dir.exists():
            return []
        
        configs = []
        for file in self.training_dir.glob("*.yaml"):
            configs.append(file.stem)
        
        return sorted(configs)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration completeness.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['portfolio', 'model', 'training']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")
        
        # Validate model section
        if 'reward' not in config['model']:
            raise ValueError("Model config missing 'reward' section")
        
        # Validate training section
        if 'total_timesteps' not in config['training']:
            raise ValueError("Training config missing 'total_timesteps'")
        
        return True
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get metadata information about a model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Model metadata
        """
        config = self.load_model_config(model_name)
        return config.get('metadata', {})
