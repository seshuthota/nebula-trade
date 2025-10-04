"""
Unified Trainer for nebula-trade.
Configuration-driven trainer that replaces all individual training scripts.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging
import json
from datetime import datetime
from multiprocessing import cpu_count

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ..rl_framework.environment import PortfolioEnvironment
from ..data_pipeline.data_manager import PortfolioDataManager
from ..core.config_manager import ConfigManager
from ..core.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """Unified configuration-driven trainer for all model types."""
    
    def __init__(
        self,
        model_name: str,
        training_config: str = "default",
        config_overrides: Optional[Dict[str, Any]] = None,
        project_root: Optional[Path] = None
    ):
        """Initialize unified trainer.
        
        Args:
            model_name: Model to train (e.g., 'v1_momentum', 'v5_tuned')
            training_config: Training configuration to use
            config_overrides: Additional configuration overrides
            project_root: Project root directory
        """
        # Setup paths
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root)
        
        # Load configuration
        self.config_manager = ConfigManager(self.project_root)
        self.model_registry = ModelRegistry(self.project_root)
        
        self.model_name = model_name
        self.config = self.config_manager.load_complete_config(
            model_name=model_name,
            training_name=training_config,
            overrides=config_overrides
        )
        
        # Create output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = self.config['metadata'].get('version', 'unknown')
        
        self.output_dir = self.project_root / "models" / "production" / f"{model_name}_{self.timestamp}"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.output_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self._save_config()
        
        # Hardware configuration
        self._setup_hardware()
        
        # Load and prepare data
        self._load_data()
        
        # Create environments
        self._create_environments()
        
        # Create agent
        self._create_agent()
        
        logger.info("=" * 80)
        logger.info(f"UNIFIED TRAINER INITIALIZED: {model_name}")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name} ({model_version})")
        logger.info(f"Training config: {training_config}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Training samples: {len(self.train_data)}")
        logger.info(f"Validation samples: {len(self.val_data)}")
        logger.info(f"Total timesteps: {self.config['training']['total_timesteps']:,}")
        logger.info("=" * 80)
    
    def _save_config(self):
        """Save complete configuration to output directory."""
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_file}")
    
    def _setup_hardware(self):
        """Setup hardware configuration."""
        training_config = self.config['training']
        optimization = training_config.get('optimization', {})
        
        # Determine number of parallel environments
        n_envs = training_config.get('n_parallel_envs', 6)
        available_cpus = cpu_count()
        self.n_envs = min(n_envs, max(1, available_cpus - 2))
        
        # Determine device
        use_gpu = optimization.get('use_gpu', True)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        logger.info("Hardware Configuration:")
        logger.info(f"  CPU cores: {available_cpus}")
        logger.info(f"  Parallel environments: {self.n_envs}")
        logger.info(f"  Device: {self.device}")
        
        if self.use_gpu:
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def _load_data(self):
        """Load and split training data."""
        data_config = self.config['model'].get('data', {})
        data_split = self.config['training'].get('data_split', {'train': 0.90, 'val': 0.10})
        
        # Check if we need to load balanced data
        balancing = data_config.get('balancing', {})
        
        if balancing.get('enabled', False):
            # Load pre-balanced data with weights
            data_period = data_config.get('period', '2015-2024')
            strategy = balancing.get('strategy', 'bear_oversampling')
            weight = balancing.get('weight_multiplier', 1.0)
            
            # Load from pre-generated balanced data
            data_file = self.project_root / "data" / f"balanced_data_{data_period.replace('-', '_')}.csv"
            weights_file = self.project_root / "data" / f"sample_weights_{data_period.replace('-', '_')}.npy"
            
            if data_file.exists() and weights_file.exists():
                logger.info(f"Loading balanced data: {data_file}")
                data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                self.sample_weights = np.load(weights_file)
                logger.info(f"Loaded {len(data)} samples with {strategy} strategy ({weight}x weighting)")
            else:
                logger.warning(f"Balanced data not found, falling back to standard data")
                data_manager = PortfolioDataManager(
                    config_path=str(self.project_root / "config" / "portfolio.yaml")
                )
                data, _ = data_manager.process_and_initialize()
                self.sample_weights = None
        else:
            # Load standard data
            data_manager = PortfolioDataManager(
                config_path=str(self.project_root / "config" / "portfolio.yaml")
            )
            data, _ = data_manager.process_and_initialize()
            self.sample_weights = None
        
        # Split data
        n = len(data)
        train_split = data_split.get('train', 0.90)
        train_end = int(n * train_split)
        
        self.train_data = data.iloc[:train_end]
        self.val_data = data.iloc[train_end:]
        
        logger.info(f"Data loaded: {n} total samples")
        logger.info(f"  Train: {len(self.train_data)} samples ({train_split:.0%})")
        logger.info(f"  Validation: {len(self.val_data)} samples ({1-train_split:.0%})")
        logger.info(f"  Period: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Save data splits
        self.train_data.to_csv(self.logs_dir / f"train_data_{self.timestamp}.csv")
        self.val_data.to_csv(self.logs_dir / f"val_data_{self.timestamp}.csv")
    
    def _make_env(self, data: pd.DataFrame, rank: int = 0):
        """Create a single environment."""
        def _init():
            env = PortfolioEnvironment(data, lookback_window=30)
            env = Monitor(env)
            return env
        return _init
    
    def _create_environments(self):
        """Create training and validation environments."""
        optimization = self.config['training'].get('optimization', {})
        use_vec_normalize = optimization.get('vec_normalize', True)
        
        # Create training environment
        if self.n_envs == 1:
            vec_env = DummyVecEnv([self._make_env(self.train_data, 0)])
        else:
            vec_env = SubprocVecEnv([
                self._make_env(self.train_data, i) for i in range(self.n_envs)
            ])
        
        # Wrap with VecNormalize if enabled
        if use_vec_normalize:
            self.env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.99,
                epsilon=1e-8,
            )
        else:
            self.env = vec_env
        
        # Create evaluation environment
        eval_vec_env = DummyVecEnv([self._make_env(self.val_data, 0)])
        
        if use_vec_normalize:
            self.eval_env = VecNormalize(
                eval_vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.99,
                epsilon=1e-8,
                training=False,  # Don't update stats during eval
            )
        else:
            self.eval_env = eval_vec_env
        
        logger.info("Environments created successfully")
    
    def _create_agent(self):
        """Create SAC agent."""
        model_config = self.config['model']
        
        # SAC hyperparameters
        learning_rate = model_config.get('learning_rate', 0.0003)
        buffer_size = model_config.get('buffer_size', 1000000)
        batch_size = model_config.get('batch_size', 256)
        gamma = model_config.get('gamma', 0.99)
        tau = model_config.get('tau', 0.005)
        
        self.agent = SAC(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            device=self.device,
            verbose=1,
            tensorboard_log=str(self.logs_dir / "tensorboard")
        )
        
        logger.info("SAC agent created")
    
    def _create_callbacks(self):
        """Create training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint = self.config['training'].get('checkpoint', {})
        if checkpoint.get('save_freq', 0) > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint['save_freq'],
                save_path=str(self.output_dir / "checkpoints"),
                name_prefix="model"
            )
            callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if checkpoint.get('eval_freq', 0) > 0:
            n_eval_episodes = self.config['training']['optimization'].get('n_eval_episodes', 5)
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.output_dir),
                log_path=str(self.logs_dir),
                eval_freq=checkpoint['eval_freq'],
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # WandB callback (if enabled)
        logging_config = self.config['training'].get('logging', {})
        if logging_config.get('use_wandb', False):
            try:
                from .callbacks import WandBLoggingCallback
                wandb_callback = WandBLoggingCallback(
                    project=logging_config.get('project', 'nebula-trade'),
                    config=self.config,
                    model_name=self.model_name
                )
                callbacks.append(wandb_callback)
            except ImportError:
                logger.warning("WandB not available, skipping WandB logging")
        
        return callbacks
    
    def train(self):
        """Train the model."""
        total_timesteps = self.config['training']['total_timesteps']
        
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Train
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )
        
        logger.info("Training completed!")
        
        # Save final model
        self._save_model()
        
        # Register model
        self._register_model()
    
    def _save_model(self):
        """Save trained model and metadata."""
        # Save model
        model_path = self.output_dir / "final_model.zip"
        self.agent.save(str(model_path))
        logger.info(f"Model saved to: {model_path}")
        
        # Save VecNormalize stats
        if isinstance(self.env, VecNormalize):
            vec_normalize_path = self.output_dir / "vec_normalize.pkl"
            self.env.save(str(vec_normalize_path))
            logger.info(f"VecNormalize stats saved to: {vec_normalize_path}")
        
        # Save training summary
        summary = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'config': self.config,
            'training_completed': datetime.now().isoformat(),
            'total_timesteps': self.config['training']['total_timesteps'],
            'device': self.device,
            'n_parallel_envs': self.n_envs,
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
    
    def _register_model(self):
        """Register trained model in the registry."""
        try:
            model_id = self.model_registry.register_model(
                model_name=self.model_name,
                model_version=self.timestamp,
                model_dir=self.output_dir,
                metadata=self.config.get('metadata', {}),
                training_config=self.config
            )
            logger.info(f"Model registered with ID: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
    
    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """Evaluate trained model.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("EVALUATING MODEL")
        logger.info("=" * 80)
        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, dones, info = self.eval_env.step(action)
                episode_return += reward[0]
                episode_length += 1
                done = dones[0]
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        metrics = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_length': np.mean(episode_lengths),
        }
        
        logger.info(f"Evaluation Results ({n_episodes} episodes):")
        logger.info(f"  Mean Return: {metrics['mean_return']:.4f} ± {metrics['std_return']:.4f}")
        logger.info(f"  Mean Episode Length: {metrics['mean_length']:.1f}")
        
        # Save evaluation results
        eval_file = self.output_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
