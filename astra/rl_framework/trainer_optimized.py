import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import logging
from multiprocessing import cpu_count

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .environment import PortfolioEnvironment
from ..data_pipeline.data_manager import PortfolioDataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitorCallback(BaseCallback):
    """Monitor training performance and hardware utilization."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None
        
    def _on_training_start(self):
        import time
        self.start_time = time.time()
        
        # Log hardware info
        device = self.model.device
        logger.info(f"Training device: {device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            import time
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            logger.info(f"Steps: {self.num_timesteps} | Steps/sec: {steps_per_sec:.1f}")
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        return True


class WandBLoggingCallback(BaseCallback):
    """Custom callback for logging to Weights & Biases."""

    def __init__(self, verbose=0):
        super(WandBLoggingCallback, self).__init__(verbose)
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project="portfolio-rl-astra", config={
                "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                "n_cpus": cpu_count(),
            })
        except ImportError:
            self.wandb = None

    def _on_step(self) -> bool:
        if self.wandb:
            # Log portfolio value and other metrics from info
            info = self.locals.get('infos', [{}])[0]
            if info:
                self.wandb.log({
                    'portfolio_value': info.get('portfolio_value', 0),
                    'episode': self.num_timesteps,
                })

        return True


class PortfolioTrainerOptimized:
    """Optimized trainer class for portfolio RL agents with GPU and multiprocessing support."""

    def __init__(self, 
                 config_path: str = None,
                 data_path: str = "data/portfolio_data_processed.csv",
                 n_envs: int = None,
                 use_gpu: bool = True):
        """
        Initialize optimized trainer.
        
        Args:
            config_path: Path to config file
            data_path: Path to processed data
            n_envs: Number of parallel environments (None = auto-detect)
            use_gpu: Whether to use GPU if available
        """
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

        # Load processed data
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Split train/test
        split_idx = int(len(self.data) * (1 - self.config['training']['test_split_ratio']))
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]

        # Determine optimal number of environments
        if n_envs is None:
            available_cpus = cpu_count()
            # Use 50-75% of CPUs, leave some for system
            self.n_envs = max(1, min(available_cpus - 2, 8))
        else:
            self.n_envs = n_envs

        # Determine device
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'

        logger.info(f"Hardware Configuration:")
        logger.info(f"  CPU cores: {cpu_count()}")
        logger.info(f"  Parallel environments: {self.n_envs}")
        logger.info(f"  Device: {self.device}")
        
        if self.use_gpu:
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Initialize environment
        self.env = self._create_vectorized_env(self.train_data, self.n_envs, normalize=True)

        # Create eval environment (also needs VecNormalize wrapper for consistency)
        eval_vec_env = DummyVecEnv([self._make_env(self.test_data, 0)])
        # Wrap eval env with VecNormalize (training=False so it uses training stats)
        self.eval_env = VecNormalize(
            eval_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
            training=False,  # Don't update stats during evaluation
        )

        # Initialize agent
        self.agent = self._create_agent()

        logger.info(f"Trainer initialized with {len(self.train_data)} train samples, {len(self.test_data)} test samples")

    def _make_env(self, data: pd.DataFrame, rank: int = 0):
        """Create a single environment (for parallel execution)."""
        def _init():
            env = PortfolioEnvironment(data, lookback_window=30)
            env = Monitor(env)
            return env
        return _init

    def _create_env(self, data: pd.DataFrame) -> PortfolioEnvironment:
        """Create single portfolio environment."""
        env = PortfolioEnvironment(data, lookback_window=30)
        env = Monitor(env)
        return env

    def _create_vectorized_env(self, data: pd.DataFrame, n_envs: int, normalize: bool = True):
        """Create vectorized environment for parallel training with normalization."""
        if n_envs == 1:
            # Single environment
            vec_env = DummyVecEnv([self._make_env(data, 0)])
        else:
            # Multiple parallel environments
            logger.info(f"Creating {n_envs} parallel environments using SubprocVecEnv")
            vec_env = SubprocVecEnv([self._make_env(data, i) for i in range(n_envs)])

        # Wrap with VecNormalize for observation and reward normalization
        if normalize:
            logger.info("Wrapping environment with VecNormalize for obs/reward normalization")
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,          # Normalize observations
                norm_reward=True,       # Normalize rewards
                clip_obs=10.0,          # Clip normalized observations
                clip_reward=10.0,       # Clip normalized rewards
                gamma=0.99,             # Discount for reward normalization
                epsilon=1e-8,           # Numerical stability
            )

        return vec_env

    def _create_agent(self) -> SAC:
        """Create optimized SAC agent with improved hyperparameters."""
        # More conservative network size for stability
        if self.use_gpu:
            # Moderate networks for GPU - smaller for better stability
            net_arch = [256, 256]
            batch_size = 256
            buffer_size = 200000
        else:
            # Smaller for CPU
            net_arch = [256, 256]
            batch_size = 128
            buffer_size = 100000

        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=torch.nn.ReLU,
            # Add gradient clipping to prevent exploding gradients
            normalize_images=False,  # We handle normalization ourselves
        )

        # Adjust learning starts based on number of envs
        learning_starts = 1000 * self.n_envs

        agent = SAC(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            # Lower learning rate for stability
            learning_rate=1e-4,  # Reduced from 3e-4
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            # Faster soft updates for better tracking
            tau=0.01,  # Increased from 0.005
            gamma=0.99,
            # More conservative gradient updates
            train_freq=(1, "step"),
            gradient_steps=1,  # Changed from -1 for stability
            device=self.device,
            tensorboard_log="./tensorboard_logs/",
            # Entropy tuning with bounds
            ent_coef='auto_0.1',  # Auto-tune with initial value 0.1
            target_entropy='auto',
            use_sde=False,
            sde_sample_freq=-1,
        )

        logger.info(f"Agent created with improved hyperparameters:")
        logger.info(f"  Network: {net_arch}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Buffer size: {buffer_size}")
        logger.info(f"  Learning rate: 1e-4")
        logger.info(f"  Gradient steps: 1 (conservative)")
        logger.info(f"  Tau: 0.01 (faster soft updates)")
        logger.info(f"  Device: {self.device}")

        return agent

    def train(self, 
              total_timesteps: int = 100000, 
              log_interval: int = 10,
              save_path: str = "models/portfolio_agent",
              eval_freq: int = 10000) -> None:
        """
        Train the agent with optimizations.
        
        Args:
            total_timesteps: Total training steps
            log_interval: Logging frequency
            save_path: Where to save models
            eval_freq: How often to evaluate (0 = disable)
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=max(eval_freq, 10000),
            save_path=save_path,
            name_prefix="portfolio_model",
            verbose=1
        )
        callbacks.append(checkpoint_callback)

        # Performance monitor
        perf_callback = PerformanceMonitorCallback(check_freq=1000, verbose=1)
        callbacks.append(perf_callback)

        # Evaluation callback (optional)
        if eval_freq > 0:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq // self.n_envs,  # Adjust for parallel envs
                deterministic=True,
                render=False,
                verbose=1,
                n_eval_episodes=10,  # Increased from 5 for better statistics
                warn=True,
            )
            callbacks.append(eval_callback)

        # WandB callback
        wandb_callback = WandBLoggingCallback()
        if wandb_callback.wandb:
            callbacks.append(wandb_callback)

        logger.info("=" * 80)
        logger.info(f"Starting optimized training for {total_timesteps} timesteps")
        logger.info(f"  Parallel environments: {self.n_envs}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Evaluation every: {eval_freq} steps")
        logger.info("=" * 80)

        # Train
        self.agent.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        self.agent.save(f"{save_path}/final_model.zip")
        logger.info(f"Training complete. Model saved to {save_path}/final_model.zip")

        # Save VecNormalize statistics if using normalization
        if isinstance(self.env, VecNormalize):
            self.env.save(f"{save_path}/vec_normalize.pkl")
            logger.info(f"VecNormalize stats saved to {save_path}/vec_normalize.pkl")

    def evaluate(self, data: Optional[pd.DataFrame] = None, n_episodes: int = 5) -> Dict:
        """Evaluate agent performance."""
        if data is None:
            data = self.test_data

        # Create eval environment with VecNormalize wrapper for consistency
        eval_vec_env = DummyVecEnv([self._make_env(data, 0)])
        eval_env = VecNormalize(
            eval_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
            training=False,  # Use fixed normalization stats
        )

        # Sync normalization stats from training env
        if isinstance(self.env, VecNormalize):
            eval_env.obs_rms = self.env.obs_rms
            eval_env.ret_rms = self.env.ret_rms

        episode_rewards = []
        episode_values = []

        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            final_value = 0

            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(action)

                # VecEnv returns arrays, extract first element
                reward = rewards[0]
                done = dones[0]
                info = infos[0]

                episode_reward += reward
                final_value = info.get('portfolio_value', 0)

            episode_rewards.append(episode_reward)
            episode_values.append(final_value)

            logger.info(f"Evaluation episode {episode+1}: reward={episode_reward:.4f}, final_value={final_value:.2f}")

        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_final_value': np.mean(episode_values),
            'std_final_value': np.std(episode_values),
            'episodes': n_episodes
        }

        logger.info(f"Evaluation results: {results}")
        return results

    def backtest(self, data: Optional[pd.DataFrame] = None,
                save_path: str = "results/backtest_results.csv") -> pd.DataFrame:
        """Run backtest with trained agent."""
        if data is None:
            data = self.test_data

        # Create backtest environment with VecNormalize wrapper for consistency
        backtest_vec_env = DummyVecEnv([self._make_env(data, 0)])
        env = VecNormalize(
            backtest_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
            training=False,
        )

        # Sync normalization stats from training env
        if isinstance(self.env, VecNormalize):
            env.obs_rms = self.env.obs_rms
            env.ret_rms = self.env.ret_rms

        obs = env.reset()

        results = []
        done = False

        while not done:
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            # VecEnv returns arrays, extract first element
            reward = rewards[0]
            done = dones[0]
            info = infos[0]

            info['action'] = action[0].tolist()
            info['obs'] = obs[0].tolist()
            info['reward'] = float(reward)
            results.append(info)

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path)
        logger.info(f"Backtest results saved to {save_path}")

        return results_df


if __name__ == "__main__":
    # Example usage with optimizations
    trainer = PortfolioTrainerOptimized(
        n_envs=6,  # Use 6 parallel environments
        use_gpu=True  # Enable GPU
    )
    
    # Train with GPU and parallel environments
    trainer.train(
        total_timesteps=50000,
        eval_freq=5000
    )

    # Evaluate
    eval_results = trainer.evaluate()
    print("Evaluation:", eval_results)
