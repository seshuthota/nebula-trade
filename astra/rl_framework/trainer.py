import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import logging

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import PortfolioEnvironment
from ..data_pipeline.data_manager import PortfolioDataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WandBLoggingCallback(BaseCallback):
    """Custom callback for logging to Weights & Biases."""

    def __init__(self, verbose=0):
        super(WandBLoggingCallback, self).__init__(verbose)
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project="portfolio-rl-astra")
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

class PortfolioTrainer:
    """Trainer class for portfolio RL agents."""

    def __init__(self, config_path: str = None,
                 data_path: str = "data/portfolio_data_processed.csv"):
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

        # Split train/test (e.g., 80/20)
        split_idx = int(len(self.data) * (1 - self.config['training']['test_split_ratio']))
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]

        # Initialize environment
        self.env = self._create_env(self.train_data)

        # Initialize agent
        self.agent = self._create_agent()

        logger.info(f"Trainer initialized with {len(self.train_data)} train samples, {len(self.test_data)} test samples")

    def _create_env(self, data: pd.DataFrame) -> PortfolioEnvironment:
        """Create portfolio environment."""
        env = PortfolioEnvironment(data, lookback_window=30)
        env = Monitor(env)  # Add monitoring
        return env

    def _create_agent(self) -> SAC:
        """Create SAC agent."""
        # Use continuous action space for portfolio weights
        policy_kwargs = dict(net_arch=[256, 256])  # Actor-critic network architecture

        agent = SAC(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            # Improved exploration parameters
            ent_coef='auto',  # Auto-tune entropy coefficient
            target_entropy='auto',  # Automatically set target entropy
        )

        return agent

    def train(self, total_timesteps: int = 100000, log_interval: int = 10,
              save_path: str = "models/portfolio_agent") -> None:
        """Train the agent."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=log_interval * 1000,
            save_path=save_path,
            name_prefix="portfolio_model"
        )

        wandb_callback = WandBLoggingCallback()

        callbacks = [checkpoint_callback]
        if wandb_callback.wandb:
            callbacks.append(wandb_callback)

        logger.info(f"Starting training for {total_timesteps} timesteps")

        # Train
        self.agent.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callbacks
        )

        # Save final model
        self.agent.save(f"{save_path}/final_model.zip")
        logger.info(f"Training complete. Model saved to {save_path}/final_model.zip")

    def evaluate(self, data: Optional[pd.DataFrame] = None, n_episodes: int = 5) -> Dict:
        """Evaluate agent performance."""
        if data is None:
            data = self.test_data

        eval_env = self._create_env(data)

        episode_rewards = []
        episode_values = []

        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            final_value = 0

            while not (terminated or truncated):
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
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

        env = self._create_env(data)
        obs, _ = env.reset()

        results = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            info['action'] = action.tolist()
            info['obs'] = obs.tolist()
            info['reward'] = reward
            results.append(info)

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path)
        logger.info(f"Backtest results saved to {save_path}")

        return results_df

if __name__ == "__main__":
    trainer = PortfolioTrainer()
    trainer.train(total_timesteps=10000)  # Quick test

    # Evaluate
    eval_results = trainer.evaluate()
    print("Evaluation:", eval_results)

    # Backtest
    backtest_results = trainer.backtest()
    print("Backtest shape:", backtest_results.shape)
