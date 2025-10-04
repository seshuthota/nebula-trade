#!/usr/bin/env python3
"""
Resume training from a checkpoint.
Loads a saved model and continues training.
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml
import logging

from stable_baselines3 import SAC
from astra.rl_framework.trainer_optimized import PortfolioTrainerOptimized

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resume_training(checkpoint_path: str, 
                   additional_steps: int = 600000,
                   config_path: str = "config/portfolio.yaml"):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint .zip file
        additional_steps: How many MORE steps to train
        config_path: Path to config file
    """
    
    checkpoint_path = Path(checkpoint_path)
    project_root = Path(__file__).resolve().parent
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info("=" * 80)
    logger.info("RESUMING TRAINING FROM CHECKPOINT")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Additional steps: {additional_steps:,}")
    
    # Extract original training timestamp from path
    original_timestamp = checkpoint_path.parent.name
    
    # Load data (we need the same data the original training used)
    config_path = project_root / config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['portfolio']
    
    # Check if we have the original training data
    original_data_path = project_root / f"results/{original_timestamp}/train_data_processed.csv"
    
    if original_data_path.exists():
        logger.info(f"Using original training data from: {original_data_path}")
        data_path = str(original_data_path)
    else:
        logger.warning("Original training data not found, using default processed data")
        data_path = "data/portfolio_data_processed.csv"
    
    # Initialize trainer (to get environment setup)
    logger.info("Initializing trainer environment...")
    trainer = PortfolioTrainerOptimized(
        config_path=str(config_path),
        data_path=data_path,
        n_envs=None,  # Auto-detect
        use_gpu=True
    )
    
    # Load the checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path.name}")

    # Check if VecNormalize stats exist and load them
    vec_normalize_path = checkpoint_path.parent / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        logger.info(f"Loading VecNormalize stats from: {vec_normalize_path}")
        from stable_baselines3.common.vec_env import VecNormalize

        # The trainer already wrapped envs with VecNormalize
        # We need to get the base env, load stats onto it
        if isinstance(trainer.env, VecNormalize):
            # Get the base vectorized env (SubprocVecEnv or DummyVecEnv)
            base_env = trainer.env.venv
            # Load VecNormalize with saved stats around the base env
            trainer.env = VecNormalize.load(str(vec_normalize_path), base_env)
            logger.info("VecNormalize stats loaded for training env")

            # Do the same for eval env
            if isinstance(trainer.eval_env, VecNormalize):
                base_eval_env = trainer.eval_env.venv
                trainer.eval_env = VecNormalize.load(str(vec_normalize_path), base_eval_env)
                trainer.eval_env.training = False  # Keep eval env in non-training mode
                logger.info("VecNormalize stats loaded for eval env")
        else:
            # If not wrapped (shouldn't happen), wrap it
            trainer.env = VecNormalize.load(str(vec_normalize_path), trainer.env)
            logger.info("VecNormalize stats loaded for training env")

    agent = SAC.load(
        str(checkpoint_path),
        env=trainer.env,
        device='cuda'  # Make sure to use GPU
    )

    # Replace trainer's agent with loaded one
    trainer.agent = agent
    
    logger.info("✓ Checkpoint loaded successfully!")
    logger.info("=" * 80)
    
    # Create new save directory
    from datetime import datetime
    new_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = project_root / "models" / f"{new_timestamp}_resumed"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save info about resume
    resume_info = f"""
Resumed Training Information
============================
Original checkpoint: {checkpoint_path}
Original timestamp: {original_timestamp}
Resume timestamp: {new_timestamp}
Additional steps: {additional_steps}
Total steps (estimated): {checkpoint_path.stem.split('_')[-2] if 'steps' in checkpoint_path.stem else 'unknown'} + {additional_steps}
"""
    
    with open(save_path / "RESUME_INFO.txt", 'w') as f:
        f.write(resume_info)
    
    logger.info(resume_info)
    
    # Continue training
    logger.info("=" * 80)
    logger.info("CONTINUING TRAINING")
    logger.info("=" * 80)
    
    trainer.train(
        total_timesteps=additional_steps,
        log_interval=10,
        save_path=str(save_path),
        eval_freq=10000
    )
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Models saved to: {save_path}")
    logger.info("=" * 80)
    
    return str(save_path / "final_model.zip")


def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument('checkpoint', type=str,
                       help='Path to checkpoint file (e.g., models/TIMESTAMP/portfolio_model_400000_steps.zip)')
    parser.add_argument('--steps', type=int, default=600000,
                       help='Additional steps to train (default: 600000)')
    parser.add_argument('--config', type=str, default='config/portfolio.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    final_model = resume_training(
        checkpoint_path=args.checkpoint,
        additional_steps=args.steps,
        config_path=args.config
    )
    
    logger.info(f"\n✓ Training complete! Final model: {final_model}")


if __name__ == "__main__":
    main()
