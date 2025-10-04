#!/usr/bin/env python3
"""
Unified Training Script for nebula-trade.
Train any model using configuration files.

Usage:
    python scripts/train.py --model v1_momentum
    python scripts/train.py --model v5_tuned --training production
    python scripts/train.py --model v2_defensive --training quick_test --timesteps 50000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from astra.training.unified_trainer import UnifiedTrainer
from astra.core.config_manager import ConfigManager


def main():
    parser = argparse.ArgumentParser(
        description="Train portfolio optimization models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train v1 momentum model with default settings
  python scripts/train.py --model v1_momentum

  # Train v5 with production training config
  python scripts/train.py --model v5_tuned --training production

  # Quick test run with fewer steps
  python scripts/train.py --model v2_defensive --training quick_test

  # Override timesteps via environment or argument
  NEBULA_TIMESTEPS=50000 python scripts/train.py --model v1_momentum
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        help='Model to train (e.g., v1_momentum, v5_tuned)'
    )
    
    parser.add_argument(
        '--training',
        type=str,
        default='default',
        help='Training configuration to use (default: "default")'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Override total training timesteps'
    )
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU training (disable GPU)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available training configs and exit'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show model information and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize config manager
    config_manager = ConfigManager(project_root)
    
    # Handle list commands
    if args.list_models:
        print("Available models:")
        for model in config_manager.list_available_models():
            print(f"  - {model}")
        return
    
    if args.list_configs:
        print("Available training configs:")
        for config in config_manager.list_available_training_configs():
            print(f"  - {config}")
        return
    
    # Show model info
    if args.info:
        try:
            info = config_manager.get_model_info(args.model)
            print(f"\nModel: {args.model}")
            print("=" * 60)
            print(f"Version: {info.get('version', 'unknown')}")
            print(f"Description: {info.get('description', 'No description')}")
            print(f"Strategy: {info.get('strategy', 'unknown')}")
            print(f"Status: {info.get('status', 'active')}")
            print(f"\nTags: {', '.join(info.get('tags', []))}")
            
            performance = info.get('performance', {})
            if performance:
                print("\nExpected Performance:")
                for key, value in performance.items():
                    print(f"  {key}: {value}")
            
            notes = info.get('notes', '')
            if notes:
                print(f"\nNotes:\n{notes}")
            
        except FileNotFoundError:
            print(f"Error: Model '{args.model}' not found")
            print("\nAvailable models:")
            for model in config_manager.list_available_models():
                print(f"  - {model}")
        return
    
    # Prepare config overrides
    overrides = {}
    
    if args.timesteps:
        overrides['training'] = {'total_timesteps': args.timesteps}
    
    if args.no_wandb:
        if 'training' not in overrides:
            overrides['training'] = {}
        overrides['training']['logging'] = {'use_wandb': False}
    
    if args.no_gpu:
        if 'training' not in overrides:
            overrides['training'] = {}
        overrides['training']['optimization'] = {'use_gpu': False}
    
    # Create trainer
    print("=" * 80)
    print(f"TRAINING MODEL: {args.model}")
    print(f"Training Config: {args.training}")
    print("=" * 80)
    
    try:
        trainer = UnifiedTrainer(
            model_name=args.model,
            training_config=args.training,
            config_overrides=overrides if overrides else None,
            project_root=project_root
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        trainer.evaluate(n_episodes=20)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Model saved to: {trainer.output_dir}")
        print(f"Logs saved to: {trainer.logs_dir}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nAvailable models:")
        for model in config_manager.list_available_models():
            print(f"  - {model}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
