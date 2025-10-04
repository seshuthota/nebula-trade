#!/usr/bin/env python3
"""
Nebula Trade CLI - Main entry point for all operations.

Usage:
    nebula train --model v1_momentum
    nebula evaluate --model v1_momentum
    nebula compare --models v1,v2,v4,v5
    nebula list models
    nebula list configs
    nebula info --model v5_tuned
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def train_command(args):
    """Handle train command."""
    from scripts.train import main as train_main
    
    # Reconstruct sys.argv for train script
    sys.argv = ['train.py', '--model', args.model]
    
    if args.training:
        sys.argv.extend(['--training', args.training])
    if args.timesteps:
        sys.argv.extend(['--timesteps', str(args.timesteps)])
    if args.no_wandb:
        sys.argv.append('--no-wandb')
    if args.no_gpu:
        sys.argv.append('--no-gpu')
    
    train_main()


def evaluate_command(args):
    """Handle evaluate command."""
    from scripts.evaluate import main as evaluate_main
    
    # Reconstruct sys.argv for evaluate script
    sys.argv = ['evaluate.py']
    
    if args.model:
        sys.argv.extend(['--model', args.model])
    if args.episodes:
        sys.argv.extend(['--episodes', str(args.episodes)])
    if args.period:
        sys.argv.extend(['--period', args.period])
    if args.output:
        sys.argv.extend(['--output', args.output])
    
    evaluate_main()


def compare_command(args):
    """Handle compare command."""
    from scripts.evaluate import main as evaluate_main
    
    # Reconstruct sys.argv for evaluate script
    sys.argv = ['evaluate.py', '--compare', args.models]
    
    if args.episodes:
        sys.argv.extend(['--episodes', str(args.episodes)])
    if args.period:
        sys.argv.extend(['--period', args.period])
    if args.output:
        sys.argv.extend(['--output', args.output])
    
    evaluate_main()


def list_command(args):
    """Handle list command."""
    from astra.core.config_manager import ConfigManager
    from astra.core.model_registry import ModelRegistry
    
    config_manager = ConfigManager(project_root)
    registry = ModelRegistry(project_root)
    
    if args.type == 'models':
        print("Available Model Configurations:")
        print("=" * 80)
        for model in config_manager.list_available_models():
            info = config_manager.get_model_info(model)
            status = info.get('status', 'active')
            desc = info.get('description', 'No description')
            print(f"  {model:<25} [{status:<10}] {desc}")
        
        print("\nTrained Models:")
        print("=" * 80)
        models = registry.list_models()
        if not models:
            print("  No trained models found.")
        else:
            for model_info in models[:10]:  # Show last 10
                print(f"  {model_info['id']:<40} [{model_info.get('status', 'active')}]")
    
    elif args.type == 'configs':
        print("Available Training Configurations:")
        print("=" * 80)
        for config in config_manager.list_available_training_configs():
            print(f"  - {config}")
    
    else:
        print(f"Unknown list type: {args.type}")
        print("Available types: models, configs")


def info_command(args):
    """Handle info command."""
    from astra.core.config_manager import ConfigManager
    
    config_manager = ConfigManager(project_root)
    
    try:
        info = config_manager.get_model_info(args.model)
        print(f"\nModel: {args.model}")
        print("=" * 80)
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


def main():
    parser = argparse.ArgumentParser(
        prog='nebula',
        description='Nebula Trade - Portfolio Optimization CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  nebula train --model v1_momentum
  nebula train --model v5_tuned --training production

  # Evaluate models
  nebula evaluate --model models/production/v1_momentum_20251004_120000
  nebula compare --models v1_momentum,v2_defensive,v5_tuned

  # List available configurations
  nebula list models
  nebula list configs

  # Get model information
  nebula info --model v5_tuned

For more help on specific commands:
  nebula train --help
  nebula evaluate --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, required=True, help='Model to train')
    train_parser.add_argument('--training', type=str, default='default', help='Training config')
    train_parser.add_argument('--timesteps', type=int, help='Override total timesteps')
    train_parser.add_argument('--no-wandb', action='store_true', help='Disable WandB')
    train_parser.add_argument('--no-gpu', action='store_true', help='Force CPU training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model', type=str, required=True, help='Model path')
    eval_parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    eval_parser.add_argument('--period', type=str, default='validation', help='Data period')
    eval_parser.add_argument('--output', type=str, help='Output file (JSON)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', type=str, required=True, help='Comma-separated model names')
    compare_parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    compare_parser.add_argument('--period', type=str, default='validation', help='Data period')
    compare_parser.add_argument('--output', type=str, help='Output file (JSON)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available items')
    list_parser.add_argument('type', choices=['models', 'configs'], help='What to list')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', type=str, required=True, help='Model name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'compare':
        compare_command(args)
    elif args.command == 'list':
        list_command(args)
    elif args.command == 'info':
        info_command(args)


if __name__ == '__main__':
    main()
