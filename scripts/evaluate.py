#!/usr/bin/env python3
"""
Unified Evaluation Script for nebula-trade.
Evaluate trained models and compare performance.

Usage:
    python scripts/evaluate.py --model production/v1_momentum_20251004_120000
    python scripts/evaluate.py --compare v1,v2,v4,v5 --period 2025_ytd
    python scripts/evaluate.py --list
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from astra.rl_framework.environment import PortfolioEnvironment
from astra.core.model_registry import ModelRegistry
from astra.data_pipeline.data_manager import PortfolioDataManager
from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester


class ModelEvaluator:
    """Evaluate trained models."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.registry = ModelRegistry(project_root)
        self.data_manager = PortfolioDataManager(
            config_path=str(project_root / "config" / "portfolio.yaml")
        )
    
    def load_model(self, model_path: Path):
        """Load a trained model."""
        # Load model
        model = SAC.load(str(model_path / "final_model.zip"))
        
        # Load VecNormalize if it exists
        vec_normalize_path = model_path / "vec_normalize.pkl"
        if vec_normalize_path.exists():
            vec_normalize = VecNormalize.load(str(vec_normalize_path), DummyVecEnv([lambda: None]))
            return model, vec_normalize
        
        return model, None
    
    def evaluate_model(
        self,
        model_path: Path,
        test_data: pd.DataFrame,
        n_episodes: int = 20
    ) -> Dict[str, Any]:
        """Evaluate a model on test data."""
        print(f"Evaluating model: {model_path.name}")
        
        # Load model
        model, vec_normalize = self.load_model(model_path)
        
        # Create environment
        env = PortfolioEnvironment(test_data, lookback_window=30)
        vec_env = DummyVecEnv([lambda: env])
        
        # Wrap with VecNormalize if available
        if vec_normalize is not None:
            vec_normalize.set_venv(vec_env)
            vec_normalize.training = False
            vec_env = vec_normalize
        
        # Run evaluation episodes
        episode_returns = []
        episode_lengths = []
        final_values = []
        
        for episode in range(n_episodes):
            obs = vec_env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                episode_return += reward[0]
                episode_length += 1
                done = dones[0]
                
                if done and infos:
                    final_values.append(infos[0].get('portfolio_value', 0))
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        # Compute metrics
        metrics = {
            'model_name': model_path.name,
            'n_episodes': n_episodes,
            'mean_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            'min_return': float(np.min(episode_returns)),
            'max_return': float(np.max(episode_returns)),
            'mean_length': float(np.mean(episode_lengths)),
            'mean_final_value': float(np.mean(final_values)) if final_values else 0.0,
        }
        
        return metrics
    
    def compare_models(
        self,
        model_paths: List[Path],
        test_data: pd.DataFrame,
        n_episodes: int = 20
    ) -> Dict[str, Any]:
        """Compare multiple models."""
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        results = {}
        
        for model_path in model_paths:
            metrics = self.evaluate_model(model_path, test_data, n_episodes)
            results[model_path.name] = metrics
        
        # Print comparison table
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Model':<40} {'Mean Return':>12} {'Std Return':>12} {'Final Value':>15}")
        print("-" * 80)
        
        for model_name, metrics in results.items():
            print(
                f"{model_name:<40} "
                f"{metrics['mean_return']:>12.4f} "
                f"{metrics['std_return']:>12.4f} "
                f"₹{metrics['mean_final_value']:>14,.0f}"
            )
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate portfolio optimization models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a specific model
  python scripts/evaluate.py --model models/production/v1_momentum_20251004_120000

  # Compare multiple models
  python scripts/evaluate.py --compare v1_momentum,v2_defensive,v5_tuned

  # List all available models
  python scripts/evaluate.py --list

  # Evaluate with more episodes
  python scripts/evaluate.py --model v1_momentum --episodes 50
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model directory to evaluate'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        help='Comma-separated list of model names to compare'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes (default: 20)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='validation',
        help='Data period to use: validation, 2025_ytd, q1_2025, q3_2025'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(project_root)
    
    # Handle list command
    if args.list:
        print("Available models:")
        print("=" * 80)
        
        models = evaluator.registry.list_models()
        if not models:
            print("No models found in registry.")
            print("\nTip: Train a model first with: python scripts/train.py --model v1_momentum")
        else:
            for model_info in models:
                print(f"  {model_info['id']}")
                print(f"    Path: {model_info['path']}")
                print(f"    Status: {model_info.get('status', 'unknown')}")
                print(f"    Registered: {model_info['registered_at']}")
                print()
        return
    
    # Load test data
    print("Loading data...")
    data, _ = evaluator.data_manager.process_and_initialize()
    
    # Select data period
    if args.period == 'validation':
        # Use last 10% as validation
        split_idx = int(len(data) * 0.90)
        test_data = data.iloc[split_idx:]
    elif args.period == '2025_ytd':
        test_data = data.loc['2025-01-01':]
    elif args.period == 'q1_2025':
        test_data = data.loc['2025-01-01':'2025-03-31']
    elif args.period == 'q3_2025':
        test_data = data.loc['2025-07-01':'2025-09-30']
    else:
        print(f"Unknown period: {args.period}")
        sys.exit(1)
    
    print(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print(f"Test samples: {len(test_data)}")
    print()
    
    # Evaluate single model
    if args.model:
        model_path = project_root / args.model
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)
        
        metrics = evaluator.evaluate_model(model_path, test_data, args.episodes)
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model: {metrics['model_name']}")
        print(f"Episodes: {metrics['n_episodes']}")
        print(f"Mean Return: {metrics['mean_return']:.4f} ± {metrics['std_return']:.4f}")
        print(f"Return Range: [{metrics['min_return']:.4f}, {metrics['max_return']:.4f}]")
        print(f"Mean Episode Length: {metrics['mean_length']:.1f}")
        print(f"Mean Final Value: ₹{metrics['mean_final_value']:,.0f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    # Compare multiple models
    elif args.compare:
        model_names = [name.strip() for name in args.compare.split(',')]
        model_paths = []
        
        for name in model_names:
            # Try to find latest version of this model
            latest = evaluator.registry.get_latest_model(name)
            if latest:
                model_path = project_root / latest['path']
                model_paths.append(model_path)
            else:
                print(f"Warning: Model '{name}' not found in registry")
        
        if not model_paths:
            print("Error: No valid models found")
            sys.exit(1)
        
        results = evaluator.compare_models(model_paths, test_data, args.episodes)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    else:
        print("Error: Must specify --model or --compare")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
