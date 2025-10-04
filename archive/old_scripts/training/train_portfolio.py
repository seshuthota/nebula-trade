#!/usr/bin/env python3
"""
Complete Training Pipeline for Portfolio Optimization RL Agent.
Trains RL models, evaluates performance, and compares with classical methods.
"""

import os
import json
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from astra.rl_framework.trainer import PortfolioTrainer
try:
    from astra.rl_framework.trainer_optimized import PortfolioTrainerOptimized
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    PortfolioTrainerOptimized = PortfolioTrainer
from astra.rl_framework.environment import PortfolioEnvironment
from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester
from astra.data_pipeline.data_manager import PortfolioDataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioTrainingPipeline:
    """Complete training and evaluation pipeline for portfolio optimization."""

    def __init__(self, config_path: str = "config/portfolio.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']
        
        self.assets = self.config['assets']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.model_dir = self.project_root / "models" / self.timestamp
        self.results_dir = self.project_root / "results" / self.timestamp
        self.plots_dir = self.results_dir / "plots"
        
        for dir_path in [self.model_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training pipeline initialized")
        logger.info(f"Models will be saved to: {self.model_dir}")
        logger.info(f"Results will be saved to: {self.results_dir}")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split data into train/validation/test sets."""
        logger.info("=" * 60)
        logger.info("STEP 1: Data Preparation")
        logger.info("=" * 60)
        
        data_manager = PortfolioDataManager(config_path=str(self.config_path))
        data, _ = data_manager.process_and_initialize()
        
        # Split: 70% train, 15% validation, 15% test
        n = len(data)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        logger.info(f"Total samples: {n}")
        logger.info(f"Train: {len(train_data)} ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        logger.info(f"Validation: {len(val_data)} ({val_data.index[0].date()} to {val_data.index[-1].date()})")
        logger.info(f"Test: {len(test_data)} ({test_data.index[0].date()} to {test_data.index[-1].date()})")
        
        # Save splits
        train_data.to_csv(self.results_dir / "train_data.csv")
        val_data.to_csv(self.results_dir / "val_data.csv")
        test_data.to_csv(self.results_dir / "test_data.csv")
        
        return train_data, val_data, test_data

    def train_classical_baselines(self, train_data: pd.DataFrame) -> Dict:
        """Train classical portfolio optimization baselines."""
        logger.info("=" * 60)
        logger.info("STEP 2: Training Classical Baselines")
        logger.info("=" * 60)
        
        returns_cols = [f"{asset}_returns" for asset in self.assets]
        returns_data = train_data[returns_cols].dropna()
        
        optimizer = ClassicalPortfolioOptimizer(returns_data)
        portfolios = optimizer.get_all_portfolios()
        
        logger.info("Classical Portfolio Results:")
        for name, portfolio in portfolios.items():
            logger.info(f"  {name:20s} | Sharpe: {portfolio['sharpe_ratio']:6.3f} | "
                       f"Return: {portfolio['expected_return']:7.4f} | "
                       f"Vol: {portfolio['expected_volatility']:7.4f}")
        
        # Save results
        classical_results = {}
        for name, portfolio in portfolios.items():
            classical_results[name] = {
                'weights': portfolio['weights'].tolist(),
                'sharpe_ratio': float(portfolio['sharpe_ratio']),
                'expected_return': float(portfolio['expected_return']),
                'expected_volatility': float(portfolio['expected_volatility'])
            }
        
        with open(self.results_dir / "classical_portfolios.json", 'w') as f:
            json.dump(classical_results, f, indent=2)
        
        return portfolios

    def train_rl_agent(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                      total_timesteps: int = 100000, use_optimized: bool = True) -> PortfolioTrainer:
        """Train RL agent with proper validation."""
        logger.info("=" * 60)
        logger.info("STEP 3: Training RL Agent")
        logger.info("=" * 60)
        
        # Save data for trainer
        train_path = self.results_dir / "train_data_processed.csv"
        train_data.to_csv(train_path)
        
        # Initialize trainer (optimized version if available)
        if use_optimized and OPTIMIZED_AVAILABLE:
            logger.info("Using OPTIMIZED trainer with GPU and multiprocessing")
            trainer = PortfolioTrainerOptimized(
                config_path=str(self.config_path),
                data_path=str(train_path),
                n_envs=None,  # Auto-detect
                use_gpu=True
            )
        else:
            if use_optimized:
                logger.warning("Optimized trainer not available, falling back to standard trainer")
            logger.info("Using standard trainer")
            trainer = PortfolioTrainer(
                config_path=str(self.config_path),
                data_path=str(train_path)
            )
        
        # Override train/test split since we're managing it externally
        trainer.train_data = train_data
        trainer.test_data = val_data
        
        logger.info(f"Training for {total_timesteps} timesteps...")
        logger.info(f"Using SAC algorithm with continuous action space")
        
        # Train
        trainer.train(
            total_timesteps=total_timesteps,
            log_interval=10,
            save_path=str(self.model_dir)
        )
        
        logger.info("RL training completed!")
        return trainer

    def evaluate_agent(self, trainer: PortfolioTrainer, 
                      data: pd.DataFrame, 
                      name: str = "test",
                      n_episodes: int = 10) -> Dict:
        """Evaluate trained agent on given data."""
        logger.info(f"Evaluating on {name} set ({n_episodes} episodes)...")
        
        env = PortfolioEnvironment(data, lookback_window=30)
        
        episode_results = []
        portfolio_trajectories = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_trajectory = []
            
            while not (terminated or truncated):
                action, _ = trainer.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                episode_trajectory.append({
                    'step': info.get('current_step', 0),
                    'portfolio_value': info.get('portfolio_value', 0),
                    'cash': info.get('cash', 0),
                    'weights': info.get('weights', {}),
                    'reward': reward
                })
            
            episode_results.append({
                'episode': episode,
                'total_reward': episode_reward,
                'final_value': episode_trajectory[-1]['portfolio_value'],
                'total_return': (episode_trajectory[-1]['portfolio_value'] / 
                               episode_trajectory[0]['portfolio_value'] - 1),
                'steps': len(episode_trajectory)
            })
            
            portfolio_trajectories.append(episode_trajectory)
        
        # Calculate statistics
        results_df = pd.DataFrame(episode_results)
        
        summary = {
            'mean_reward': results_df['total_reward'].mean(),
            'std_reward': results_df['total_reward'].std(),
            'mean_return': results_df['total_return'].mean(),
            'std_return': results_df['total_return'].std(),
            'mean_final_value': results_df['final_value'].mean(),
            'std_final_value': results_df['final_value'].std(),
            'n_episodes': n_episodes
        }
        
        logger.info(f"  Mean Reward: {summary['mean_reward']:.4f} ± {summary['std_reward']:.4f}")
        logger.info(f"  Mean Return: {summary['mean_return']:.2%} ± {summary['std_return']:.2%}")
        logger.info(f"  Mean Final Value: ₹{summary['mean_final_value']:,.0f}")
        
        return {
            'summary': summary,
            'episodes': episode_results,
            'trajectories': portfolio_trajectories
        }

    def backtest_classical(self, portfolios: Dict, data: pd.DataFrame) -> Dict:
        """Backtest classical portfolios."""
        logger.info("Backtesting classical portfolios...")
        
        backtester = PortfolioBacktester(data)
        results = {}
        
        for name, portfolio in portfolios.items():
            backtest_result = backtester.backtest_portfolio(
                weights=portfolio['weights'],
                rebalance_freq='daily',
                transaction_cost=self.config['rebalance']['transaction_cost']
            )
            
            results[name] = backtest_result
            logger.info(f"  {name:20s} | Final Value: ₹{backtest_result['final_value']:,.0f} | "
                       f"Return: {backtest_result['total_return']:7.2%} | "
                       f"Sharpe: {backtest_result['sharpe_ratio']:6.3f}")
        
        return results

    def compare_methods(self, rl_results: Dict, classical_results: Dict, 
                       dataset_name: str = "test") -> pd.DataFrame:
        """Compare RL vs classical methods."""
        logger.info(f"Comparing methods on {dataset_name} set...")
        
        comparison_data = []
        
        # RL results
        comparison_data.append({
            'Method': 'RL (SAC)',
            'Type': 'RL',
            'Mean Return': rl_results['summary']['mean_return'],
            'Std Return': rl_results['summary']['std_return'],
            'Mean Final Value': rl_results['summary']['mean_final_value'],
            'Mean Reward': rl_results['summary']['mean_reward']
        })
        
        # Classical results
        for name, result in classical_results.items():
            comparison_data.append({
                'Method': name.replace('_', ' ').title(),
                'Type': 'Classical',
                'Mean Return': result['total_return'],
                'Std Return': 0,  # Single run
                'Mean Final Value': result['final_value'],
                'Sharpe Ratio': result['sharpe_ratio']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.results_dir / f"comparison_{dataset_name}.csv", index=False)
        
        logger.info(f"\nPerformance Comparison ({dataset_name}):")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df

    def plot_results(self, rl_results: Dict, classical_results: Dict, 
                    comparison_df: pd.DataFrame):
        """Generate visualization plots."""
        logger.info("Generating plots...")
        
        # 1. Portfolio value trajectories
        plt.figure(figsize=(14, 6))
        
        # Plot RL trajectories (sample a few episodes)
        for i, trajectory in enumerate(rl_results['trajectories'][:3]):
            values = [t['portfolio_value'] for t in trajectory]
            plt.plot(values, alpha=0.7, label=f'RL Episode {i+1}')
        
        # Plot classical portfolios
        for name, result in classical_results.items():
            if 'portfolio_values' in result:
                plt.plot(result['portfolio_values'], 
                        label=name.replace('_', ' ').title(),
                        linestyle='--', linewidth=2)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value (₹)')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / "portfolio_trajectories.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Returns comparison
        plt.figure(figsize=(10, 6))
        methods = comparison_df['Method'].tolist()
        returns = comparison_df['Mean Return'].tolist()
        colors = ['#2ecc71' if t == 'RL' else '#3498db' 
                 for t in comparison_df['Type']]
        
        bars = plt.bar(range(len(methods)), returns, color=colors)
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        plt.ylabel('Mean Return')
        plt.title('Returns Comparison: RL vs Classical Methods')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "returns_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. RL learning curve (from episode rewards)
        if len(rl_results['episodes']) > 1:
            plt.figure(figsize=(10, 6))
            episodes = [e['episode'] for e in rl_results['episodes']]
            rewards = [e['total_reward'] for e in rl_results['episodes']]
            returns = [e['total_return'] for e in rl_results['episodes']]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            ax1.plot(episodes, rewards, marker='o')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('RL Agent Evaluation: Episode Rewards')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(episodes, returns, marker='o', color='green')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Total Return')
            ax2.set_title('RL Agent Evaluation: Episode Returns')
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / "rl_evaluation_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Plots saved to {self.plots_dir}")

    def generate_report(self, rl_val_results: Dict, rl_test_results: Dict,
                       classical_val: Dict, classical_test: Dict):
        """Generate comprehensive training report."""
        logger.info("Generating final report...")
        
        report = {
            'timestamp': self.timestamp,
            'config': self.config,
            'validation_results': {
                'rl': rl_val_results['summary'],
                'classical': {name: {
                    'total_return': r['total_return'],
                    'sharpe_ratio': r['sharpe_ratio'],
                    'final_value': r['final_value']
                } for name, r in classical_val.items()}
            },
            'test_results': {
                'rl': rl_test_results['summary'],
                'classical': {name: {
                    'total_return': r['total_return'],
                    'sharpe_ratio': r['sharpe_ratio'],
                    'final_value': r['final_value']
                } for name, r in classical_test.items()}
            }
        }
        
        with open(self.results_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        with open(self.results_dir / "REPORT.md", 'w') as f:
            f.write("# Portfolio Optimization Training Report\n\n")
            f.write(f"**Training Date:** {self.timestamp}\n\n")
            f.write(f"**Assets:** {', '.join(self.assets)}\n\n")
            
            f.write("## Validation Results\n\n")
            f.write("### RL Agent (SAC)\n")
            f.write(f"- Mean Return: {rl_val_results['summary']['mean_return']:.2%}\n")
            f.write(f"- Std Return: {rl_val_results['summary']['std_return']:.2%}\n")
            f.write(f"- Mean Reward: {rl_val_results['summary']['mean_reward']:.4f}\n\n")
            
            f.write("### Classical Methods\n")
            for name, result in classical_val.items():
                f.write(f"#### {name.replace('_', ' ').title()}\n")
                f.write(f"- Total Return: {result['total_return']:.2%}\n")
                f.write(f"- Sharpe Ratio: {result['sharpe_ratio']:.3f}\n")
                f.write(f"- Final Value: ₹{result['final_value']:,.0f}\n\n")
            
            f.write("## Test Results\n\n")
            f.write("### RL Agent (SAC)\n")
            f.write(f"- Mean Return: {rl_test_results['summary']['mean_return']:.2%}\n")
            f.write(f"- Std Return: {rl_test_results['summary']['std_return']:.2%}\n")
            f.write(f"- Mean Reward: {rl_test_results['summary']['mean_reward']:.4f}\n\n")
            
            f.write("### Classical Methods\n")
            for name, result in classical_test.items():
                f.write(f"#### {name.replace('_', ' ').title()}\n")
                f.write(f"- Total Return: {result['total_return']:.2%}\n")
                f.write(f"- Sharpe Ratio: {result['sharpe_ratio']:.3f}\n")
                f.write(f"- Final Value: ₹{result['final_value']:,.0f}\n\n")
        
        logger.info(f"Report saved to {self.results_dir}/REPORT.md")

    def run_complete_training(self, timesteps: int = 100000, use_optimized: bool = True):
        """Run the complete training pipeline."""
        logger.info("=" * 80)
        logger.info("PORTFOLIO OPTIMIZATION TRAINING PIPELINE")
        logger.info("=" * 80)
        
        # 1. Prepare data
        train_data, val_data, test_data = self.prepare_data()
        
        # 2. Train classical baselines
        classical_portfolios = self.train_classical_baselines(train_data)
        
        # 3. Train RL agent
        trainer = self.train_rl_agent(train_data, val_data, total_timesteps=timesteps, use_optimized=use_optimized)
        
        # 4. Evaluate on validation set
        logger.info("=" * 60)
        logger.info("STEP 4: Validation Set Evaluation")
        logger.info("=" * 60)
        
        rl_val_results = self.evaluate_agent(trainer, val_data, name="validation", n_episodes=10)
        classical_val_results = self.backtest_classical(classical_portfolios, val_data)
        val_comparison = self.compare_methods(rl_val_results, classical_val_results, "validation")
        
        # 5. Evaluate on test set
        logger.info("=" * 60)
        logger.info("STEP 5: Test Set Evaluation")
        logger.info("=" * 60)
        
        rl_test_results = self.evaluate_agent(trainer, test_data, name="test", n_episodes=10)
        classical_test_results = self.backtest_classical(classical_portfolios, test_data)
        test_comparison = self.compare_methods(rl_test_results, classical_test_results, "test")
        
        # 6. Generate plots
        logger.info("=" * 60)
        logger.info("STEP 6: Generating Visualizations")
        logger.info("=" * 60)
        self.plot_results(rl_test_results, classical_test_results, test_comparison)
        
        # 7. Generate report
        logger.info("=" * 60)
        logger.info("STEP 7: Generating Final Report")
        logger.info("=" * 60)
        self.generate_report(rl_val_results, rl_test_results, 
                           classical_val_results, classical_test_results)
        
        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Models saved to: {self.model_dir}")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"Plots saved to: {self.plots_dir}")
        logger.info("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train portfolio optimization RL agent")
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps (default: 100000)')
    parser.add_argument('--config', type=str, default='config/portfolio.yaml',
                       help='Path to config file (default: config/portfolio.yaml)')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Disable GPU and multiprocessing optimizations')
    
    args = parser.parse_args()
    
    pipeline = PortfolioTrainingPipeline(config_path=args.config)
    pipeline.run_complete_training(timesteps=args.timesteps, use_optimized=not args.no_optimize)


if __name__ == "__main__":
    main()
