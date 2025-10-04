#!/usr/bin/env python3
"""
Evaluation script for trained portfolio optimization models.
Load a saved model and evaluate its performance on test data.
"""

import os
import json
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from astra.rl_framework.environment import PortfolioEnvironment
from astra.evaluation.optimizer import ClassicalPortfolioOptimizer, PortfolioBacktester
from astra.data_pipeline.data_manager import PortfolioDataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioEvaluator:
    """Evaluate trained portfolio models."""

    def __init__(self, model_path: str, config_path: str = "config/portfolio.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['portfolio']
        
        self.assets = self.config['assets']
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.agent = SAC.load(str(self.model_path))
        
        # Create results directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.project_root / "evaluation_results" / self.timestamp
        self.plots_dir = self.results_dir / "plots"
        
        for dir_path in [self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluation results will be saved to: {self.results_dir}")

    def load_test_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load test data."""
        if data_path is None:
            logger.info("Loading and processing data...")
            data_manager = PortfolioDataManager(config_path=str(self.config_path))
            data, _ = data_manager.process_and_initialize()
            
            # Use last 20% as test data
            test_split = int(len(data) * 0.8)
            test_data = data.iloc[test_split:]
        else:
            logger.info(f"Loading data from {data_path}")
            test_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        logger.info(f"Test data: {len(test_data)} samples "
                   f"({test_data.index[0].date()} to {test_data.index[-1].date()})")
        
        return test_data

    def evaluate_rl_agent(self, test_data: pd.DataFrame, n_episodes: int = 20) -> Dict:
        """Evaluate RL agent on test data."""
        logger.info(f"Evaluating RL agent ({n_episodes} episodes)...")
        
        env = PortfolioEnvironment(test_data, lookback_window=30)
        
        episode_results = []
        all_trajectories = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            trajectory = []
            actions_taken = []
            
            while not (terminated or truncated):
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                actions_taken.append(action.copy())
                
                trajectory.append({
                    'step': info.get('current_step', 0),
                    'portfolio_value': info.get('portfolio_value', 0),
                    'cash': info.get('cash', 0),
                    'weights': info.get('weights', {}).copy(),
                    'reward': reward,
                    'action': action.tolist()
                })
            
            final_value = trajectory[-1]['portfolio_value']
            initial_value = trajectory[0]['portfolio_value']
            total_return = (final_value / initial_value - 1)
            
            # Calculate Sharpe ratio from episode returns
            episode_returns = [t['reward'] for t in trajectory]
            sharpe = np.mean(episode_returns) / (np.std(episode_returns) + 1e-8) * np.sqrt(252)
            
            episode_results.append({
                'episode': episode,
                'total_reward': episode_reward,
                'final_value': final_value,
                'initial_value': initial_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'steps': len(trajectory)
            })
            
            all_trajectories.append(trajectory)
            
            if (episode + 1) % 5 == 0:
                logger.info(f"  Episode {episode+1}/{n_episodes}: "
                          f"Return={total_return:.2%}, Sharpe={sharpe:.3f}")
        
        results_df = pd.DataFrame(episode_results)
        
        summary = {
            'mean_reward': results_df['total_reward'].mean(),
            'std_reward': results_df['total_reward'].std(),
            'mean_return': results_df['total_return'].mean(),
            'std_return': results_df['total_return'].std(),
            'mean_sharpe': results_df['sharpe_ratio'].mean(),
            'std_sharpe': results_df['sharpe_ratio'].std(),
            'mean_final_value': results_df['final_value'].mean(),
            'std_final_value': results_df['final_value'].std(),
            'n_episodes': n_episodes
        }
        
        logger.info("RL Agent Performance:")
        logger.info(f"  Mean Return: {summary['mean_return']:.2%} ± {summary['std_return']:.2%}")
        logger.info(f"  Mean Sharpe: {summary['mean_sharpe']:.3f} ± {summary['std_sharpe']:.3f}")
        logger.info(f"  Mean Final Value: ₹{summary['mean_final_value']:,.0f}")
        
        return {
            'summary': summary,
            'episodes': episode_results,
            'trajectories': all_trajectories
        }

    def evaluate_classical_methods(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate classical portfolio methods."""
        logger.info("Evaluating classical methods...")
        
        # Use full historical data for optimization (in practice, would use train data)
        returns_cols = [f"{asset}_returns" for asset in self.assets]
        returns_data = test_data[returns_cols].dropna()
        
        optimizer = ClassicalPortfolioOptimizer(returns_data)
        portfolios = optimizer.get_all_portfolios()
        
        backtester = PortfolioBacktester(test_data)
        results = {}
        
        logger.info("Classical Methods Performance:")
        for name, portfolio in portfolios.items():
            backtest_result = backtester.backtest_portfolio(
                weights=portfolio['weights'],
                rebalance_freq='daily',
                transaction_cost=self.config['rebalance']['transaction_cost']
            )
            
            results[name] = backtest_result
            logger.info(f"  {name:20s} | Return: {backtest_result['total_return']:7.2%} | "
                       f"Sharpe: {backtest_result['sharpe_ratio']:6.3f} | "
                       f"Final: ₹{backtest_result['final_value']:,.0f}")
        
        return results

    def compare_and_visualize(self, rl_results: Dict, classical_results: Dict):
        """Compare and visualize results."""
        logger.info("Generating comparison plots...")
        
        # 1. Returns comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns bar chart
        methods = ['RL (SAC)'] + [n.replace('_', ' ').title() for n in classical_results.keys()]
        returns = [rl_results['summary']['mean_return']] + \
                 [r['total_return'] for r in classical_results.values()]
        colors = ['#e74c3c'] + ['#3498db'] * len(classical_results)
        
        axes[0, 0].bar(range(len(methods)), returns, color=colors)
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Total Return')
        axes[0, 0].set_title('Returns Comparison')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add RL std as error bar
        axes[0, 0].errorbar([0], [rl_results['summary']['mean_return']], 
                           yerr=[rl_results['summary']['std_return']], 
                           fmt='none', color='black', capsize=5)
        
        # Sharpe ratio comparison
        sharpe_values = [rl_results['summary']['mean_sharpe']] + \
                       [r['sharpe_ratio'] for r in classical_results.values()]
        
        axes[0, 1].bar(range(len(methods)), sharpe_values, color=colors)
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Sharpe Ratio Comparison')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Portfolio value trajectories
        for i, trajectory in enumerate(rl_results['trajectories'][:5]):
            values = [t['portfolio_value'] for t in trajectory]
            axes[1, 0].plot(values, alpha=0.5, color='#e74c3c', 
                          label='RL' if i == 0 else '')
        
        for name, result in classical_results.items():
            if 'portfolio_values' in result:
                axes[1, 0].plot(result['portfolio_values'], 
                              label=name.replace('_', ' ').title(),
                              linestyle='--', linewidth=2)
        
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Portfolio Value (₹)')
        axes[1, 0].set_title('Portfolio Value Trajectories')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Episode performance distribution (RL only)
        episode_returns = [e['total_return'] for e in rl_results['episodes']]
        axes[1, 1].hist(episode_returns, bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(rl_results['summary']['mean_return'], color='black', 
                          linestyle='--', linewidth=2, label='Mean')
        axes[1, 1].set_xlabel('Total Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('RL Episode Returns Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "evaluation_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Weight allocation over time (RL)
        if len(rl_results['trajectories']) > 0:
            trajectory = rl_results['trajectories'][0]  # Use first episode
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Extract weights for each asset
            asset_weights = {asset: [] for asset in self.assets}
            asset_weights['cash'] = []
            
            for step_data in trajectory:
                weights = step_data['weights']
                for asset in self.assets:
                    asset_weights[asset].append(weights.get(asset, 0))
                asset_weights['cash'].append(weights.get('cash', 0))
            
            # Stacked area plot
            steps = range(len(trajectory))
            bottom = np.zeros(len(steps))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.assets) + 1))
            
            for i, asset in enumerate(self.assets + ['cash']):
                ax.fill_between(steps, bottom, bottom + asset_weights[asset],
                               label=asset, alpha=0.7, color=colors[i])
                bottom += asset_weights[asset]
            
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Portfolio Weight')
            ax.set_title('RL Agent Portfolio Weight Allocation Over Time')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / "weight_allocation.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Plots saved to {self.plots_dir}")

    def save_results(self, rl_results: Dict, classical_results: Dict):
        """Save evaluation results."""
        logger.info("Saving results...")
        
        # Save summary
        summary = {
            'timestamp': self.timestamp,
            'model_path': str(self.model_path),
            'rl_performance': rl_results['summary'],
            'classical_performance': {
                name: {
                    'total_return': r['total_return'],
                    'sharpe_ratio': r['sharpe_ratio'],
                    'final_value': r['final_value'],
                    'max_drawdown': r['max_drawdown']
                } for name, r in classical_results.items()
            }
        }
        
        with open(self.results_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed episode results
        episodes_df = pd.DataFrame(rl_results['episodes'])
        episodes_df.to_csv(self.results_dir / "rl_episodes.csv", index=False)
        
        # Generate markdown report
        with open(self.results_dir / "EVALUATION_REPORT.md", 'w') as f:
            f.write("# Portfolio Model Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {self.timestamp}\n\n")
            f.write(f"**Model:** {self.model_path.name}\n\n")
            f.write(f"**Assets:** {', '.join(self.assets)}\n\n")
            
            f.write("## RL Agent Performance\n\n")
            f.write(f"- Mean Return: {rl_results['summary']['mean_return']:.2%} "
                   f"± {rl_results['summary']['std_return']:.2%}\n")
            f.write(f"- Mean Sharpe Ratio: {rl_results['summary']['mean_sharpe']:.3f} "
                   f"± {rl_results['summary']['std_sharpe']:.3f}\n")
            f.write(f"- Mean Final Value: ₹{rl_results['summary']['mean_final_value']:,.0f}\n")
            f.write(f"- Episodes Evaluated: {rl_results['summary']['n_episodes']}\n\n")
            
            f.write("## Classical Methods Performance\n\n")
            for name, result in classical_results.items():
                f.write(f"### {name.replace('_', ' ').title()}\n")
                f.write(f"- Total Return: {result['total_return']:.2%}\n")
                f.write(f"- Sharpe Ratio: {result['sharpe_ratio']:.3f}\n")
                f.write(f"- Final Value: ₹{result['final_value']:,.0f}\n")
                f.write(f"- Max Drawdown: {result['max_drawdown']:.2%}\n\n")
            
            f.write("## Comparison\n\n")
            best_classical = max(classical_results.items(), 
                               key=lambda x: x[1]['total_return'])
            
            rl_return = rl_results['summary']['mean_return']
            classical_return = best_classical[1]['total_return']
            
            if rl_return > classical_return:
                f.write(f"✅ **RL agent outperformed the best classical method "
                       f"({best_classical[0]}) by "
                       f"{(rl_return - classical_return):.2%}**\n")
            else:
                f.write(f"⚠️ **RL agent underperformed the best classical method "
                       f"({best_classical[0]}) by "
                       f"{(classical_return - rl_return):.2%}**\n")
        
        logger.info(f"Results saved to {self.results_dir}")

    def run_evaluation(self, test_data_path: Optional[str] = None, n_episodes: int = 20):
        """Run complete evaluation."""
        logger.info("=" * 80)
        logger.info("PORTFOLIO MODEL EVALUATION")
        logger.info("=" * 80)
        
        # Load test data
        test_data = self.load_test_data(test_data_path)
        
        # Evaluate RL agent
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating RL Agent")
        logger.info("=" * 60)
        rl_results = self.evaluate_rl_agent(test_data, n_episodes=n_episodes)
        
        # Evaluate classical methods
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating Classical Methods")
        logger.info("=" * 60)
        classical_results = self.evaluate_classical_methods(test_data)
        
        # Compare and visualize
        logger.info("\n" + "=" * 60)
        logger.info("Generating Visualizations")
        logger.info("=" * 60)
        self.compare_and_visualize(rl_results, classical_results)
        
        # Save results
        logger.info("\n" + "=" * 60)
        logger.info("Saving Results")
        logger.info("=" * 60)
        self.save_results(rl_results, classical_results)
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained portfolio model")
    parser.add_argument('model_path', type=str,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data CSV (optional)')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--config', type=str, default='config/portfolio.yaml',
                       help='Path to config file (default: config/portfolio.yaml)')
    
    args = parser.parse_args()
    
    evaluator = PortfolioEvaluator(
        model_path=args.model_path,
        config_path=args.config
    )
    evaluator.run_evaluation(
        test_data_path=args.test_data,
        n_episodes=args.episodes
    )


if __name__ == "__main__":
    main()
