import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

from src.config import Config
from convolutional_counting import ObjectCounter

class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.counter = ObjectCounter(config)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Save config
        self._save_config()

    def _save_config(self):
        """Save configuration to results directory."""
        config_dict = {
            'model': {
                'name': self.config.model.model_name,
                'resize_dim': self.config.model.resize_dim,
                'map_keys': self.config.model.map_keys,
                'lora_weights': self.config.model.lora_weights
            },
            'processing': {
                'divide_et_impera': self.config.processing.divide_et_impera,
                'divide_et_impera_twice': self.config.processing.divide_et_impera_twice,
                'filter_background': self.config.processing.filter_background,
                'ellipse_normalization': self.config.processing.ellipse_normalization,
                'ellipse_kernel_cleaning': self.config.processing.ellipse_kernel_cleaning,
                'threshold': self.config.processing.threshold
            },
            'data': {
                'img_dir': self.config.data.img_dir,
                'annotation_file': self.config.data.annotation_file,
                'splits_file': self.config.data.splits_file
            }
        }
        
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

    def evaluate(self) -> Dict:
        """Run evaluation and return metrics."""
        results = self.counter.evaluate()
        
        # Calculate metrics
        metrics = {
            'mae': np.mean([r['mae'] for r in results]),
            'mse': np.mean([r['mse'] for r in results]),
            'rmse': np.mean([r['rmse'] for r in results]),
            'r2': self._calculate_r2(results)
        }
        
        # Save detailed results
        self._save_results(results, metrics)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        return metrics

    def _calculate_r2(self, results: List[Dict]) -> float:
        """Calculate R² score."""
        y_true = [r['ground_truth_count'] for r in results]
        y_pred = [r['predicted_count'] for r in results]
        return np.corrcoef(y_true, y_pred)[0, 1] ** 2

    def _save_results(self, results: List[Dict], metrics: Dict):
        """Save evaluation results in multiple formats."""
        # Save detailed results as JSON
        with open(self.run_dir / 'detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics as JSON
        with open(self.run_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save results as CSV
        df = pd.DataFrame(results)
        df.to_csv(self.run_dir / 'results.csv', index=False)
        
        # Save summary as text
        with open(self.run_dir / 'summary.txt', 'w') as f:
            f.write("Evaluation Summary\n")
            f.write("=================\n\n")
            f.write(f"Model: {self.config.model.model_name}\n")
            if self.config.model.lora_weights:
                f.write(f"LoRA Weights: {self.config.model.lora_weights}\n")
            f.write(f"Total images evaluated: {len(results)}\n")
            f.write(f"MAE: {metrics['mae']:.2f}\n")
            f.write(f"MSE: {metrics['mse']:.2f}\n")
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"R²: {metrics['r2']:.2f}\n")

    def _generate_visualizations(self, results: List[Dict]):
        """Generate visualization plots."""
        # Create plots directory
        plots_dir = self.run_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Scatter plot of predicted vs ground truth
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=[r['ground_truth_count'] for r in results],
            y=[r['predicted_count'] for r in results]
        )
        plt.plot([0, max(r['ground_truth_count'] for r in results)],
                [0, max(r['ground_truth_count'] for r in results)],
                'r--')
        plt.xlabel('Ground Truth Count')
        plt.ylabel('Predicted Count')
        plt.title('Predicted vs Ground Truth Counts')
        plt.savefig(plots_dir / 'scatter_plot.png')
        plt.close()
        
        # Error distribution
        plt.figure(figsize=(10, 6))
        errors = [r['predicted_count'] - r['ground_truth_count'] for r in results]
        sns.histplot(errors, bins=30)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.savefig(plots_dir / 'error_distribution.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate Object Counting Model')
    parser.add_argument('--model_name', type=str, default='dinov2_vits14_reg')
    parser.add_argument('--resize_dim', type=int, default=840)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--annotation', type=str, required=True)
    parser.add_argument('--splits', type=str, required=True)
    parser.add_argument('--lora_weights', type=str, default='',
                      help='Path to LoRA weights file')
    args = parser.parse_args()

    # Create config
    config = Config(
        model=ModelConfig(
            model_name=args.model_name,
            resize_dim=args.resize_dim,
            lora_weights=args.lora_weights
        ),
        data=DataConfig(
            img_dir=args.img_dir,
            annotation_file=args.annotation,
            splits_file=args.splits
        )
    )

    # Run evaluation
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate()
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Model: {config.model.model_name}")
    if config.model.lora_weights:
        print(f"LoRA Weights: {config.model.lora_weights}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R²: {metrics['r2']:.2f}")
    print(f"\nDetailed results saved in: {evaluator.run_dir}")

if __name__ == '__main__':
    main() 