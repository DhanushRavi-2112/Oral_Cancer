import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from scipy import stats
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.preprocessing import create_data_loaders
from src.models.architectures import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = {}
        
    def load_model(
        self,
        model_type: str,
        checkpoint_path: str
    ) -> nn.Module:
        """Load a trained model from checkpoint."""
        # Create model
        model, _ = create_model(model_type, pretrained=False, device=self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        
        return model
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader,
        model_name: str
    ) -> Dict:
        """Evaluate a single model comprehensively."""
        logger.info(f"Evaluating {model_name}...")
        
        all_labels = []
        all_probs = []
        all_preds = []
        inference_times = []
        
        model.eval()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.numpy()
                
                # Measure inference time
                start_time = time.time()
                outputs = model(images).squeeze()
                torch.cuda.synchronize() if self.device == 'cuda' else None
                inference_time = time.time() - start_time
                inference_times.append(inference_time / len(images))
                
                # Get predictions
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_labels.extend(labels)
                all_probs.extend(probs)
                all_preds.extend(preds)
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        # Calculate metrics
        results = self._calculate_comprehensive_metrics(
            all_labels, all_probs, all_preds, model_name
        )
        
        # Add inference time
        results['avg_inference_time'] = np.mean(inference_times)
        results['std_inference_time'] = np.std(inference_times)
        
        # Store raw predictions for statistical tests
        results['labels'] = all_labels
        results['probs'] = all_probs
        results['preds'] = all_preds
        
        return results
    
    def _calculate_comprehensive_metrics(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        preds: np.ndarray,
        model_name: str
    ) -> Dict:
        """Calculate comprehensive metrics for evaluation."""
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        metrics = {
            'model_name': model_name,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
            'confusion_matrix': [[tn, fp], [fn, tp]]
        }
        
        # ROC and AUC
        fpr, tpr, roc_thresholds = roc_curve(labels, probs)
        metrics['auc_roc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['roc_thresholds'] = roc_thresholds
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
        metrics['auc_pr'] = average_precision_score(labels, probs)
        metrics['precision_curve'] = precision
        metrics['recall_curve'] = recall
        metrics['pr_thresholds'] = pr_thresholds
        
        # Matthews Correlation Coefficient
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0
        
        # Cohen's Kappa
        po = metrics['accuracy']
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / ((tp + tn + fp + fn) ** 2)
        metrics['cohen_kappa'] = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
        
        return metrics
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Compare multiple models and perform statistical tests."""
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Sensitivity': f"{results['sensitivity']:.4f}",
                'Specificity': f"{results['specificity']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'AUC-ROC': f"{results['auc_roc']:.4f}",
                'AUC-PR': f"{results['auc_pr']:.4f}",
                'MCC': f"{results['mcc']:.4f}",
                'Inference Time (ms)': f"{results['avg_inference_time']*1000:.2f}±{results['std_inference_time']*1000:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Perform statistical tests if we have two models
        if len(results_dict) == 2:
            model_names = list(results_dict.keys())
            self._perform_statistical_tests(
                results_dict[model_names[0]],
                results_dict[model_names[1]]
            )
        
        return comparison_df
    
    def _perform_statistical_tests(self, results1: Dict, results2: Dict):
        """Perform statistical tests between two models."""
        logger.info("\nStatistical Comparison:")
        
        # McNemar's test
        # Create contingency table
        correct1_correct2 = np.sum((results1['preds'] == results1['labels']) & 
                                  (results2['preds'] == results2['labels']))
        correct1_wrong2 = np.sum((results1['preds'] == results1['labels']) & 
                                (results2['preds'] != results2['labels']))
        wrong1_correct2 = np.sum((results1['preds'] != results1['labels']) & 
                                (results2['preds'] == results2['labels']))
        wrong1_wrong2 = np.sum((results1['preds'] != results1['labels']) & 
                              (results2['preds'] != results2['labels']))
        
        # McNemar's test statistic
        n12 = correct1_wrong2
        n21 = wrong1_correct2
        
        if n12 + n21 > 0:
            mcnemar_stat = ((abs(n12 - n21) - 1) ** 2) / (n12 + n21)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            logger.info(f"McNemar's test: χ² = {mcnemar_stat:.4f}, p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                logger.info("Significant difference between models (p < 0.05)")
            else:
                logger.info("No significant difference between models (p >= 0.05)")
        
        # DeLong's test for AUC comparison (simplified version)
        # This is a simplified implementation - for production use a proper implementation
        auc_diff = abs(results1['auc_roc'] - results2['auc_roc'])
        logger.info(f"\nAUC difference: {auc_diff:.4f}")
        
        # Bootstrap confidence intervals for metrics
        self._bootstrap_confidence_intervals(results1, results2)
    
    def _bootstrap_confidence_intervals(
        self,
        results1: Dict,
        results2: Dict,
        n_bootstrap: int = 1000
    ):
        """Calculate bootstrap confidence intervals for metric differences."""
        logger.info("\nBootstrap Confidence Intervals (95%):")
        
        n_samples = len(results1['labels'])
        metrics_to_compare = ['sensitivity', 'specificity', 'f1_score']
        
        for metric in metrics_to_compare:
            differences = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                
                # Recalculate metrics for bootstrap sample
                labels_boot = results1['labels'][indices]
                preds1_boot = results1['preds'][indices]
                preds2_boot = results2['preds'][indices]
                
                # Simple metric calculation for bootstrap
                if metric == 'sensitivity':
                    value1 = np.sum((preds1_boot == 1) & (labels_boot == 1)) / np.sum(labels_boot == 1)
                    value2 = np.sum((preds2_boot == 1) & (labels_boot == 1)) / np.sum(labels_boot == 1)
                elif metric == 'specificity':
                    value1 = np.sum((preds1_boot == 0) & (labels_boot == 0)) / np.sum(labels_boot == 0)
                    value2 = np.sum((preds2_boot == 0) & (labels_boot == 0)) / np.sum(labels_boot == 0)
                else:  # f1_score
                    tp1 = np.sum((preds1_boot == 1) & (labels_boot == 1))
                    fp1 = np.sum((preds1_boot == 1) & (labels_boot == 0))
                    fn1 = np.sum((preds1_boot == 0) & (labels_boot == 1))
                    value1 = 2 * tp1 / (2 * tp1 + fp1 + fn1) if (2 * tp1 + fp1 + fn1) > 0 else 0
                    
                    tp2 = np.sum((preds2_boot == 1) & (labels_boot == 1))
                    fp2 = np.sum((preds2_boot == 1) & (labels_boot == 0))
                    fn2 = np.sum((preds2_boot == 0) & (labels_boot == 1))
                    value2 = 2 * tp2 / (2 * tp2 + fp2 + fn2) if (2 * tp2 + fp2 + fn2) > 0 else 0
                
                differences.append(value1 - value2)
            
            # Calculate confidence interval
            ci_lower = np.percentile(differences, 2.5)
            ci_upper = np.percentile(differences, 97.5)
            
            logger.info(f"{metric} difference CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    def plot_results(
        self,
        results_dict: Dict[str, Dict],
        save_dir: str = 'outputs/figures'
    ):
        """Create comprehensive visualization of results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. ROC Curves
        plt.figure(figsize=(10, 8))
        for model_name, results in results_dict.items():
            plt.plot(
                results['fpr'], 
                results['tpr'], 
                label=f"{model_name} (AUC = {results['auc_roc']:.3f})",
                linewidth=2
            )
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        for model_name, results in results_dict.items():
            plt.plot(
                results['recall_curve'][:-1], 
                results['precision_curve'][:-1], 
                label=f"{model_name} (AP = {results['auc_pr']:.3f})",
                linewidth=2
            )
        
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrices
        fig, axes = plt.subplots(1, len(results_dict), figsize=(6*len(results_dict), 5))
        if len(results_dict) == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            cm = results['confusion_matrix']
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Healthy', 'Cancer'],
                yticklabels=['Healthy', 'Cancer'],
                ax=axes[idx],
                cbar_kws={'label': 'Count'}
            )
            axes[idx].set_title(f'{model_name} Confusion Matrix', fontsize=12)
            axes[idx].set_xlabel('Predicted', fontsize=11)
            axes[idx].set_ylabel('Actual', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Metrics Comparison Bar Plot
        metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc_roc']
        metric_names = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
        
        data_for_plot = []
        for model_name, results in results_dict.items():
            for metric, display_name in zip(metrics_to_plot, metric_names):
                data_for_plot.append({
                    'Model': model_name,
                    'Metric': display_name,
                    'Value': results[metric]
                })
        
        df_plot = pd.DataFrame(data_for_plot)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=df_plot, x='Metric', y='Value', hue='Model')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.ylim(0, 1.1)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.legend(title='Model', fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Inference Time Comparison
        plt.figure(figsize=(8, 6))
        models = list(results_dict.keys())
        times = [results_dict[m]['avg_inference_time'] * 1000 for m in models]
        stds = [results_dict[m]['std_inference_time'] * 1000 for m in models]
        
        bars = plt.bar(models, times, yerr=stds, capsize=10, alpha=0.7, edgecolor='black')
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.ylabel('Inference Time (ms)', fontsize=12)
        plt.title('Average Inference Time per Image', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (time, std) in enumerate(zip(times, stds)):
            plt.text(i, time + std + 0.5, f'{time:.2f}±{std:.2f}', ha='center', va='bottom')
        
        plt.savefig(os.path.join(save_dir, 'inference_times.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_dir}")
    
    def generate_report(
        self,
        results_dict: Dict[str, Dict],
        comparison_df: pd.DataFrame,
        save_path: str = 'outputs/evaluation_report.txt'
    ):
        """Generate comprehensive evaluation report."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ORAL CANCER DETECTION MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Model Comparison Table
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # Detailed Results
            f.write("DETAILED RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            for model_name, results in results_dict.items():
                f.write(f"{model_name.upper()} RESULTS:\n")
                f.write(f"  Confusion Matrix:\n")
                f.write(f"                 Predicted\n")
                f.write(f"                 Healthy  Cancer\n")
                f.write(f"  Actual Healthy   {results['confusion_matrix'][0][0]:5d}  {results['confusion_matrix'][0][1]:5d}\n")
                f.write(f"         Cancer   {results['confusion_matrix'][1][0]:5d}  {results['confusion_matrix'][1][1]:5d}\n")
                f.write(f"\n")
                f.write(f"  Key Metrics:\n")
                f.write(f"    - Accuracy:    {results['accuracy']:.4f}\n")
                f.write(f"    - Sensitivity: {results['sensitivity']:.4f} (True Positive Rate)\n")
                f.write(f"    - Specificity: {results['specificity']:.4f} (True Negative Rate)\n")
                f.write(f"    - Precision:   {results['precision']:.4f}\n")
                f.write(f"    - F1-Score:    {results['f1_score']:.4f}\n")
                f.write(f"    - NPV:         {results['npv']:.4f} (Negative Predictive Value)\n")
                f.write(f"    - AUC-ROC:     {results['auc_roc']:.4f}\n")
                f.write(f"    - AUC-PR:      {results['auc_pr']:.4f}\n")
                f.write(f"    - MCC:         {results['mcc']:.4f} (Matthews Correlation Coefficient)\n")
                f.write(f"    - Cohen's kappa:   {results['cohen_kappa']:.4f}\n")
                f.write(f"\n")
                f.write(f"  Performance:\n")
                f.write(f"    - Avg Inference Time: {results['avg_inference_time']*1000:.2f} ± {results['std_inference_time']*1000:.2f} ms\n")
                f.write(f"\n" + "-"*40 + "\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            
            # Find best model based on different criteria
            best_sensitivity = max(results_dict.items(), key=lambda x: x[1]['sensitivity'])
            best_specificity = max(results_dict.items(), key=lambda x: x[1]['specificity'])
            best_f1 = max(results_dict.items(), key=lambda x: x[1]['f1_score'])
            best_speed = min(results_dict.items(), key=lambda x: x[1]['avg_inference_time'])
            
            f.write(f"Best model for sensitivity (cancer detection): {best_sensitivity[0]} ({best_sensitivity[1]['sensitivity']:.4f})\n")
            f.write(f"Best model for specificity (avoiding false positives): {best_specificity[0]} ({best_specificity[1]['specificity']:.4f})\n")
            f.write(f"Best model for overall balance (F1-Score): {best_f1[0]} ({best_f1[1]['f1_score']:.4f})\n")
            f.write(f"Fastest model: {best_speed[0]} ({best_speed[1]['avg_inference_time']*1000:.2f} ms)\n")
            
            # Clinical considerations
            f.write("\n\nCLINICAL DEPLOYMENT CONSIDERATIONS:\n")
            f.write("-"*40 + "\n")
            
            for model_name, results in results_dict.items():
                if results['sensitivity'] >= 0.95:
                    f.write(f"✓ {model_name} meets target sensitivity (>95%)\n")
                else:
                    f.write(f"✗ {model_name} below target sensitivity (current: {results['sensitivity']:.2%}, target: >95%)\n")
                
                if results['specificity'] >= 0.90:
                    f.write(f"✓ {model_name} meets target specificity (>90%)\n")
                else:
                    f.write(f"✗ {model_name} below target specificity (current: {results['specificity']:.2%}, target: >90%)\n")
                f.write("\n")
        
        logger.info(f"Evaluation report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare trained models')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--model_dir', type=str, default='outputs/models',
                       help='Directory containing trained models')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run evaluation on')
    parser.add_argument('--save_dir', type=str, default='outputs',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        logger.warning("CUDA not available, using CPU")
    
    # Create evaluator
    evaluator = ModelEvaluator(device=device)
    
    # Create test data loader
    _, _, test_loader, _ = create_data_loaders(
        args.data_dir,
        batch_size=32,
        model_type='regnet'  # Both models use same preprocessing
    )
    
    # Dictionary to store results
    results_dict = {}
    
    # Evaluate RegNetY320
    regnet_checkpoint = os.path.join(args.model_dir, 'RegNetY-320MF', 'best_model.pth')
    if os.path.exists(regnet_checkpoint):
        regnet_model = evaluator.load_model('regnet', regnet_checkpoint)
        results_dict['RegNetY-320'] = evaluator.evaluate_model(
            regnet_model, test_loader, 'RegNetY-320'
        )
    else:
        logger.warning(f"RegNet checkpoint not found at {regnet_checkpoint}")
    
    # Evaluate VGG16
    vgg_checkpoint = os.path.join(args.model_dir, 'VGG16', 'best_model.pth')
    if os.path.exists(vgg_checkpoint):
        vgg_model = evaluator.load_model('vgg16', vgg_checkpoint)
        results_dict['VGG16'] = evaluator.evaluate_model(
            vgg_model, test_loader, 'VGG16'
        )
    else:
        logger.warning(f"VGG16 checkpoint not found at {vgg_checkpoint}")
    
    if len(results_dict) == 0:
        logger.error("No models found for evaluation!")
        return
    
    # Compare models
    comparison_df = evaluator.compare_models(results_dict)
    
    # Generate visualizations
    evaluator.plot_results(results_dict, save_dir=os.path.join(args.save_dir, 'figures'))
    
    # Generate report
    evaluator.generate_report(
        results_dict, 
        comparison_df,
        save_path=os.path.join(args.save_dir, 'evaluation_report.txt')
    )
    
    # Save raw results
    results_path = os.path.join(args.save_dir, 'results', 'comparison_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for model_name, results in results_dict.items():
        json_results[model_name] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in results.items()
            if k not in ['labels', 'probs', 'preds', 'fpr', 'tpr', 'roc_thresholds', 
                        'precision_curve', 'recall_curve', 'pr_thresholds']
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    logger.info("Model comparison completed!")
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()