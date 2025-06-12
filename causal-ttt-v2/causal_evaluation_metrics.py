"""
Comprehensive evaluation metrics for causal inference in time series forecasting.

This module provides various metrics for evaluating causal models including:
1. Precision in Estimation of Heterogeneous Effect (PEHE)
2. Average Treatment Effect (ATE) estimation
3. Policy value evaluation
4. Confounding robustness metrics
5. Uncertainty calibration metrics
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CausalMetric(ABC):
    """Abstract base class for causal inference metrics."""
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, 
                treatments: np.ndarray, **kwargs) -> Dict[str, float]:
        """Compute the metric."""
        pass


class PEHEMetric(CausalMetric):
    """Precision in Estimation of Heterogeneous Effect (PEHE) metric."""
    
    def __init__(self, sqrt: bool = True):
        self.sqrt = sqrt
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, 
                treatments: np.ndarray, true_effects: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, float]:
        """
        Compute PEHE metric.
        
        Args:
            y_true: Ground truth outcomes [batch_size, 1]
            y_pred: Predicted outcomes [batch_size, num_treatments]
            treatments: Treatment assignments [batch_size]
            true_effects: True treatment effects if available [batch_size, num_treatments]
        
        Returns:
            Dictionary with PEHE metrics
        """
        if true_effects is None:
            logger.warning("True effects not provided, computing approximation")
            # Approximate true effects using observed data
            num_treatments = y_pred.shape[1]
            true_effects = np.zeros((len(y_true), num_treatments))
            
            for t in range(num_treatments):
                mask = treatments == t
                if np.sum(mask) > 0:
                    true_effects[:, t] = np.mean(y_true[mask])
        
        # Compute Individual Treatment Effects (ITE)
        num_treatments = y_pred.shape[1]
        true_ite = true_effects - true_effects[:, 0:1]  # Relative to control
        pred_ite = y_pred - y_pred[:, 0:1]
        
        # PEHE computation
        pehe_squared = np.mean((true_ite - pred_ite) ** 2)
        pehe = np.sqrt(pehe_squared) if self.sqrt else pehe_squared
        
        # Per-treatment PEHE
        per_treatment_pehe = {}
        for t in range(1, num_treatments):
            pehe_t = np.mean((true_ite[:, t] - pred_ite[:, t]) ** 2)
            pehe_t = np.sqrt(pehe_t) if self.sqrt else pehe_t
            per_treatment_pehe[f'pehe_treatment_{t}'] = pehe_t
        
        result = {
            'pehe': pehe,
            'pehe_squared': pehe_squared,
            **per_treatment_pehe
        }
        
        return result


class ATEMetric(CausalMetric):
    """Average Treatment Effect (ATE) metric."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, 
                treatments: np.ndarray, true_ate: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, float]:
        """
        Compute ATE metrics.
        
        Args:
            y_true: Ground truth outcomes
            y_pred: Predicted counterfactual outcomes [batch_size, num_treatments]
            treatments: Treatment assignments
            true_ate: True ATE values if available
        
        Returns:
            Dictionary with ATE metrics
        """
        num_treatments = y_pred.shape[1]
        
        # Predicted ATE (relative to control treatment 0)
        pred_ate = np.mean(y_pred, axis=0) - np.mean(y_pred[:, 0])
        
        # True ATE from observed data (biased estimate)
        observed_ate = []
        for t in range(num_treatments):
            mask = treatments == t
            if np.sum(mask) > 0:
                observed_ate.append(np.mean(y_true[mask]))
            else:
                observed_ate.append(0.0)
        
        observed_ate = np.array(observed_ate) - observed_ate[0]
        
        # ATE bias (if true ATE provided)
        result = {
            'pred_ate': pred_ate.tolist(),
            'observed_ate': observed_ate.tolist(),
        }
        
        if true_ate is not None:
            ate_bias = pred_ate - true_ate
            ate_mse = np.mean(ate_bias ** 2)
            ate_mae = np.mean(np.abs(ate_bias))
            
            result.update({
                'true_ate': true_ate.tolist(),
                'ate_bias': ate_bias.tolist(),
                'ate_mse': ate_mse,
                'ate_mae': ate_mae
            })
        
        return result


class PolicyValueMetric(CausalMetric):
    """Policy value evaluation metric."""
    
    def __init__(self, policy_fn=None):
        self.policy_fn = policy_fn or self._default_policy
    
    def _default_policy(self, y_pred: np.ndarray) -> np.ndarray:
        """Default policy: select treatment with highest predicted outcome."""
        return np.argmax(y_pred, axis=1)
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, 
                treatments: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Compute policy value metrics.
        
        Args:
            y_true: Ground truth outcomes
            y_pred: Predicted counterfactual outcomes
            treatments: Actual treatment assignments
        
        Returns:
            Dictionary with policy value metrics
        """
        # Get policy recommendations
        recommended_treatments = self.policy_fn(y_pred)
        
        # Policy value: expected outcome under the recommended policy
        policy_value = np.mean([y_pred[i, recommended_treatments[i]] 
                               for i in range(len(y_pred))])
        
        # Random policy baseline
        random_policy_value = np.mean(y_pred)
        
        # Current policy value (observed treatments)
        current_policy_value = np.mean([y_pred[i, treatments[i]] 
                                      for i in range(len(y_pred))])
        
        # Policy improvement
        policy_improvement = policy_value - current_policy_value
        
        # Treatment distribution under recommended policy
        treatment_dist = np.bincount(recommended_treatments, 
                                   minlength=y_pred.shape[1]) / len(recommended_treatments)
        
        return {
            'policy_value': policy_value,
            'random_policy_value': random_policy_value,
            'current_policy_value': current_policy_value,
            'policy_improvement': policy_improvement,
            'treatment_distribution': treatment_dist.tolist()
        }


class ConfoundingRobustnessMetric(CausalMetric):
    """Metrics for evaluating robustness to confounding."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, 
                treatments: np.ndarray, covariates: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, float]:
        """
        Compute confounding robustness metrics.
        
        Args:
            y_true: Ground truth outcomes
            y_pred: Predicted outcomes
            treatments: Treatment assignments
            covariates: Observed covariates
        
        Returns:
            Dictionary with robustness metrics
        """
        # Treatment assignment balance
        treatment_counts = np.bincount(treatments)
        treatment_balance = np.std(treatment_counts) / np.mean(treatment_counts)
        
        # Outcome variance by treatment
        outcome_variances = []
        for t in range(len(treatment_counts)):
            mask = treatments == t
            if np.sum(mask) > 1:
                outcome_variances.append(np.var(y_true[mask]))
            else:
                outcome_variances.append(0.0)
        
        outcome_variance_ratio = np.max(outcome_variances) / (np.min(outcome_variances) + 1e-8)
        
        result = {
            'treatment_balance': treatment_balance,
            'outcome_variance_ratio': outcome_variance_ratio,
            'treatment_counts': treatment_counts.tolist()
        }
        
        # Covariate balance if available
        if covariates is not None:
            covariate_imbalance = []
            for feature_idx in range(covariates.shape[1]):
                feature_stds = []
                for t in range(len(treatment_counts)):
                    mask = treatments == t
                    if np.sum(mask) > 1:
                        feature_stds.append(np.std(covariates[mask, feature_idx]))
                
                if len(feature_stds) > 1:
                    covariate_imbalance.append(np.std(feature_stds))
            
            result['covariate_imbalance'] = np.mean(covariate_imbalance) if covariate_imbalance else 0.0
        
        return result


class UncertaintyCalibrationMetric(CausalMetric):
    """Metrics for evaluating uncertainty calibration."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, 
                treatments: np.ndarray, uncertainties: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, float]:
        """
        Compute uncertainty calibration metrics.
        
        Args:
            y_true: Ground truth outcomes
            y_pred: Predicted outcomes
            treatments: Treatment assignments
            uncertainties: Predicted uncertainties
        
        Returns:
            Dictionary with calibration metrics
        """
        if uncertainties is None:
            logger.warning("No uncertainties provided for calibration metrics")
            return {'calibration_error': np.nan}
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Convert uncertainties to confidence scores
        # Assuming uncertainties are standard deviations
        confidences = 1.0 / (1.0 + uncertainties.flatten())
        
        # Compute errors for factual predictions
        factual_pred = y_pred[np.arange(len(treatments)), treatments].flatten()
        errors = np.abs(factual_pred - y_true.flatten())
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (errors[in_bin] < np.std(y_true)).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Reliability diagram data
        reliability_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                accuracy = (errors[in_bin] < np.std(y_true)).mean()
                confidence = confidences[in_bin].mean()
                count = in_bin.sum()
                reliability_data.append({
                    'confidence': confidence,
                    'accuracy': accuracy,
                    'count': count
                })
        
        return {
            'ece': ece,
            'reliability_data': reliability_data
        }


class CausalEvaluator:
    """Comprehensive evaluator for causal inference models."""
    
    def __init__(self, metrics: Optional[List[CausalMetric]] = None):
        if metrics is None:
            self.metrics = [
                PEHEMetric(),
                ATEMetric(),
                PolicyValueMetric(),
                ConfoundingRobustnessMetric(),
                UncertaintyCalibrationMetric()
            ]
        else:
            self.metrics = metrics
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 treatments: np.ndarray, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of causal model predictions.
        
        Args:
            y_true: Ground truth outcomes [batch_size, 1]
            y_pred: Predicted counterfactual outcomes [batch_size, num_treatments]
            treatments: Treatment assignments [batch_size]
            **kwargs: Additional arguments for specific metrics
        
        Returns:
            Dictionary with results from all metrics
        """
        results = {}
        
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.replace('Metric', '').lower()
            try:
                metric_results = metric.compute(y_true, y_pred, treatments, **kwargs)
                results[metric_name] = metric_results
                logger.info(f"Computed {metric_name} metrics")
            except Exception as e:
                logger.error(f"Error computing {metric_name} metrics: {e}")
                results[metric_name] = {'error': str(e)}
        
        return results
    
    def create_evaluation_report(self, results: Dict[str, Dict[str, float]], 
                                save_path: Optional[str] = None) -> str:
        """Create a formatted evaluation report."""
        report = "=== Causal Inference Evaluation Report ===\n\n"
        
        for metric_name, metric_results in results.items():
            if 'error' in metric_results:
                report += f"{metric_name.upper()}:\n"
                report += f"  Error: {metric_results['error']}\n\n"
                continue
            
            report += f"{metric_name.upper()}:\n"
            
            if metric_name == 'pehe':
                report += f"  PEHE: {metric_results.get('pehe', 'N/A'):.4f}\n"
                report += f"  PEHE (squared): {metric_results.get('pehe_squared', 'N/A'):.4f}\n"
                
                # Per-treatment PEHE
                for key, value in metric_results.items():
                    if key.startswith('pehe_treatment_'):
                        treatment_id = key.split('_')[-1]
                        report += f"  PEHE Treatment {treatment_id}: {value:.4f}\n"
            
            elif metric_name == 'ate':
                pred_ate = metric_results.get('pred_ate', [])
                observed_ate = metric_results.get('observed_ate', [])
                
                report += f"  Predicted ATE: {pred_ate}\n"
                report += f"  Observed ATE: {observed_ate}\n"
                
                if 'ate_mse' in metric_results:
                    report += f"  ATE MSE: {metric_results['ate_mse']:.4f}\n"
                    report += f"  ATE MAE: {metric_results['ate_mae']:.4f}\n"
            
            elif metric_name == 'policyvalue':
                report += f"  Policy Value: {metric_results.get('policy_value', 'N/A'):.4f}\n"
                report += f"  Policy Improvement: {metric_results.get('policy_improvement', 'N/A'):.4f}\n"
                report += f"  Treatment Distribution: {metric_results.get('treatment_distribution', [])}\n"
            
            elif metric_name == 'confoundingrobustness':
                report += f"  Treatment Balance: {metric_results.get('treatment_balance', 'N/A'):.4f}\n"
                report += f"  Outcome Variance Ratio: {metric_results.get('outcome_variance_ratio', 'N/A'):.4f}\n"
                
                if 'covariate_imbalance' in metric_results:
                    report += f"  Covariate Imbalance: {metric_results['covariate_imbalance']:.4f}\n"
            
            elif metric_name == 'uncertaintycalibration':
                report += f"  Expected Calibration Error: {metric_results.get('ece', 'N/A'):.4f}\n"
            
            report += "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def plot_evaluation_results(self, results: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None):
        """Create visualizations of evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Causal Inference Evaluation Results', fontsize=16)
        
        # PEHE plot
        if 'pehe' in results and 'pehe' in results['pehe']:
            ax = axes[0, 0]
            pehe_value = results['pehe']['pehe']
            ax.bar(['PEHE'], [pehe_value], color='skyblue')
            ax.set_ylabel('PEHE')
            ax.set_title('Precision in Estimation of Heterogeneous Effect')
            ax.grid(True, alpha=0.3)
        
        # ATE comparison plot
        if 'ate' in results:
            ax = axes[0, 1]
            ate_data = results['ate']
            
            if 'pred_ate' in ate_data and 'observed_ate' in ate_data:
                treatments = range(len(ate_data['pred_ate']))
                width = 0.35
                
                ax.bar([t - width/2 for t in treatments], ate_data['pred_ate'], 
                      width, label='Predicted ATE', alpha=0.7)
                ax.bar([t + width/2 for t in treatments], ate_data['observed_ate'], 
                      width, label='Observed ATE', alpha=0.7)
                
                ax.set_xlabel('Treatment')
                ax.set_ylabel('Average Treatment Effect')
                ax.set_title('ATE Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Policy value plot
        if 'policyvalue' in results:
            ax = axes[1, 0]
            policy_data = results['policyvalue']
            
            values = [
                policy_data.get('current_policy_value', 0),
                policy_data.get('policy_value', 0),
                policy_data.get('random_policy_value', 0)
            ]
            labels = ['Current Policy', 'Recommended Policy', 'Random Policy']
            colors = ['orange', 'green', 'red']
            
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_ylabel('Policy Value')
            ax.set_title('Policy Value Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Calibration plot
        if 'uncertaintycalibration' in results and 'reliability_data' in results['uncertaintycalibration']:
            ax = axes[1, 1]
            reliability_data = results['uncertaintycalibration']['reliability_data']
            
            if reliability_data:
                confidences = [d['confidence'] for d in reliability_data]
                accuracies = [d['accuracy'] for d in reliability_data]
                
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                ax.scatter(confidences, accuracies, s=50, alpha=0.7, label='Observed')
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Accuracy')
                ax.set_title('Reliability Diagram')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        return fig 