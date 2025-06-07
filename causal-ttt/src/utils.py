"""
Utility functions for data preprocessing and evaluation.
"""
import numpy as np
import torch
import torchcde


def prepare_cde_data(data, time_points, device):
    """
    Prepare data for CDE models by computing coefficients for interpolation.
    
    Args:
        data: Time series data (batch_size, time_steps, features)
        time_points: Time points for each observation (batch_size, time_steps)
        device: Device to use
        
    Returns:
        coeffs: Tensor of coefficients for interpolation
    """
    # Add time dimension to the data
    batch_size, seq_len, channels = data.shape
    data_with_time = torch.cat([time_points.unsqueeze(-1), data], dim=2)
    
    # Create coefficients for interpolation
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data_with_time)
    return coeffs.to(device)


def evaluate_counterfactual_prediction(true_outcomes, predicted_outcomes, metrics=None):
    """
    Evaluate counterfactual predictions using various metrics.
    
    Args:
        true_outcomes: Ground truth outcomes
        predicted_outcomes: Predicted outcomes
        metrics: List of metrics to compute ('mse', 'rmse', 'mae')
        
    Returns:
        Dictionary of computed metrics
    """
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae']
    
    results = {}
    
    # Convert to numpy if tensor
    if isinstance(true_outcomes, torch.Tensor):
        true_outcomes = true_outcomes.detach().cpu().numpy()
    if isinstance(predicted_outcomes, torch.Tensor):
        predicted_outcomes = predicted_outcomes.detach().cpu().numpy()
    
    # Compute metrics
    if 'mse' in metrics:
        results['mse'] = np.mean((predicted_outcomes - true_outcomes) ** 2)
    
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(np.mean((predicted_outcomes - true_outcomes) ** 2))
    
    if 'mae' in metrics:
        results['mae'] = np.mean(np.abs(predicted_outcomes - true_outcomes))
    
    return results


def compute_average_treatment_effect(counterfactuals, baseline_treatment=None):
    """
    Compute the Average Treatment Effect (ATE) from counterfactual predictions.
    
    Args:
        counterfactuals: Dictionary of counterfactual predictions for different treatments
        baseline_treatment: Index of treatment to use as baseline (if None, use treatment_0)
        
    Returns:
        Dictionary of ATE values for each treatment relative to baseline
    """
    # Default baseline is treatment_0
    baseline_key = f'treatment_{baseline_treatment}' if baseline_treatment is not None else 'treatment_0'
    baseline = counterfactuals[baseline_key]
    
    ate_results = {}
    
    for treatment_id, cf_pred in counterfactuals.items():
        if treatment_id != baseline_key:
            # ATE is the average difference between treatment and baseline
            ate = np.mean(cf_pred - baseline)
            ate_results[treatment_id] = ate
            
    return ate_results


def get_treatment_assignment_accuracy(true_treatments, predicted_treatment_logits):
    """
    Calculate accuracy of treatment assignment prediction.
    
    Args:
        true_treatments: True treatment assignments (one-hot encoded)
        predicted_treatment_logits: Predicted treatment logits
        
    Returns:
        Accuracy score
    """
    if isinstance(true_treatments, torch.Tensor):
        true_treatments = true_treatments.detach().cpu().numpy()
    
    if isinstance(predicted_treatment_logits, torch.Tensor):
        predicted_treatment_logits = predicted_treatment_logits.detach().cpu().numpy()
    
    # Get the indices of the highest values (predicted class)
    true_indices = np.argmax(true_treatments, axis=1)
    predicted_indices = np.argmax(predicted_treatment_logits, axis=1)
    
    # Calculate accuracy
    correct = (predicted_indices == true_indices).sum()
    total = len(true_indices)
    
    return correct / total
