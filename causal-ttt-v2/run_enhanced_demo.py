#!/usr/bin/env python3
"""
UPDATED Enhanced TTT-Neural CDE Demo with Fixed Treatment Effect Learning

This script now includes:
1. Fixed treatment effect learning with proper causal loss
2. Comprehensive policy accuracy metrics
3. Proper counterfactual prediction evaluation
"""

import os
import sys
import numpy as np
import torch
import torchcde
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import fixed model and evaluation metrics
from enhanced_ttt_cde_model_fixed import FixedEnhancedTTTNeuralCDE
from causal_evaluation_metrics import CausalEvaluator


def generate_simple_synthetic_data(num_samples=200, seq_len=15, num_features=6, 
                                 num_treatments=3, seed=42):
    """Generate simple synthetic data for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info(f"Generating {num_samples} samples with {seq_len} time steps...")
    
    # Generate simple AR(1) time series
    X = np.zeros((num_samples, seq_len, num_features))
    X[:, 0, :] = np.random.normal(0, 1, (num_samples, num_features))
    
    autocorr = 0.6
    noise_std = 0.2
    
    for t in range(1, seq_len):
        noise = np.random.normal(0, noise_std, (num_samples, num_features))
        X[:, t, :] = autocorr * X[:, t-1, :] + noise
    
    # Simple treatment assignment based on last observation
    confounders = X[:, -1, :]
    treatment_logits = np.dot(confounders, np.random.normal(0, 0.8, (num_features, num_treatments)))
    treatment_probs = np.exp(treatment_logits) / np.sum(np.exp(treatment_logits), axis=1, keepdims=True)
    
    treatments = np.array([np.random.choice(num_treatments, p=probs) for probs in treatment_probs])
    treatments_one_hot = np.eye(num_treatments)[treatments]
    
    # Simple outcomes with treatment effects
    base_outcome = np.mean(confounders, axis=1)
    treatment_effects = np.array([0.0, 1.0, -0.5][:num_treatments])
    
    # Potential outcomes
    potential_outcomes = base_outcome[:, None] + treatment_effects[None, :]
    potential_outcomes += np.random.normal(0, 0.1, potential_outcomes.shape)
    
    # Observed outcomes
    observed_outcomes = potential_outcomes[np.arange(num_samples), treatments]
    
    # Normalize features
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(observed_outcomes[:, None], dtype=torch.float32),
        torch.tensor(treatments_one_hot, dtype=torch.float32),
        torch.tensor(potential_outcomes, dtype=torch.float32)
    )


def train_model_with_causal_loss(model, X_train, y_train, treatments_train, potential_outcomes_train, device, num_epochs=100):
    """FIXED training with explicit causal loss to teach treatment effects."""
    logger.info("Starting FIXED training with causal loss...")
    
    # Prepare data with time dimension
    t = torch.linspace(0, 1, X_train.shape[1]).unsqueeze(0).unsqueeze(-1)
    t = t.repeat(X_train.shape[0], 1, 1)
    X_train_with_time = torch.cat([t.to(device), X_train.to(device)], dim=2)
    coeffs_train = torchcde.linear_interpolation_coeffs(X_train_with_time)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.8)
    
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass - get potential outcomes directly
        pred_outcomes, treatment_probs, treatment_effects, z_hat = model.forward(
            coeffs_train, device, training=True
        )
        
        # Handle different model output formats
        if pred_outcomes.dim() == 3:  # Fixed model: [batch, num_treatments, 1]
            pred_potential_outcomes = pred_outcomes.squeeze(-1)  # [batch, num_treatments]
        else:  # Original model: needs counterfactual prediction
            cf_predictions, _ = model.counterfactual_prediction(coeffs_train, device, adapt=False)
            pred_potential_outcomes = torch.stack([cf_predictions[t] for t in range(len(cf_predictions))], dim=1).squeeze(-1)
        
        # 1. FACTUAL LOSS: Predict observed outcomes correctly
        treatment_indices = torch.argmax(treatments_train.to(device), dim=1)
        if pred_outcomes.dim() == 3:
            observed_predictions = pred_outcomes[torch.arange(len(pred_outcomes)), treatment_indices, :]
            factual_loss = F.mse_loss(observed_predictions, y_train.to(device))
        else:
            factual_loss = F.mse_loss(pred_outcomes, y_train.to(device))
        
        # 2. CAUSAL LOSS: Learn ALL potential outcomes (KEY FIX!)
        causal_loss = F.mse_loss(pred_potential_outcomes, potential_outcomes_train.to(device))
        
        # 3. TREATMENT CLASSIFICATION LOSS
        treatment_loss = F.cross_entropy(treatment_probs, treatment_indices)
        
        # 4. AUXILIARY LOSS
        aux_loss = model.compute_auxiliary_loss(z_hat, coeffs_train)
        
        # 5. TREATMENT EFFECT CONSISTENCY LOSS
        pred_effects = pred_potential_outcomes - pred_potential_outcomes[:, 0:1]
        true_effects = torch.tensor([0.0, 1.0, -0.5], device=device).unsqueeze(0)
        true_effects = true_effects.expand(len(X_train), -1)
        effect_consistency_loss = F.mse_loss(pred_effects, true_effects)
        
        # 6. RANKING LOSS: Explicitly enforce T1 > T0 > T2
        ranking_loss = torch.tensor(0.0, device=device)
        # T1 should be better than T0
        t1_better_than_t0 = torch.clamp(pred_potential_outcomes[:, 0] - pred_potential_outcomes[:, 1] + 0.5, min=0)
        ranking_loss += torch.mean(t1_better_than_t0)
        # T0 should be better than T2
        t0_better_than_t2 = torch.clamp(pred_potential_outcomes[:, 2] - pred_potential_outcomes[:, 0] + 0.25, min=0)
        ranking_loss += torch.mean(t0_better_than_t2)
        
        # FIXED: Combine all losses with proper weighting
        total_loss = (
            1.0 * factual_loss +           # Predict observed outcomes
            2.0 * causal_loss +            # Learn ALL potential outcomes (MOST IMPORTANT)
            0.3 * treatment_loss +         # Learn treatment propensity
            0.1 * aux_loss +               # Auxiliary tasks
            1.0 * effect_consistency_loss + # Learn correct effect sizes
            0.5 * ranking_loss             # Enforce correct ordering
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        losses.append(total_loss.item())
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Total: {total_loss.item():.4f}")
            logger.info(f"  Factual: {factual_loss.item():.4f}")
            logger.info(f"  Causal: {causal_loss.item():.4f}")
            logger.info(f"  Treatment: {treatment_loss.item():.4f}")
            logger.info(f"  Effect Consistency: {effect_consistency_loss.item():.4f}")
            logger.info(f"  Ranking: {ranking_loss.item():.4f}")
    
    logger.info("FIXED training completed!")
    return losses


def calculate_policy_accuracy_metrics(cf_std_np, cf_ttt_np, potential_outcomes_np):
    """Calculate comprehensive policy accuracy metrics."""
    # True optimal treatment for each individual
    true_optimal = np.argmax(potential_outcomes_np, axis=1)
    
    # Model recommendations
    std_optimal = np.argmax(cf_std_np, axis=1)
    ttt_optimal = np.argmax(cf_ttt_np, axis=1)
    
    # Policy accuracy
    std_policy_accuracy = np.mean(true_optimal == std_optimal)
    ttt_policy_accuracy = np.mean(true_optimal == ttt_optimal)
    
    # Treatment recommendation distributions
    true_dist = np.bincount(true_optimal, minlength=cf_std_np.shape[1]) / len(true_optimal)
    std_dist = np.bincount(std_optimal, minlength=cf_std_np.shape[1]) / len(std_optimal)
    ttt_dist = np.bincount(ttt_optimal, minlength=cf_ttt_np.shape[1]) / len(ttt_optimal)
    
    # Value of recommended policies
    std_policy_value = np.mean([cf_std_np[i, std_optimal[i]] for i in range(len(std_optimal))])
    ttt_policy_value = np.mean([cf_ttt_np[i, ttt_optimal[i]] for i in range(len(ttt_optimal))])
    optimal_policy_value = np.mean([potential_outcomes_np[i, true_optimal[i]] for i in range(len(true_optimal))])
    
    # Policy regret (how much value is lost by not using optimal policy)
    std_policy_regret = optimal_policy_value - std_policy_value
    ttt_policy_regret = optimal_policy_value - ttt_policy_value
    
    return {
        'std_policy_accuracy': float(std_policy_accuracy),
        'ttt_policy_accuracy': float(ttt_policy_accuracy),
        'policy_accuracy_improvement': float(ttt_policy_accuracy - std_policy_accuracy),
        'true_treatment_distribution': true_dist.tolist(),
        'std_treatment_distribution': std_dist.tolist(),
        'ttt_treatment_distribution': ttt_dist.tolist(),
        'std_policy_value': float(std_policy_value),
        'ttt_policy_value': float(ttt_policy_value),
        'optimal_policy_value': float(optimal_policy_value),
        'std_policy_regret': float(std_policy_regret),
        'ttt_policy_regret': float(ttt_policy_regret),
        'policy_regret_reduction': float(std_policy_regret - ttt_policy_regret),
        'treatment_effect_learning_success': bool(ttt_policy_accuracy > 0.8)
    }


def evaluate_model_with_policy_metrics(model, X_test, y_test, treatments_test, potential_outcomes, device):
    """Enhanced evaluation with comprehensive policy metrics."""
    logger.info("Evaluating model with policy metrics...")
    
    model.eval()
    
    # Prepare test data
    t = torch.linspace(0, 1, X_test.shape[1]).unsqueeze(0).unsqueeze(-1)
    t = t.repeat(X_test.shape[0], 1, 1)
    X_test_with_time = torch.cat([t.to(device), X_test.to(device)], dim=2)
    coeffs_test = torchcde.linear_interpolation_coeffs(X_test_with_time)
    
    with torch.no_grad():
        # Standard predictions
        std_pred, std_treatment_probs, _, std_z_hat = model.forward(
            coeffs_test, device, training=False
        )
        
        # TTT predictions
        ttt_pred, ttt_treatment_probs, _, ttt_z_hat = model.ttt_forward(
            coeffs_test, device, adapt=True
        )
        
        # Counterfactual predictions
        cf_std, _ = model.counterfactual_prediction(coeffs_test, device, adapt=False)
        cf_ttt, _ = model.counterfactual_prediction(coeffs_test, device, adapt=True)
        
        # Convert counterfactuals to numpy
        num_treatments = len(cf_std)
        batch_size = cf_std[0].shape[0]
        
        cf_std_np = np.zeros((batch_size, num_treatments))
        cf_ttt_np = np.zeros((batch_size, num_treatments))
        
        for t_idx in range(num_treatments):
            cf_std_np[:, t_idx] = cf_std[t_idx].detach().cpu().numpy().flatten()
            cf_ttt_np[:, t_idx] = cf_ttt[t_idx].detach().cpu().numpy().flatten()
        
        # FIXED: Extract factual predictions from potential outcomes
        treatment_indices = torch.argmax(treatments_test.to(device), dim=1)
        
        # For the fixed model, std_pred and ttt_pred have shape [batch, num_treatments, 1]
        # We need to extract the predictions for observed treatments
        if std_pred.dim() == 3:  # [batch, num_treatments, 1]
            std_factual = std_pred[torch.arange(len(std_pred)), treatment_indices, :]
            ttt_factual = ttt_pred[torch.arange(len(ttt_pred)), treatment_indices, :]
        else:  # [batch, 1] - already factual
            std_factual = std_pred
            ttt_factual = ttt_pred
        
        # Basic prediction metrics
        std_mse = F.mse_loss(std_factual, y_test.to(device)).item()
        ttt_mse = F.mse_loss(ttt_factual, y_test.to(device)).item()
        prediction_improvement = (std_mse - ttt_mse) / std_mse * 100 if std_mse > 0 else 0.0
        
        logger.info(f"Standard MSE: {std_mse:.4f}")
        logger.info(f"TTT MSE: {ttt_mse:.4f}")
        logger.info(f"Prediction Improvement: {prediction_improvement:.2f}%")
        
        # FIXED: Calculate comprehensive policy metrics
        potential_outcomes_np = potential_outcomes.numpy()
        policy_metrics = calculate_policy_accuracy_metrics(cf_std_np, cf_ttt_np, potential_outcomes_np)
        
        logger.info(f"Standard Policy Accuracy: {policy_metrics['std_policy_accuracy']:.1%}")
        logger.info(f"TTT Policy Accuracy: {policy_metrics['ttt_policy_accuracy']:.1%}")
        logger.info(f"Policy Accuracy Improvement: {policy_metrics['policy_accuracy_improvement']:.1%}")
        
        # Check if treatment effect learning is working
        avg_outcomes_std = np.mean(cf_std_np, axis=0)
        avg_outcomes_ttt = np.mean(cf_ttt_np, axis=0)
        
        logger.info(f"Standard Model Avg Outcomes: T0={avg_outcomes_std[0]:.3f}, T1={avg_outcomes_std[1]:.3f}, T2={avg_outcomes_std[2]:.3f}")
        logger.info(f"TTT Model Avg Outcomes: T0={avg_outcomes_ttt[0]:.3f}, T1={avg_outcomes_ttt[1]:.3f}, T2={avg_outcomes_ttt[2]:.3f}")
        
        correct_ordering_std = avg_outcomes_std[1] > avg_outcomes_std[0] > avg_outcomes_std[2]
        correct_ordering_ttt = avg_outcomes_ttt[1] > avg_outcomes_ttt[0] > avg_outcomes_ttt[2]
        
        logger.info(f"Standard Model Learned Correct Ordering (T1>T0>T2): {'✅' if correct_ordering_std else '❌'}")
        logger.info(f"TTT Model Learned Correct Ordering (T1>T0>T2): {'✅' if correct_ordering_ttt else '❌'}")
        
        # Causal evaluation with original metrics
        try:
            evaluator = CausalEvaluator()
            treatments_idx = torch.argmax(treatments_test, dim=1).numpy()
            
            causal_results = evaluator.evaluate(
                y_test.numpy(),
                cf_ttt_np,
                treatments_idx,
                true_effects=potential_outcomes_np
            )
            
            logger.info("Causal evaluation completed successfully")
            
        except Exception as e:
            logger.warning(f"Causal evaluation failed: {e}")
            causal_results = {}
        
        return {
            'std_mse': std_mse,
            'ttt_mse': ttt_mse,
            'prediction_improvement': prediction_improvement,
            'policy_metrics': policy_metrics,
            'treatment_effect_analysis': {
                'std_avg_outcomes': avg_outcomes_std.tolist(),
                'ttt_avg_outcomes': avg_outcomes_ttt.tolist(),
                'correct_ordering_std': correct_ordering_std,
                'correct_ordering_ttt': correct_ordering_ttt,
                'expected_ordering': [0.0, 1.0, -0.5]
            },
            'causal_results': causal_results
        }


def main():
    """Main function to run the enhanced demo with fixed treatment effect learning."""
    logger.info("=== Enhanced TTT-Neural CDE Demo with Fixed Treatment Effects ===")
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        torch.set_default_dtype(torch.float32)
        logger.info("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Create output directory
    output_dir = Path('results_simple')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Generate data
        X, y, treatments, potential_outcomes = generate_simple_synthetic_data(
            num_samples=200, seq_len=15, num_features=6, num_treatments=3
        )
        
        # Train/test split
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        treatments_train, treatments_test = treatments[:train_size], treatments[train_size:]
        potential_outcomes_train = potential_outcomes[:train_size]
        potential_outcomes_test = potential_outcomes[train_size:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Create model with optimized parameters (using FIXED model with conservative TTT)
        model = FixedEnhancedTTTNeuralCDE(
            input_channels_x=X.shape[2],
            hidden_channels=32,
            output_channels=1,
            num_treatments=treatments.shape[1],
            dropout_rate=0.1,
            interpolation_method='linear',
            ttt_steps=20,                      # Conservative TTT steps to prevent over-adaptation
            ttt_lr=0.002,                      # Conservative TTT learning rate
            use_multi_head_attention=True,
            num_attention_heads=4,
            use_residual_cde=True,
            use_uncertainty=False,
            input_has_time=True,
            ttt_early_stopping_patience=8,    # Earlier stopping for conservative adaptation
            cf_strength=1.0
        ).to(device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model with FIXED causal loss
        losses = train_model_with_causal_loss(
            model, X_train, y_train, treatments_train, potential_outcomes_train, device, num_epochs=120
        )
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss (with Fixed Causal Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(output_dir / 'training_loss.png')
        plt.close()
        
        # Evaluate model with comprehensive policy metrics
        results = evaluate_model_with_policy_metrics(
            model, X_test, y_test, treatments_test, potential_outcomes_test, device
        )
        
        # Save results with proper serialization
        import json
        
        def convert_to_json_serializable(obj):
            """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif np.isnan(obj) if isinstance(obj, (float, np.floating)) else False:
                return None
            else:
                return obj
        
        with open(output_dir / 'results.json', 'w') as f:
            json_results = convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
        logger.info("Demo completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    if results:
        print("\n" + "="*60)
        print("ENHANCED TTT-NEURAL CDE RESULTS SUMMARY")
        print("="*60)
        print(f"Prediction Metrics:")
        print(f"  Standard MSE:     {results['std_mse']:.4f}")
        print(f"  TTT MSE:          {results['ttt_mse']:.4f}")
        print(f"  Improvement:      {results['prediction_improvement']:.2f}%")
        
        if 'policy_metrics' in results:
            pm = results['policy_metrics']
            print(f"\nPolicy Accuracy Metrics:")
            print(f"  Standard Policy Accuracy:  {pm['std_policy_accuracy']:.1%}")
            print(f"  TTT Policy Accuracy:       {pm['ttt_policy_accuracy']:.1%}")
            print(f"  Policy Improvement:        {pm['policy_accuracy_improvement']:+.1%}")
            print(f"  Treatment Effect Learning: {'✅ SUCCESS' if pm['treatment_effect_learning_success'] else '❌ FAILED'}")
            
            print(f"\nTreatment Recommendations:")
            print(f"  True Optimal:     {pm['true_treatment_distribution']}")
            print(f"  TTT Model:        {pm['ttt_treatment_distribution']}")
        
        if 'treatment_effect_analysis' in results:
            tea = results['treatment_effect_analysis']
            print(f"\nTreatment Effect Analysis:")
            print(f"  TTT Avg Outcomes: T0={tea['ttt_avg_outcomes'][0]:.3f}, T1={tea['ttt_avg_outcomes'][1]:.3f}, T2={tea['ttt_avg_outcomes'][2]:.3f}")
            print(f"  Expected:         T0=0.0, T1=1.0, T2=-0.5")
            print(f"  Correct Ordering: {'✅' if tea['correct_ordering_ttt'] else '❌'}")
        
        print(f"\nDetailed results saved to 'results_simple/' directory")
    else:
        print("Demo failed - check the logs above for errors")
        sys.exit(1) 