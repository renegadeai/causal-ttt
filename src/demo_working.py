"""
Working demo script for TTT-Neural CDE model with synthetic data
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchcde
import logging

# Set up logging
class NoisyInterpolationFilter(logging.Filter):
    def filter(self, record):
        # Suppress messages from root logger containing 'Using linear interpolation'
        return not (record.name == 'root' and 'Using linear interpolation' in record.getMessage())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Get the root logger and add the filter to it
root_logger = logging.getLogger()
root_logger.addFilter(NoisyInterpolationFilter())

logger = logging.getLogger(__name__)

# Import our model
from models.ttt_cde_model import TTTNeuralCDE


def generate_synthetic_data(num_samples=100, seq_len=10, num_features=5, 
                           output_dim=1, treatment_types=4, seed=42):
    """Generate synthetic time series data with treatments and outcomes."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random time series
    X = torch.randn(num_samples, seq_len, num_features)
    
    # Normalize features to prevent numerical overflow
    X = X * 0.1  # Scale down the feature values

    # Generate random treatment assignments
    treatment_idx = torch.randint(0, treatment_types, (num_samples,))
    treatments_one_hot = torch.zeros(num_samples, treatment_types)
    treatments_one_hot.scatter_(1, treatment_idx.unsqueeze(1), 1)
    
    # Generate outcomes based on the time series and treatments
    # More controlled outcome generation
    treatment_effects_vals = torch.tensor([-0.5, 0.8, 1.5, -0.3]) # Define specific effects for each treatment type
    if treatment_types > len(treatment_effects_vals):
        # Extend if more treatment types than defined effects (e.g. with zeros or random)
        treatment_effects_vals = torch.cat((treatment_effects_vals, torch.zeros(treatment_types - len(treatment_effects_vals))))
    else:
        treatment_effects_vals = treatment_effects_vals[:treatment_types]

    # Amplify treatment effects to make them more pronounced
    treatment_effects_vals = treatment_effects_vals * 10.0
    logger.debug(f"Amplified treatment_effects_vals: {treatment_effects_vals}")

    # Use a more stable way to generate baseline outcomes
    # Consider using a subset of features or a simpler combination
    baseline_y = torch.mean(X[:, -1, :min(num_features, 4)], dim=1) # Use mean of last time step of first few features
    noise = torch.randn(num_samples) * 0.05 # Smaller noise
    
    # Apply treatment effects
    y_treatment_effect = treatment_effects_vals[treatment_idx]
    y = baseline_y + y_treatment_effect + noise
    y = y.unsqueeze(1) # Ensure y has shape [num_samples, output_dim]

    # Ensure outcomes have reasonable scale
    y = y * 0.1  # Scale down outcomes
    
    return X, y, treatments_one_hot


def run_demo():
    """Run a demo of the TTT-Neural CDE model on synthetic data."""
    print("--- SCRIPT START ---")
    print(f"Logger level: {logger.level}, Root logger level: {root_logger.level}, Effective logger level: {logger.getEffectiveLevel()}")
    # Check for MPS (Apple Silicon) availability first, then CUDA, otherwise fall back to CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        # Set default tensor type to float32 for MPS compatibility
        torch.set_default_dtype(torch.float32)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directories
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate larger dataset
    logger.info("Generating synthetic data...")
    num_samples = 500  # Increased from 100
    seq_len = 20      # Doubled sequence length for better time series modeling
    num_features = 8   # More features for richer data
    output_dim = 1
    treatment_types = 4
    
    X, y, treatments = generate_synthetic_data(
        num_samples=num_samples, seq_len=seq_len, num_features=num_features,
        output_dim=output_dim, treatment_types=treatment_types
    )
    
    # Train/test split
    train_ratio = 0.8
    train_size = int(train_ratio * num_samples)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    treatments_train, treatments_test = treatments[:train_size], treatments[train_size:]
    
    # Add time dimension for CDE
    t = torch.linspace(0, 1, seq_len)
    X_train_with_time = torch.cat([
        t.unsqueeze(0).unsqueeze(-1).repeat(X_train.shape[0], 1, 1),
        X_train
    ], dim=2)
    X_test_with_time = torch.cat([
        t.unsqueeze(0).unsqueeze(-1).repeat(X_test.shape[0], 1, 1),
        X_test
    ], dim=2)
    
    # Create interpolation coefficients
    logger.info("Preparing CDE data...")
    # Using linear interpolation for simplicity
    coeffs_train = torchcde.linear_interpolation_coeffs(X_train_with_time).to(device)
    coeffs_test = torchcde.linear_interpolation_coeffs(X_test_with_time).to(device)
    
    # Create model with enhanced architecture
    logger.info("Creating the TTT-Neural CDE model...")
    model = TTTNeuralCDE(
        input_channels_x=num_features,  # Number of features excluding time
        hidden_channels=64,            # Increased hidden channels
        output_channels=output_dim,
        num_treatments=treatment_types,
        dropout_rate=0.2,              # Slightly higher dropout for regularization
        interpolation_method='linear',
        ttt_steps=15,                  # Adjusted TTT steps
        ttt_lr=0.01,                   # Higher learning rate for adaptation
        ttt_loss_weight=0.2,           # Increased weight for auxiliary task
        include_treatment_in_aux=False,  # Simplified aux task does not include treatment
        use_attention=True,            # Use attention mechanism
        ttt_early_stopping_patience=5, # Early stopping for TTT
        ttt_lr_decay=0.9,             # Learning rate decay for TTT
        cf_strength=15.0              # Amplify counterfactual differences (increased)
    ).to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) # Reverted LR, Added weight decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5) # Adjusted LR scheduler
    mse_criterion = torch.nn.MSELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()
    alpha_treatment_loss_weight = 5.0 # Weight for treatment prediction loss
     
    # Training loop
    logger.info("Beginning training...")
    num_epochs = 10  # Increased from 20 for more thorough training
    batch_size = 32   # Larger batch size for more stable gradients
    train_losses = []
    
    for epoch in range(num_epochs):
        # Process in batches
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle indices
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            # Get batch indices
            batch_indices = indices[i:i+batch_size]
            
            # Get batch data
            batch_coeffs = coeffs_train[batch_indices]
            batch_y = y_train[batch_indices].to(device)
            batch_treatments = treatments_train[batch_indices].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_y, pred_a_softmax, pred_a, z_hat = model(batch_coeffs, device)
            
            # Compute losses
            outcome_loss = mse_criterion(pred_y, batch_y)
            treatment_indices = torch.argmax(batch_treatments, dim=1)
            treatment_loss = ce_criterion(pred_a, treatment_indices)
            
            # Compute auxiliary loss for TTT
            auxiliary_loss = model.auxiliary_task(z_hat, batch_coeffs)
            
            # Total loss
            loss = outcome_loss + (alpha_treatment_loss_weight * treatment_loss) + 0.1 * auxiliary_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        scheduler.step() # Step the scheduler
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(model_dir, 'ttt_cde_model.pt'))
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    
    # Evaluation
    logger.info("Running evaluation...")
    model.eval()
    
    # Move test data to device
    y_test = y_test.to(device)
    
    # Enhanced evaluation
    logger.info("Running enhanced evaluation...")
    model.eval()
    
    # Define multiple metrics
    def calculate_metrics(pred, target):
        # Calculate metrics as tensors first
        mse_tensor = mse_criterion(pred, target)
        mae_tensor = torch.mean(torch.abs(pred - target))
        rmse_tensor = torch.sqrt(mse_tensor)  # sqrt needs a tensor
        r2_tensor = 1 - torch.sum((target - pred)**2) / torch.sum((target - torch.mean(target))**2)
        
        # Convert to Python floats for return
        return {
            'mse': mse_tensor.item(),
            'mae': mae_tensor.item(),
            'rmse': rmse_tensor.item(),
            'r2': r2_tensor.item()
        }
    
    # Standard predictions (without TTT)
    with torch.no_grad():
        std_pred, std_treatment_probs, _, std_hidden = model.forward(coeffs_test, device)
    
    # Calculate standard metrics
    std_metrics = calculate_metrics(std_pred, y_test.to(device))
    logger.info(f"Standard prediction metrics:")
    logger.info(f"  MSE: {std_metrics['mse']:.4f}")
    logger.info(f"  MAE: {std_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {std_metrics['rmse']:.4f}")
    logger.info(f"  R²: {std_metrics['r2']:.4f}")
    
    # Calculate treatment accuracy - convert treatments_test to indices if it's one-hot encoded
    _, pred_treatments = torch.max(std_treatment_probs, dim=1)
    
    # Check if treatments_test is one-hot encoded
    if len(treatments_test.shape) > 1 and treatments_test.shape[1] > 1:
        # Convert from one-hot to indices
        actual_treatments = torch.argmax(treatments_test, dim=1).to(device)
    else:
        # Already indices
        actual_treatments = treatments_test.to(device)
        
    treatment_accuracy = (pred_treatments == actual_treatments).float().mean().item()
    logger.info(f"Treatment prediction accuracy: {treatment_accuracy:.4f}")
    
    # TTT predictions with adaptation
    ttt_pred, ttt_treatment_probs, _, ttt_hidden = model.ttt_forward(coeffs_test, device, adapt=True)
    
    # Calculate TTT metrics
    ttt_metrics = calculate_metrics(ttt_pred, y_test.to(device))
    logger.info(f"TTT prediction metrics:")
    logger.info(f"  MSE: {ttt_metrics['mse']:.4f}")
    logger.info(f"  MAE: {ttt_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {ttt_metrics['rmse']:.4f}")
    logger.info(f"  R²: {ttt_metrics['r2']:.4f}")
    
    # Calculate improvement for each metric
    for metric in ['mse', 'mae', 'rmse']:
        improvement = (std_metrics[metric] - ttt_metrics[metric]) / std_metrics[metric] * 100
        logger.info(f"Improvement in {metric.upper()}: {improvement:.2f}%")
    
    # R² improvement is different since higher is better
    r2_improvement = (ttt_metrics['r2'] - std_metrics['r2']) / abs(std_metrics['r2']) * 100
    logger.info(f"Improvement in R²: {r2_improvement:.2f}%")
    
    # Compare representations before and after adaptation
    rep_diff = torch.mean(torch.norm(ttt_hidden - std_hidden, dim=1)).item()
    logger.info(f"Mean representation change after adaptation: {rep_diff:.4f}")

    # Calculate Average Treatment Effects (ATE)
    logger.info("Calculating Average Treatment Effects (ATE)...")
    all_cf_outcomes_no_ttt = {i: [] for i in range(treatment_types)}
    all_cf_outcomes_ttt = {i: [] for i in range(treatment_types)}

    model.eval() # Ensure model is in eval mode for counterfactuals
    with torch.no_grad():
        for i in range(len(coeffs_test)):
            coeffs_sample = coeffs_test[i:i+1].to(device)
            
            # Counterfactuals without TTT adaptation
            cf_outcomes_sample_no_ttt, _ = model.counterfactual_prediction(coeffs_sample, adapt=False, device=device)
            for t_idx in range(treatment_types):
                all_cf_outcomes_no_ttt[t_idx].append(cf_outcomes_sample_no_ttt[t_idx].item())
            
            # Counterfactuals with TTT adaptation
            # For TTT-adapted ATE, we need to run ttt_forward first for the specific sample to get adapted params
            # This is simplified here by using the model's state *after* batch TTT adaptation on the whole test set.
            # A more rigorous sample-specific TTT ATE would re-adapt for each sample, which is computationally intensive.
            # Here, we use the already adapted model state from ttt_forward(coeffs_test, ...)
            # and apply counterfactual_prediction. This assumes the adaptation is somewhat general.
            # For a more sample-specific approach, one would call model.ttt_forward(coeffs_sample, adapt=True) 
            # then model.counterfactual_prediction(coeffs_sample, adapt=False) (as adapt=True in cf_pred re-runs ttt_forward)
            # The current model.counterfactual_prediction(adapt=True) re-runs TTT for that sample.
            cf_outcomes_sample_ttt, _ = model.counterfactual_prediction(coeffs_sample, adapt=True, device=device)
            for t_idx in range(treatment_types):
                all_cf_outcomes_ttt[t_idx].append(cf_outcomes_sample_ttt[t_idx].item())

    for t_idx in range(treatment_types):
        all_cf_outcomes_no_ttt[t_idx] = np.array(all_cf_outcomes_no_ttt[t_idx])
        all_cf_outcomes_ttt[t_idx] = np.array(all_cf_outcomes_ttt[t_idx])

    logger.info("Average Treatment Effects (ATE) - No TTT (vs Treatment 0):")
    for t_idx in range(1, treatment_types):
        ate_no_ttt = np.mean(all_cf_outcomes_no_ttt[t_idx] - all_cf_outcomes_no_ttt[0])
        logger.info(f"  ATE for Treatment {t_idx} vs Treatment 0: {ate_no_ttt:.4f}")

    logger.info("Average Treatment Effects (ATE) - With TTT (vs Treatment 0):")
    for t_idx in range(1, treatment_types):
        ate_ttt = np.mean(all_cf_outcomes_ttt[t_idx] - all_cf_outcomes_ttt[0])
        logger.info(f"  ATE for Treatment {t_idx} vs Treatment 0: {ate_ttt:.4f}")

    # Save results for visualization later
    results = {
        'standard': {
            'pred': std_pred.detach().cpu().numpy(),
            'metrics': std_metrics
        },
        'ttt': {
            'pred': ttt_pred.detach().cpu().numpy(),
            'metrics': ttt_metrics
        },
        'true': y_test.cpu().numpy()
    }
    np.save(os.path.join(output_dir, 'prediction_results.npy'), results)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # Get Axes objects
    
    # Show only first 20 samples for clarity
    plot_size = min(20, len(y_test))
    
    # Standard Prediction Plot
    ax1.scatter(y_test[:plot_size].cpu().numpy(), 
                std_pred[:plot_size].detach().cpu().numpy(), alpha=0.7)
    ax1.plot([y_test.min().item(), y_test.max().item()], 
             [y_test.min().item(), y_test.max().item()], 'r--')
    ax1.set_title('Standard Prediction')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    # Add metrics text for standard prediction
    std_metrics_text = (f"R²: {std_metrics['r2']:.2f}\n"
                        f"MAE: {std_metrics['mae']:.2f}\n"
                        f"MSE: {std_metrics['mse']:.2f}")
    ax1.text(0.05, 0.95, std_metrics_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    
    # TTT Prediction Plot
    ax2.scatter(y_test[:plot_size].cpu().numpy(), 
                ttt_pred[:plot_size].detach().cpu().numpy(), alpha=0.7)
    ax2.plot([y_test.min().item(), y_test.max().item()], 
             [y_test.min().item(), y_test.max().item()], 'r--')
    ax2.set_title('TTT Prediction')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    # Add metrics text for TTT prediction
    ttt_metrics_text = (f"R²: {ttt_metrics['r2']:.2f}\n"
                        f"MAE: {ttt_metrics['mae']:.2f}\n"
                        f"MSE: {ttt_metrics['mse']:.2f}")
    ax2.text(0.05, 0.95, ttt_metrics_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'))
    
    # Counterfactual predictions for a few samples
    logger.info("Generating counterfactual predictions for a few samples (with and without TTT adaptation)...")
    num_plot_samples = min(5, X_test.shape[0]) # Plot for at most 5 samples
    
    # Select subset of data for plotting
    # Ensure coeffs_test_subset is correctly sliced whether it's a tensor or dict of tensors
    if isinstance(coeffs_test, dict):
        coeffs_test_subset_for_plot = {k: v[:num_plot_samples].clone().to(device) for k, v in coeffs_test.items()} 
    else:
        coeffs_test_subset_for_plot = coeffs_test[:num_plot_samples].clone().to(device)
    
    y_test_subset_for_plot = y_test[:num_plot_samples].clone().to(device)
    treatments_test_subset_for_plot = treatments_test[:num_plot_samples].clone().to(device)

    # Generate counterfactual predictions with TTT adaptation (adapt=True)
    logger.info("Generating counterfactuals with adapt=True (TTT)")
    cf_predictions_ttt, z_hat_for_cf_ttt = model.counterfactual_prediction(coeffs_test_subset_for_plot, device, adapt=True)
    
    # Generate counterfactual predictions without TTT adaptation (adapt=False)
    logger.info("Generating counterfactuals with adapt=False (No TTT)")
    # Ensure model is in eval mode and gradients are not computed for this pass if not already handled
    with torch.no_grad():
        # Temporarily store original parameters if model was adapted by ttt_forward in cf_predictions_ttt
        # The counterfactual_prediction method itself handles parameter restoration if adapt=True was used internally.
        # For adapt=False, it uses self.forward which doesn't change params.
        cf_predictions_no_ttt, z_hat_for_cf_no_ttt = model.counterfactual_prediction(coeffs_test_subset_for_plot, device, adapt=False)

    # Log z_hat statistics for debugging counterfactuals
    logger.info(f"z_hat_for_cf_ttt shape: {z_hat_for_cf_ttt.shape}, mean: {z_hat_for_cf_ttt.mean().item():.4f}, std: {z_hat_for_cf_ttt.std().item():.4f}")
    logger.info(f"z_hat_for_cf_no_ttt shape: {z_hat_for_cf_no_ttt.shape}, mean: {z_hat_for_cf_no_ttt.mean().item():.4f}, std: {z_hat_for_cf_no_ttt.std().item():.4f}")
    z_hat_diff_norm = torch.norm(z_hat_for_cf_ttt - z_hat_for_cf_no_ttt, p=2).item()
    logger.info(f"Norm of difference between z_hat_for_cf_ttt and z_hat_for_cf_no_ttt: {z_hat_diff_norm:.4f}")

    # Plot counterfactuals
    def plot_counterfactuals(coeffs_test_subset, cf_predictions_ttt, cf_predictions_no_ttt, 
                             y_test_subset, treatments_test_subset, output_dir, num_plot_samples):
        fig, axes = plt.subplots(num_plot_samples, 2, figsize=(20, 4 * num_plot_samples), squeeze=False)
        # squeeze=False ensures axes is always a 2D array, even if num_plot_samples is 1
        
        for i in range(num_plot_samples):
            # Plot for adapt=True (TTT)
            ax_ttt = axes[i, 0]
            sample_y_true = y_test_subset[i].item()
            sample_treatment_true_idx = torch.argmax(treatments_test_subset[i]).item()

            sample_cf_outcomes_ttt = {tid: outcomes[i].item() for tid, outcomes in cf_predictions_ttt.items()}
            treatments_ttt = sorted(sample_cf_outcomes_ttt.keys())
            outcomes_ttt = [sample_cf_outcomes_ttt[tid] for tid in treatments_ttt]

            bars_ttt = ax_ttt.bar([f"T{t}" for t in treatments_ttt], outcomes_ttt, color='skyblue')
            for bar in bars_ttt:
                yval = bar.get_height()
                ax_ttt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * np.sign(yval) if yval != 0 else 0.01, f'{yval:.2f}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)
            ax_ttt.axhline(sample_y_true, color='r', linestyle='--', label=f"Actual (T{sample_treatment_true_idx}): {sample_y_true:.2f}")
            ax_ttt.set_title(f"Sample {i+1} - TTT (adapt=True)\n(Factual T={sample_treatment_true_idx})")
            ax_ttt.set_ylabel("Predicted Outcome")
            ax_ttt.set_xlabel("Treatment Type")
            ax_ttt.legend()
            ax_ttt.grid(axis='y', linestyle='--', alpha=0.7)

            # Plot for adapt=False (No TTT)
            ax_no_ttt = axes[i, 1]
            sample_cf_outcomes_no_ttt = {tid: outcomes[i].item() for tid, outcomes in cf_predictions_no_ttt.items()}
            treatments_no_ttt = sorted(sample_cf_outcomes_no_ttt.keys())
            outcomes_no_ttt = [sample_cf_outcomes_no_ttt[tid] for tid in treatments_no_ttt]

            bars_no_ttt = ax_no_ttt.bar([f"T{t}" for t in treatments_no_ttt], outcomes_no_ttt, color='lightcoral')
            for bar in bars_no_ttt:
                yval = bar.get_height()
                ax_no_ttt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * np.sign(yval) if yval != 0 else 0.01, f'{yval:.2f}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)
            ax_no_ttt.axhline(sample_y_true, color='r', linestyle='--', label=f"Actual (T{sample_treatment_true_idx}): {sample_y_true:.2f}")
            ax_no_ttt.set_title(f"Sample {i+1} - No TTT (adapt=False)\n(Factual T={sample_treatment_true_idx})")
            ax_no_ttt.set_ylabel("Predicted Outcome")
            ax_no_ttt.set_xlabel("Treatment Type")
            ax_no_ttt.legend()
            ax_no_ttt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'counterfactual_comparison.png'))
    
    plot_counterfactuals(coeffs_test_subset_for_plot, cf_predictions_ttt, cf_predictions_no_ttt, 
                         y_test_subset_for_plot, treatments_test_subset_for_plot, 
                         output_dir, num_plot_samples)
    logger.info(f"Counterfactual plots comparing TTT and No TTT saved to {output_dir}")
    
    logger.info(f"Demo completed! Results saved to {output_dir}")
    print("--- SCRIPT END ---")
    

if __name__ == "__main__":
    run_demo()
