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
        hidden_channels=64,             # Larger hidden dimension
        output_channels=output_dim,
        num_treatments=treatment_types,
        dropout_rate=0.2,              # Slightly higher dropout for regularization
        interpolation_method='linear',
        ttt_steps=20,                  # More steps for test-time training
        ttt_lr=0.01,                   # Higher learning rate for adaptation
        ttt_loss_weight=0.2,           # Increased weight for auxiliary task
        # Custom weights for auxiliary task components
        aux_task_weights={
            'reconstruction': 0.3,
            'forecasting': 0.4,        # Prioritize forecasting more
            'temporal': 0.2,
            'disentanglement': 0.1
        },
        use_attention=True,            # Use attention mechanism
        forecast_horizon=3,            # Multi-step forecasting in auxiliary task
        include_treatment_in_aux=True, # Include treatment in auxiliary task
        ttt_early_stopping_patience=5, # Early stopping for TTT
        ttt_lr_decay=0.9              # Learning rate decay for TTT
    ).to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mse_criterion = torch.nn.MSELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("Beginning training...")
    num_epochs = 50  # Increased from 20 for more thorough training
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
            loss = outcome_loss + 0.5 * treatment_loss + 0.1 * auxiliary_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
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
    plt.figure(figsize=(12, 5))
    
    # Show only first 20 samples for clarity
    plot_size = min(20, len(y_test))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:plot_size].cpu().numpy(), 
                std_pred[:plot_size].detach().cpu().numpy(), alpha=0.7)
    plt.plot([y_test.min().item(), y_test.max().item()], 
             [y_test.min().item(), y_test.max().item()], 'r--')
    plt.title('Standard Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:plot_size].cpu().numpy(), 
                ttt_pred[:plot_size].detach().cpu().numpy(), alpha=0.7)
    plt.plot([y_test.min().item(), y_test.max().item()], 
             [y_test.min().item(), y_test.max().item()], 'r--')
    plt.title('TTT Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'))
    
    # Counterfactual prediction
    logger.info("Generating counterfactual predictions...")
    sample_idx = 0
    coeffs_sample = coeffs_test[sample_idx:sample_idx+1]
    
    # Get counterfactuals with TTT
    counterfactuals = model.counterfactual_prediction(coeffs_sample, device, adapt=True)
    
    # Create bar chart of counterfactuals
    plt.figure(figsize=(10, 6))
    
    factual_value = y_test[sample_idx].item()
    plt.axhline(y=factual_value, color='k', linestyle='-', label='Factual Outcome')
    
    bars = []
    bar_labels = []
    
    for treatment_id, cf_value in counterfactuals.items():
        bars.append(cf_value.item())
        bar_labels.append(f"T{treatment_id}")
    
    plt.bar(range(len(bars)), bars, alpha=0.7, tick_label=bar_labels)
    plt.title('Counterfactual Outcomes')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'counterfactuals.png'))
    
    logger.info(f"Demo completed! Results saved to {output_dir}")
    

if __name__ == "__main__":
    run_demo()
