"""
Demo script to run the TTT-Neural CDE model with synthetic data.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchcde
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import our model and utilities
from models.ttt_cde_model import TTTNeuralCDE
from utils import evaluate_counterfactual_prediction


def prepare_cde_data(data, time_points, device):
    """
    Prepare data for CDE models by computing coefficients for interpolation.
    
    Args:
        data: Time series data (batch_size, time_steps, features)
        time_points: Time points for each observation
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


def generate_synthetic_data(num_samples=100, seq_len=24, num_features=5, 
                           output_dim=1, treatment_types=4, seed=42):
    """
    Generate synthetic time series data for causal inference.
    
    Args:
        num_samples: Number of samples to generate
        seq_len: Length of each time series
        num_features: Number of features per timestep
        output_dim: Dimension of the outcome
        treatment_types: Number of possible treatments
        seed: Random seed for reproducibility
        
    Returns:
        X: Input time series (num_samples, seq_len, num_features)
        y: Outcomes (num_samples, output_dim)
        treatments: Treatment assignments (num_samples, treatment_types)
        time_points: Time points (num_samples, seq_len)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random time series
    X = torch.randn(num_samples, seq_len, num_features)
    
    # Generate time points (evenly spaced between 0 and 1)
    time_points = torch.linspace(0, 1, seq_len).repeat(num_samples, 1)
    
    # Generate random treatment assignments (one-hot encoded)
    treatments = torch.zeros(num_samples, treatment_types)
    treatment_idx = torch.randint(0, treatment_types, (num_samples,))
    treatments.scatter_(1, treatment_idx.unsqueeze(1), 1)
    
    # Generate outcomes based on the time series and treatments
    # We'll make this a function of the time series, the treatment, and some random noise
    # This creates a causal relationship between treatment and outcome
    
    # Effect of each feature over time (varies by treatment)
    feature_effects = []
    for t in range(treatment_types):
        # Different treatment leads to different feature importance
        effect = torch.randn(num_features, output_dim) 
        feature_effects.append(effect)
    
    # Calculate treatment-specific outcomes
    y = torch.zeros(num_samples, output_dim)
    for i in range(num_samples):
        # Get the treatment index for this sample
        t_idx = treatment_idx[i].item()
        
        # Get the effect for this treatment
        effect = feature_effects[t_idx]
        
        # Calculate outcome based on last 5 timesteps (to create temporal dependency)
        # We use the last few timesteps to simulate an effect over time
        recent_values = X[i, -5:, :]
        y[i] = torch.mean(torch.matmul(recent_values, effect), dim=0) + torch.randn(output_dim) * 0.1
    
    # Scale the outcome to be more realistic
    y = y * 10.0
    
    return X, y, treatments, time_points


def run_demo():
    """Run a demo of the TTT-Neural CDE model on synthetic data."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    num_samples = 100
    seq_len = 24
    num_features = 5
    output_dim = 1
    treatment_types = 4
    
    X, y, treatments, time_points = generate_synthetic_data(
        num_samples=num_samples,
        seq_len=seq_len,
        num_features=num_features,
        output_dim=output_dim,
        treatment_types=treatment_types
    )
    
    # Split into train and test sets
    train_ratio = 0.8
    train_size = int(train_ratio * num_samples)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    treatments_train, treatments_test = treatments[:train_size], treatments[train_size:]
    time_points_train, time_points_test = time_points[:train_size], time_points[train_size:]
    
    # Create model
    logger.info("Creating the TTT-Neural CDE model...")
    hidden_dim = 32
    model = TTTNeuralCDE(
        input_channels_x=num_features,
        hidden_channels_x=hidden_dim,
        output_channels=output_dim,
        interpolation='linear',
        ttt_lr=0.001,
        ttt_steps=5,
        ttt_loss_weight=0.1,
    ).to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Create output directory for results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info("Beginning training...")
    num_epochs = 50
    batch_size = 16
    
    train_losses = []
    
    for epoch in range(num_epochs):
        # Shuffle the training data
        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        treatments_train = treatments_train[indices]
        time_points_train = time_points_train[indices]
        
        # Process in batches
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            # Get batch
            X_batch = X_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device)
            treatments_batch = treatments_train[i:i+batch_size].to(device)
            time_points_batch = time_points_train[i:i+batch_size].to(device)
            
            # Prepare data for CDE
            coeffs_x = prepare_cde_data(X_batch, time_points_batch, device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_y, pred_a_softmax, pred_a, z_hat = model(coeffs_x, device)
            
            # Compute main loss (outcome prediction)
            main_loss = criterion(pred_y, y_batch)
            
            # Add treatment prediction loss (cross-entropy)
            treatment_indices = torch.argmax(treatments_batch, dim=1)
            treatment_loss = torch.nn.CrossEntropyLoss()(pred_a, treatment_indices)
            
            # Compute auxiliary task loss for TTT
            aux_loss = model.auxiliary_task(z_hat, coeffs_x)
            
            # Combine losses
            loss = main_loss + treatment_loss + 0.1 * aux_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        epoch_loss /= num_batches
        train_losses.append(epoch_loss)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    
    # Save model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
    }, os.path.join(model_dir, 'demo_model.pt'))
    
    logger.info("Model training completed and saved!")
    
    # Evaluation
    logger.info("Running evaluation...")
    model.eval()
    
    # Evaluate on test set - compare with and without TTT
    # First, prepare all test data
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    treatments_test = treatments_test.to(device)
    time_points_test = time_points_test.to(device)
    coeffs_x_test = prepare_cde_data(X_test, time_points_test, device)
    
    # Run prediction with standard forward
    with torch.no_grad():
        pred_y_standard, _, _, _ = model(coeffs_x_test, device)
        standard_mse = torch.nn.MSELoss()(pred_y_standard, y_test).item()
    
    logger.info(f"Standard prediction MSE: {standard_mse:.4f}")
    
    # Run prediction with TTT forward
    pred_y_ttt, _, _, _ = model.ttt_forward(coeffs_x_test, device, adapt=True)
    ttt_mse = torch.nn.MSELoss()(pred_y_ttt, y_test).item()
    
    logger.info(f"TTT prediction MSE: {ttt_mse:.4f}")
    logger.info(f"Improvement with TTT: {(standard_mse - ttt_mse) / standard_mse * 100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Take first 10 samples for visualization
    sample_idx = min(10, len(y_test))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:sample_idx].cpu().numpy(), 
               pred_y_standard[:sample_idx].detach().cpu().numpy(), alpha=0.7)
    plt.plot([y_test.min().item(), y_test.max().item()], 
             [y_test.min().item(), y_test.max().item()], 'r--')
    plt.title('Standard Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:sample_idx].cpu().numpy(), 
               pred_y_ttt[:sample_idx].detach().cpu().numpy(), alpha=0.7)
    plt.plot([y_test.min().item(), y_test.max().item()], 
             [y_test.min().item(), y_test.max().item()], 'r--')
    plt.title('TTT Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'))
    
    # Generate counterfactuals for a sample case
    logger.info("Generating counterfactuals for a sample case...")
    sample_idx = 0
    
    X_sample = X_test[sample_idx:sample_idx+1]
    time_points_sample = time_points_test[sample_idx:sample_idx+1]
    coeffs_sample = prepare_cde_data(X_sample, time_points_sample, device)
    
    # Get counterfactuals with TTT
    counterfactuals = model.counterfactual_prediction(coeffs_sample, device, adapt=True)
    
    # Plot counterfactuals
    plt.figure(figsize=(10, 6))
    
    # Get the factual outcome
    factual_value = y_test[sample_idx].item()
    plt.axhline(y=factual_value, color='k', linestyle='-', label='Factual Outcome')
    
    # Plot each counterfactual
    for treatment_id, cf_value in counterfactuals.items():
        plt.bar(treatment_id, cf_value.item(), alpha=0.7)
    
    plt.title('Counterfactual Outcomes for Different Treatments')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'counterfactuals.png'))
    
    logger.info(f"Demo completed! Results saved to {output_dir}")


if __name__ == "__main__":
    run_demo()
