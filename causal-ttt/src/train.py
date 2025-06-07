"""
Training script for the TTT-Neural CDE Model for causal time series forecasting.
"""
import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import the TTT-CDE model
from models.ttt_cde_model import TTTNeuralCDE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


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


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=100,
    aux_loss_weight=0.1,
    checkpoint_dir='checkpoints',
    use_ttt_during_training=False,
):
    """
    Train the TTT-Neural CDE model.
    
    Args:
        model: TTTNeuralCDE model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epochs: Number of epochs to train for
        aux_loss_weight: Weight for the auxiliary task loss
        checkpoint_dir: Directory to save checkpoints
        use_ttt_during_training: Whether to use test-time training during training
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_aux_loss = 0.0
        
        for batch_idx, (X, y, treatment) in enumerate(train_loader):
            X, y, treatment = X.to(device), y.to(device), treatment.to(device)
            
            # Create time points (assuming equally spaced)
            batch_size, seq_len, _ = X.shape
            time_points = torch.linspace(0, 1, seq_len).repeat(batch_size, 1).to(device)
            
            # Prepare data for CDE
            coeffs_x = prepare_cde_data(X, time_points, device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_y, pred_a_softmax, pred_a, z_hat = model(coeffs_x, device)
            
            # Compute main loss (outcome prediction)
            main_loss = criterion(pred_y, y)
            
            # Add treatment prediction loss (cross-entropy)
            treatment_loss = nn.CrossEntropyLoss()(pred_a, treatment.argmax(dim=1))
            
            # Compute auxiliary task loss for TTT
            aux_loss = model.auxiliary_task(z_hat, coeffs_x)
            
            # Combine losses
            loss = main_loss + treatment_loss + aux_loss_weight * aux_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_aux_loss += aux_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X, y, treatment in val_loader:
                X, y, treatment = X.to(device), y.to(device), treatment.to(device)
                
                # Create time points
                batch_size, seq_len, _ = X.shape
                time_points = torch.linspace(0, 1, seq_len).repeat(batch_size, 1).to(device)
                
                # Prepare data for CDE
                coeffs_x = prepare_cde_data(X, time_points, device)
                
                # Forward pass (with TTT if enabled)
                if use_ttt_during_training:
                    pred_y, _, _, _ = model.ttt_forward(coeffs_x, device, adapt=True)
                else:
                    pred_y, _, _, _ = model(coeffs_x, device)
                
                # Compute loss
                loss = criterion(pred_y, y)
                val_loss += loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        train_aux_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Train Aux Loss: {train_aux_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            logger.info(f"Saved checkpoint at epoch {epoch+1} with val_loss {val_loss:.4f}")
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt'))


def main():
    parser = argparse.ArgumentParser(description='Train TTT-Neural CDE Model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='directory containing the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='hidden dimension size')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='output dimension size')
    parser.add_argument('--ttt_lr', type=float, default=0.001,
                        help='learning rate for test-time training')
    parser.add_argument('--ttt_steps', type=int, default=5,
                        help='number of adaptation steps in TTT')
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                        help='weight for auxiliary loss')
    parser.add_argument('--interpolation', type=str, default='linear',
                        choices=['linear', 'cubic'],
                        help='interpolation type for CDE')
    parser.add_argument('--use_ttt_during_training', action='store_true',
                        help='whether to use TTT during training')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data (placeholder - you'll need to adapt this to your actual data loading)
    # This is just a sketch of how the data loading might look
    # X_train: shape (num_samples, seq_len, features)
    # y_train: shape (num_samples, output_dim)
    # treatment_train: shape (num_samples, num_treatments) - one-hot encoded
    X_train = torch.randn(500, 24, 10)  # Example dimensions
    y_train = torch.randn(500, args.output_dim)
    treatment_train = torch.zeros(500, 4)
    treatment_train[:, 0] = 1  # Example: all samples get treatment 0
    
    X_val = torch.randn(100, 24, 10)
    y_val = torch.randn(100, args.output_dim)
    treatment_val = torch.zeros(100, 4)
    treatment_val[:, 0] = 1
    
    train_dataset = TensorDataset(X_train, y_train, treatment_train)
    val_dataset = TensorDataset(X_val, y_val, treatment_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    input_channels = X_train.shape[2]  # Number of features
    model = TTTNeuralCDE(
        input_channels_x=input_channels,
        hidden_channels_x=args.hidden_dim,
        output_channels=args.output_dim,
        interpolation=args.interpolation,
        ttt_lr=args.ttt_lr,
        ttt_steps=args.ttt_steps,
        ttt_loss_weight=args.aux_loss_weight,
    ).to(device)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Train the model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        aux_loss_weight=args.aux_loss_weight,
        checkpoint_dir=args.checkpoint_dir,
        use_ttt_during_training=args.use_ttt_during_training,
    )


if __name__ == "__main__":
    main()
