"""
Inference script for the TTT-Neural CDE Model demonstrating counterfactual predictions
with test-time adaptation.
"""
import argparse
import logging
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
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


def generate_counterfactuals(
    model,
    test_data,
    device,
    use_ttt=True,
    save_plots=True,
    output_dir='results',
):
    """
    Generate and visualize counterfactual predictions.
    
    Args:
        model: Trained TTTNeuralCDE model
        test_data: Test data (X, y, treatment)
        device: Device to run on
        use_ttt: Whether to use test-time training
        save_plots: Whether to save visualization plots
        output_dir: Directory to save results
    """
    model.eval()
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    X, y, treatment = test_data
    X, y, treatment = X.to(device), y.to(device), treatment.to(device)
    
    # Create time points (assuming equally spaced)
    batch_size, seq_len, _ = X.shape
    time_points = torch.linspace(0, 1, seq_len).repeat(batch_size, 1).to(device)
    
    # Prepare data for CDE
    coeffs_x = prepare_cde_data(X, time_points, device)
    
    # Get factual predictions
    if use_ttt:
        logger.info("Generating predictions with test-time adaptation...")
        pred_y, _, _, _ = model.ttt_forward(coeffs_x, device, adapt=True)
    else:
        logger.info("Generating predictions without test-time adaptation...")
        pred_y, _, _, _ = model(coeffs_x, device)
    
    # Generate counterfactuals for all treatment options
    counterfactuals = model.counterfactual_prediction(coeffs_x, device, adapt=use_ttt)
    
    # Convert predictions to numpy for easier handling
    pred_y_np = pred_y.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    cf_results = {}
    for treatment_id, cf_pred in counterfactuals.items():
        cf_results[treatment_id] = cf_pred.detach().cpu().numpy()
    
    # Calculate metrics
    mse = ((pred_y_np - y_np) ** 2).mean()
    logger.info(f"Mean Squared Error on test data: {mse:.4f}")
    
    # Visualize results for a few examples
    num_examples = min(5, batch_size)
    for i in range(num_examples):
        plt.figure(figsize=(12, 6))
        
        # Plot factual outcome
        plt.plot(range(y_np.shape[1]), y_np[i], 'k-', label='Ground Truth', linewidth=2)
        plt.plot(range(pred_y_np.shape[1]), pred_y_np[i], 'b--', label='Factual Prediction')
        
        # Plot counterfactuals
        colors = ['r-', 'g-', 'y-', 'm-']
        for j, (treatment_id, cf_pred) in enumerate(cf_results.items()):
            plt.plot(range(cf_pred.shape[1]), cf_pred[i], colors[j], label=f'Counterfactual {treatment_id}')
        
        plt.title(f'Example {i+1}: Factual vs. Counterfactual Predictions{"with TTT" if use_ttt else ""}')
        plt.xlabel('Time')
        plt.ylabel('Outcome')
        plt.legend()
        plt.grid(True)
        
        if save_plots:
            plt.savefig(os.path.join(output_dir, f'counterfactual_example_{i+1}_ttt_{use_ttt}.png'))
            plt.close()
        else:
            plt.show()
    
    return {
        'factual': pred_y_np,
        'ground_truth': y_np,
        'counterfactuals': cf_results,
        'mse': mse,
    }


def ablation_study(
    model,
    test_data,
    device,
    output_dir='results',
):
    """
    Perform an ablation study comparing performance with and without TTT.
    
    Args:
        model: Trained TTTNeuralCDE model
        test_data: Test data (X, y, treatment)
        device: Device to run on
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions with and without TTT
    results_with_ttt = generate_counterfactuals(
        model, test_data, device,
        use_ttt=True, save_plots=True, output_dir=output_dir,
    )
    
    results_without_ttt = generate_counterfactuals(
        model, test_data, device,
        use_ttt=False, save_plots=True, output_dir=output_dir,
    )
    
    # Compare MSE
    mse_with_ttt = results_with_ttt['mse']
    mse_without_ttt = results_without_ttt['mse']
    
    logger.info(f"MSE with TTT: {mse_with_ttt:.4f}")
    logger.info(f"MSE without TTT: {mse_without_ttt:.4f}")
    logger.info(f"Improvement with TTT: {(mse_without_ttt - mse_with_ttt) / mse_without_ttt * 100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['With TTT', 'Without TTT'], [mse_with_ttt, mse_without_ttt])
    plt.title('MSE Comparison: With vs. Without Test-Time Training')
    plt.ylabel('Mean Squared Error')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, 'ttt_comparison.png'))
    
    return {
        'mse_with_ttt': mse_with_ttt,
        'mse_without_ttt': mse_without_ttt,
        'improvement': (mse_without_ttt - mse_with_ttt) / mse_without_ttt * 100,
    }


def main():
    parser = argparse.ArgumentParser(description='Inference with TTT-Neural CDE Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='directory to save results')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='hidden dimension size (must match trained model)')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='output dimension size (must match trained model)')
    parser.add_argument('--interpolation', type=str, default='linear',
                        choices=['linear', 'cubic'],
                        help='interpolation type for CDE')
    parser.add_argument('--ttt_lr', type=float, default=0.001,
                        help='learning rate for test-time training')
    parser.add_argument('--ttt_steps', type=int, default=5,
                        help='number of adaptation steps in TTT')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for evaluation')
    parser.add_argument('--run_ablation', action='store_true',
                        help='run ablation study comparing with/without TTT')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data (placeholder - you'll need to adapt this to your actual data loading)
    # X_test: shape (num_samples, seq_len, features)
    # y_test: shape (num_samples, output_dim)
    # treatment_test: shape (num_samples, num_treatments) - one-hot encoded
    X_test = torch.randn(20, 24, 10)  # Example dimensions
    y_test = torch.randn(20, args.output_dim)
    treatment_test = torch.zeros(20, 4)
    treatment_test[:, 0] = 1  # Example: all samples get treatment 0
    
    test_data = (X_test, y_test, treatment_test)
    
    # Create model with same hyperparameters as training
    input_channels = X_test.shape[2]  # Number of features
    model = TTTNeuralCDE(
        input_channels_x=input_channels,
        hidden_channels_x=args.hidden_dim,
        output_channels=args.output_dim,
        interpolation=args.interpolation,
        ttt_lr=args.ttt_lr,
        ttt_steps=args.ttt_steps,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {args.checkpoint}")
    
    if args.run_ablation:
        results = ablation_study(
            model=model,
            test_data=test_data,
            device=device,
            output_dir=args.output_dir,
        )
        logger.info(f"Completed ablation study. Results saved to {args.output_dir}")
    else:
        results = generate_counterfactuals(
            model=model,
            test_data=test_data,
            device=device,
            use_ttt=True,
            save_plots=True,
            output_dir=args.output_dir,
        )
        logger.info(f"Generated counterfactuals. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
