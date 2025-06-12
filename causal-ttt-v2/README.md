# Enhanced TTT-Neural CDE Framework

An enhanced implementation of Test-Time Training (TTT) for causal time series forecasting using Neural Controlled Differential Equations (CDEs).

## Key Improvements Over Original Implementation

### 🚀 Model Architecture Enhancements
- **Multi-task Auxiliary Learning**: Multiple auxiliary tasks for better TTT adaptation
- **Enhanced CDE Function**: Residual connections and layer normalization
- **Multi-Head Attention**: Better temporal modeling with attention mechanisms
- **Treatment Effect Network**: Specialized network for heterogeneous treatment effects
- **Uncertainty Quantification**: Built-in uncertainty estimates

### 📊 Evaluation Improvements
- **Comprehensive Causal Metrics**: PEHE, ATE, Policy Value, Confounding Robustness
- **Uncertainty Calibration**: Expected Calibration Error (ECE) and reliability diagrams
- **Automated Reporting**: Detailed evaluation reports and visualizations

### 🔧 Data Generation Enhancements
- **Realistic Temporal Dynamics**: AR processes, seasonal patterns, trends
- **Confounding Mechanisms**: Multiple treatment assignment strategies
- **Distribution Shifts**: Built-in covariate and temporal shifts
- **Missing Data Patterns**: Various missingness mechanisms

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup
1. Navigate to the enhanced framework directory:
```bash
cd causal-ttt-v2
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install torchcde if not automatically installed:
```bash
pip install torchcde
```

## Quick Start

### Option 1: Simple Demo (Recommended for first run)
Run the simplified demo script:
```bash
python run_enhanced_demo.py
```

This will:
- Generate synthetic causal time series data
- Train the enhanced TTT-Neural CDE model
- Compare standard vs TTT-adapted predictions
- Output results to `results_simple/` directory

### Option 2: Full Featured Demo
For a comprehensive comparison with the original model:
```bash
python enhanced_demo.py
```

This includes:
- Comparison with baseline TTT-Neural CDE
- Comprehensive causal evaluation metrics
- Advanced plotting and reporting
- Results saved to `results_enhanced/` directory

## Usage Examples

### Basic Model Usage
```python
from enhanced_ttt_cde_model import EnhancedTTTNeuralCDE
import torch
import torchcde

# Create model
model = EnhancedTTTNeuralCDE(
    input_channels_x=8,      # Number of features
    hidden_channels=64,      # Hidden dimension
    output_channels=1,       # Output dimension
    num_treatments=4,        # Number of treatments
    ttt_steps=20,           # TTT adaptation steps
    use_multi_head_attention=True,
    use_uncertainty=True
)

# Prepare data (with time dimension)
X_with_time = torch.cat([time_tensor, features_tensor], dim=2)
coeffs = torchcde.linear_interpolation_coeffs(X_with_time)

# Standard forward pass
pred_y, treatment_probs, treatment_effects, z_hat = model.forward(
    coeffs, device, training=False
)

# TTT-adapted forward pass
ttt_pred_y, ttt_treatment_probs, ttt_effects, ttt_z_hat = model.ttt_forward(
    coeffs, device, adapt=True
)

# Counterfactual predictions
counterfactuals, _ = model.counterfactual_prediction(
    coeffs, device, adapt=True
)
```

### Custom Data Generation
```python
from enhanced_data_generation import DataGenerationConfig, EnhancedDataGenerator

# Configure data generation
config = DataGenerationConfig(
    num_samples=1000,
    seq_len=20,
    num_features=8,
    num_treatments=4,
    treatment_assignment_type="confounded",
    confounding_strength=2.0,
    enable_distribution_shift=True
)

# Generate dataset
generator = EnhancedDataGenerator(config)
dataset = generator.generate_dataset()

# Access data
X = dataset['X']                    # Features [samples, time, features]
y = dataset['y']                    # Outcomes [samples, 1]
treatments = dataset['treatments']   # Treatment assignments [samples]
potential_outcomes = dataset['potential_outcomes']  # All potential outcomes
```

### Comprehensive Evaluation
```python
from causal_evaluation_metrics import CausalEvaluator

# Create evaluator
evaluator = CausalEvaluator()

# Evaluate predictions
results = evaluator.evaluate(
    y_true=observed_outcomes,
    y_pred=counterfactual_predictions,
    treatments=treatment_assignments,
    true_effects=true_potential_outcomes  # If available
)

# Generate report
report = evaluator.create_evaluation_report(results)
print(report)

# Create plots
evaluator.plot_evaluation_results(results, save_path="evaluation_plots.png")
```

## Architecture Overview

```
Enhanced TTT-Neural CDE Architecture:

Input Time Series (X) 
         ↓
   Feature Embedding
         ↓
Enhanced CDE Function (with residuals & normalization)
         ↓
   Multi-Head Attention (optional)
         ↓
    Hidden State (z)
         ↓
    ┌─────────────────┐
    ↓                 ↓
Outcome Network   Treatment Network
    ↓                 ↓
Predicted Y      Treatment Effects
                     ↓
               Counterfactuals

TTT Adaptation:
- Reconstruction Task
- Forecasting Task  
- Temporal Consistency
- Causal Contrastive Learning
```

## Key Features

### 1. Multi-Task Auxiliary Learning
- **Reconstruction**: Reconstruct final observed features
- **Forecasting**: Predict future time steps
- **Temporal Consistency**: Maintain consistency across adaptations
- **Causal Contrastive**: Separate treatment representations

### 2. Enhanced Model Components
- **Residual CDE Function**: Better gradient flow and stability
- **Multi-Head Attention**: Capture complex temporal dependencies
- **Treatment-Specific Networks**: Specialized modeling per treatment
- **Uncertainty Quantification**: Built-in uncertainty estimates

### 3. Comprehensive Evaluation
- **PEHE**: Precision in Estimation of Heterogeneous Effect
- **ATE**: Average Treatment Effect estimation
- **Policy Value**: Treatment recommendation quality
- **Calibration**: Uncertainty calibration metrics

## Configuration Options

### Model Configuration
```python
model = EnhancedTTTNeuralCDE(
    # Architecture
    hidden_channels=64,              # Hidden dimension
    use_multi_head_attention=True,   # Enable attention
    num_attention_heads=8,           # Number of attention heads
    use_residual_cde=True,          # Enhanced CDE function
    use_uncertainty=True,            # Uncertainty quantification
    
    # TTT Parameters
    ttt_steps=20,                   # Adaptation steps
    ttt_lr=0.01,                    # TTT learning rate
    ttt_early_stopping_patience=5,  # Early stopping
    
    # Auxiliary Tasks
    auxiliary_task_weights={
        'reconstruction': 0.4,
        'forecasting': 0.3,
        'temporal_consistency': 0.2,
        'causal_contrastive': 0.1
    }
)
```

### Data Generation Configuration
```python
config = DataGenerationConfig(
    # Data size
    num_samples=1000,
    seq_len=20,
    num_features=8,
    
    # Treatment assignment
    treatment_assignment_type="confounded",  # "random", "confounded", "policy"
    confounding_strength=1.5,
    
    # Temporal dynamics
    temporal_autocorr=0.7,
    temporal_noise_std=0.1,
    
    # Distribution shift
    enable_distribution_shift=True,
    shift_magnitude=0.5,
    shift_start_ratio=0.7
)
```

## Output Structure

After running the demo, you'll find:

```
results_simple/  (or results_enhanced/)
├── training_loss.png          # Training curve
├── results.json              # Numerical results
├── evaluation_report.txt     # Detailed report
├── comparison_plots.png      # Model comparison
└── enhanced_causal_evaluation.png  # Causal metrics plots
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd causal-ttt-v2
   # Try installing missing packages
   pip install torch torchcde matplotlib seaborn scikit-learn
   ```

2. **CUDA/MPS Issues**
   ```python
   # The code automatically detects and uses the best available device
   # Force CPU if needed by modifying the device selection in the script
   device = torch.device('cpu')
   ```

3. **Memory Issues**
   ```python
   # Reduce model size or batch size
   model = EnhancedTTTNeuralCDE(
       hidden_channels=32,  # Reduce from 64
       num_attention_heads=4,  # Reduce from 8
       ttt_steps=10  # Reduce from 20
   )
   ```

4. **Slow Training**
   ```python
   # Reduce dataset size for testing
   X, y, treatments, potential_outcomes = generate_simple_synthetic_data(
       num_samples=100,  # Reduce from 200
       seq_len=10       # Reduce from 15
   )
   ```

## Comparison with Original

| Feature | Original TTT-CDE | Enhanced TTT-CDE |
|---------|------------------|------------------|
| Auxiliary Tasks | Single reconstruction | Multi-task learning |
| CDE Function | Basic MLP | Residual + LayerNorm |
| Attention | Simple pooling | Multi-head attention |
| Treatment Effects | Basic modeling | Specialized network |
| Uncertainty | None | Built-in quantification |
| Evaluation | Basic metrics | Comprehensive causal metrics |
| Data Generation | Simple synthetic | Realistic with confounding |

## Citation

If you use this enhanced framework, please cite:

```bibtex
@misc{enhanced_ttt_cde_2024,
  title={Enhanced Test-Time Training for Causal Time Series Forecasting},
  author={Gibran Hasan},
  year={2025},
  note={Enhanced implementation based on TTT forecasting research}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 