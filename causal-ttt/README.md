# Causal Time Series Forecasting with Test-Time Training

This repository implements a novel approach for causal time series forecasting by combining Neural Controlled Differential Equations (Neural CDEs) with Test-Time Training (TTT). The model can generate both factual and counterfactual predictions while adapting to individual samples during inference.

## Background

This implementation combines two key approaches:

1. **Neural CDEs for Causal Inference**: Based on the paper "Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations", this approach uses neural CDEs to model the evolution of time series data under different treatment interventions.

2. **Test-Time Training (TTT)**: Based on concepts from the "Test-Time Training for Forecasting" paper, this approach enables adaptation to individual test samples by performing mini-training steps during inference, leading to more accurate predictions.

## Model Architecture

The core model (`TTTNeuralCDE`) extends the Neural CDE approach with test-time training capabilities:

### Key Components:

1. **Neural CDE Base**: Models time series as a solution to a controlled differential equation.
2. **Test-Time Training**: Mini-adaptation during inference using self-supervised auxiliary tasks.
3. **Counterfactual Generator**: Produces predictions under different treatment scenarios.

## Project Structure

```
causal-ttt/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── ttt_cde_model.py    # Core model implementation
│   ├── __init__.py
│   ├── train.py                # Training script
│   ├── inference.py            # Inference and counterfactual generation
│   └── utils.py                # Utility functions for data processing
└── README.md
```

## Implementation Details

### Core Model (`ttt_cde_model.py`)

The `TTTNeuralCDE` class implements the combined approach with key methods:

- `forward()`: Standard forward pass for training and evaluation
- `ttt_forward()`: Forward pass with test-time adaptation
- `auxiliary_task()`: Self-supervised task used for test-time adaptation
- `counterfactual_prediction()`: Generates counterfactual outcomes under different treatments

### Training (`train.py`)

The training script demonstrates how to:
- Train the model with both primary and auxiliary losses
- Optionally incorporate test-time training during validation
- Save model checkpoints

### Inference (`inference.py`)

The inference script shows how to:
- Generate counterfactual predictions
- Compare performance with and without test-time training
- Visualize factual vs. counterfactual outcomes

## Usage

### Training

```bash
python src/train.py --data_dir path/to/data --batch_size 32 --epochs 100 --hidden_dim 64 --ttt_steps 5 --aux_loss_weight 0.1
```

### Inference

```bash
python src/inference.py --checkpoint path/to/checkpoint --output_dir results --run_ablation
```

## Key Advantages

1. **Improved Accuracy**: Test-time adaptation improves prediction accuracy by tuning the model for each individual sample.
2. **Better Counterfactuals**: The approach generates more reliable counterfactual predictions by adapting to the specific time series.
3. **Self-Supervised Learning**: No need for ground truth counterfactuals during training.

## Requirements

- PyTorch
- torchcde
- torchdiffeq
- numpy
- matplotlib

## References

- "Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations"
- "Test-Time Training for Forecasting"
