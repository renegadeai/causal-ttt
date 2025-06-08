"""
Causal Time Series Forecasting with Test-Time Training using Neural CDEs.
This model combines the Neural CDE approach for causal inference with test-time training.
"""
import logging
import torch
import torch.nn as nn
import torchcde
from torch.nn import functional as F


class CDEFunc(torch.nn.Module):
    """
    Neural CDE function f_θ.
    
    Defines the f_θ in the CDE definition:
    z_t = z_0 + \\int_0^t f_θ(z_s) dX_s
    """
    def __init__(self, hidden_channels, hidden_hidden, input_size):
        super(CDEFunc, self).__init__()
        self.hidden_channels = hidden_channels
        self.hidden_hidden = hidden_hidden
        self.input_size = input_size

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_hidden)
        self.linear2 = torch.nn.Linear(hidden_hidden, hidden_channels * input_size)

    def forward(self, t, z):
        """
        CDEFunc forward pass.
        
        Args:
            t: Current time
            z: Hidden state z_t
            
        Returns:
            f_θ(z_t): Shaped as a matrix to perform matrix-vector product with dX_t
        """
        z = self.linear1(z)
        z = torch.relu(z)
        z = self.linear2(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_size)
        return z


class TTTNeuralCDE(torch.nn.Module):
    """
    Implementation of a Neural CDE with test-time training.

This module contains the implementation of a Test-Time Training Neural Controlled Differential Equation (TTT-Neural CDE) model. It is designed for time series forecasting under distributional shifts, allowing the model to adapt to new data at test time. The model leverages Neural CDEs for continuous-time dynamics and incorporates a test-time adaptation mechanism based on self-supervised learning.

The TTTNeuralCDE class provides:
1. Standard Neural CDE functionality for time series modeling
2. Test-time adaptation through a self-supervised auxiliary task
3. Counterfactual prediction capabilities for different treatment scenarios

Key features:
- Neural CDE base model for time series modeling
- Self-supervised auxiliary task for test-time adaptation
- Support for counterfactual prediction
"""
    def __init__(
        self, 
        input_channels_x, 
        hidden_channels, 
        output_channels,
        num_treatments,
        dropout_rate=0.1,
        interpolation_method="linear",
        ttt_steps=20,  
        ttt_lr=0.01,
        ttt_loss_weight=0.1,
        aux_task_weights=None,
        use_attention=True,  
        forecast_horizon=3,
        input_has_time=True,
        include_treatment_in_aux=True,
        ttt_early_stopping_patience=5,
        ttt_lr_decay=0.9,
        cf_strength=1.0  # New parameter for counterfactual strength
    ):
        """
        Args:
            input_channels_x: Dimension of input data (excluding time).
            hidden_channels: Dimension of hidden state.
            output_channels: Dimension of output (usually 1 for scalar outcome).
            forecast_horizon: Number of future steps to predict in auxiliary task.
            num_treatments: Number of possible treatments.
            interpolation_method: Type of interpolation ('cubic' or 'linear').
            ttt_lr: Learning rate for test-time training.
            ttt_steps: Number of gradient steps for test-time training.
            ttt_loss_weight: Weight of the auxiliary loss during test-time training.
            input_has_time: Whether input data includes time as the first dimension.
            dropout_rate: Rate for dropout regularization.
            use_attention: Whether to use attention mechanism in the model.
        """
        super(TTTNeuralCDE, self).__init__()
        
        self.input_channels_x = input_channels_x
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_treatments = num_treatments
        self.interpolation_method = interpolation_method
        self.use_attention = use_attention
        self.input_has_time = input_has_time
        self.forecast_horizon = forecast_horizon
        self.include_treatment_in_aux = include_treatment_in_aux
        
        # Default weights for the auxiliary loss components if not provided
        if aux_task_weights is None:
            self.aux_task_weights = {
                'reconstruction': 0.35,
                'forecasting': 0.35,
                'temporal': 0.20,
                'disentanglement': 0.10
            }
        else:
            self.aux_task_weights = aux_task_weights
            
        # TTT optimization settings
        self.ttt_early_stopping_patience = ttt_early_stopping_patience
        self.ttt_lr_decay = ttt_lr_decay
        
        # TTT hyperparameters
        self.ttt_lr = ttt_lr
        self.ttt_steps = ttt_steps
        self.ttt_loss_weight = ttt_loss_weight
        
        # Store cf_strength
        self.cf_strength = cf_strength
        
        # Input embedding with batch normalization for stable training
        self.embed_x = torch.nn.Sequential(
            torch.nn.Linear(self.input_channels_x, self.hidden_channels),
            torch.nn.BatchNorm1d(self.hidden_channels),
            torch.nn.ReLU()
        )

        # The CDE function - if input has time as first dimension, we need to account for it
        input_size = input_channels_x + 1 if input_has_time else input_channels_x
        self.cde_func = CDEFunc(self.hidden_channels, self.hidden_channels // 2, input_size)
        
        # Attention mechanism for focusing on important time steps
        if self.use_attention:
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels // 2),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_channels // 2, 1)
            )
            # Add forecast_horizon parameter for multi-step prediction
            self.forecast_horizon = forecast_horizon
        
        # Define the enhanced auxiliary network with multiple outputs
        # 1. Reconstruction of current observation
        # 2. Multi-step forecasting (forecast_horizon steps)
        # 3. Feature disentanglement (predictable vs unpredictable components)
        # Calculate output size for auxiliary network: reconstructs current features
        output_size = self.input_channels_x
        
        self.auxiliary_network = nn.Sequential(
            nn.Linear(hidden_channels, 3 * hidden_channels),
            nn.BatchNorm1d(3 * hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(3 * hidden_channels, 2 * hidden_channels),
            nn.BatchNorm1d(2 * hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * hidden_channels, output_size)
        )
        
        # Outcome network
        self.outcome_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels // 2, output_channels)
        )
        
        # Treatment network
        self.treatment_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels // 2, num_treatments)
        )
        
        # Treatment gate for better counterfactual modeling
        self.treatment_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),  # Takes concatenated [z_hat, t_embedding]
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels),
        )
        
        # Dropout for regularization
        self.dropout_layer = torch.nn.Dropout(p=dropout_rate)
        
        # For counterfactual prediction: treatment embedding with normalization
        self.treatment_embedding = torch.nn.Parameter(
            torch.randn(num_treatments, hidden_channels) / hidden_channels ** 0.5
        )
        
        # Log interpolation type
        # logging.info(f"Interpolation type: {interpolation_method}")

    def get_interpolation(self, coeffs_x):
        """Create appropriate interpolation based on the specified type."""
        if self.interpolation_method == "cubic":
            logging.info("Using cubic interpolation")
            return torchcde.NaturalCubicSpline(coeffs_x)
        elif self.interpolation_method == "linear":
            logging.info("Using linear interpolation")
            return torchcde.LinearInterpolation(coeffs_x)
        else:
            raise ValueError(
                f"Unknown interpolation type: {self.interpolation_method}. Expected 'cubic' or 'linear'."
            )

    def forward(self, coeffs_x, device, mcd=True):
        """
        Forward pass through the model.
        
        Args:
            coeffs_x: Tensor containing coefficients for interpolation
            device: Device to run the model on
            mcd: Whether to use Monte Carlo Dropout during prediction
            
        Returns:
            pred_y: Predicted outcome
            pred_a_softmax: Predicted treatment probabilities
            pred_a: Predicted treatment logits
            z_hat: Final hidden state
        """
        # Create the interpolation
        x = self.get_interpolation(coeffs_x)
        
        # Calculate initial hidden state from initial values
        initial_x = x.evaluate(x.interval[0])
        
        # If input has time as first channel, remove it before embedding
        if initial_x.shape[-1] > self.input_channels_x:
            initial_x = initial_x[..., 1:]
        
        # Embed initial observation to create initial hidden state
        z_x = self.embed_x(initial_x).to(device)
        
        # Solve the controlled differential equation
        z_hat = torchcde.cdeint(
            X=x,
            z0=z_x,
            func=self.cde_func,
            t=x.grid_points,
            rtol=torch.tensor(1e-3, dtype=torch.float32),
            atol=torch.tensor(1e-5, dtype=torch.float32),
            method='rk4'
        )
        
        # Apply attention if enabled
        if self.use_attention:
            # Calculate attention weights across the sequence
            batch_size, seq_len, hidden_dim = z_hat.shape
            
            # Reshape for attention calculation
            z_flat = z_hat.reshape(-1, hidden_dim)
            attention_scores = self.attention(z_flat).reshape(batch_size, seq_len)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
            
            # Apply attention weights to sequence
            z_hat = torch.sum(z_hat * attention_weights, dim=1)
        else:
            # Just take the last hidden state if not using attention
            z_hat = z_hat[:, -1]
        
        # Apply dropout if in training mode
        if mcd:
            z_hat = self.dropout_layer(z_hat)
        
        # Predict outcome and treatment using enhanced networks
        pred_y = self.outcome_net(z_hat)
        pred_a = self.treatment_net(z_hat)
        pred_a_softmax = F.softmax(pred_a, dim=1)
        
        return pred_y, pred_a_softmax, pred_a, z_hat

    def auxiliary_task(self, z_hat, coeffs_x):
        """
        Simplified self-supervised auxiliary task for test-time training.
        This task aims to reconstruct the features of the final observed time point.
        
        Args:
            z_hat: Hidden state at the final timestep from the CDE.
            coeffs_x: Tensor containing coefficients for interpolation of the input path X.
            
        Returns:
            loss: Reconstruction loss for the final observed features.
        """
        # Get the interpolation of the input path X
        x_path = self.get_interpolation(coeffs_x)
        grid_points = x_path.grid_points
        
        # Ensure there's at least one point in the path
        if len(grid_points) == 0:
            return torch.tensor(0.0, device=z_hat.device)
        
        # Get the features of the last observed point in the time series
        # grid_points are sorted, so grid_points[-1] is the last time t_m
        last_observed_t = grid_points[-1]
        last_observed_features = x_path.evaluate(last_observed_t)
        
        # Remove time dimension if it was part of the input features to CDE
        if self.input_has_time and last_observed_features.shape[-1] > self.input_channels_x:
            last_observed_features = last_observed_features[..., 1:] # Assuming time is the first channel
            
        # The auxiliary network predicts the features of the last observed point
        predicted_features = self.auxiliary_network(z_hat)
        
        # Ensure shapes match for loss calculation
        if predicted_features.shape != last_observed_features.shape:
            # This might happen if batch sizes differ or there's a configuration mismatch.
            # Log an error or warning, and return a zero loss to avoid crashing.
            # For a robust implementation, consider how to handle or prevent this.
            # logging.warning(f"Shape mismatch in auxiliary task: pred {predicted_features.shape}, actual {last_observed_features.shape}")
            return torch.tensor(0.0, device=z_hat.device)
        
        # Calculate reconstruction loss (Mean Squared Error)
        loss = F.mse_loss(predicted_features, last_observed_features)
        
        return loss
        
    def ttt_forward(self, coeffs_x, device, adapt=True):
        """
        Enhanced forward pass with test-time training.
        
        This enhanced version includes:
        1. Learning rate scheduling with exponential decay
        2. Early stopping with configurable patience
        3. Gradient norm clipping for stability
        4. State dict saving for best model during adaptation
        
        Args:
            coeffs_x: Tensor containing coefficients for interpolation
            device: Device to run the model on
            adapt: Whether to perform test-time adaptation
            
        Returns:
            pred_y: Predicted outcome
            pred_a_softmax: Predicted treatment probabilities
            pred_a: Predicted treatment logits
            z_hat: Final hidden state after adaptation
        """
        # If no adaptation requested, just use the normal forward pass
        if not adapt:
            return self.forward(coeffs_x, device)
            
        # Store original parameters to restore later
        original_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Create parameter groups for optimization during TTT
        # We're focusing adaptation on representation learning components
        param_groups = [
            {'params': self.auxiliary_network.parameters(), 'lr': self.ttt_lr},
            {'params': self.cde_func.parameters(), 'lr': self.ttt_lr * 0.1},
            {'params': self.embed_x.parameters(), 'lr': self.ttt_lr * 0.1},
        ]
        
        # If using attention, include those parameters
        if self.use_attention:
            param_groups.append({'params': self.attention.parameters(), 'lr': self.ttt_lr * 0.1})
        
        # Create optimizer for TTT
        optimizer = torch.optim.Adam(param_groups)
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.ttt_lr_decay)
        
        # Variables for early stopping
        best_loss = float('inf')
        patience = self.ttt_early_stopping_patience
        counter = 0
        best_state_dict = None
        
        # Store performance metrics
        ttt_losses = []
        
        # Perform test-time training steps
        for step in range(self.ttt_steps):
            # Get the interpolation
            x = self.get_interpolation(coeffs_x)
            
            # Get initial hidden state
            initial_x = x.evaluate(x.interval[0])
            if initial_x.shape[-1] > self.input_channels_x:
                initial_x = initial_x[..., 1:]
            
            # Compute the embedding and CDE
            z_x = self.embed_x(initial_x).to(device)
            
            # Solve the CDE
            z_hat = torchcde.cdeint(
                X=x, 
                z0=z_x, 
                func=self.cde_func, 
                t=x.grid_points,
                rtol=torch.tensor(1e-3, dtype=torch.float32), 
                atol=torch.tensor(1e-5, dtype=torch.float32),
                method='rk4'
            )
            
            # Apply attention if used
            if self.use_attention:
                # Compute attention weights
                batch_size, seq_len, hidden_dim = z_hat.shape
                z_flat = z_hat.reshape(-1, hidden_dim)
                attn_weights = self.attention(z_flat).reshape(batch_size, seq_len, 1)
                attn_weights = F.softmax(attn_weights, dim=1)
                
                # Apply attention weights
                z_hat = torch.sum(z_hat * attn_weights, dim=1)
            else:
                # Just use the last hidden state if not using attention
                z_hat = z_hat[:, -1]
            
            # Compute auxiliary loss
            auxiliary_loss = self.auxiliary_task(z_hat, coeffs_x)
            
            # Store loss value
            current_loss = auxiliary_loss.item()
            ttt_losses.append(current_loss)
            
            # Optimize
            optimizer.zero_grad()
            auxiliary_loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Early stopping logic with model state saving
            if current_loss < best_loss:
                best_loss = current_loss
                counter = 0
                # Save the best model state
                best_state_dict = {name: param.clone() for name, param in self.named_parameters()}
            else:
                counter += 1
            
            # Update learning rate
            scheduler.step()
            
            # Stop if no improvement for 'patience' steps
            if counter >= patience:
                logging.debug(f"Early stopping at step {step} with best loss: {best_loss:.4f}")
                break
                
        # Load the best model parameters if we have them
        if best_state_dict is not None:
            for name, param in self.named_parameters():
                param.data = best_state_dict[name].data
                
        # Forward pass with adapted weights
        pred_y, pred_a_softmax, pred_a, z_hat = self.forward(coeffs_x, device)
        
        # Restore original parameters
        for name, param in self.named_parameters():
            param.data = original_params[name].data
        
        # Log the TTT adaptation performance
        if len(ttt_losses) > 1:
            initial_loss = ttt_losses[0]
            final_loss = ttt_losses[-1]
            logging.debug(f"TTT adaptation: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}, reduction={(initial_loss - final_loss) / initial_loss * 100:.2f}%")
            
        return pred_y, pred_a_softmax, pred_a, z_hat
    
    def counterfactual_prediction(self, coeffs_x, device, adapt=True):
        """
        Generate counterfactual predictions for all possible treatments.
        
        This enhanced version uses the test-time adapted representation when adapt=True
        and incorporates better treatment effect modeling.
        
        Args:
            coeffs_x: Tensor containing coefficients for interpolation
            device: Device to run the model on
            adapt: Whether to perform test-time adaptation
            
        Returns:
            counterfactuals: Dictionary mapping treatment IDs to predicted outcomes
            z_hat: The latent representation used to generate these counterfactuals
        """
        # Run forward pass to get hidden state with adaptation if requested
        if adapt:
            _, _, _, z_hat = self.ttt_forward(coeffs_x, device)
        else:
            _, _, _, z_hat = self.forward(coeffs_x, device)
        
        # Get batch size
        batch_size = z_hat.shape[0]
        
        # Create counterfactuals dictionary
        counterfactuals = {}
        
        # Normalize treatment embeddings for more stable counterfactual predictions
        normalized_embeddings = F.normalize(self.treatment_embedding, dim=1)
        
        # For each possible treatment
        for treatment_id in range(self.num_treatments):
            # Get treatment embedding
            t_embedding = normalized_embeddings[treatment_id].unsqueeze(0).repeat(batch_size, 1)
            
            # Modify hidden state with treatment embedding using an improved gating mechanism
            # This allows for more nuanced counterfactual predictions
            gate = torch.sigmoid(self.treatment_gate(torch.cat([z_hat, t_embedding], dim=1)))
            cf_hidden = z_hat + self.cf_strength * gate * t_embedding
            
            # Predict outcome for this counterfactual
            cf_outcome = self.outcome_net(cf_hidden)
            
            # Add to counterfactuals dictionary
            counterfactuals[treatment_id] = cf_outcome
        
        return counterfactuals, z_hat
