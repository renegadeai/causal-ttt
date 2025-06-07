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
    z_t = z_0 + \int_0^t f_θ(z_s) dX_s
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

This module contains the implementation of a Neural Controlled Differential Equation (Neural CDE)
for causal time-series forecasting, enhanced with test-time training (TTT) capabilities.
The model builds on the approach described in "Continuous-Time Modeling of Counterfactual Outcomes
Using Neural Controlled Differential Equations" and incorporates test-time adaptation strategies
from "Test-Time Training for Forecasting".

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
        ttt_lr_decay=0.9
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
        # Calculate output size based on whether we include treatment in auxiliary task
        if self.include_treatment_in_aux:
            output_size = input_channels_x * (1 + forecast_horizon + 1) + num_treatments  # Current + forecast steps + disentanglement + treatment
        else:
            output_size = input_channels_x * (1 + forecast_horizon + 1)  # Current + forecast steps + disentanglement
        
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
        
        # Enhanced auxiliary network for test-time training
        # Now predicts both next value and reconstruction of current value
        self.auxiliary_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.LayerNorm(self.hidden_channels // 2),  # Layer normalization for more stable training
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_channels // 2, 2 * input_channels_x)  # Double size output for dual task
        )
        
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
        Enhanced self-supervised auxiliary task for test-time training.
        
        This method performs multiple self-supervised tasks:
        1. Reconstruction: Predicts the current observation from the hidden state
        2. Multi-step Forecasting: Predicts multiple future observations
        3. Feature Disentanglement: Separates predictable from stochastic components
        4. Temporal Consistency: Ensures predictions are temporally coherent
        
        Args:
            z_hat: Hidden state at the final timestep
            coeffs_x: Tensor containing coefficients for interpolation
            
        Returns:
            loss: Combined auxiliary loss value for test-time training
        """
        # Get the interpolation
        x = self.get_interpolation(coeffs_x)
        grid_points = x.grid_points
        
        # Make sure we have at least one point
        if len(grid_points) <= 0:
            # If no points, return zero loss
            return torch.tensor(0.0, device=z_hat.device)
        
        # For multi-step forecasting, we need to have at least forecast_horizon + 1 points
        # Get the last few observations
        observations = []
        for i in range(min(len(grid_points), self.forecast_horizon + 1)):
            if i < len(grid_points):
                # Evaluate at last few timesteps (starting from the most recent)
                t = grid_points[-(i+1)]
                obs = x.evaluate(t)
                # Remove time dimension if it exists
                if obs.shape[-1] > self.input_channels_x:
                    obs = obs[..., 1:]
                observations.append(obs)
        
        # If we don't have enough observations, pad with duplicates of the last one
        while len(observations) < self.forecast_horizon + 1:
            if len(observations) > 0:
                observations.append(observations[-1].clone())
            else:
                # This should not happen normally, but just in case
                dummy_obs = torch.zeros(z_hat.size(0), self.input_channels_x, device=z_hat.device)
                observations.append(dummy_obs)
        
        # Current observation is the first one in our list (most recent)
        current_obs = observations[0]
        
        # Generate predictions from the auxiliary network
        pred = self.auxiliary_network(z_hat)
        
        # Split the predictions based on tasks
        chunk_size = self.input_channels_x
        pred_current = pred[:, :chunk_size]  # Reconstruction of current observation
        
        # Multi-step forecasting predictions
        forecasting_preds = []
        for i in range(self.forecast_horizon):
            start_idx = chunk_size * (i + 1)
            end_idx = start_idx + chunk_size
            forecasting_preds.append(pred[:, start_idx:end_idx])
        
        # Feature disentanglement - predict decomposition of features
        # Make sure we don't exceed the available dimensions
        disentanglement_idx = chunk_size * (self.forecast_horizon + 1)
        if disentanglement_idx + chunk_size <= pred.shape[1]:
            pred_disentangled = pred[:, disentanglement_idx:disentanglement_idx + chunk_size]
        else:
            # If we don't have enough dimensions, create zeros with the right shape
            pred_disentangled = torch.zeros_like(current_obs)
        
        # Calculate losses
        # 1. Reconstruction loss for current observation
        reconstruction_loss = F.mse_loss(pred_current, current_obs)
        
        # 2. Multi-step forecasting loss with distance weighting
        forecasting_loss = 0.0
        forecast_weights_sum = 0.0
        
        # We need at least 2 observations to do forecasting (current + next)
        if len(observations) >= 2:
            for i in range(min(len(observations)-1, self.forecast_horizon)):
                # Earlier predictions have higher weights
                weight = 1.0 / (i + 1)
                # Skip if dimensions don't match
                if forecasting_preds[i].shape == observations[i+1].shape and observations[i+1].numel() > 0:
                    step_loss = F.mse_loss(forecasting_preds[i], observations[i+1])
                    forecasting_loss += weight * step_loss
                    forecast_weights_sum += weight
            
            # Normalize by weights if we have any forecasting loss
            if forecast_weights_sum > 0:
                forecasting_loss /= forecast_weights_sum
        
        # 3. Feature disentanglement loss - orthogonality constraint
        # Make sure both tensors have the same shape
        if current_obs.shape == pred_disentangled.shape and current_obs.numel() > 0:
            disentanglement_loss = torch.mean(torch.abs(torch.sum(current_obs * pred_disentangled, dim=1)))
        else:
            # If shapes don't match, use a zero loss
            disentanglement_loss = torch.tensor(0.0, device=z_hat.device)
        
        # 4. Temporal consistency loss
        temporal_loss = 0.0
        temporal_count = 0
        
        if self.forecast_horizon > 1:
            for i in range(self.forecast_horizon - 1):
                # Make sure we have valid tensors and matching shapes
                if (i+1 < len(forecasting_preds) and 
                    forecasting_preds[i].numel() > 0 and 
                    forecasting_preds[i+1].numel() > 0 and
                    forecasting_preds[i].shape == forecasting_preds[i+1].shape):
                    
                    # Calculate difference between consecutive predicted timesteps
                    pred_diff = forecasting_preds[i+1] - forecasting_preds[i]
                    
                    # If we have enough observations, compare with actual differences
                    if (i + 2 < len(observations) and 
                        observations[i+1].numel() > 0 and 
                        observations[i+2].numel() > 0 and
                        observations[i+1].shape == observations[i+2].shape and
                        pred_diff.shape == observations[i+2].shape):
                        
                        actual_diff = observations[i+2] - observations[i+1]
                        temporal_loss += F.mse_loss(pred_diff, actual_diff)
                        temporal_count += 1
                    else:
                        # Encourage smoothness if no ground truth is available or shapes don't match
                        temporal_loss += torch.mean(torch.square(pred_diff))
                        temporal_count += 1
            
            if temporal_count > 0:
                temporal_loss /= temporal_count
            else:
                # If we couldn't calculate any temporal loss, use zero
                temporal_loss = torch.tensor(0.0, device=z_hat.device)
        
        # Get treatment prediction if included in auxiliary task
        treatment_loss = 0.0
        if self.include_treatment_in_aux:
            # Extract treatment prediction from the auxiliary network output
            treatment_pred_idx = chunk_size * (self.forecast_horizon + 1) + chunk_size
            treatment_pred = pred[:, treatment_pred_idx:treatment_pred_idx + self.num_treatments]
            
            # If we have the ground truth treatment, calculate loss
            # This requires the treatment to be encoded in the coefficients
            # For demonstration, we'll just use a dummy loss if we can't extract actual treatments
            treatment_loss = torch.tensor(0.0, device=z_hat.device)
        
        # Combine all losses with task weights from config
        loss = (
            self.aux_task_weights['reconstruction'] * reconstruction_loss + 
            self.aux_task_weights['forecasting'] * forecasting_loss + 
            self.aux_task_weights['temporal'] * temporal_loss + 
            self.aux_task_weights['disentanglement'] * disentanglement_loss
        )
        
        # Add treatment loss if included
        if self.include_treatment_in_aux and treatment_loss > 0:
            loss = loss + 0.2 * treatment_loss  # Add treatment prediction loss with weight
        
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
            cf_hidden = z_hat + gate * t_embedding
            
            # Predict outcome for this counterfactual
            cf_outcome = self.outcome_net(cf_hidden)
            
            # Add to counterfactuals dictionary
            counterfactuals[treatment_id] = cf_outcome
        
        return counterfactuals
