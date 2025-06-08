"""
Causal Time Series Forecasting with Test-Time Training using Neural CDEs.
This model combines the Neural CDE approach for causal inference with test-time training.
"""
import logging
import torch
import torch.nn as nn
import torchcde
from torch.nn import functional as F

logger = logging.getLogger(__name__)

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
    # Store original parameters as class variable
    _original_state = None

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
        cf_strength=1.0,  # New parameter for counterfactual strength
        cde_rtol=1e-5,
        cde_atol=1e-5,
        cde_method='rk4'
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

        # CDE solver parameters
        self.cde_rtol = cde_rtol
        self.cde_atol = cde_atol
        self.cde_method = cde_method
        
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
        
    def auxiliary_task(self, z_hat, coeffs_x):
        """
        Self-supervised auxiliary task for TTT. Reconstructs final observed features.
        Args:
            z_hat: Hidden state from CDE.
            coeffs_x: Path object or Tensor for input path X.
        Returns:
            loss: Reconstruction loss.
        """
        aux_net_device = next(self.auxiliary_network.parameters()).device
        z_hat = z_hat.to(aux_net_device)

        if isinstance(coeffs_x, torch.Tensor):
            coeffs_x_on_device = coeffs_x.to(aux_net_device)
            x_path = self.get_interpolation(coeffs_x_on_device)
        else: 
            x_path = self.get_interpolation(coeffs_x)

        grid_points = x_path.grid_points
        if len(grid_points) == 0:
            logger.warning("TTT: Auxiliary task received path with no grid points. Returning zero loss.")
            return torch.tensor(0.0, device=aux_net_device, requires_grad=True)
        
        last_observed_t = grid_points[-1]
        last_observed_features = x_path.evaluate(last_observed_t).to(aux_net_device)
        
        if self.input_has_time and last_observed_features.shape[-1] > self.input_channels_x:
            last_observed_features = last_observed_features[..., 1:]
            
        predicted_features = self.auxiliary_network(z_hat)
        
        if predicted_features.shape != last_observed_features.shape:
            logger.warning(f"TTT: Shape mismatch in auxiliary task. Pred: {predicted_features.shape}, Actual: {last_observed_features.shape}. Returning zero loss.")
            return torch.tensor(0.0, device=aux_net_device, requires_grad=True)
        
        loss = F.mse_loss(predicted_features, last_observed_features)
        logger.debug(f"TTT: Aux loss: {loss.item():.4f}, grad: {loss.requires_grad}, grad_fn: {loss.grad_fn is not None}")
        return loss

    def ttt_forward(self, coeffs_x, device, adapt=True, force_bn_eval_if_bs_is_one=False, store_original_state=True):
        """
        Forward pass with Test-Time Training (TTT).
        Adapts model parameters to minimize a self-supervised auxiliary loss.
        Args:
            coeffs_x: Path object or Tensor for input path X.
            device: PyTorch device for computation.
            adapt (bool): If True, performs TTT.
            force_bn_eval_if_bs_is_one (bool): If True and batch size is 1, forces BatchNorm layers in eval mode.
            store_original_state (bool): If True, stores original model state before adaptation.
        Returns:
            Tuple (pred_y, pred_a_softmax, pred_a, z_hat_final): Predictions and final hidden state.
        """

        # Perform an initial forward pass.
        # coeffs_x is assumed to be on the correct device or a path object.
        # self.forward's mcd parameter defaults to True, which is consistent with
        # how it would have been called if adapt=False previously (implicitly).
        
        # Check for batch size 1 to handle BatchNorm layers
        initial_bn_states = {}
        if force_bn_eval_if_bs_is_one:
            if isinstance(coeffs_x, torch.Tensor):
                batch_size = coeffs_x.shape[0]
            else:
                # For path object
                batch_size = coeffs_x[0].shape[0] if len(coeffs_x) > 0 else 0
            
            # If batch size is 1, force BatchNorm layers to eval mode
            if batch_size == 1:
                # Find all BatchNorm layers and store their current training states
                for name, module in self.named_modules():
                    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        initial_bn_states[name] = module.training
                        module.eval()
                logger.info("Initial forward pass with batch size 1: Setting BatchNorm layers to eval mode")
        
        # Perform initial forward pass
        try:
            pred_y_initial, pred_a_softmax_initial, pred_a_initial, z_hat_initial = self.forward(coeffs_x, device=device)
        finally:
            # Restore original BatchNorm states if changed
            if initial_bn_states:
                for name, training in initial_bn_states.items():
                    if name in dict(self.named_modules()):
                        dict(self.named_modules())[name].train(training)

        if not adapt:
            return pred_y_initial, pred_a_softmax_initial, pred_a_initial, z_hat_initial
            
        # Store original model state if requested and not already stored
        if store_original_state and self.__class__._original_state is None:
            logger.info("Storing original model state before first TTT adaptation")
            self.__class__._original_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            
        # Forcefully enable gradients for the TTT scope
        original_grad_enabled_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            # Configure parameters for TTT adaptation
            modules_to_adapt_config = [
                (self.auxiliary_network, self.ttt_lr, "auxiliary_network"),
                (self.cde_func, self.ttt_lr, "cde_func"),
                (self.embed_x, self.ttt_lr, "embed_x"),
                (self.treatment_gate, self.ttt_lr, "treatment_gate"),
                (self.outcome_net, self.ttt_lr, "outcome_net")
            ]
            if self.use_attention and self.attention is not None:
                modules_to_adapt_config.append((self.attention, self.ttt_lr, "attention"))

            params_to_adapt_list = []
            param_groups_for_optimizer = []

            for module, lr, name in modules_to_adapt_config:
                if module is not None:
                    module_params = list(module.parameters())
                    if not module_params:
                        logger.warning(f"TTT: Module {name} has no parameters to adapt.")
                        continue
                    # Ensure parameters require gradients
                    for p in module_params: 
                        p.requires_grad_(True)
                    params_to_adapt_list.extend(module_params)
                    param_groups_for_optimizer.append({'params': module_params, 'lr': lr})
                else:
                    logger.warning(f"TTT: Module {name} is None and cannot be adapted.")

            if not params_to_adapt_list:
                logger.warning("TTT: No parameters to adapt. Skipping TTT optimization.")
                # TTT skipped as no parameters to adapt, return initial predictions.
                torch.set_grad_enabled(original_grad_enabled_state) # Restore grad state before early return
                return pred_y_initial, pred_a_softmax_initial, pred_a_initial, z_hat_initial
            
            optimizer = torch.optim.Adam(param_groups_for_optimizer)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.ttt_lr_decay)

            best_loss = float('inf')
            patience_counter = 0
            initial_adapted_params_state = {name: p.clone().detach() for name, p in self.named_parameters() if any(p is p_adapted for p_adapted in params_to_adapt_list)}
            best_adapted_state_dict = {k: v.clone() for k, v in initial_adapted_params_state.items()}

            # Store original training modes and set to train() for TTT
            adapted_modules_for_mode_change = [m_config[0] for m_config in modules_to_adapt_config]
            original_modes = {}
            
            # This inner try/finally handles module mode restoration
            try: 
                for module_to_set_mode in adapted_modules_for_mode_change:
                    if isinstance(module_to_set_mode, torch.nn.Module):
                        original_modes[module_to_set_mode] = module_to_set_mode.training
                        module_to_set_mode.train()

                if isinstance(coeffs_x, tuple):
                    batch_size = coeffs_x[0].shape[0]
                elif isinstance(coeffs_x, torch.Tensor):
                    batch_size = coeffs_x.shape[0]
                else:
                    logger.warning("TTT: Could not determine batch size from coeffs_x type.")
                    batch_size = -1 

                if force_bn_eval_if_bs_is_one and batch_size == 1:
                    logger.debug("TTT: Batch size is 1 and force_bn_eval_if_bs_is_one is True. Setting BatchNorm layers to eval mode.")
                    for main_module in adapted_modules_for_mode_change:
                        if isinstance(main_module, torch.nn.Module):
                            for sub_m in main_module.modules():
                                if isinstance(sub_m, torch.nn.modules.batchnorm._BatchNorm):
                                    if sub_m not in original_modes: 
                                        original_modes[sub_m] = sub_m.training
                                    sub_m.eval() 

                # TTT optimization loop
                for step in range(self.ttt_steps):
                    optimizer.zero_grad()
                    
                    current_coeffs_x_for_cde = coeffs_x.to(device) if isinstance(coeffs_x, torch.Tensor) else coeffs_x
                    x_cde_path = self.get_interpolation(current_coeffs_x_for_cde)
                    
                    initial_x_for_embed = x_cde_path.evaluate(x_cde_path.interval[0]).to(device)
                    if self.input_has_time and initial_x_for_embed.shape[-1] > self.input_channels_x:
                        initial_x_for_embed = initial_x_for_embed[..., 1:]
                    
                    z0_cde = self.embed_x(initial_x_for_embed)
                    
                    z_hat_sequence = torchcde.cdeint(
                        X=x_cde_path, z0=z0_cde, func=self.cde_func, 
                        t=x_cde_path.grid_points.to(device),
                        rtol=self.cde_rtol, atol=self.cde_atol, method=self.cde_method
                    )
                    
                    if self.use_attention and self.attention is not None:
                        b, s, h = z_hat_sequence.shape
                        attn_weights = self.attention(z_hat_sequence.reshape(-1, h)).view(b, s, 1)
                        attn_weights = F.softmax(attn_weights, dim=1)
                        z_hat = torch.sum(z_hat_sequence * attn_weights, dim=1)
                    else:
                        z_hat = z_hat_sequence[:, -1]

                    auxiliary_loss = self.auxiliary_task(z_hat, coeffs_x) 
                    current_loss_val = auxiliary_loss.item()

                    logger.debug(f"TTT (Iter {step + 1}/{self.ttt_steps}): Loss: {current_loss_val:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                    if not auxiliary_loss.requires_grad:
                        logger.error(f"TTT CRITICAL: aux_loss no grad @ step {step+1}. grad_fn: {auxiliary_loss.grad_fn}. Stop TTT.")
                        break
                    
                    auxiliary_loss.backward()
                    torch.nn.utils.clip_grad_norm_(params_to_adapt_list, max_norm=getattr(self, 'ttt_grad_clip_norm', 1.0))
                    optimizer.step()

                    if current_loss_val < best_loss:
                        best_loss = current_loss_val
                        patience_counter = 0
                        for name, p_model in initial_adapted_params_state.items(): 
                            best_adapted_state_dict[name] = self.state_dict()[name].clone().detach()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.ttt_early_stopping_patience:
                            logger.info(f"TTT: Early stop @ step {step + 1}. Best loss: {best_loss:.4f}")
                            break
                    scheduler.step()
            finally:
                # Restore original training modes
                for module_to_restore_mode, original_training_state in original_modes.items():
                    if isinstance(module_to_restore_mode, torch.nn.Module):
                         module_to_restore_mode.training = original_training_state

            final_load_state = self.state_dict()
            if best_loss != float('inf') and best_adapted_state_dict:
                logger.info(f"TTT: Loading best adapted params (loss: {best_loss:.4f}).")
                for name, param_tensor in best_adapted_state_dict.items():
                    final_load_state[name] = param_tensor.to(device) 
            else:
                logger.info("TTT: No improvement or TTT skipped. Restoring initial state.")
                for name, param_tensor in initial_adapted_params_state.items():
                    if name in final_load_state: 
                        final_load_state[name] = param_tensor.to(device)
                    else:
                        logger.warning(f"TTT: Parameter {name} from initial_adapted_params_state not found in current model state_dict during restore.")
            self.load_state_dict(final_load_state)

        finally:
            torch.set_grad_enabled(original_grad_enabled_state) # Restore original grad state

        # Perform a final forward pass with the adapted model
        with torch.no_grad(): # Ensure this pass doesn't affect gradients or require training mode
            # Check if we need to handle BatchNorm for batch size 1
            bn_states = {}
            if force_bn_eval_if_bs_is_one:
                # Get batch size from coeffs_x
                if isinstance(coeffs_x, torch.Tensor):
                    batch_size = coeffs_x.shape[0]
                else:
                    # For path object
                    batch_size = coeffs_x[0].shape[0] if len(coeffs_x) > 0 else 0
                
                # If batch size is 1, force BatchNorm layers to eval mode
                if batch_size == 1:
                    # Find all BatchNorm layers and store their current training states
                    for name, module in self.named_modules():
                        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                            bn_states[name] = module.training
                            module.eval()
                    logger.info("Final forward pass with batch size 1: Setting BatchNorm layers to eval mode")
            
            # Using mcd=False for deterministic output after adaptation
            pred_y_final, pred_a_softmax_final, pred_a_final, z_hat_final = self.forward(coeffs_x, device=device, mcd=False)
            
            # Restore original BatchNorm states if changed
            if bn_states:
                for name, training in bn_states.items():
                    if name in dict(self.named_modules()):
                        dict(self.named_modules())[name].train(training)
                        
            # If we restored the original state for adapt=False, restore the adapted state now
            if not adapt and self.__class__._original_state is not None and 'current_state' in locals():
                logger.info("Restoring adapted model state after non-adapted counterfactual prediction")
                self.load_state_dict({k: v.to(device) for k, v in current_state.items()})
                        
        return pred_y_final, pred_a_softmax_final, pred_a_final, z_hat_final

    def counterfactual_prediction(self, coeffs_x, device, adapt=True):

        """
        Generate counterfactual predictions for all possible treatments.

        This enhanced version uses the test-time adapted representation when adapt=True
        and incorporates better treatment effect modeling. When adapt=False, it ensures
        the original non-adapted model state is used.

        Args:
            coeffs_x: Tensor containing coefficients for interpolation or Path object.
            device: Device to run the model on.
            adapt: Whether to perform test-time adaptation.
            Returns:
                counterfactuals: Dictionary mapping treatment IDs to predicted outcomes.
                z_hat: The latent representation used to generate these counterfactuals.
            """
        self.to(device) # Ensure model is on the correct device
        # Ensure coeffs_x is on the correct device if it's a tensor, or handled if it's a path object
        coeffs_x_device = coeffs_x.to(device) if isinstance(coeffs_x, torch.Tensor) else coeffs_x

        # Run forward pass to get hidden state with adaptation if requested
        if adapt:
            # ttt_forward is expected to return: pred_y, pred_a_softmax, pred_a, z_hat_final
            # ttt_forward already has BatchNorm handling for batch size 1
            _, _, _, z_hat = self.ttt_forward(coeffs_x_device, device=device, adapt=True, force_bn_eval_if_bs_is_one=True)
        else:
            # For adapt=False, restore original model state if available
            if self.__class__._original_state is not None:
                logger.info("Restoring original non-adapted model state for counterfactual prediction")
                # Save current state for restoration after counterfactual computation
                current_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                # Load original non-adapted state
                self.load_state_dict({k: v.to(device) for k, v in self.__class__._original_state.items()})
            # Get batch size from coeffs_x_device
            if isinstance(coeffs_x_device, torch.Tensor):
                batch_size = coeffs_x_device.shape[0]
            else:
                # For path object
                batch_size = coeffs_x_device[0].shape[0] if len(coeffs_x_device) > 0 else 0
            
            # If batch size is 1, temporarily set BatchNorm layers to eval mode
            bn_states = {}
            if batch_size == 1:
                # Find all BatchNorm layers and store their current training states
                for name, module in self.named_modules():
                    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        bn_states[name] = module.training
                        module.eval()
                logger.info("Counterfactual with batch size 1 (no TTT): Setting BatchNorm layers to eval mode")
            
            # Run forward pass
            try:
                # forward is expected to return: pred_y, pred_a_softmax, pred_a, z_hat
                _, _, _, z_hat = self.forward(coeffs_x_device, device=device)
            finally:
                # Restore original BatchNorm states if changed
                if bn_states:
                    for name, training in bn_states.items():
                        if name in dict(self.named_modules()):
                            dict(self.named_modules())[name].train(training)
        
        # Get batch size from z_hat (which should be on the correct device)
        batch_size = z_hat.shape[0]
        
        # Create counterfactuals dictionary
        counterfactuals = {}
        
        # Normalize treatment embeddings for more stable counterfactual predictions
        # Ensure treatment_embedding is on the correct device
        treatment_embedding_device = self.treatment_embedding.to(device)
        normalized_embeddings = F.normalize(treatment_embedding_device, dim=1)
        
        # For each possible treatment
        for treatment_id in range(self.num_treatments):
            # Get treatment embedding, ensuring it's on the correct device
            t_embedding = normalized_embeddings[treatment_id].unsqueeze(0).repeat(batch_size, 1)
            # t_embedding is derived from normalized_embeddings which is already on 'device'
            
            # Modify hidden state with treatment embedding using an improved gating mechanism
            # Ensure all inputs to treatment_gate (z_hat, t_embedding) are on the same device.
            # self.treatment_gate itself should be on 'device' due to self.to(device).
            gate_input = torch.cat([z_hat, t_embedding], dim=1)
            gate = torch.sigmoid(self.treatment_gate(gate_input))
            cf_hidden = z_hat + self.cf_strength * gate * t_embedding
            
            # Predict outcome for this counterfactual
            # self.outcome_net should be on 'device'. cf_hidden is on 'device'.
            cf_outcome = self.outcome_net(cf_hidden)
            
            # Add to counterfactuals dictionary
            counterfactuals[treatment_id] = cf_outcome
        
        return counterfactuals, z_hat

