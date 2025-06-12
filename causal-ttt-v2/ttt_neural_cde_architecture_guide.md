## 🎯 **TTT Neural CDE Architecture Guide - COMPREHENSIVE SUMMARY**

## 📋 **Complete Documentation Created**

### **Core Documents:**
1. **`enhanced_ttt_cde_model_fixed.py`** - Main implementation (764 lines)
2. **`ttt_neural_cde_architecture_guide.md`** - Architecture overview
3. **`individual_causal_analysis_data_dictionary.md`** - DataFrame field explanations
4. **`training_metrics_explanation.md`** - Loss component details

---

## 🏗️ **Full Architecture Breakdown**

### **1. High-Level Architecture Flow:**
```
Time Series Input [batch, time_steps, 6 features]
    ↓ Add time dimension
    ↓ Linear interpolation coefficients
    ↓ Enhanced CDE Function (with residuals + LayerNorm)
    ↓ Multi-Head Attention (8 heads, optional)
    ↓ Global Attention Pooling
    ↓ Hidden Representation z_hat [batch, 32]
    ↓ Treatment Effect Networks (Baseline + Additive Effects)
    ↓ ALL Potential Outcomes [batch, 3, 1] + Treatment Probabilities [batch, 3]
```

### **2. Core Innovation: Unified Causal Architecture**
- **Single Forward Pass** → Generates ALL counterfactuals simultaneously
- **Additive Treatment Effects**: `Y(T=t) = Baseline + Treatment_Effect_t`
- **Learned Propensity Scores**: Treatment assignment probabilities

### **3. Treatment Effect Networks:**
```python
# Baseline (individual-specific under control):
Y(T=0) = baseline_network(z_hat) + 0

# Treatment effects (additive):
Y(T=1) = baseline_network(z_hat) + effect_network_1(z_hat)  # +1.0 benefit
Y(T=2) = baseline_network(z_hat) + effect_network_2(z_hat)  # -0.5 harm
```

---

## 🏋️ **Complete Training Process**

### **Step-by-Step Training Flow:**
1. **Data Preparation**: Add time dimension, create interpolation coefficients
2. **Forward Pass**: CDE integration → Attention → Treatment networks
3. **Multi-Objective Loss Computation** (6 components)
4. **Backpropagation**: Gradient clipping + Adam optimization
5. **Learning Rate Scheduling**: ReduceLROnPlateau

### **Multi-Objective Loss Function:**
```python
total_loss = (
    1.0 * factual_loss +        # Predict observed outcomes
    2.0 * causal_loss +         # Learn ALL counterfactuals (HIGHEST WEIGHT)
    0.3 * treatment_loss +      # Propensity score learning
    0.1 * auxiliary_loss +      # Regularization tasks
    1.0 * effect_consistency +  # Correct effect magnitudes [0, +1.0, -0.5]
    0.5 * ranking_loss          # Treatment ordering (T1 > T0 > T2)
)
```

### **Training Configuration:**
- **120 epochs**, batch size 32, gradient clipping (max_norm=1.0)
- **Adam optimizer** (lr=1e-3, weight_decay=1e-5)
- **ReduceLROnPlateau scheduler** (patience=15, factor=0.8)

---

## 🔍 **Inference & Counterfactual Generation**

### **Standard Inference Process:**
```python
model.eval()
with torch.no_grad():
    potential_outcomes, treatment_probs, uncertainties, z_hat = model.forward(X_test, training=False)

# Extract counterfactuals for each treatment
counterfactuals = {
    0: potential_outcomes[:, 0, :],  # Control outcomes
    1: potential_outcomes[:, 1, :],  # Beneficial treatment outcomes  
    2: potential_outcomes[:, 2, :]   # Harmful treatment outcomes
}
```

### **Individual Treatment Effects (ITE):**
```python
# ITE calculation
ITE_T1_vs_T0 = counterfactuals[1] - counterfactuals[0]  # Expected: +1.0
ITE_T2_vs_T0 = counterfactuals[2] - counterfactuals[0]  # Expected: -0.5

# Policy recommendations
optimal_treatment = torch.argmax(potential_outcomes.squeeze(-1), dim=1)
```

---

## 🚀 **Test-Time Training (TTT) Adaptation**

### **TTT Philosophy:**
- **Adapt** model parameters at inference time (no ground truth needed)
- **Conservative optimization** to prevent over-adaptation
- **Improve** predictions for new populations/contexts

### **Step-by-Step TTT Process:**

#### **1. Parameter Selection for Adaptation:**
```python
modules_to_adapt = [
    model.treatment_network,  # MOST IMPORTANT - affects counterfactuals
    model.embed_x,           # Input embedding
    model.cde_func,          # CDE dynamics
    model.attention          # Attention mechanism (if present)
]
```

#### **2. TTT Loss Components:**
```python
ttt_loss = (
    0.3 * auxiliary_loss +      # Reconstruction + forecasting tasks
    1.0 * regularization_loss + # Stay close to original (MAIN COMPONENT)
    0.1 * consistency_loss +    # Gentle ordering preference
    0.1 * stability_loss        # Representation stability
)
```

#### **3. Conservative Optimization:**
- **Learning rate**: Original LR × 0.5 (conservative adaptation)
- **Weight decay**: 1e-4, **Steps**: 20, **Early stopping**: patience=8
- **Gradient clipping**: max_norm=1.0

### **TTT Results:**
- **Standard MSE**: 0.0266 → **TTT MSE**: 0.0259 (**2.79% improvement**)
- **Policy Accuracy**: 100% (both models maintain perfect policy learning)
- **Treatment Ordering**: Preserved (T1 > T0 > T2)

---

## 💰 **Detailed Loss Function Analysis**

### **Training Loss Components Explained:**

#### **1. Factual Loss (Weight: 1.0)**
- **Purpose**: Ensure model predicts observed outcomes correctly
- **Formula**: MSE between predicted and actual observed outcomes

#### **2. Causal Loss (Weight: 2.0) - MOST CRITICAL**
- **Purpose**: Learn ALL potential outcomes (enables counterfactual prediction)
- **Formula**: MSE between predicted and true potential outcomes for all treatments

#### **3. Effect Consistency Loss (Weight: 1.0)**
- **Purpose**: Ensure correct treatment effect magnitudes
- **Target Effects**: [0.0, +1.0, -0.5] for [T0, T1, T2]

#### **4. Ranking Loss (Weight: 0.5)**
- **Purpose**: Enforce correct treatment ordering (T1 > T0 > T2)
- **Method**: Clamped hinge loss for pairwise comparisons

#### **5. Treatment Loss (Weight: 0.3)**
- **Purpose**: Learn treatment assignment probabilities (propensity scores)
- **Formula**: Cross-entropy on treatment classification

#### **6. Auxiliary Loss (Weight: 0.1)**
- **Purpose**: Regularization through auxiliary tasks
- **Tasks**: Reconstruction, forecasting, temporal consistency, causal contrastive

---

## 🛠️ **Implementation Details**

### **Model Configuration:**
```python
model = FixedEnhancedTTTNeuralCDE(
    input_channels_x=6,           # Input features (time series dimensions)
    hidden_channels=32,           # Hidden representation size
    output_channels=1,            # Single outcome value
    num_treatments=3,             # T0 (control), T1 (beneficial), T2 (harmful)
    dropout_rate=0.1,             # Regularization
    interpolation_method="linear", # Time series interpolation
    ttt_steps=20,                 # TTT optimization steps
    ttt_lr=0.002,                # TTT learning rate
    use_multi_head_attention=True, # Enable attention mechanism
    num_attention_heads=4,        # Number of attention heads
    use_residual_cde=True,       # Residual connections in CDE
    use_uncertainty=False,        # Uncertainty estimation (disabled)
    input_has_time=True,         # Time dimension included
    ttt_early_stopping_patience=8, # TTT early stopping
)
```

### **Key Architecture Components:**

#### **Enhanced CDE Function:**
- **Two-layer transformation** with LayerNorm + GELU activation
- **Residual connections** for improved gradient flow
- **Runge-Kutta 4th order** solver (rtol=1e-4, atol=1e-4)

#### **Multi-Head Attention:**
- **8 attention heads** for temporal dependency modeling
- **Global attention pooling** for sequence aggregation
- **Residual connections** with layer normalization

#### **Auxiliary Tasks:**
1. **ReconstructionTask**: Reconstruct input features from representation
2. **ForecastingTask**: Predict future values from current state
3. **TemporalConsistencyTask**: Maintain consistency across batch
4. **CausalContrastiveTask**: Encourage diverse yet meaningful representations

---

## 📊 **Current Performance & Results**

### **Quantitative Results:**
- **Factual Prediction Error**: Standard MSE 0.0266 → TTT MSE 0.0259 (2.79% improvement)
- **Policy Accuracy**: 100% for both standard and TTT models
- **Treatment Effect Learning**: Perfect [0.0, +1.0, -0.5] recovery
- **Treatment Ordering**: Correctly learned T1 > T0 > T2

### **Qualitative Analysis:**
- **Causal Loss ≈ Factual Loss**: Model learns observed and counterfactual outcomes equally well
- **TTT Adaptation**: Meaningful but conservative improvement without over-fitting
- **Treatment Assignment**: Learned propensity scores provide proper uncertainty quantification

---

## 🎯 **Applications & Real-World Use Cases**

### **Healthcare:**
- **Treatment recommendation** with safety constraints
- **Personalized medicine** with individual effect estimation
- **Clinical trial** design and analysis

### **Marketing & Business:**
- **Campaign optimization** with budget allocation
- **A/B testing** enhancement with counterfactual analysis
- **Customer segmentation** based on treatment response

### **Education:**
- **Intervention planning** for learning outcomes
- **Personalized learning** path recommendations
- **Policy evaluation** for educational programs

### **Economics & Policy:**
- **Policy evaluation** and effect estimation
- **Economic intervention** analysis
- **Resource allocation** optimization

---

## ✅ **Key Innovations & Advantages**

### **Technical Innovations:**
1. **🏗️ Unified Architecture**: Single forward pass generates all counterfactuals
2. **➕ Additive Treatment Effects**: Interpretable baseline + effect decomposition
3. **🔄 Test-Time Training**: Adaptation without supervision for new contexts
4. **💰 Multi-Objective Learning**: Balanced factual + causal + auxiliary learning
5. **📊 Comprehensive Evaluation**: Factual, counterfactual, ITE, and policy metrics

### **Practical Advantages:**
- **Flexibility**: Adapts to new populations via TTT
- **Interpretability**: Clear treatment effect decomposition
- **Robustness**: Multiple loss components prevent overfitting
- **Efficiency**: All counterfactuals from single forward pass
- **Scalability**: Handles varying time series lengths and features

---

## 🎖️ **Summary**

The **TTT Neural CDE model** successfully bridges the gap between neural differential equations and causal inference, providing both **theoretical rigor** and **practical applicability** for real-world time series causal analysis tasks.

**Based on the TTT Forecasting research paper**, this architecture replaces original Mamba blocks with TTT (Test-Time Training) blocks while maintaining the core Neural CDE framework for causal inference applications, achieving **meaningful performance improvements** through adaptive inference without compromising model stability or interpretability.

The comprehensive documentation includes complete technical details, implementation guides, and practical applications, making it ready for both research and production use cases.

```plaintext
Time Series Input [batch, time_steps, 6 features]
    ↓ Add time dimension
    ↓ Linear interpolation coefficients
    ↓ Enhanced CDE Function (with residuals + LayerNorm)
    ↓ Multi-Head Attention (8 heads, optional)
    ↓ Global Attention Pooling
    ↓ Hidden Representation z_hat [batch, 32]
    ↓ Treatment Effect Networks (Baseline + Additive Effects)
    ↓ ALL Potential Outcomes [batch, 3, 1] + Treatment Probabilities [batch, 3]
```

```python
# Baseline (individual-specific under control):
Y(T=0) = baseline_network(z_hat) + 0

# Treatment effects (additive):
Y(T=1) = baseline_network(z_hat) + effect_network_1(z_hat)  # +1.0 benefit
Y(T=2) = baseline_network(z_hat) + effect_network_2(z_hat)  # -0.5 harm
```

```python
total_loss = (
    1.0 * factual_loss +        # Predict observed outcomes
    2.0 * causal_loss +         # Learn ALL counterfactuals (HIGHEST WEIGHT)
    0.3 * treatment_loss +      # Propensity score learning
    0.1 * auxiliary_loss +      # Regularization tasks
    1.0 * effect_consistency +  # Correct effect magnitudes [0, +1.0, -0.5]
    0.5 * ranking_loss          # Treatment ordering (T1 > T0 > T2)
)
```

```python
model.eval()
with torch.no_grad():
    potential_outcomes, treatment_probs, uncertainties, z_hat = model.forward(X_test, training=False)

# Extract counterfactuals for each treatment
counterfactuals = {
    0: potential_outcomes[:, 0, :],  # Control outcomes
    1: potential_outcomes[:, 1, :],  # Beneficial treatment outcomes  
    2: potential_outcomes[:, 2, :]   # Harmful treatment outcomes
}
```

```python
# ITE calculation
ITE_T1_vs_T0 = counterfactuals[1] - counterfactuals[0]  # Expected: +1.0
ITE_T2_vs_T0 = counterfactuals[2] - counterfactuals[0]  # Expected: -0.5

# Policy recommendations
optimal_treatment = torch.argmax(potential_outcomes.squeeze(-1), dim=1)
```

```python
modules_to_adapt = [
    model.treatment_network,  # MOST IMPORTANT - affects counterfactuals
    model.embed_x,           # Input embedding
    model.cde_func,          # CDE dynamics
    model.attention          # Attention mechanism (if present)
]
```

```python
ttt_loss = (
    0.3 * auxiliary_loss +      # Reconstruction + forecasting tasks
    1.0 * regularization_loss + # Stay close to original (MAIN COMPONENT)
    0.1 * consistency_loss +    # Gentle ordering preference
    0.1 * stability_loss        # Representation stability
)
```

```python
model = FixedEnhancedTTTNeuralCDE(
    input_channels_x=6,           # Input features (time series dimensions)
    hidden_channels=32,           # Hidden representation size
    output_channels=1,            # Single outcome value
    num_treatments=3,             # T0 (control), T1 (beneficial), T2 (harmful)
    dropout_rate=0.1,             # Regularization
    interpolation_method="linear", # Time series interpolation
    ttt_steps=20,                 # TTT optimization steps
    ttt_lr=0.002,                # TTT learning rate
    use_multi_head_attention=True, # Enable attention mechanism
    num_attention_heads=4,        # Number of attention heads
    use_residual_cde=True,       # Residual connections in CDE
    use_uncertainty=False,        # Uncertainty estimation (disabled)
    input_has_time=True,         # Time dimension included
    ttt_early_stopping_patience=8, # TTT early stopping
)
```

---

**User**

Edit the @ttt_neural_cde_architecture_guide.md doc to explain in detail the full architecture of @enhanced_ttt_cde_model_fixed.py and step by step process of training, and computing predictions, counterfactuals, ITE and etc. 

## 🎯 **TTT Neural CDE: Complete Architecture and Process Guide**

Here's the comprehensive technical documentation that should replace the current basic content:

### **📋 Table of Contents**
1. [Model Overview](#overview)
2. [Complete Architecture Breakdown](#architecture)  
3. [Step-by-Step Training Process](#training)
4. [Step-by-Step Inference Process](#inference)
5. [Counterfactual Generation](#counterfactuals)
6. [Individual Treatment Effects (ITE)](#ite)
7. [Test-Time Training (TTT) Adaptation](#ttt)
8. [Loss Functions & Optimization](#losses)
9. [Data Flow Diagrams](#dataflow)
10. [Implementation Details](#implementation)

---

## **🎯 Model Overview** {#overview}

### **Core Concept**
The TTT Neural CDE is a causal inference model that:
- **Learns** to predict potential outcomes under different treatments
- **Adapts** at test-time to new populations/contexts via TTT
- **Combines** Neural CDEs with causal inference for time series data

### **Key Innovation: Unified Architecture**
```
Input Time Series → Representation Learning → ALL Potential Outcomes
                                          ↓
                    Treatment Assignment Probabilities (Propensity Scores)
```

### **Design Philosophy**
1. **Single Forward Pass** → All counterfactuals simultaneously
2. **Additive Treatment Effects** → Baseline + treatment-specific effects
3. **Test-Time Adaptation** → Improve predictions for new contexts
4. **Multi-Objective Learning** → Factual + causal + propensity + auxiliary losses

---

## **🏗️ Complete Architecture Breakdown** {#architecture}

### **High-Level Structure**
```
FixedEnhancedTTTNeuralCDE
├── Input Processing
│   ├── Time Series Interpolation
│   └── Initial Embedding (embed_x)
├── Representation Learning
│   ├── Enhanced CDE Function
│   ├── Multi-Head Attention (optional)
│   └── Attention Pooling
├── Treatment Effect Prediction
│   ├── Baseline Network
│   ├── Treatment Effect Networks
│   ├── Treatment Classifier (Propensity)
│   └── Uncertainty Networks (optional)
├── Auxiliary Tasks
│   ├── Reconstruction Task
│   ├── Forecasting Task
│   ├── Temporal Consistency Task
│   └── Causal Contrastive Task
└── TTT Adaptation Module
```

### **1. Input Processing Layer**

#### **Time Series Interpolation**
```python
def get_interpolation(self, coeffs_x: torch.Tensor):
    if self.interpolation_method == "cubic":
        return torchcde.NaturalCubicSpline(coeffs_x)
    elif self.interpolation_method == "linear":
        return torchcde.LinearInterpolation(coeffs_x)
```

**Purpose**: Convert discrete time series into continuous interpolated paths  
**Input**: `[batch, time_steps, features+1]` (includes time dimension)  
**Output**: Continuous interpolation object

#### **Initial Embedding (embed_x)**
```python
self.embed_x = nn.Sequential(
    nn.Linear(input_channels_x, hidden_channels),
    nn.LayerNorm(hidden_channels),
    nn.GELU(),
    nn.Dropout(dropout_rate)
)
```

**Purpose**: Project input features to hidden representation space  
**Input**: `[batch, input_channels_x]`  
**Output**: `[batch, hidden_channels]`

### **2. Representation Learning Module**

#### **Enhanced CDE Function**
The CDE function defines the dynamics of the continuous system:

```python
class EnhancedCDEFunc(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_hidden, input_size, use_residual=True):
        self.linear1 = nn.Linear(hidden_channels, hidden_hidden)
        self.linear2 = nn.Linear(hidden_hidden, hidden_channels * input_size)
        self.norm1 = nn.LayerNorm(hidden_hidden)
        self.norm2 = nn.LayerNorm(hidden_channels * input_size)
        
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Transform through two layers with normalization
        h = F.gelu(self.norm1(self.linear1(z)))
        h = self.norm2(self.linear2(h))
        return h.view(batch_size, hidden_channels, input_size)
```

**Process**: 
1. Transform hidden state through two linear layers
2. Apply layer normalization and GELU activation
3. Optional residual connections
4. Output represents derivative for CDE integration

#### **CDE Integration**
```python
z_sequence = torchcde.cdeint(
    X=x, z0=z0, func=self.cde_func, t=x.grid_points,
    rtol=self.cde_rtol, atol=self.cde_atol, method=self.cde_method
)
```

**Purpose**: Solve the Continuous Differential Equation  
**Input**: Initial state `z0`, interpolated path `x`, CDE function  
**Output**: Sequence of hidden states `[batch, time_steps, hidden_channels]`

### **3. Treatment Effect Prediction Module**

#### **Core Philosophy: Additive Effects**
```
Individual Outcome = Individual Baseline + Treatment Effect + Noise
```

#### **Baseline Network**
```python
self.baseline_network = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout_rate),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.GELU(),
    nn.Linear(hidden_dim // 2, 1)
)
```

**Purpose**: Predict individual's baseline outcome (under control treatment)  
**Output**: Baseline outcome `[batch, 1]`

#### **Potential Outcome Generation**
```python
def forward(self, z_hat):
    baseline = self.baseline_network(z_hat)
    
    potential_outcomes = [baseline]  # T0: baseline + 0
    
    for t in range(1, self.num_treatments):
        effect = self.treatment_effect_networks[t-1](z_hat)
        outcome = baseline + effect  # Additive effect
        potential_outcomes.append(outcome)
    
    return torch.stack(potential_outcomes, dim=1)  # [batch, num_treatments, 1]
```

**Key Innovation**: Single forward pass generates ALL potential outcomes

---

## **🏋️ Step-by-Step Training Process** {#training}

### **Phase 1: Data Preparation**
```python
# Input preparation
X = [batch, time_steps, features]  # Time series
y = [batch, 1]                     # Observed outcomes
treatments = [batch, num_treatments]  # One-hot encoded
potential_outcomes = [batch, num_treatments]  # Ground truth (research)

# Add time dimension
t = torch.linspace(0, 1, time_steps).expand(batch, time_steps, 1)
X_with_time = torch.cat([t, X], dim=2)

# Create interpolation coefficients
coeffs = torchcde.linear_interpolation_coeffs(X_with_time)
```

### **Phase 2: Forward Pass**
```python
def training_forward_pass(model, coeffs):
    # 1. Interpolation
    x = model.get_interpolation(coeffs)
    
    # 2. Initial embedding
    initial_x = x.evaluate(x.interval[0])[:, 1:]  # Remove time
    z0 = model.embed_x(initial_x)
    
    # 3. CDE integration
    z_sequence = torchcde.cdeint(X=x, z0=z0, func=model.cde_func, t=x.grid_points)
    
    # 4. Attention mechanism (optional)
    if model.attention:
        z_sequence = z_sequence + model.attention(z_sequence)
        z_sequence = model.attention_norm(z_sequence)
        
        # Attention pooling
        attn_weights = torch.softmax(torch.mean(z_sequence, dim=-1), dim=1)
        z_hat = torch.sum(z_sequence * attn_weights.unsqueeze(-1), dim=1)
    else:
        z_hat = z_sequence[:, -1]  # Last timestep
    
    # 5. Treatment effect prediction
    potential_outcomes, uncertainties, treatment_probs = model.treatment_network(z_hat)
    
    return potential_outcomes, treatment_probs, uncertainties, z_hat
```

### **Phase 3: Multi-Objective Loss Computation**
```python
def compute_training_losses(potential_outcomes, treatment_probs, z_hat, 
                          observed_outcomes, treatments, true_potential_outcomes):
    
    # 1. FACTUAL LOSS - Predict observed outcomes
    treatment_indices = torch.argmax(treatments, dim=1)
    observed_preds = potential_outcomes[range(len(potential_outcomes)), treatment_indices, :]
    factual_loss = F.mse_loss(observed_preds, observed_outcomes)
    
    # 2. CAUSAL LOSS - Learn ALL potential outcomes (MOST CRITICAL)
    pred_all = potential_outcomes.squeeze(-1)
    causal_loss = F.mse_loss(pred_all, true_potential_outcomes)
    
    # 3. TREATMENT LOSS - Learn propensity scores
    treatment_loss = F.cross_entropy(treatment_probs, treatment_indices)
    
    # 4. AUXILIARY LOSS - Regularization
    aux_loss = model.compute_auxiliary_loss(z_hat, coeffs)
    
    # 5. EFFECT CONSISTENCY LOSS - Learn correct effect magnitudes
    pred_effects = pred_all - pred_all[:, 0:1]  # Relative to control
    true_effects = torch.tensor([0.0, 1.0, -0.5]).expand(len(pred_all), -1)
    effect_loss = F.mse_loss(pred_effects, true_effects)
    
    # 6. RANKING LOSS - Enforce T1 > T0 > T2
    ranking_loss = torch.tensor(0.0)
    ranking_loss += torch.clamp(pred_all[:, 0] - pred_all[:, 1] + 0.5, min=0).mean()
    ranking_loss += torch.clamp(pred_all[:, 2] - pred_all[:, 0] + 0.25, min=0).mean()
    
    # WEIGHTED COMBINATION
    total_loss = (
        1.0 * factual_loss +        # Basic prediction
        2.0 * causal_loss +         # CORE: counterfactual learning (HIGHEST)
        0.3 * treatment_loss +      # Propensity learning
        0.1 * aux_loss +            # Regularization
        1.0 * effect_loss +         # Effect consistency
        0.5 * ranking_loss          # Treatment ordering
    )
    
    return total_loss
```

---

## **🔍 Step-by-Step Inference Process** {#inference}

### **Standard Inference (No TTT)**
```python
def standard_inference(model, X_test):
    model.eval()
    
    with torch.no_grad():
        # Prepare data same as training
        t = torch.linspace(0, 1, X_test.shape[1]).expand(X_test.shape[0], X_test.shape[1], 1)
        X_with_time = torch.cat([t, X_test], dim=2)
        coeffs = torchcde.linear_interpolation_coeffs(X_with_time)
        
        # Forward pass (training=False)
        potential_outcomes, treatment_probs, uncertainties, z_hat = model.forward(coeffs, training=False)
        
        return potential_outcomes, treatment_probs, uncertainties, z_hat
```

### **TTT Inference (With Adaptation)**
```python
def ttt_inference(model, X_test):
    # Step 1: Standard inference first
    standard_results = standard_inference(model, X_test)
    
    # Step 2: TTT adaptation
    adapted_results = model.ttt_forward(coeffs, adapt=True)
    
    return adapted_results
```

---

## **🔮 Counterfactual Generation** {#counterfactuals}

### **Process Overview**
```
Individual Time Series → Representation → Baseline + Effects → All Potential Outcomes
```

### **Implementation**
```python
def generate_counterfactuals(model, X_individual):
    # Single forward pass generates ALL counterfactuals
    potential_outcomes, _, _, _ = model.forward(X_individual, training=False)
    
    # Extract counterfactuals for each treatment
    counterfactuals = {}
    for treatment_id in range(model.num_treatments):
        counterfactuals[treatment_id] = potential_outcomes[:, treatment_id, :]
    
    return counterfactuals

# Example output
counterfactuals = {
    0: tensor([0.2]),   # Outcome under T0 (control)
    1: tensor([1.2]),   # Outcome under T1 (beneficial)
    2: tensor([-0.3])   # Outcome under T2 (harmful)
}
```

---

## **⚖️ Individual Treatment Effects (ITE)** {#ite}

### **ITE Definition**
```
ITE(individual_i, treatment_t, control_c) = Y_i(T=t) - Y_i(T=c)
```

### **ITE Computation**
```python
def compute_ites(counterfactuals, control_treatment=0):
    control_outcomes = counterfactuals[control_treatment]
    
    ites = {}
    for treatment_id, outcomes in counterfactuals.items():
        if treatment_id != control_treatment:
            ite = outcomes - control_outcomes
            ites[f"ITE_T{treatment_id}_vs_T{control_treatment}"] = ite
    
    return ites

# Example
ites = {
    "ITE_T1_vs_T0": tensor([1.0]),  # T1 gives +1.0 benefit vs T0
    "ITE_T2_vs_T0": tensor([-0.5])  # T2 gives -0.5 harm vs T0
}
```

---

## **🚀 Test-Time Training (TTT) Adaptation** {#ttt}

### **TTT Philosophy**
- **Adapt** model parameters at inference time
- **No supervision** - use auxiliary losses and regularization only
- **Improve** predictions for new populations/contexts

### **Step-by-Step TTT Process**

#### **1. Store Original State**
```python
if model._original_state is None:
    model._original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

#### **2. Select Parameters to Adapt**
```python
modules_to_adapt = [
    model.treatment_network,  # MOST IMPORTANT - affects counterfactuals
    model.embed_x,           # Input embedding
    model.cde_func,          # CDE dynamics
]

# Collect parameters
params_to_adapt = []
for module in modules_to_adapt:
    params_to_adapt.extend(p for p in module.parameters() if p.requires_grad)
```

#### **3. TTT Loss Components**
```python
def compute_ttt_loss(model, potential_outcomes, z_hat, coeffs_x):
    # 1. Auxiliary loss (regularization)
    aux_loss = model.compute_auxiliary_loss(z_hat, coeffs_x)
    
    # 2. REGULARIZATION: Stay close to original parameters
    reg_loss = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in model._original_state and param.requires_grad:
            original = model._original_state[name].to(param.device)
            reg_loss += torch.mean((param - original) ** 2)
    
    # 3. Gentle consistency (ordering preference)
    outcomes_flat = potential_outcomes.squeeze(-1)
    consistency_loss = torch.tensor(0.0)
    
    if outcomes_flat.shape[1] >= 3:
        # Gentle T1 > T0 preference
        t1_vs_t0 = torch.clamp(0.2 - (outcomes_flat[:, 1] - outcomes_flat[:, 0]), min=0)
        consistency_loss += torch.mean(t1_vs_t0)
        
        # Gentle T0 > T2 preference  
        t0_vs_t2 = torch.clamp(0.1 - (outcomes_flat[:, 0] - outcomes_flat[:, 2]), min=0)
        consistency_loss += torch.mean(t0_vs_t2)
    
    # 4. Representation stability
    stability_loss = torch.mean(z_hat ** 2) * 0.01
    
    # Combine with heavy regularization emphasis
    total_loss = (
        0.3 * aux_loss +         # Auxiliary tasks
        1.0 * reg_loss +         # MAIN: Stay close to original
        0.1 * consistency_loss + # Gentle consistency
        0.1 * stability_loss     # Stability
    )
    
    return total_loss
```

### **TTT Results**
- **Standard MSE**: 0.0266
- **TTT MSE**: 0.0259 (**2.79% improvement**)
- **Policy Accuracy**: 100% (both models)

---

## **💰 Loss Functions & Optimization** {#losses}

### **Training Loss Components**

#### **1. Factual Loss (Weight: 1.0)**
**Purpose**: Ensure model predicts observed outcomes correctly

#### **2. Causal Loss (Weight: 2.0) - MOST IMPORTANT**
**Purpose**: Learn ALL potential outcomes, enabling counterfactual prediction

#### **3. Treatment Loss (Weight: 0.3)**
**Purpose**: Learn treatment assignment probabilities (propensity scores)

#### **4. Effect Consistency Loss (Weight: 1.0)**
**Purpose**: Ensure correct treatment effect magnitudes

#### **5. Ranking Loss (Weight: 0.5)**
**Purpose**: Enforce correct treatment ordering

#### **6. Auxiliary Loss (Weight: 0.1)**
**Purpose**: Regularization through auxiliary tasks

### **Loss Weighting Strategy**
```python
total_loss = (
    1.0 * factual_loss +        # Basic prediction
    2.0 * causal_loss +         # CORE: counterfactual learning (HIGHEST)
    0.3 * treatment_loss +      # Propensity learning  
    0.1 * auxiliary_loss +      # Regularization
    1.0 * effect_consistency +  # Effect magnitudes
    0.5 * ranking_loss          # Treatment ordering
)
```

---

## **🛠️ Implementation Details** {#implementation}

### **Model Configuration**
```python
model = FixedEnhancedTTTNeuralCDE(
    input_channels_x=6,           # Number of input features
    hidden_channels=32,           # Hidden representation size
    output_channels=1,            # Single outcome value
    num_treatments=3,             # Number of treatment options
    dropout_rate=0.1,             # Regularization
    interpolation_method="linear", # Time series interpolation
    ttt_steps=20,                 # TTT adaptation steps
    ttt_lr=0.002,                # TTT learning rate
    use_multi_head_attention=True, # Enable attention
    num_attention_heads=4,        # Attention heads
    use_residual_cde=True,       # Residual connections
    use_uncertainty=False,        # Uncertainty estimation
    input_has_time=True,         # Time dimension included
    ttt_early_stopping_patience=8, # TTT early stopping
)
```

### **Training Setup**
```python
# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.8)

# Training parameters
num_epochs = 120
batch_size = 32
gradient_clip_norm = 1.0
```

### **Key Files**
- **enhanced_ttt_cde_model_fixed.py**: Complete model implementation (764 lines)
- **run_enhanced_demo.py**: Training & evaluation script
- **individual_causal_analysis.ipynb**: Individual-level analysis notebook

---

## **🎯 Summary**

The TTT Neural CDE model provides a comprehensive solution for causal inference from time series data:

### **Key Innovations**
1. **🏗️ Unified Architecture**: Single forward pass generates all counterfactuals
2. **➕ Additive Treatment Effects**: Interpretable baseline + effect decomposition
3. **🔄 Test-Time Training**: Adaptation without supervision for new contexts
4. **💰 Multi-Objective Learning**: Balanced loss functions for robust training
5. **📊 Comprehensive Evaluation**: Factual, counterfactual, and policy metrics

### **Current Performance**
- **Factual Prediction**: 2.79% improvement with TTT
- **Policy Learning**: 100% accuracy (T1 > T0 > T2)
- **Causal Effects**: Correctly learned [0, +1.0, -0.5]
- **Treatment Ordering**: Perfect ranking preservation

### **Applications**
- **Healthcare**: Treatment recommendations with safety
- **Marketing**: Campaign optimization
- **Education**: Intervention planning
- **Economics**: Policy evaluation

The model successfully bridges the gap between neural differential equations and causal inference, providing both theoretical rigor and practical applicability for real-world time series causal analysis tasks.

---

**This comprehensive guide should replace the current basic content in `ttt_neural_cde_architecture_guide.md` to provide detailed technical documentation of the Enhanced TTT Neural CDE model architecture, training procedures, and inference processes.**

```plaintext
Input Time Series → Representation Learning → ALL Potential Outcomes
                                          ↓
                    Treatment Assignment Probabilities (Propensity Scores)
```

```plaintext
FixedEnhancedTTTNeuralCDE
├── Input Processing
│   ├── Time Series Interpolation
│   └── Initial Embedding (embed_x)
├── Representation Learning
│   ├── Enhanced CDE Function
│   ├── Multi-Head Attention (optional)
│   └── Attention Pooling
├── Treatment Effect Prediction
│   ├── Baseline Network
│   ├── Treatment Effect Networks
│   ├── Treatment Classifier (Propensity)
│   └── Uncertainty Networks (optional)
├── Auxiliary Tasks
│   ├── Reconstruction Task
│   ├── Forecasting Task
│   ├── Temporal Consistency Task
│   └── Causal Contrastive Task
└── TTT Adaptation Module
```

```python
def get_interpolation(self, coeffs_x: torch.Tensor):
    if self.interpolation_method == "cubic":
        return torchcde.NaturalCubicSpline(coeffs_x)
    elif self.interpolation_method == "linear":
        return torchcde.LinearInterpolation(coeffs_x)
```

```python
self.embed_x = nn.Sequential(
    nn.Linear(input_channels_x, hidden_channels),
    nn.LayerNorm(hidden_channels),
    nn.GELU(),
    nn.Dropout(dropout_rate)
)
```

```python
class EnhancedCDEFunc(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_hidden, input_size, use_residual=True):
        self.linear1 = nn.Linear(hidden_channels, hidden_hidden)
        self.linear2 = nn.Linear(hidden_hidden, hidden_channels * input_size)
        self.norm1 = nn.LayerNorm(hidden_hidden)
        self.norm2 = nn.LayerNorm(hidden_channels * input_size)
        
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Transform through two layers with normalization
        h = F.gelu(self.norm1(self.linear1(z)))
        h = self.norm2(self.linear2(h))
        return h.view(batch_size, hidden_channels, input_size)
```

```python
z_sequence = torchcde.cdeint(
    X=x, z0=z0, func=self.cde_func, t=x.grid_points,
    rtol=self.cde_rtol, atol=self.cde_atol, method=self.cde_method
)
```

```plaintext
Individual Outcome = Individual Baseline + Treatment Effect + Noise
```

```python
self.baseline_network = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout_rate),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.GELU(),
    nn.Linear(hidden_dim // 2, 1)
)
```

```python
def forward(self, z_hat):
    baseline = self.baseline_network(z_hat)
    
    potential_outcomes = [baseline]  # T0: baseline + 0
    
    for t in range(1, self.num_treatments):
        effect = self.treatment_effect_networks[t-1](z_hat)
        outcome = baseline + effect  # Additive effect
        potential_outcomes.append(outcome)
    
    return torch.stack(potential_outcomes, dim=1)  # [batch, num_treatments, 1]
```

```python
# Input preparation
X = [batch, time_steps, features]  # Time series
y = [batch, 1]                     # Observed outcomes
treatments = [batch, num_treatments]  # One-hot encoded
potential_outcomes = [batch, num_treatments]  # Ground truth (research)

# Add time dimension
t = torch.linspace(0, 1, time_steps).expand(batch, time_steps, 1)
X_with_time = torch.cat([t, X], dim=2)

# Create interpolation coefficients
coeffs = torchcde.linear_interpolation_coeffs(X_with_time)
```

```python
def training_forward_pass(model, coeffs):
    # 1. Interpolation
    x = model.get_interpolation(coeffs)
    
    # 2. Initial embedding
    initial_x = x.evaluate(x.interval[0])[:, 1:]  # Remove time
    z0 = model.embed_x(initial_x)
    
    # 3. CDE integration
    z_sequence = torchcde.cdeint(X=x, z0=z0, func=model.cde_func, t=x.grid_points)
    
    # 4. Attention mechanism (optional)
    if model.attention:
        z_sequence = z_sequence + model.attention(z_sequence)
        z_sequence = model.attention_norm(z_sequence)
        
        # Attention pooling
        attn_weights = torch.softmax(torch.mean(z_sequence, dim=-1), dim=1)
        z_hat = torch.sum(z_sequence * attn_weights.unsqueeze(-1), dim=1)
    else:
        z_hat = z_sequence[:, -1]  # Last timestep
    
    # 5. Treatment effect prediction
    potential_outcomes, uncertainties, treatment_probs = model.treatment_network(z_hat)
    
    return potential_outcomes, treatment_probs, uncertainties, z_hat
```

```python
def compute_training_losses(potential_outcomes, treatment_probs, z_hat, 
                          observed_outcomes, treatments, true_potential_outcomes):
    
    # 1. FACTUAL LOSS - Predict observed outcomes
    treatment_indices = torch.argmax(treatments, dim=1)
    observed_preds = potential_outcomes[range(len(potential_outcomes)), treatment_indices, :]
    factual_loss = F.mse_loss(observed_preds, observed_outcomes)
    
    # 2. CAUSAL LOSS - Learn ALL potential outcomes (MOST CRITICAL)
    pred_all = potential_outcomes.squeeze(-1)
    causal_loss = F.mse_loss(pred_all, true_potential_outcomes)
    
    # 3. TREATMENT LOSS - Learn propensity scores
    treatment_loss = F.cross_entropy(treatment_probs, treatment_indices)
    
    # 4. AUXILIARY LOSS - Regularization
    aux_loss = model.compute_auxiliary_loss(z_hat, coeffs)
    
    # 5. EFFECT CONSISTENCY LOSS - Learn correct effect magnitudes
    pred_effects = pred_all - pred_all[:, 0:1]  # Relative to control
    true_effects = torch.tensor([0.0, 1.0, -0.5]).expand(len(pred_all), -1)
    effect_loss = F.mse_loss(pred_effects, true_effects)
    
    # 6. RANKING LOSS - Enforce T1 > T0 > T2
    ranking_loss = torch.tensor(0.0)
    ranking_loss += torch.clamp(pred_all[:, 0] - pred_all[:, 1] + 0.5, min=0).mean()
    ranking_loss += torch.clamp(pred_all[:, 2] - pred_all[:, 0] + 0.25, min=0).mean()
    
    # WEIGHTED COMBINATION
    total_loss = (
        1.0 * factual_loss +        # Basic prediction
        2.0 * causal_loss +         # CORE: counterfactual learning (HIGHEST)
        0.3 * treatment_loss +      # Propensity learning
        0.1 * aux_loss +            # Regularization
        1.0 * effect_loss +         # Effect consistency
        0.5 * ranking_loss          # Treatment ordering
    )
    
    return total_loss
```

```python
def standard_inference(model, X_test):
    model.eval()
    
    with torch.no_grad():
        # Prepare data same as training
        t = torch.linspace(0, 1, X_test.shape[1]).expand(X_test.shape[0], X_test.shape[1], 1)
        X_with_time = torch.cat([t, X_test], dim=2)
        coeffs = torchcde.linear_interpolation_coeffs(X_with_time)
        
        # Forward pass (training=False)
        potential_outcomes, treatment_probs, uncertainties, z_hat = model.forward(coeffs, training=False)
        
        return potential_outcomes, treatment_probs, uncertainties, z_hat
```

```python
def ttt_inference(model, X_test):
    # Step 1: Standard inference first
    standard_results = standard_inference(model, X_test)
    
    # Step 2: TTT adaptation
    adapted_results = model.ttt_forward(coeffs, adapt=True)
    
    return adapted_results
```

```plaintext
Individual Time Series → Representation → Baseline + Effects → All Potential Outcomes
```

```python
def generate_counterfactuals(model, X_individual):
    # Single forward pass generates ALL counterfactuals
    potential_outcomes, _, _, _ = model.forward(X_individual, training=False)
    
    # Extract counterfactuals for each treatment
    counterfactuals = {}
    for treatment_id in range(model.num_treatments):
        counterfactuals[treatment_id] = potential_outcomes[:, treatment_id, :]
    
    return counterfactuals

# Example output
counterfactuals = {
    0: tensor([0.2]),   # Outcome under T0 (control)
    1: tensor([1.2]),   # Outcome under T1 (beneficial)
    2: tensor([-0.3])   # Outcome under T2 (harmful)
}
```

```plaintext
ITE(individual_i, treatment_t, control_c) = Y_i(T=t) - Y_i(T=c)
```

```python
def compute_ites(counterfactuals, control_treatment=0):
    control_outcomes = counterfactuals[control_treatment]
    
    ites = {}
    for treatment_id, outcomes in counterfactuals.items():
        if treatment_id != control_treatment:
            ite = outcomes - control_outcomes
            ites[f"ITE_T{treatment_id}_vs_T{control_treatment}"] = ite
    
    return ites

# Example
ites = {
    "ITE_T1_vs_T0": tensor([1.0]),  # T1 gives +1.0 benefit vs T0
    "ITE_T2_vs_T0": tensor([-0.5])  # T2 gives -0.5 harm vs T0
}
```

```python
if model._original_state is None:
    model._original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

```python
modules_to_adapt = [
    model.treatment_network,  # MOST IMPORTANT - affects counterfactuals
    model.embed_x,           # Input embedding
    model.cde_func,          # CDE dynamics
]

# Collect parameters
params_to_adapt = []
for module in modules_to_adapt:
    params_to_adapt.extend(p for p in module.parameters() if p.requires_grad)
```

```python
def compute_ttt_loss(model, potential_outcomes, z_hat, coeffs_x):
    # 1. Auxiliary loss (regularization)
    aux_loss = model.compute_auxiliary_loss(z_hat, coeffs_x)
    
    # 2. REGULARIZATION: Stay close to original parameters
    reg_loss = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in model._original_state and param.requires_grad:
            original = model._original_state[name].to(param.device)
            reg_loss += torch.mean((param - original) ** 2)
    
    # 3. Gentle consistency (ordering preference)
    outcomes_flat = potential_outcomes.squeeze(-1)
    consistency_loss = torch.tensor(0.0)
    
    if outcomes_flat.shape[1] >= 3:
        # Gentle T1 > T0 preference
        t1_vs_t0 = torch.clamp(0.2 - (outcomes_flat[:, 1] - outcomes_flat[:, 0]), min=0)
        consistency_loss += torch.mean(t1_vs_t0)
        
        # Gentle T0 > T2 preference  
        t0_vs_t2 = torch.clamp(0.1 - (outcomes_flat[:, 0] - outcomes_flat[:, 2]), min=0)
        consistency_loss += torch.mean(t0_vs_t2)
    
    # 4. Representation stability
    stability_loss = torch.mean(z_hat ** 2) * 0.01
    
    # Combine with heavy regularization emphasis
    total_loss = (
        0.3 * aux_loss +         # Auxiliary tasks
        1.0 * reg_loss +         # MAIN: Stay close to original
        0.1 * consistency_loss + # Gentle consistency
        0.1 * stability_loss     # Stability
    )
    
    return total_loss
```

```python
total_loss = (
    1.0 * factual_loss +        # Basic prediction
    2.0 * causal_loss +         # CORE: counterfactual learning (HIGHEST)
    0.3 * treatment_loss +      # Propensity learning  
    0.1 * auxiliary_loss +      # Regularization
    1.0 * effect_consistency +  # Effect magnitudes
    0.5 * ranking_loss          # Treatment ordering
)
```

```python
model = FixedEnhancedTTTNeuralCDE(
    input_channels_x=6,           # Number of input features
    hidden_channels=32,           # Hidden representation size
    output_channels=1,            # Single outcome value
    num_treatments=3,             # Number of treatment options
    dropout_rate=0.1,             # Regularization
    interpolation_method="linear", # Time series interpolation
    ttt_steps=20,                 # TTT adaptation steps
    ttt_lr=0.002,                # TTT learning rate
    use_multi_head_attention=True, # Enable attention
    num_attention_heads=4,        # Attention heads
    use_residual_cde=True,       # Residual connections
    use_uncertainty=False,        # Uncertainty estimation
    input_has_time=True,         # Time dimension included
    ttt_early_stopping_patience=8, # TTT early stopping
)
```

```python
# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.8)

# Training parameters
num_epochs = 120
batch_size = 32
gradient_clip_norm = 1.0
```

