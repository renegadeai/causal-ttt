# Training Metrics Detailed Explanation

This document provides comprehensive explanations for the training loss components displayed during model training.

## 📊 **Training Log Example**
```
2025-06-09 17:03:42,805 - INFO - Epoch 51/120
2025-06-09 17:03:42,805 - INFO -   Total: 0.4125
2025-06-09 17:03:42,806 - INFO -   Factual: 0.0259
2025-06-09 17:03:42,806 - INFO -   Causal: 0.0305
2025-06-09 17:03:42,806 - INFO -   Treatment: 1.0437
2025-06-09 17:03:42,806 - INFO -   Effect Consistency: [value]
```

---

## 🎯 **1. Total Loss (0.4125)**

### **Description**
The **combined weighted loss** that the optimizer minimizes during training. This is the actual value used for backpropagation.

### **Formula**
```python
total_loss = (
    1.0 * factual_loss +           # Predict observed outcomes
    2.0 * causal_loss +            # Learn ALL potential outcomes (MOST IMPORTANT)
    0.3 * treatment_loss +         # Learn treatment propensity
    0.1 * aux_loss +               # Auxiliary tasks
    1.0 * effect_consistency_loss + # Learn correct effect sizes
    0.5 * ranking_loss             # Enforce correct ordering
)
```

### **Weighting Strategy**
- **Causal Loss (2.0x)**: Highest weight because learning potential outcomes is the core causal task
- **Factual + Effect Consistency (1.0x each)**: Essential for basic prediction and correct treatment effects
- **Ranking (0.5x)**: Moderate weight to enforce T1 > T0 > T2 ordering
- **Treatment (0.3x)**: Lower weight since propensity learning is secondary
- **Auxiliary (0.1x)**: Small weight for regularization only

### **Interpretation**
- **Decreasing**: Model is learning successfully
- **Range**: Typically starts around 2-5, converges to 0.3-0.8
- **Target**: Lower is better, but focus on individual components

---

## 🎯 **2. Factual Loss (0.0259)**

### **Description**
Measures how well the model predicts **observed outcomes** - the basic prediction task.

### **Formula**
```python
# Extract observed predictions for each individual's actual treatment
treatment_indices = torch.argmax(treatments, dim=1)
observed_predictions = potential_outcomes[range(batch_size), treatment_indices]
factual_loss = F.mse_loss(observed_predictions, observed_outcomes)
```

### **What It Measures**
- **Input**: Individual's covariates + their actual treatment
- **Output**: Predicted outcome for that treatment
- **Target**: The outcome that was actually observed
- **Metric**: Mean Squared Error (MSE)

### **Interpretation**
- **Low values (0.01-0.05)**: Good factual prediction
- **High values (>0.1)**: Poor basic prediction ability
- **Trend**: Should decrease steadily during training

### **Why Important**
If the model can't predict what actually happened, it definitely can't predict counterfactuals. This is the foundation.

---

## 🔮 **3. Causal Loss (0.0305)**

### **Description**
**Most critical loss** - teaches the model to predict potential outcomes under ALL treatments, not just observed ones.

### **Formula**
```python
# Predict potential outcomes for ALL treatments
pred_potential_outcomes = model.get_all_potential_outcomes(x)  # [batch, num_treatments]
true_potential_outcomes = ground_truth_potentials             # [batch, num_treatments]
causal_loss = F.mse_loss(pred_potential_outcomes, true_potential_outcomes)
```

### **What It Measures**
- **Input**: Individual's covariates
- **Output**: Predicted outcomes under T0, T1, T2
- **Target**: True potential outcomes under all treatments
- **Scope**: ALL treatments, including unobserved counterfactuals

### **Why Critical**
This is what makes it a **causal model**:
- Without this, the model only learns correlations from observed data
- With this, the model learns the true causal mechanisms
- Enables counterfactual prediction and policy learning

### **Interpretation**
- **Low values (0.02-0.05)**: Good causal learning
- **High values (>0.1)**: Poor counterfactual estimation
- **Should be similar to factual loss**: Both measure outcome prediction quality

---

## 👥 **4. Treatment Loss (1.0437)**

### **Description**
Learns **treatment assignment probabilities** (propensity scores) - which treatments are likely for which individuals.

### **Formula**
```python
# Model predicts treatment assignment probabilities
treatment_logits = model.treatment_classifier(representations)
treatment_probs = F.softmax(treatment_logits, dim=1)
treatment_loss = F.cross_entropy(treatment_probs, actual_treatments)
```

### **What It Measures**
- **Input**: Individual's covariates
- **Output**: Probability distribution over treatments [P(T0), P(T1), P(T2)]
- **Target**: The treatment that was actually assigned
- **Metric**: Cross-entropy (classification loss)

### **Purpose**
1. **Confounding Control**: Understanding why treatments were assigned
2. **Propensity Scores**: For doubly robust estimation
3. **Model Interpretability**: Understanding treatment assignment patterns

### **Interpretation**
- **Range**: 0 to ∞ (cross-entropy has no upper bound)
- **Values around 1.0**: Reasonable classification performance
- **High values (>2.0)**: Poor treatment assignment modeling
- **Low values (<0.5)**: Very good propensity estimation

### **Note on High Values**
Treatment loss can be higher than other losses because:
- Cross-entropy vs MSE have different scales
- Treatment assignment may have inherent randomness
- Focus on trend (decreasing) rather than absolute value

---

## ⚖️ **5. Effect Consistency Loss**

### **Description**
Ensures the model learns the **correct treatment effect magnitudes** relative to known ground truth.

### **Formula**
```python
# Predicted effects relative to control (T0)
pred_effects = pred_potential_outcomes - pred_potential_outcomes[:, 0:1]
# True effects: [0.0, 1.0, -0.5]
true_effects = torch.tensor([0.0, 1.0, -0.5]).expand_as(pred_effects)
effect_consistency_loss = F.mse_loss(pred_effects, true_effects)
```

### **What It Measures**
- **Input**: Predicted potential outcomes
- **Output**: Treatment effects relative to T0
- **Target**: Known true effects [0.0, +1.0, -0.5]
- **Purpose**: Explicit supervision on effect sizes

### **Why Needed**
Without this supervision, the model might learn:
- Correct ordering (T1 > T0 > T2) but wrong magnitudes
- Shifted outcomes (all +10 higher) with correct relative effects
- This loss anchors the absolute effect sizes

### **Interpretation**
- **Low values (0.01-0.05)**: Learning correct effect sizes
- **High values (>0.1)**: Poor effect estimation
- **Should decrease**: As model learns true treatment effects

---

## 📈 **6. Ranking Loss**

### **Description**
Explicitly enforces the **correct treatment ordering**: T1 > T0 > T2.

### **Formula**
```python
ranking_loss = 0
# T1 should be better than T0
t1_better_than_t0 = torch.clamp(pred_T0 - pred_T1 + margin, min=0)
ranking_loss += torch.mean(t1_better_than_t0)

# T0 should be better than T2  
t0_better_than_t2 = torch.clamp(pred_T2 - pred_T0 + margin, min=0)
ranking_loss += torch.mean(t0_better_than_t2)
```

### **What It Measures**
- **Violations**: When predicted ordering doesn't match T1 > T0 > T2
- **Margins**: Ensures sufficient separation between treatments
- **Penalty**: Higher when ordering is violated

### **Why Important**
- **Policy Learning**: Correct ordering is essential for treatment recommendations
- **Robustness**: Prevents local minima with wrong treatment rankings
- **Consistency**: Works with effect consistency loss for comprehensive supervision

### **Interpretation**
- **Zero**: Perfect ordering achieved
- **Low values (0.01-0.1)**: Minor ordering violations
- **High values (>0.2)**: Significant ranking problems

---

## 🔄 **7. Auxiliary Loss (0.1 weight)**

### **Description**
Regularization through auxiliary tasks like reconstruction, forecasting, and consistency.

### **Components**
```python
aux_loss = (
    reconstruction_loss +      # Reconstruct input features
    forecasting_loss +        # Predict future values
    temporal_consistency_loss + # Consistent representations
    causal_contrastive_loss   # Diverse representations
)
```

### **Purpose**
- **Regularization**: Prevent overfitting
- **Representation Learning**: Encourage useful hidden representations
- **Stability**: More robust training dynamics

### **Interpretation**
- **Small weight (0.1)**: Should not dominate training
- **Steady values**: Provides consistent regularization
- **Not critical**: Main focus should be on causal losses

---

## 📊 **Training Progress Analysis**

### **Healthy Training Patterns**
```
Epoch 1:   Total: 3.2508, Factual: 0.8123, Causal: 0.9456, Treatment: 1.2345
Epoch 20:  Total: 1.1234, Factual: 0.2567, Causal: 0.2890, Treatment: 1.1234
Epoch 50:  Total: 0.4125, Factual: 0.0259, Causal: 0.0305, Treatment: 1.0437
Epoch 100: Total: 0.3567, Factual: 0.0189, Causal: 0.0234, Treatment: 0.9876
```

### **Red Flags**
- **Factual/Causal loss not decreasing**: Poor outcome learning
- **Treatment loss increasing**: Degrading propensity estimation  
- **Total loss plateauing early**: Potential underfitting
- **Massive spikes**: Gradient explosion or learning rate issues

### **Success Indicators**
- **Steady decrease in all losses**: Good convergence
- **Factual ≈ Causal**: Consistent outcome prediction
- **Treatment loss stabilizing**: Good propensity learning
- **Total < 0.5**: Generally good performance

---

## 🎯 **Optimization Strategy**

### **Loss Weighting Rationale**
1. **Causal (2.0)**: Most important - core counterfactual learning
2. **Factual (1.0)**: Foundation - must predict observed data well
3. **Effect Consistency (1.0)**: Critical - ensures correct effect sizes
4. **Ranking (0.5)**: Important - enforces treatment ordering
5. **Treatment (0.3)**: Useful - propensity score learning
6. **Auxiliary (0.1)**: Regularization - prevents overfitting

### **Monitoring Priority**
1. **Causal Loss**: Primary metric for counterfactual quality
2. **Factual Loss**: Basic prediction capability
3. **Effect Consistency**: Treatment effect accuracy
4. **Total Loss**: Overall convergence
5. **Treatment Loss**: Propensity estimation quality

---

## 📚 **Key Takeaways**

1. **Causal Loss is King**: The most important metric for causal inference quality
2. **Multiple Objectives**: Each loss serves a specific purpose in causal learning
3. **Balanced Training**: Weights ensure no single objective dominates
4. **Trend Matters**: Focus on decreasing trends rather than absolute values
5. **Domain Knowledge**: Effect consistency and ranking encode expert knowledge
6. **Holistic View**: All losses together create a robust causal model

This multi-objective approach ensures the TTT Neural CDE learns not just to predict, but to understand and model causal relationships effectively. 