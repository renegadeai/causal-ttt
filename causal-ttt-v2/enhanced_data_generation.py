"""
Enhanced data generation utilities for causal time series forecasting.

This module provides sophisticated synthetic data generation with:
1. Realistic temporal dependencies
2. Confounding variables
3. Heterogeneous treatment effects
4. Distribution shifts
5. Missing data patterns
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration for data generation."""
    num_samples: int = 1000
    seq_len: int = 20
    num_features: int = 8
    num_treatments: int = 4
    output_dim: int = 1
    
    # Treatment assignment parameters
    treatment_assignment_type: str = "confounded"  # "random", "confounded", "policy"
    confounding_strength: float = 1.0
    
    # Temporal dynamics
    temporal_autocorr: float = 0.7
    temporal_noise_std: float = 0.1
    
    # Treatment effects
    treatment_effect_heterogeneity: float = 1.0
    treatment_effect_temporal: bool = True
    
    # Distribution shift parameters
    enable_distribution_shift: bool = False
    shift_magnitude: float = 0.5
    shift_start_ratio: float = 0.7
    
    # Missing data
    missing_data_prob: float = 0.0
    missing_pattern: str = "random"  # "random", "temporal", "feature_dependent"
    
    # Outcome model complexity
    outcome_nonlinearity: str = "moderate"  # "linear", "moderate", "high"
    
    # Seed for reproducibility
    seed: Optional[int] = None


class TemporalDynamicsGenerator:
    """Generates realistic temporal dynamics for time series."""
    
    def __init__(self, autocorr: float = 0.7, noise_std: float = 0.1):
        self.autocorr = autocorr
        self.noise_std = noise_std
    
    def generate_ar_process(self, num_samples: int, seq_len: int, 
                           num_features: int) -> np.ndarray:
        """Generate AR(1) process for each feature."""
        X = np.zeros((num_samples, seq_len, num_features))
        
        # Initialize with random values
        X[:, 0, :] = np.random.normal(0, 1, (num_samples, num_features))
        
        # Generate AR(1) process
        for t in range(1, seq_len):
            noise = np.random.normal(0, self.noise_std, (num_samples, num_features))
            X[:, t, :] = self.autocorr * X[:, t-1, :] + noise
        
        return X
    
    def add_seasonal_pattern(self, X: np.ndarray, period: int = 5,
                           amplitude: float = 0.5) -> np.ndarray:
        """Add seasonal patterns to time series."""
        seq_len = X.shape[1]
        time_points = np.arange(seq_len)
        
        # Add sinusoidal seasonal pattern
        seasonal = amplitude * np.sin(2 * np.pi * time_points / period)
        X_seasonal = X + seasonal[None, :, None]
        
        return X_seasonal
    
    def add_trend(self, X: np.ndarray, trend_strength: float = 0.1) -> np.ndarray:
        """Add linear trend to time series."""
        seq_len = X.shape[1]
        trend = trend_strength * np.arange(seq_len)
        X_trend = X + trend[None, :, None]
        
        return X_trend


class TreatmentAssignmentMechanism:
    """Generates treatment assignments with various mechanisms."""
    
    def __init__(self, assignment_type: str = "confounded", 
                 confounding_strength: float = 1.0):
        self.assignment_type = assignment_type
        self.confounding_strength = confounding_strength
    
    def assign_treatments(self, X: np.ndarray, 
                         num_treatments: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign treatments based on the specified mechanism.
        
        Args:
            X: Time series features [num_samples, seq_len, num_features]
            num_treatments: Number of treatment options
        
        Returns:
            treatments: Treatment assignments [num_samples]
            propensity_scores: Treatment propensity scores [num_samples, num_treatments]
        """
        num_samples = X.shape[0]
        
        if self.assignment_type == "random":
            return self._random_assignment(num_samples, num_treatments)
        elif self.assignment_type == "confounded":
            return self._confounded_assignment(X, num_treatments)
        elif self.assignment_type == "policy":
            return self._policy_based_assignment(X, num_treatments)
        else:
            raise ValueError(f"Unknown assignment type: {self.assignment_type}")
    
    def _random_assignment(self, num_samples: int, 
                          num_treatments: int) -> Tuple[np.ndarray, np.ndarray]:
        """Random treatment assignment."""
        treatments = np.random.randint(0, num_treatments, num_samples)
        propensity_scores = np.ones((num_samples, num_treatments)) / num_treatments
        
        return treatments, propensity_scores
    
    def _confounded_assignment(self, X: np.ndarray, 
                             num_treatments: int) -> Tuple[np.ndarray, np.ndarray]:
        """Treatment assignment based on observed confounders."""
        # Use last time step features as confounders
        confounders = X[:, -1, :]  # [num_samples, num_features]
        
        # Create confounding effects
        confounding_weights = np.random.normal(
            0, self.confounding_strength, 
            (confounders.shape[1], num_treatments)
        )
        
        # Compute logits
        logits = np.dot(confounders, confounding_weights)
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        propensity_scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Sample treatments
        treatments = np.array([
            np.random.choice(num_treatments, p=props) 
            for props in propensity_scores
        ])
        
        return treatments, propensity_scores
    
    def _policy_based_assignment(self, X: np.ndarray, 
                               num_treatments: int) -> Tuple[np.ndarray, np.ndarray]:
        """Treatment assignment based on a simple policy."""
        # Simple policy: assign treatment based on feature threshold
        feature_sum = np.sum(X[:, -1, :], axis=1)
        
        # Divide into quantiles
        quantiles = np.quantile(feature_sum, np.linspace(0, 1, num_treatments + 1))
        
        treatments = np.zeros(len(feature_sum), dtype=int)
        propensity_scores = np.zeros((len(feature_sum), num_treatments))
        
        for i in range(num_treatments):
            mask = (feature_sum >= quantiles[i]) & (feature_sum < quantiles[i + 1])
            treatments[mask] = i
            propensity_scores[mask, i] = 1.0
        
        # Handle edge case for last quantile
        treatments[feature_sum >= quantiles[-1]] = num_treatments - 1
        propensity_scores[feature_sum >= quantiles[-1], :] = 0.0
        propensity_scores[feature_sum >= quantiles[-1], -1] = 1.0
        
        return treatments, propensity_scores


class OutcomeGenerator:
    """Generates outcomes with heterogeneous treatment effects."""
    
    def __init__(self, nonlinearity: str = "moderate", 
                 heterogeneity: float = 1.0,
                 temporal_effects: bool = True):
        self.nonlinearity = nonlinearity
        self.heterogeneity = heterogeneity
        self.temporal_effects = temporal_effects
    
    def generate_outcomes(self, X: np.ndarray, treatments: np.ndarray,
                         num_treatments: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate outcomes with heterogeneous treatment effects.
        
        Args:
            X: Time series features [num_samples, seq_len, num_features]
            treatments: Treatment assignments [num_samples]
            num_treatments: Number of treatment options
        
        Returns:
            outcomes: Observed outcomes [num_samples, 1]
            potential_outcomes: All potential outcomes [num_samples, num_treatments]
        """
        num_samples = X.shape[0]
        
        # Base outcome from features (confounding effect)
        base_outcome = self._compute_base_outcome(X)
        
        # Treatment effects
        treatment_effects = self._compute_treatment_effects(X, num_treatments)
        
        # Compute potential outcomes
        potential_outcomes = base_outcome[:, None] + treatment_effects
        
        # Add noise
        noise = np.random.normal(0, 0.1, potential_outcomes.shape)
        potential_outcomes += noise
        
        # Observed outcomes
        observed_outcomes = potential_outcomes[np.arange(num_samples), treatments]
        
        return observed_outcomes[:, None], potential_outcomes
    
    def _compute_base_outcome(self, X: np.ndarray) -> np.ndarray:
        """Compute base outcome from features."""
        # Use summary statistics from time series
        mean_features = np.mean(X, axis=1)
        std_features = np.std(X, axis=1)
        trend_features = X[:, -1, :] - X[:, 0, :]
        
        # Combine features
        combined_features = np.concatenate([
            mean_features, std_features, trend_features
        ], axis=1)
        
        # Random weights for base outcome
        weights = np.random.normal(0, 0.5, combined_features.shape[1])
        base_outcome = np.dot(combined_features, weights)
        
        # Apply nonlinearity
        if self.nonlinearity == "moderate":
            base_outcome = np.tanh(base_outcome)
        elif self.nonlinearity == "high":
            base_outcome = np.sin(base_outcome) + 0.1 * base_outcome**3
        
        return base_outcome
    
    def _compute_treatment_effects(self, X: np.ndarray, 
                                 num_treatments: int) -> np.ndarray:
        """Compute heterogeneous treatment effects."""
        num_samples = X.shape[0]
        
        # Base treatment effects
        base_effects = np.array([0.0, 1.0, -0.5, 0.8][:num_treatments])
        
        # Individual modifiers based on features
        individual_modifiers = np.zeros((num_samples, num_treatments))
        
        for t in range(num_treatments):
            # Use different features for different treatments
            feature_idx = t % X.shape[2]
            modifier_base = X[:, -1, feature_idx]
            
            # Add heterogeneity
            individual_modifiers[:, t] = (
                modifier_base * self.heterogeneity * 
                np.random.normal(1.0, 0.2, num_samples)
            )
        
        # Temporal effects
        if self.temporal_effects:
            temporal_modifier = np.mean(X[:, -3:, :], axis=(1, 2))
            individual_modifiers += 0.3 * temporal_modifier[:, None]
        
        # Combine base effects with individual modifiers
        treatment_effects = base_effects[None, :] + individual_modifiers
        
        return treatment_effects


class DistributionShiftGenerator:
    """Generates distribution shifts for testing robustness."""
    
    def __init__(self, shift_magnitude: float = 0.5):
        self.shift_magnitude = shift_magnitude
    
    def apply_covariate_shift(self, X: np.ndarray, 
                            shift_start_idx: int) -> np.ndarray:
        """Apply covariate shift to features."""
        X_shifted = X.copy()
        
        # Apply shift to features after shift_start_idx
        shift_vector = np.random.normal(
            0, self.shift_magnitude, X.shape[2]
        )
        
        X_shifted[shift_start_idx:, :, :] += shift_vector[None, None, :]
        
        return X_shifted
    
    def apply_temporal_shift(self, X: np.ndarray, 
                           shift_start_idx: int) -> np.ndarray:
        """Apply temporal pattern shift."""
        X_shifted = X.copy()
        
        # Change temporal dynamics after shift point
        for i in range(shift_start_idx, len(X)):
            # Apply different autocorrelation
            new_autocorr = 0.9  # Higher autocorrelation
            for t in range(1, X.shape[1]):
                noise = np.random.normal(0, 0.05, X.shape[2])
                X_shifted[i, t, :] = (
                    new_autocorr * X_shifted[i, t-1, :] + 
                    (1 - new_autocorr) * X_shifted[i, 0, :] + 
                    noise
                )
        
        return X_shifted


class MissingDataGenerator:
    """Generates missing data patterns."""
    
    def __init__(self, missing_prob: float = 0.1, 
                 pattern: str = "random"):
        self.missing_prob = missing_prob
        self.pattern = pattern
    
    def introduce_missing_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Introduce missing data patterns."""
        X_missing = X.copy()
        missing_mask = np.zeros_like(X, dtype=bool)
        
        if self.pattern == "random":
            missing_mask = np.random.random(X.shape) < self.missing_prob
        elif self.pattern == "temporal":
            # Missing entire time steps
            missing_timesteps = np.random.random(X.shape[1]) < self.missing_prob
            missing_mask[:, missing_timesteps, :] = True
        elif self.pattern == "feature_dependent":
            # Some features more likely to be missing
            feature_missing_probs = np.random.uniform(
                0, 2 * self.missing_prob, X.shape[2]
            )
            for f in range(X.shape[2]):
                missing_mask[:, :, f] = (
                    np.random.random((X.shape[0], X.shape[1])) < 
                    feature_missing_probs[f]
                )
        
        # Apply missing mask
        X_missing[missing_mask] = np.nan
        
        return X_missing, missing_mask


class EnhancedDataGenerator:
    """Enhanced data generator for causal time series forecasting."""
    
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        
        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
        
        # Initialize generators
        self.temporal_gen = TemporalDynamicsGenerator(
            config.temporal_autocorr, config.temporal_noise_std
        )
        self.treatment_gen = TreatmentAssignmentMechanism(
            config.treatment_assignment_type, config.confounding_strength
        )
        self.outcome_gen = OutcomeGenerator(
            config.outcome_nonlinearity, 
            config.treatment_effect_heterogeneity,
            config.treatment_effect_temporal
        )
        
        if config.enable_distribution_shift:
            self.shift_gen = DistributionShiftGenerator(config.shift_magnitude)
        
        if config.missing_data_prob > 0:
            self.missing_gen = MissingDataGenerator(
                config.missing_data_prob, config.missing_pattern
            )
    
    def generate_dataset(self) -> Dict[str, np.ndarray]:
        """Generate complete synthetic dataset."""
        logger.info(f"Generating dataset with {self.config.num_samples} samples")
        
        # 1. Generate temporal features
        X = self.temporal_gen.generate_ar_process(
            self.config.num_samples, 
            self.config.seq_len,
            self.config.num_features
        )
        
        # Add seasonal patterns and trends
        X = self.temporal_gen.add_seasonal_pattern(X)
        X = self.temporal_gen.add_trend(X)
        
        # 2. Generate treatment assignments
        treatments, propensity_scores = self.treatment_gen.assign_treatments(
            X, self.config.num_treatments
        )
        
        # 3. Generate outcomes
        observed_outcomes, potential_outcomes = self.outcome_gen.generate_outcomes(
            X, treatments, self.config.num_treatments
        )
        
        # 4. Apply distribution shift if enabled
        if self.config.enable_distribution_shift:
            shift_start_idx = int(self.config.shift_start_ratio * self.config.num_samples)
            X = self.shift_gen.apply_covariate_shift(X, shift_start_idx)
        
        # 5. Introduce missing data if enabled
        missing_mask = None
        if self.config.missing_data_prob > 0:
            X, missing_mask = self.missing_gen.introduce_missing_data(X)
        
        # 6. Create treatment one-hot encoding
        treatments_one_hot = np.zeros((self.config.num_samples, self.config.num_treatments))
        treatments_one_hot[np.arange(self.config.num_samples), treatments] = 1
        
        # Prepare dataset
        dataset = {
            'X': X.astype(np.float32),
            'y': observed_outcomes.astype(np.float32),
            'treatments': treatments,
            'treatments_one_hot': treatments_one_hot.astype(np.float32),
            'potential_outcomes': potential_outcomes.astype(np.float32),
            'propensity_scores': propensity_scores.astype(np.float32),
        }
        
        if missing_mask is not None:
            dataset['missing_mask'] = missing_mask
        
        # Add metadata
        dataset['metadata'] = {
            'config': self.config,
            'true_ate': np.mean(potential_outcomes, axis=0) - np.mean(potential_outcomes[:, 0]),
            'treatment_distribution': np.bincount(treatments) / len(treatments)
        }
        
        logger.info("Dataset generation completed")
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Outcomes shape: {observed_outcomes.shape}")
        logger.info(f"Treatment distribution: {dataset['metadata']['treatment_distribution']}")
        logger.info(f"True ATE: {dataset['metadata']['true_ate']}")
        
        return dataset
    
    def generate_train_test_split(self, train_ratio: float = 0.8) -> Tuple[Dict, Dict]:
        """Generate dataset with train/test split."""
        full_dataset = self.generate_dataset()
        
        # Split index
        split_idx = int(train_ratio * self.config.num_samples)
        
        # Create train set
        train_set = {}
        test_set = {}
        
        for key, value in full_dataset.items():
            if key == 'metadata':
                train_set[key] = value
                test_set[key] = value
            elif isinstance(value, np.ndarray) and len(value) == self.config.num_samples:
                train_set[key] = value[:split_idx]
                test_set[key] = value[split_idx:]
            else:
                train_set[key] = value
                test_set[key] = value
        
        logger.info(f"Train set: {split_idx} samples, Test set: {self.config.num_samples - split_idx} samples")
        
        return train_set, test_set


# Convenience functions for common scenarios
def generate_simple_dataset(num_samples: int = 1000, 
                          seq_len: int = 20,
                          num_features: int = 8,
                          num_treatments: int = 4,
                          seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Generate a simple dataset with default parameters."""
    config = DataGenerationConfig(
        num_samples=num_samples,
        seq_len=seq_len,
        num_features=num_features,
        num_treatments=num_treatments,
        seed=seed
    )
    
    generator = EnhancedDataGenerator(config)
    return generator.generate_dataset()


def generate_confounded_dataset(num_samples: int = 1000,
                              confounding_strength: float = 2.0,
                              seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Generate a dataset with strong confounding."""
    config = DataGenerationConfig(
        num_samples=num_samples,
        treatment_assignment_type="confounded",
        confounding_strength=confounding_strength,
        seed=seed
    )
    
    generator = EnhancedDataGenerator(config)
    return generator.generate_dataset()


def generate_distribution_shift_dataset(num_samples: int = 1000,
                                       shift_magnitude: float = 1.0,
                                       shift_start_ratio: float = 0.7,
                                       seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Generate a dataset with distribution shift."""
    config = DataGenerationConfig(
        num_samples=num_samples,
        enable_distribution_shift=True,
        shift_magnitude=shift_magnitude,
        shift_start_ratio=shift_start_ratio,
        seed=seed
    )
    
    generator = EnhancedDataGenerator(config)
    return generator.generate_dataset() 