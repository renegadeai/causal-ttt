"""
Enhanced Causal TTT Neural CDE Framework

This package provides an enhanced implementation of Test-Time Training (TTT) 
for causal time series forecasting using Neural Controlled Differential Equations (CDEs).

Key Features:
- Multi-task auxiliary learning for better TTT adaptation
- Enhanced CDE architecture with attention mechanisms
- Comprehensive causal inference evaluation metrics
- Realistic synthetic data generation
- Uncertainty quantification
"""

__version__ = "2.0.0"
__author__ = "Enhanced TTT CDE Team"

from .enhanced_ttt_cde_model_fixed import (
    FixedEnhancedTTTNeuralCDE,
    FixedTreatmentEffectNetwork,
    AuxiliaryTask,
    ReconstructionTask,
    ForecastingTask,
    TemporalConsistencyTask,
    CausalContrastiveTask,
    EnhancedCDEFunc,
    MultiHeadAttention
)

from .causal_evaluation_metrics import (
    CausalEvaluator,
    PEHEMetric,
    ATEMetric,
    PolicyValueMetric,
    ConfoundingRobustnessMetric,
    UncertaintyCalibrationMetric
)

from .enhanced_data_generation import (
    DataGenerationConfig,
    EnhancedDataGenerator,
    generate_simple_dataset,
    generate_confounded_dataset,
    generate_distribution_shift_dataset
)

__all__ = [
    # Model components
    'FixedEnhancedTTTNeuralCDE',
    'FixedTreatmentEffectNetwork',
    'AuxiliaryTask',
    'ReconstructionTask', 
    'ForecastingTask',
    'TemporalConsistencyTask',
    'CausalContrastiveTask',
    'EnhancedCDEFunc',
    'MultiHeadAttention',
    
    # Evaluation metrics
    'CausalEvaluator',
    'PEHEMetric',
    'ATEMetric', 
    'PolicyValueMetric',
    'ConfoundingRobustnessMetric',
    'UncertaintyCalibrationMetric',
    
    # Data generation
    'DataGenerationConfig',
    'EnhancedDataGenerator',
    'generate_simple_dataset',
    'generate_confounded_dataset',
    'generate_distribution_shift_dataset'
] 