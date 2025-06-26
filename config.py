"""
Configuration file for the Currency Detection System
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Supported currencies
SUPPORTED_CURRENCIES = ['USD', 'SAR']

# Image processing parameters
IMAGE_SIZE = (224, 224)  # Standard size for deep learning models
GRAYSCALE_THRESHOLD = 127
EDGE_DETECTION_THRESHOLD = (50, 150)

# Traditional CV parameters
TEXTURE_ANALYSIS = {
    'glcm_distances': [1, 2, 3],
    'glcm_angles': [0, 45, 90, 135],
    'lbp_radius': 3,
    'lbp_n_points': 24
}

# Machine Learning parameters
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    'feature_selection_k': 20
}

# Deep Learning parameters
DL_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'patience': 10,
    'validation_split': 0.2
}

# Security features to detect (can be extended)
SECURITY_FEATURES = {
    'USD': [
        'watermark',
        'security_thread',
        'microprinting',
        'color_changing_ink',
        'raised_printing'
    ],
    'SAR': [
        'watermark',
        'security_thread',
        'microprinting',
        'holographic_strip',
        'tactile_features'
    ]
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high_confidence': 0.8,
    'medium_confidence': 0.6,
    'low_confidence': 0.4
}