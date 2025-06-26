"""
Main Currency Detection System
Integrates traditional CV, ML, and Deep Learning approaches
"""
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from pathlib import Path

from algorithms.traditional_cv import TraditionalCVDetector
from algorithms.machine_learning import MLCurrencyDetector
from algorithms.deep_learning import DeepLearningDetector
from utils.data_loader import CurrencyDataLoader
from config import SUPPORTED_CURRENCIES, CONFIDENCE_THRESHOLDS, RESULTS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('currency_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CurrencyDetectionSystem:
    """
    Comprehensive Currency Detection System
    Combines multiple detection approaches for robust authentication
    """
    
    def __init__(self):
        self.cv_detector = TraditionalCVDetector()
        self.ml_detector = MLCurrencyDetector()
        self.dl_detector = DeepLearningDetector()
        self.data_loader = CurrencyDataLoader()
        
        self.is_trained = False
        self.supported_currencies = SUPPORTED_CURRENCIES
        self.confidence_thresholds = CONFIDENCE_THRESHOLDS
        
        logger.info("Currency Detection System initialized")
    
    def load_and_prepare_data(self, currency: str = 'USD') -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare training data for the specified currency"""
        logger.info(f"Loading data for {currency}...")
        
        # Try to load real data first
        images, labels, filenames = self.data_loader.load_currency_dataset(currency)
        
        # If no real data found, generate synthetic data for demonstration
        if images is None or len(images) == 0:
            logger.warning(f"No real data found for {currency}. Generating synthetic data...")
            images, labels = self.data_loader.generate_synthetic_data(currency, 500)
        
        logger.info(f"Loaded {len(images)} images for {currency}")
        return images, labels
    
    def extract_traditional_features(self, images: np.ndarray, currency: str = 'USD') -> Dict[str, List[float]]:
        """Extract traditional CV features from all images"""
        logger.info(f"Extracting traditional CV features for {len(images)} images...")
        
        all_features = {}
        
        for i, image in enumerate(images):
            if i % 50 == 0:
                logger.info(f"Processing image {i+1}/{len(images)}")
            
            try:
                features = self.cv_detector.extract_all_features(image, currency)
                
                # Initialize feature lists if first image
                if i == 0:
                    all_features = {key: [] for key in features.keys()}
                
                # Add features to lists
                for key, value in features.items():
                    all_features[key].append(value)
                    
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # Add NaN values for failed extractions
                if all_features:
                    for key in all_features.keys():
                        all_features[key].append(0.0)  # Use 0 instead of NaN
        
        logger.info(f"Extracted {len(all_features)} features from {len(images)} images")
        return all_features
    
    def train_system(self, currency: str = 'USD', test_size: float = 0.2):
        """Train the complete detection system"""
        logger.info(f"Training currency detection system for {currency}...")
        
        if currency not in self.supported_currencies:
            raise ValueError(f"Currency {currency} not supported. Use one of: {self.supported_currencies}")
        
        # Load data
        images, labels = self.load_and_prepare_data(currency)
        
        # Split data
        split_idx = int((1 - test_size) * len(images))
        train_images, test_images = images[:split_idx], images[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        
        # Further split training data for validation
        val_split_idx = int(0.8 * len(train_images))
        train_images_final, val_images = train_images[:val_split_idx], train_images[val_split_idx:]
        train_labels_final, val_labels = train_labels[:val_split_idx], train_labels[val_split_idx:]
        
        logger.info(f"Data split - Train: {len(train_images_final)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        # Extract traditional features
        train_features = self.extract_traditional_features(train_images_final, currency)
        
        # Train ML models
        logger.info("Training machine learning models...")
        self.ml_detector.train(train_features, train_labels_final.tolist(), currency)
        
        # Train deep learning models
        logger.info("Training deep learning models...")
        self.dl_detector.train_all_models(train_images_final, train_labels_final, val_images, val_labels, currency)
        
        # Store test data for evaluation
        self.test_data = {
            currency: {
                'images': test_images,
                'labels': test_labels,
                'features': self.extract_traditional_features(test_images, currency)
            }
        }
        
        self.is_trained = True
        logger.info(f"Training completed for {currency}")
    
    def predict_single_image(self, image: np.ndarray, currency: str = 'USD') -> Dict[str, Any]:
        """Predict authenticity of a single currency image"""
        if not self.is_trained:
            raise ValueError("System not trained. Call train_system() first.")
        
        start_time = time.time()
        
        # Extract traditional features
        cv_features = self.cv_detector.extract_all_features(image, currency)
        cv_features_dict = {key: [value] for key, value in cv_features.items()}
        
        # ML prediction
        try:
            ml_pred, ml_prob = self.ml_detector.predict(cv_features_dict, currency)
            ml_confidence = ml_prob[0][1] if len(ml_prob[0]) > 1 else ml_prob[0]
            ml_prediction = ml_pred[0]
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            ml_prediction = 0
            ml_confidence = 0.0
        
        # Deep learning prediction
        try:
            dl_pred, dl_prob = self.dl_detector.predict(np.expand_dims(image, axis=0), currency)
            dl_prediction = dl_pred[0]
            dl_confidence = dl_prob[0]
        except Exception as e:
            logger.error(f"DL prediction error: {e}")
            dl_prediction = 0
            dl_confidence = 0.0
        
        # Ensemble prediction (weighted average)
        # Give more weight to deep learning as it typically performs better on images
        ensemble_confidence = (0.3 * ml_confidence + 0.7 * dl_confidence)
        ensemble_prediction = 1 if ensemble_confidence > 0.5 else 0
        
        # Determine confidence level
        if ensemble_confidence >= self.confidence_thresholds['high_confidence']:
            confidence_level = 'High'
        elif ensemble_confidence >= self.confidence_thresholds['medium_confidence']:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        processing_time = time.time() - start_time
        
        result = {
            'currency': currency,
            'prediction': 'Real' if ensemble_prediction == 1 else 'Fake',
            'confidence_score': float(ensemble_confidence),
            'confidence_level': confidence_level,
            'processing_time': processing_time,
            'individual_predictions': {
                'traditional_cv_ml': {
                    'prediction': 'Real' if ml_prediction == 1 else 'Fake',
                    'confidence': float(ml_confidence)
                },
                'deep_learning': {
                    'prediction': 'Real' if dl_prediction == 1 else 'Fake',
                    'confidence': float(dl_confidence)
                }
            },
            'features_extracted': len(cv_features),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def predict_batch(self, images: List[np.ndarray], currency: str = 'USD') -> List[Dict[str, Any]]:
        """Predict authenticity for a batch of images"""
        logger.info(f"Processing batch of {len(images)} images for {currency}")
        
        results = []
        for i, image in enumerate(images):
            try:
                result = self.predict_single_image(image, currency)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'prediction': 'Error',
                    'confidence_score': 0.0
                })
        
        return results
    
    def evaluate_system(self, currency: str = 'USD') -> Dict[str, Any]:
        """Evaluate the complete detection system"""
        if not self.is_trained or currency not in self.test_data:
            raise ValueError(f"System not trained or no test data available for {currency}")
        
        logger.info(f"Evaluating system performance for {currency}...")
        
        test_images = self.test_data[currency]['images']
        test_labels = self.test_data[currency]['labels']
        test_features = self.test_data[currency]['features']
        
        # Evaluate ML models
        try:
            ml_results = self.ml_detector.evaluate_models(
                self.ml_detector.preprocess_features(
                    *self.ml_detector.prepare_data(test_features, test_labels.tolist())[:2],
                    fit=False
                ),
                test_labels,
                currency
            )
        except Exception as e:
            logger.error(f"ML evaluation error: {e}")
            ml_results = {}
        
        # Evaluate DL models
        try:
            dl_results = self.dl_detector.evaluate_models(test_images, test_labels, currency)
        except Exception as e:
            logger.error(f"DL evaluation error: {e}")
            dl_results = {}
        
        # Evaluate ensemble
        ensemble_predictions = []
        ensemble_confidences = []
        
        for image in test_images:
            try:
                result = self.predict_single_image(image, currency)
                ensemble_predictions.append(1 if result['prediction'] == 'Real' else 0)
                ensemble_confidences.append(result['confidence_score'])
            except:
                ensemble_predictions.append(0)
                ensemble_confidences.append(0.0)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate ensemble metrics
        ensemble_accuracy = np.mean(ensemble_predictions == test_labels)
        
        # Calculate precision, recall, F1 for ensemble
        tp = np.sum((ensemble_predictions == 1) & (test_labels == 1))
        fp = np.sum((ensemble_predictions == 1) & (test_labels == 0))
        fn = np.sum((ensemble_predictions == 0) & (test_labels == 1))
        
        ensemble_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        ensemble_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        ensemble_f1 = 2 * (ensemble_precision * ensemble_recall) / (ensemble_precision + ensemble_recall) if (ensemble_precision + ensemble_recall) > 0 else 0
        
        evaluation_results = {
            'currency': currency,
            'test_samples': len(test_images),
            'ml_models': ml_results,
            'dl_models': dl_results,
            'ensemble': {
                'accuracy': ensemble_accuracy,
                'precision': ensemble_precision,
                'recall': ensemble_recall,
                'f1': ensemble_f1,
                'avg_confidence': np.mean(ensemble_confidences)
            },
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save evaluation results
        results_file = Path(RESULTS_DIR) / f'evaluation_{currency}_{int(time.time())}.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_file}")
        return evaluation_results
    
    def save_models(self, currency: str = 'USD'):
        """Save all trained models"""
        if not self.is_trained:
            raise ValueError("No models to save. Train the system first.")
        
        try:
            self.ml_detector.save_models(currency)
            self.dl_detector.save_models(currency)
            logger.info(f"All models saved for {currency}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, currency: str = 'USD'):
        """Load pre-trained models"""
        try:
            self.ml_detector.load_models(currency)
            self.dl_detector.load_models(currency)
            self.is_trained = True
            logger.info(f"Models loaded for {currency}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the detection system"""
        return {
            'system_name': 'Advanced Currency Detection System',
            'version': '1.0.0',
            'approaches': ['Traditional Computer Vision', 'Machine Learning', 'Deep Learning'],
            'supported_currencies': self.supported_currencies,
            'is_trained': self.is_trained,
            'confidence_thresholds': self.confidence_thresholds,
            'features': {
                'traditional_cv': [
                    'Color analysis', 'Texture analysis', 'Edge detection',
                    'Geometric features', 'Security features'
                ],
                'machine_learning': [
                    'Random Forest', 'Gradient Boosting', 'SVM',
                    'Logistic Regression', 'K-NN', 'Naive Bayes', 'Ensemble'
                ],
                'deep_learning': [
                    'Custom CNN', 'Transfer Learning (EfficientNet, ResNet, MobileNet)',
                    'Attention Mechanism', 'Ensemble Deep Learning'
                ]
            }
        }
    
    def print_system_info(self):
        """Print detailed system information"""
        info = self.get_system_info()
        
        print("="*60)
        print(f"{info['system_name']} v{info['version']}")
        print("="*60)
        print(f"Status: {'Trained' if info['is_trained'] else 'Not Trained'}")
        print(f"Supported Currencies: {', '.join(info['supported_currencies'])}")
        print()
        
        print("Detection Approaches:")
        for approach in info['approaches']:
            print(f"  • {approach}")
        print()
        
        print("Features by Approach:")
        for approach, features in info['features'].items():
            print(f"  {approach.replace('_', ' ').title()}:")
            for feature in features:
                print(f"    - {feature}")
        print()
        
        print("Confidence Thresholds:")
        for level, threshold in info['confidence_thresholds'].items():
            print(f"  {level.replace('_', ' ').title()}: {threshold:.1%}")
        print("="*60)

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the detection system
    detector = CurrencyDetectionSystem()
    
    # Print system information
    detector.print_system_info()
    
    # Create sample dataset for demonstration
    logger.info("Creating sample dataset...")
    from utils.data_loader import create_sample_dataset
    sample_data = create_sample_dataset()
    
    # Example of how to use the system
    logger.info("This is a demonstration of the currency detection system.")
    logger.info("To use with real data:")
    logger.info("1. Add your currency images to the data/[CURRENCY]/real and data/[CURRENCY]/fake folders")
    logger.info("2. Call detector.train_system('USD') or detector.train_system('SAR')")
    logger.info("3. Use detector.predict_single_image(image, 'USD') to detect fake currency")
    
    # Demonstrate feature extraction
    sample_image = sample_data['USD'][0][0]  # First USD image
    features = detector.cv_detector.extract_all_features(sample_image, 'USD')
    logger.info(f"Extracted {len(features)} traditional CV features from sample image")
    
    print("\nSample feature extraction completed successfully!")
    print("The system is ready for training with real currency data.")