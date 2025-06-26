"""
Professional Currency Detection System
Advanced, production-ready system for USD and SAR currency authentication
Combines state-of-the-art computer vision, machine learning, and deep learning
"""
import numpy as np
import cv2
import pandas as pd
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

# Import our advanced modules
from datasets.dataset_downloader import CurrencyDatasetDownloader
from advanced_features.feature_extractor import AdvancedFeatureExtractor
from advanced_ml.ensemble_classifier import AdvancedEnsembleClassifier

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('professional_currency_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ProfessionalCurrencyDetector:
    """
    Professional-grade Currency Detection System
    
    Features:
    - Multi-currency support (USD, SAR)
    - Advanced feature extraction (200+ features)
    - Ensemble machine learning (15+ algorithms)
    - Professional evaluation metrics
    - Production-ready deployment
    """
    
    def __init__(self):
        self.dataset_downloader = CurrencyDatasetDownloader()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.classifiers = {}  # One classifier per currency
        
        self.supported_currencies = ['USD', 'SAR']
        self.is_trained = {}
        self.training_history = {}
        self.feature_names = []
        
        # Performance thresholds
        self.confidence_thresholds = {
            'high_confidence': 0.85,
            'medium_confidence': 0.70,
            'low_confidence': 0.55
        }
        
        # Model directories
        self.models_dir = Path('models/professional')
        self.results_dir = Path('results/professional')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Professional Currency Detection System initialized")
    
    def prepare_comprehensive_dataset(self, currencies: List[str] = None, 
                                    samples_per_currency: int = 2000) -> Dict[str, Dict]:
        """Prepare comprehensive datasets for training"""
        
        if currencies is None:
            currencies = self.supported_currencies
        
        logger.info(f"Preparing comprehensive datasets for {currencies}")
        
        datasets = {}
        
        for currency in currencies:
            logger.info(f"Creating advanced dataset for {currency}...")
            
            # Create advanced dataset with augmentation
            real_images, fake_images = self.dataset_downloader.create_advanced_dataset(
                currency, samples_per_currency
            )
            
            # Prepare training data
            X, y, filenames = self.dataset_downloader.prepare_training_data(
                currency, use_augmented=True
            )
            
            datasets[currency] = {
                'images': X,
                'labels': y,
                'filenames': filenames,
                'real_count': np.sum(y),
                'fake_count': len(y) - np.sum(y),
                'total_count': len(y)
            }
            
            logger.info(f"{currency} dataset: {len(y)} samples "
                       f"({np.sum(y)} real, {len(y) - np.sum(y)} fake)")
        
        return datasets
    
    def extract_comprehensive_features(self, datasets: Dict[str, Dict]) -> Dict[str, Dict]:
        """Extract comprehensive features from all datasets"""
        
        logger.info("Extracting comprehensive features...")
        
        feature_datasets = {}
        
        for currency, data in datasets.items():
            logger.info(f"Extracting features for {currency}...")
            
            images = data['images']
            labels = data['labels']
            
            # Extract features for all images
            features_list = []
            
            for i, image in enumerate(images):
                if i % 100 == 0:
                    logger.debug(f"Processing image {i+1}/{len(images)}")
                
                try:
                    features = self.feature_extractor.extract_comprehensive_features(
                        image, currency
                    )
                    features_list.append(features)
                    
                except Exception as e:
                    logger.warning(f"Feature extraction failed for image {i}: {e}")
                    # Add empty features for failed extractions
                    if features_list:
                        empty_features = {k: 0.0 for k in features_list[0].keys()}
                        features_list.append(empty_features)
                    else:
                        features_list.append({})
            
            # Convert to DataFrame for easier handling
            if features_list and features_list[0]:
                features_df = pd.DataFrame(features_list)
                
                # Handle any remaining NaN or inf values
                features_df = features_df.replace([np.inf, -np.inf], np.nan)
                features_df = features_df.fillna(features_df.median())
                
                # Store feature names (same for all currencies)
                if not self.feature_names:
                    self.feature_names = list(features_df.columns)
                
                feature_datasets[currency] = {
                    'features': features_df.values,
                    'labels': labels,
                    'feature_names': list(features_df.columns),
                    'n_features': features_df.shape[1]
                }
                
                logger.info(f"{currency}: Extracted {features_df.shape[1]} features "
                           f"from {features_df.shape[0]} images")
            else:
                logger.error(f"Feature extraction failed for {currency}")
                feature_datasets[currency] = None
        
        return feature_datasets
    
    def train_professional_models(self, feature_datasets: Dict[str, Dict]) -> Dict[str, Any]:
        """Train professional models for each currency"""
        
        logger.info("Training professional models...")
        
        training_results = {}
        
        for currency, data in feature_datasets.items():
            if data is None:
                logger.error(f"No data available for {currency}")
                continue
            
            logger.info(f"Training models for {currency}...")
            
            # Initialize classifier for this currency
            classifier = AdvancedEnsembleClassifier(currency)
            
            # Train the complete ensemble system
            X = data['features']
            y = data['labels']
            
            # Train system
            results = classifier.train_complete_system(X, y)
            
            # Store classifier and results
            self.classifiers[currency] = classifier
            self.is_trained[currency] = True
            
            training_results[currency] = {
                'classifier': classifier,
                'results': results,
                'best_model': classifier.best_model_name,
                'best_score': results[classifier.best_model_name]['mean_cv_score'],
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            }
            
            logger.info(f"{currency} training completed - Best: {classifier.best_model_name} "
                       f"(Score: {results[classifier.best_model_name]['mean_cv_score']:.4f})")
        
        self.training_history = training_results
        return training_results
    
    def comprehensive_evaluation(self, feature_datasets: Dict[str, Dict], 
                                test_size: float = 0.2) -> Dict[str, Any]:
        """Comprehensive evaluation of all models"""
        
        logger.info("Performing comprehensive evaluation...")
        
        evaluation_results = {}
        
        for currency, data in feature_datasets.items():
            if currency not in self.classifiers or data is None:
                continue
            
            logger.info(f"Evaluating {currency} models...")
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            X = data['features']
            y = data['labels']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Evaluate on test set
            classifier = self.classifiers[currency]
            test_results = classifier.evaluate_on_test_set(X_test, y_test)
            
            # Calculate comprehensive metrics
            evaluation_results[currency] = {
                'test_set_size': len(X_test),
                'test_real_count': np.sum(y_test),
                'test_fake_count': len(y_test) - np.sum(y_test),
                'model_results': test_results,
                'best_model_performance': test_results.get(classifier.best_model_name, {}),
                'feature_count': X.shape[1]
            }
            
            # Log best model performance
            if classifier.best_model_name in test_results:
                best_metrics = test_results[classifier.best_model_name]['metrics']
                logger.info(f"{currency} Best Model Performance:")
                logger.info(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {best_metrics['precision']:.4f}")
                logger.info(f"  Recall:    {best_metrics['recall']:.4f}")
                logger.info(f"  F1-Score:  {best_metrics['f1']:.4f}")
                logger.info(f"  AUC:       {best_metrics['auc']:.4f}")
        
        return evaluation_results
    
    def predict_currency_authenticity(self, image: np.ndarray, 
                                    currency: str) -> Dict[str, Any]:
        """Predict currency authenticity with comprehensive analysis"""
        
        if currency not in self.is_trained or not self.is_trained[currency]:
            raise ValueError(f"Model not trained for {currency}")
        
        start_time = time.time()
        
        # Extract features
        features = self.feature_extractor.extract_comprehensive_features(image, currency)
        
        # Convert to array format
        feature_array = np.array([list(features.values())])
        
        # Make prediction
        classifier = self.classifiers[currency]
        prediction = classifier.predict(feature_array)[0]
        probabilities = classifier.predict(feature_array, return_probabilities=True)[0]
        
        # Calculate confidence
        confidence = np.max(probabilities)
        
        # Determine confidence level
        if confidence >= self.confidence_thresholds['high_confidence']:
            confidence_level = 'High'
        elif confidence >= self.confidence_thresholds['medium_confidence']:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        processing_time = time.time() - start_time
        
        # Prepare comprehensive result
        result = {
            'currency': currency,
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'prediction_binary': int(prediction),
            'confidence_score': float(confidence),
            'confidence_level': confidence_level,
            'probabilities': {
                'fake': float(probabilities[0]),
                'real': float(probabilities[1])
            },
            'processing_time': processing_time,
            'model_used': classifier.best_model_name,
            'features_extracted': len(features),
            'feature_summary': {
                'color_features': len([k for k in features.keys() if 'color' in k.lower()]),
                'texture_features': len([k for k in features.keys() if 'texture' in k.lower()]),
                'edge_features': len([k for k in features.keys() if 'edge' in k.lower()]),
                'security_features': len([k for k in features.keys() if 'security' in k.lower() or 'watermark' in k.lower()])
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def batch_predict(self, images: List[np.ndarray], 
                     currencies: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction for multiple images"""
        
        if len(images) != len(currencies):
            raise ValueError("Number of images must match number of currencies")
        
        logger.info(f"Processing batch of {len(images)} images")
        
        results = []
        
        for i, (image, currency) in enumerate(zip(images, currencies)):
            try:
                result = self.predict_currency_authenticity(image, currency)
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'prediction': 'Error',
                    'confidence_score': 0.0,
                    'currency': currency
                })
        
        return results
    
    def generate_comprehensive_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        logger.info("Generating comprehensive performance report...")
        
        report = {
            'system_info': {
                'name': 'Professional Currency Detection System',
                'version': '2.0',
                'supported_currencies': self.supported_currencies,
                'total_features': len(self.feature_names),
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'training_summary': {},
            'evaluation_summary': {},
            'model_comparison': {},
            'feature_analysis': {},
            'recommendations': []
        }
        
        # Training summary
        for currency in self.supported_currencies:
            if currency in self.training_history:
                training_data = self.training_history[currency]
                report['training_summary'][currency] = {
                    'best_model': training_data['best_model'],
                    'best_score': training_data['best_score'],
                    'total_models_trained': len(training_data['results']),
                    'features_used': training_data['n_features'],
                    'training_samples': training_data['n_samples']
                }
        
        # Evaluation summary
        for currency, eval_data in evaluation_results.items():
            if 'best_model_performance' in eval_data and eval_data['best_model_performance']:
                metrics = eval_data['best_model_performance'].get('metrics', {})
                report['evaluation_summary'][currency] = {
                    'test_samples': eval_data['test_set_size'],
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1', 0),
                    'auc': metrics.get('auc', 0)
                }
        
        # Model comparison across currencies
        if len(evaluation_results) > 1:
            comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
            for metric in comparison_metrics:
                report['model_comparison'][metric] = {}
                for currency in evaluation_results:
                    if currency in report['evaluation_summary']:
                        report['model_comparison'][metric][currency] = \
                            report['evaluation_summary'][currency][metric]
        
        # Feature analysis
        report['feature_analysis'] = {
            'total_features': len(self.feature_names),
            'feature_categories': {
                'color_features': len([f for f in self.feature_names if 'color' in f.lower()]),
                'texture_features': len([f for f in self.feature_names if 'texture' in f.lower()]),
                'edge_features': len([f for f in self.feature_names if 'edge' in f.lower()]),
                'geometric_features': len([f for f in self.feature_names if 'geometric' in f.lower()]),
                'security_features': len([f for f in self.feature_names if any(term in f.lower() 
                                        for term in ['security', 'watermark', 'thread', 'microprint'])])
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        # Performance recommendations
        for currency, eval_data in report['evaluation_summary'].items():
            if eval_data['accuracy'] < 0.90:
                recommendations.append(f"Consider collecting more training data for {currency}")
            if eval_data['precision'] < eval_data['recall']:
                recommendations.append(f"Reduce false positives for {currency} by adjusting thresholds")
            if eval_data['recall'] < eval_data['precision']:
                recommendations.append(f"Improve fake detection for {currency} with better features")
        
        # General recommendations
        if len(self.feature_names) < 150:
            recommendations.append("Consider adding more feature types for better performance")
        
        recommendations.append("Regular model retraining recommended with new data")
        recommendations.append("Monitor model performance in production environment")
        
        report['recommendations'] = recommendations
        
        return report
    
    def save_professional_models(self) -> None:
        """Save all trained models and metadata"""
        
        logger.info("Saving professional models...")
        
        for currency, classifier in self.classifiers.items():
            if self.is_trained.get(currency, False):
                model_path = self.models_dir / f'professional_{currency}_model.joblib'
                classifier.save_model(str(model_path))
                logger.info(f"Saved {currency} model to {model_path}")
        
        # Save system metadata
        metadata = {
            'system_version': '2.0',
            'supported_currencies': self.supported_currencies,
            'feature_names': self.feature_names,
            'confidence_thresholds': self.confidence_thresholds,
            'training_history': self.training_history,
            'save_timestamp': time.time()
        }
        
        metadata_path = self.models_dir / 'system_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved system metadata to {metadata_path}")
    
    def load_professional_models(self) -> None:
        """Load trained models and metadata"""
        
        logger.info("Loading professional models...")
        
        # Load system metadata
        metadata_path = self.models_dir / 'system_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get('feature_names', [])
            self.confidence_thresholds = metadata.get('confidence_thresholds', self.confidence_thresholds)
            self.training_history = metadata.get('training_history', {})
        
        # Load individual models
        for currency in self.supported_currencies:
            model_path = self.models_dir / f'professional_{currency}_model.joblib'
            if model_path.exists():
                classifier = AdvancedEnsembleClassifier(currency)
                classifier.load_model(str(model_path))
                self.classifiers[currency] = classifier
                self.is_trained[currency] = True
                logger.info(f"Loaded {currency} model from {model_path}")
            else:
                logger.warning(f"Model file not found for {currency}: {model_path}")
    
    def create_visualization_dashboard(self, evaluation_results: Dict[str, Any]) -> None:
        """Create comprehensive visualization dashboard"""
        
        logger.info("Creating visualization dashboard...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Performance Comparison
        ax1 = plt.subplot(2, 3, 1)
        currencies = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        performance_data = []
        for currency in currencies:
            if currency in evaluation_results and 'best_model_performance' in evaluation_results[currency]:
                perf = evaluation_results[currency]['best_model_performance'].get('metrics', {})
                performance_data.append([perf.get(m, 0) for m in metrics])
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data, 
                                        index=currencies, 
                                        columns=[m.replace('_', ' ').title() for m in metrics])
            
            performance_df.plot(kind='bar', ax=ax1)
            ax1.set_title('Model Performance Comparison')
            ax1.set_ylabel('Score')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.set_ylim(0, 1)
        
        # 2. Feature Category Distribution
        ax2 = plt.subplot(2, 3, 2)
        if self.feature_names:
            feature_categories = {
                'Color': len([f for f in self.feature_names if 'color' in f.lower()]),
                'Texture': len([f for f in self.feature_names if 'texture' in f.lower()]),
                'Edge': len([f for f in self.feature_names if 'edge' in f.lower()]),
                'Geometric': len([f for f in self.feature_names if 'geometric' in f.lower()]),
                'Security': len([f for f in self.feature_names if any(term in f.lower() 
                              for term in ['security', 'watermark', 'thread'])]),
                'Statistical': len([f for f in self.feature_names if any(term in f.lower() 
                                  for term in ['mean', 'std', 'entropy'])]),
                'Other': 0
            }
            
            # Calculate 'Other' category
            total_categorized = sum(feature_categories.values()) - feature_categories['Other']
            feature_categories['Other'] = len(self.feature_names) - total_categorized
            
            # Remove categories with zero features
            feature_categories = {k: v for k, v in feature_categories.items() if v > 0}
            
            ax2.pie(feature_categories.values(), labels=feature_categories.keys(), autopct='%1.1f%%')
            ax2.set_title('Feature Categories Distribution')
        
        # 3. Training Scores Comparison
        ax3 = plt.subplot(2, 3, 3)
        if self.training_history:
            training_scores = {}
            for currency, data in self.training_history.items():
                if 'results' in data:
                    scores = {model: result['mean_cv_score'] 
                             for model, result in data['results'].items()}
                    training_scores[currency] = scores
            
            if training_scores:
                # Get common models across currencies
                all_models = set()
                for scores in training_scores.values():
                    all_models.update(scores.keys())
                
                # Create comparison data
                comparison_data = []
                model_names = []
                for model in sorted(all_models):
                    model_scores = []
                    for currency in currencies:
                        if currency in training_scores and model in training_scores[currency]:
                            model_scores.append(training_scores[currency][model])
                        else:
                            model_scores.append(0)
                    
                    if any(score > 0 for score in model_scores):
                        comparison_data.append(model_scores)
                        model_names.append(model.replace('_', ' ').title())
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data, 
                                               index=model_names, 
                                               columns=currencies)
                    
                    comparison_df.plot(kind='barh', ax=ax3)
                    ax3.set_title('Training Scores by Model')
                    ax3.set_xlabel('Cross-Validation Score')
                    ax3.legend(title='Currency')
        
        # 4. Dataset Statistics
        ax4 = plt.subplot(2, 3, 4)
        if evaluation_results:
            dataset_stats = []
            for currency, data in evaluation_results.items():
                stats = [
                    data.get('test_set_size', 0),
                    data.get('test_real_count', 0),
                    data.get('test_fake_count', 0)
                ]
                dataset_stats.append(stats)
            
            if dataset_stats:
                stats_df = pd.DataFrame(dataset_stats,
                                      index=currencies,
                                      columns=['Total', 'Real', 'Fake'])
                
                stats_df.plot(kind='bar', ax=ax4, stacked=True)
                ax4.set_title('Test Dataset Statistics')
                ax4.set_ylabel('Number of Samples')
                ax4.legend()
        
        # 5. Confidence Distribution (placeholder for future implementation)
        ax5 = plt.subplot(2, 3, 5)
        ax5.text(0.5, 0.5, 'Confidence Distribution\n(Requires prediction data)', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Confidence Distribution')
        
        # 6. System Overview
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # System info text
        system_info = f"""
        Professional Currency Detection System v2.0
        
        Supported Currencies: {', '.join(self.supported_currencies)}
        Total Features: {len(self.feature_names)}
        Models per Currency: {len(self.classifiers[currencies[0]].models) if currencies and currencies[0] in self.classifiers else 'N/A'}
        
        Training Status:
        """
        
        for currency in self.supported_currencies:
            status = "✓ Trained" if self.is_trained.get(currency, False) else "✗ Not Trained"
            system_info += f"\n        {currency}: {status}"
        
        ax6.text(0.1, 0.9, system_info, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save the dashboard
        dashboard_path = self.results_dir / 'performance_dashboard.png'
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {dashboard_path}")
        
        plt.show()
    
    def run_complete_professional_pipeline(self, samples_per_currency: int = 1000) -> Dict[str, Any]:
        """Run the complete professional pipeline"""
        
        logger.info("🚀 STARTING PROFESSIONAL CURRENCY DETECTION PIPELINE")
        logger.info("="*70)
        
        start_time = time.time()
        
        try:
            # Step 1: Prepare datasets
            logger.info("Step 1: Preparing comprehensive datasets...")
            datasets = self.prepare_comprehensive_dataset(
                self.supported_currencies, samples_per_currency
            )
            
            # Step 2: Extract features
            logger.info("Step 2: Extracting comprehensive features...")
            feature_datasets = self.extract_comprehensive_features(datasets)
            
            # Step 3: Train models
            logger.info("Step 3: Training professional models...")
            training_results = self.train_professional_models(feature_datasets)
            
            # Step 4: Evaluate models
            logger.info("Step 4: Comprehensive evaluation...")
            evaluation_results = self.comprehensive_evaluation(feature_datasets)
            
            # Step 5: Generate report
            logger.info("Step 5: Generating comprehensive report...")
            report = self.generate_comprehensive_report(evaluation_results)
            
            # Step 6: Save models
            logger.info("Step 6: Saving professional models...")
            self.save_professional_models()
            
            # Step 7: Create visualizations
            logger.info("Step 7: Creating visualization dashboard...")
            self.create_visualization_dashboard(evaluation_results)
            
            # Save comprehensive report
            report_path = self.results_dir / 'comprehensive_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            total_time = time.time() - start_time
            
            logger.info("✅ PROFESSIONAL PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Report saved to: {report_path}")
            
            return {
                'datasets': datasets,
                'feature_datasets': feature_datasets,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'report': report,
                'processing_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'system_name': 'Professional Currency Detection System',
            'version': '2.0',
            'supported_currencies': self.supported_currencies,
            'training_status': {},
            'model_info': {},
            'feature_info': {
                'total_features': len(self.feature_names),
                'feature_categories': {}
            },
            'performance_thresholds': self.confidence_thresholds
        }
        
        # Training status
        for currency in self.supported_currencies:
            status['training_status'][currency] = self.is_trained.get(currency, False)
            
            if currency in self.classifiers:
                classifier = self.classifiers[currency]
                status['model_info'][currency] = {
                    'best_model': classifier.best_model_name,
                    'total_models': len(classifier.models),
                    'is_trained': self.is_trained.get(currency, False)
                }
        
        # Feature categories
        if self.feature_names:
            categories = ['color', 'texture', 'edge', 'geometric', 'security', 'statistical']
            for category in categories:
                count = len([f for f in self.feature_names if category in f.lower()])
                status['feature_info']['feature_categories'][category] = count
        
        return status

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the professional system
    detector = ProfessionalCurrencyDetector()
    
    # Run the complete pipeline
    results = detector.run_complete_professional_pipeline(samples_per_currency=500)
    
    # Print summary
    print("\n🎉 PROFESSIONAL CURRENCY DETECTION SYSTEM")
    print("="*60)
    
    report = results['report']
    
    print(f"System: {report['system_info']['name']} v{report['system_info']['version']}")
    print(f"Features: {report['system_info']['total_features']}")
    print(f"Currencies: {', '.join(report['system_info']['supported_currencies'])}")
    
    print(f"\n📊 Performance Summary:")
    for currency, metrics in report['evaluation_summary'].items():
        print(f"  {currency}:")
        print(f"    Accuracy:  {metrics['accuracy']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1-Score:  {metrics['f1_score']:.3f}")
        print(f"    AUC:       {metrics['auc']:.3f}")
    
    print(f"\n🔧 Processing Time: {results['processing_time']:.2f} seconds")
    print(f"📁 Results saved to: results/professional/")
    print(f"💾 Models saved to: models/professional/")
    
    print("\n✅ Professional system ready for production deployment!")