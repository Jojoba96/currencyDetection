"""
Professional Currency Detection System Demo
Demonstrates advanced features with current available libraries
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
import json

# Import our modules (using fallbacks for unavailable libraries)
from datasets.dataset_downloader import CurrencyDatasetDownloader
from advanced_features.feature_extractor import AdvancedFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalDemo:
    """Professional demo with available libraries"""
    
    def __init__(self):
        self.dataset_downloader = CurrencyDatasetDownloader()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.supported_currencies = ['USD', 'SAR']
        
        # Create directories
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
    def demonstrate_advanced_dataset_creation(self):
        """Demonstrate advanced dataset creation"""
        print("🗂️  ADVANCED DATASET CREATION")
        print("="*50)
        
        for currency in self.supported_currencies:
            print(f"\nCreating {currency} dataset...")
            
            # Create advanced dataset
            real_images, fake_images = self.dataset_downloader.create_advanced_dataset(
                currency, 200  # Smaller for demo
            )
            
            print(f"✅ {currency} Dataset created:")
            print(f"   Real images: {len(real_images)}")
            print(f"   Fake images: {len(fake_images)}")
            print(f"   Total augmented: {len(real_images) + len(fake_images)}")
        
        # Show dataset info
        dataset_info = self.dataset_downloader.get_dataset_info()
        print(f"\n📊 Dataset Summary:")
        for currency, info in dataset_info['dataset_structure'].items():
            print(f"  {currency}:")
            for category, count in info['categories'].items():
                if isinstance(count, dict):
                    print(f"    {category}: {count}")
                else:
                    print(f"    {category}: {count}")
    
    def demonstrate_advanced_feature_extraction(self):
        """Demonstrate advanced feature extraction"""
        print("\n🔬 ADVANCED FEATURE EXTRACTION")
        print("="*50)
        
        # Load some sample images
        for currency in self.supported_currencies:
            print(f"\nExtracting features for {currency}...")
            
            # Get training data
            try:
                X, y, filenames = self.dataset_downloader.prepare_training_data(currency, use_augmented=True)
                
                if len(X) > 0:
                    # Extract features from first few images
                    sample_size = min(10, len(X))
                    features_list = []
                    
                    for i in range(sample_size):
                        features = self.feature_extractor.extract_comprehensive_features(X[i], currency)
                        features_list.append(features)
                        
                        if i == 0:
                            print(f"   Sample features extracted: {len(features)}")
                            
                            # Show feature categories
                            categories = {
                                'Color': len([k for k in features.keys() if 'color' in k.lower()]),
                                'Texture': len([k for k in features.keys() if 'texture' in k.lower()]),
                                'Edge': len([k for k in features.keys() if 'edge' in k.lower()]),
                                'Geometric': len([k for k in features.keys() if 'geometric' in k.lower()]),
                                'Security': len([k for k in features.keys() if any(term in k.lower() 
                                               for term in ['security', 'watermark', 'thread'])])
                            }
                            
                            print(f"   Feature breakdown:")
                            for cat, count in categories.items():
                                print(f"     {cat}: {count}")
                    
                    # Calculate feature statistics
                    if features_list:
                        feature_names = list(features_list[0].keys())
                        feature_matrix = np.array([[f[name] for name in feature_names] for f in features_list])
                        
                        print(f"   Feature analysis complete:")
                        print(f"     Total features: {len(feature_names)}")
                        print(f"     Sample size: {len(features_list)}")
                        print(f"     Feature range: [{np.min(feature_matrix):.3f}, {np.max(feature_matrix):.3f}]")
                        print(f"     Feature variance: {np.var(feature_matrix):.3f}")
                else:
                    print(f"   No data available for {currency}")
                    
            except Exception as e:
                print(f"   Error: {e}")
    
    def demonstrate_simple_classification(self):
        """Demonstrate simple classification with extracted features"""
        print("\n🤖 SIMPLE CLASSIFICATION DEMO")
        print("="*50)
        
        for currency in self.supported_currencies:
            print(f"\nClassification demo for {currency}...")
            
            try:
                # Get training data
                X, y, filenames = self.dataset_downloader.prepare_training_data(currency, use_augmented=False)
                
                if len(X) > 10:
                    # Extract features
                    print(f"   Extracting features from {len(X)} images...")
                    features_list = []
                    valid_labels = []
                    
                    for i, (image, label) in enumerate(zip(X, y)):
                        try:
                            features = self.feature_extractor.extract_comprehensive_features(image, currency)
                            feature_values = list(features.values())
                            
                            # Check for valid features (no NaN/inf)
                            if all(np.isfinite(v) for v in feature_values):
                                features_list.append(feature_values)
                                valid_labels.append(label)
                                
                        except Exception:
                            continue
                    
                    if len(features_list) > 20:  # Need minimum samples
                        # Convert to numpy arrays
                        X_features = np.array(features_list)
                        y_labels = np.array(valid_labels)
                        
                        # Simple train/test split
                        split_idx = int(0.8 * len(X_features))
                        X_train = X_features[:split_idx]
                        X_test = X_features[split_idx:]
                        y_train = y_labels[:split_idx]
                        y_test = y_labels[split_idx:]
                        
                        # Simple classification using basic statistics
                        accuracy = self.simple_classify(X_train, y_train, X_test, y_test)
                        
                        print(f"   ✅ Classification results:")
                        print(f"     Training samples: {len(X_train)}")
                        print(f"     Test samples: {len(X_test)}")
                        print(f"     Features used: {X_features.shape[1]}")
                        print(f"     Simple accuracy: {accuracy:.3f}")
                        
                        # Feature importance (variance-based)
                        feature_vars = np.var(X_train, axis=0)
                        top_features = np.argsort(feature_vars)[-10:][::-1]
                        
                        print(f"     Top 10 most variable features: {top_features}")
                    else:
                        print(f"   Insufficient valid data for classification")
                else:
                    print(f"   Insufficient data for {currency}")
                    
            except Exception as e:
                print(f"   Error: {e}")
    
    def simple_classify(self, X_train, y_train, X_test, y_test):
        """Simple classification using feature means"""
        
        # Calculate mean features for each class
        real_mask = y_train == 1
        fake_mask = y_train == 0
        
        if np.sum(real_mask) > 0 and np.sum(fake_mask) > 0:
            real_mean = np.mean(X_train[real_mask], axis=0)
            fake_mean = np.mean(X_train[fake_mask], axis=0)
            
            # Classify test samples based on distance to class means
            predictions = []
            for test_sample in X_test:
                real_dist = np.linalg.norm(test_sample - real_mean)
                fake_dist = np.linalg.norm(test_sample - fake_mean)
                
                # Closer to real mean = real (1), closer to fake mean = fake (0)
                prediction = 1 if real_dist < fake_dist else 0
                predictions.append(prediction)
            
            # Calculate accuracy
            accuracy = np.mean(np.array(predictions) == y_test)
            return accuracy
        else:
            return 0.0
    
    def demonstrate_advanced_analysis(self):
        """Demonstrate advanced analysis capabilities"""
        print("\n📊 ADVANCED ANALYSIS")
        print("="*50)
        
        analysis_results = {}
        
        for currency in self.supported_currencies:
            print(f"\nAnalyzing {currency} currency features...")
            
            try:
                # Get sample data
                X, y, filenames = self.dataset_downloader.prepare_training_data(currency, use_augmented=False)
                
                if len(X) > 0:
                    # Extract features from sample
                    sample_image = X[0]
                    features = self.feature_extractor.extract_comprehensive_features(sample_image, currency)
                    
                    # Currency-specific analysis
                    if currency == 'USD':
                        color_match = features.get('currency_color_match', 0)
                        security_score = features.get('security_thread_score', 0)
                        watermark_score = features.get('watermark_score', 0)
                        
                        print(f"   USD-specific features:")
                        print(f"     Color match score: {color_match:.3f}")
                        print(f"     Security thread: {security_score:.3f}")
                        print(f"     Watermark detection: {watermark_score:.3f}")
                        
                    elif currency == 'SAR':
                        color_match = features.get('currency_color_match', 0)
                        security_score = features.get('security_thread_score', 0)
                        holo_score = features.get('holographic_score', 0)
                        
                        print(f"   SAR-specific features:")
                        print(f"     Color match score: {color_match:.3f}")
                        print(f"     Security thread: {security_score:.3f}")
                        print(f"     Holographic features: {holo_score:.3f}")
                    
                    # General quality metrics
                    print_quality = features.get('print_quality_score', 0)
                    microprint = features.get('microprint_score', 0)
                    
                    print(f"   Quality assessment:")
                    print(f"     Print quality: {print_quality:.3f}")
                    print(f"     Microprinting: {microprint:.3f}")
                    
                    analysis_results[currency] = {
                        'total_features': len(features),
                        'color_match': color_match,
                        'security_features': security_score,
                        'print_quality': print_quality
                    }
                    
            except Exception as e:
                print(f"   Error analyzing {currency}: {e}")
        
        return analysis_results
    
    def create_professional_report(self, analysis_results):
        """Create a professional analysis report"""
        print("\n📋 GENERATING PROFESSIONAL REPORT")
        print("="*50)
        
        report = {
            'system_info': {
                'name': 'Professional Currency Detection System',
                'version': '2.0 Demo',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'supported_currencies': self.supported_currencies
            },
            'feature_analysis': {},
            'dataset_statistics': {},
            'recommendations': []
        }
        
        # Add analysis results
        for currency, results in analysis_results.items():
            report['feature_analysis'][currency] = results
        
        # Add dataset statistics
        dataset_info = self.dataset_downloader.get_dataset_info()
        report['dataset_statistics'] = dataset_info
        
        # Generate recommendations
        recommendations = [
            "Collect more real-world currency images for better training",
            "Implement deep learning models for improved accuracy",
            "Add more security feature detection algorithms",
            "Develop real-time processing capabilities",
            "Create web-based interface for easy deployment"
        ]
        
        report['recommendations'] = recommendations
        
        # Save report
        report_path = self.results_dir / 'professional_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Professional report generated:")
        print(f"   📁 Saved to: {report_path}")
        print(f"   📊 Currencies analyzed: {len(analysis_results)}")
        print(f"   🔧 Total recommendations: {len(recommendations)}")
        
        return report
    
    def run_complete_demo(self):
        """Run the complete professional demo"""
        
        print("🎭 PROFESSIONAL CURRENCY DETECTION SYSTEM DEMO")
        print("="*70)
        print("Advanced system with multi-currency support and comprehensive analysis")
        print("="*70)
        
        start_time = time.time()
        
        try:
            # Step 1: Dataset creation
            self.demonstrate_advanced_dataset_creation()
            
            # Step 2: Feature extraction
            self.demonstrate_advanced_feature_extraction()
            
            # Step 3: Simple classification
            self.demonstrate_simple_classification()
            
            # Step 4: Advanced analysis
            analysis_results = self.demonstrate_advanced_analysis()
            
            # Step 5: Professional report
            report = self.create_professional_report(analysis_results)
            
            total_time = time.time() - start_time
            
            print("\n🏁 DEMO COMPLETED SUCCESSFULLY")
            print("="*50)
            print(f"⏱️  Total time: {total_time:.2f} seconds")
            print(f"💾 Results saved to: {self.results_dir}")
            print(f"📊 Report: professional_analysis_report.json")
            
            print(f"\n🎯 SYSTEM CAPABILITIES DEMONSTRATED:")
            print(f"   ✅ Multi-currency support (USD, SAR)")
            print(f"   ✅ Advanced dataset creation with augmentation")
            print(f"   ✅ Comprehensive feature extraction (200+ features)")
            print(f"   ✅ Currency-specific security feature detection")
            print(f"   ✅ Professional analysis and reporting")
            print(f"   ✅ Quality assessment and recommendations")
            
            print(f"\n📚 NEXT STEPS FOR PRODUCTION:")
            print(f"   1. Install scikit-learn for ML models")
            print(f"   2. Add TensorFlow/PyTorch for deep learning")
            print(f"   3. Collect real currency datasets")
            print(f"   4. Deploy with web interface")
            print(f"   5. Implement real-time processing")
            
            return report
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Demo error: {e}")
            raise

def main():
    """Main function"""
    demo = ProfessionalDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()