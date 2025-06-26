"""
Demonstration script for the Currency Detection System
Shows how to use all the features of the system
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path

from currency_detector import CurrencyDetectionSystem
from utils.data_loader import create_sample_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_feature_extraction():
    """Demonstrate traditional CV feature extraction"""
    print("\n🔍 FEATURE EXTRACTION DEMONSTRATION")
    print("="*50)
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Add some patterns to make it more currency-like
    cv2.rectangle(sample_image, (50, 50), (174, 174), (255, 255, 255), 2)
    cv2.circle(sample_image, (112, 112), 30, (0, 0, 0), 1)
    cv2.putText(sample_image, "SAMPLE", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Initialize detector
    detector = CurrencyDetectionSystem()
    
    # Extract features
    print("Extracting traditional CV features...")
    features = detector.cv_detector.extract_all_features(sample_image, 'USD')
    
    print(f"✅ Extracted {len(features)} features")
    print("\n📊 Sample Features:")
    
    # Show first 10 features
    feature_items = list(features.items())[:10]
    for i, (name, value) in enumerate(feature_items, 1):
        print(f"{i:2d}. {name}: {value:.4f}")
    
    print(f"... and {len(features) - 10} more features")
    
    return sample_image, features

def demonstrate_training():
    """Demonstrate system training"""
    print("\n🎯 TRAINING DEMONSTRATION")
    print("="*50)
    
    # Create sample dataset
    print("Creating sample dataset...")
    sample_data = create_sample_dataset()
    
    # Initialize detector
    detector = CurrencyDetectionSystem()
    
    # Train for USD (using synthetic data)
    print("Training system for USD...")
    start_time = time.time()
    
    try:
        # This will use the synthetic data created above
        detector.train_system('USD')
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
        return detector
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def demonstrate_prediction(detector, sample_image):
    """Demonstrate currency prediction"""
    print("\n🔮 PREDICTION DEMONSTRATION")
    print("="*50)
    
    if detector is None:
        print("❌ Cannot demonstrate prediction - detector not trained")
        return
    
    print("Making prediction on sample image...")
    
    try:
        # Make prediction
        result = detector.predict_single_image(sample_image, 'USD')
        
        print("✅ Prediction completed!")
        print(f"📊 Results:")
        print(f"   Currency: {result['currency']}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence_score']:.2%}")
        print(f"   Confidence Level: {result['confidence_level']}")
        print(f"   Processing Time: {result['processing_time']:.3f} seconds")
        print(f"   Features Used: {result['features_extracted']}")
        
        print(f"\n🔍 Individual Model Results:")
        for model_name, pred_data in result['individual_predictions'].items():
            print(f"   {model_name.replace('_', ' ').title()}:")
            print(f"     Prediction: {pred_data['prediction']}")
            print(f"     Confidence: {pred_data['confidence']:.2%}")
        
        return result
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def demonstrate_evaluation(detector):
    """Demonstrate system evaluation"""
    print("\n📈 EVALUATION DEMONSTRATION")
    print("="*50)
    
    if detector is None:
        print("❌ Cannot demonstrate evaluation - detector not trained")
        return
    
    print("Evaluating system performance...")
    
    try:
        # Run evaluation
        evaluation_results = detector.evaluate_system('USD')
        
        print("✅ Evaluation completed!")
        print(f"\n📊 Ensemble Performance:")
        ensemble_metrics = evaluation_results['ensemble']
        print(f"   Accuracy:  {ensemble_metrics['accuracy']:.2%}")
        print(f"   Precision: {ensemble_metrics['precision']:.2%}")
        print(f"   Recall:    {ensemble_metrics['recall']:.2%}")
        print(f"   F1-Score:  {ensemble_metrics['f1']:.2%}")
        print(f"   Avg Confidence: {ensemble_metrics['avg_confidence']:.2%}")
        
        print(f"\n🤖 Individual Model Performance:")
        
        # Show ML models
        if 'ml_models' in evaluation_results:
            print("   Machine Learning Models:")
            for model_name, metrics in evaluation_results['ml_models'].items():
                print(f"     {model_name}: F1={metrics.get('f1', 0):.2%}")
        
        # Show DL models
        if 'dl_models' in evaluation_results:
            print("   Deep Learning Models:")
            for model_name, metrics in evaluation_results['dl_models'].items():
                print(f"     {model_name}: F1={metrics.get('f1', 0):.2%}")
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def demonstrate_batch_processing(detector):
    """Demonstrate batch processing"""
    print("\n📦 BATCH PROCESSING DEMONSTRATION")
    print("="*50)
    
    if detector is None:
        print("❌ Cannot demonstrate batch processing - detector not trained")
        return
    
    # Create batch of sample images
    batch_size = 5
    print(f"Creating batch of {batch_size} sample images...")
    
    batch_images = []
    for i in range(batch_size):
        # Create different sample images
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add different patterns
        if i % 2 == 0:
            cv2.rectangle(img, (30, 30), (194, 194), (255, 255, 255), 2)
        else:
            cv2.circle(img, (112, 112), 50, (0, 0, 0), 2)
        
        batch_images.append(img)
    
    print("Processing batch...")
    
    try:
        # Process batch
        start_time = time.time()
        results = detector.predict_batch(batch_images, 'USD')
        processing_time = time.time() - start_time
        
        print(f"✅ Batch processing completed in {processing_time:.3f} seconds")
        print(f"📊 Results:")
        
        real_count = sum(1 for r in results if r.get('prediction') == 'Real')
        fake_count = len(results) - real_count
        avg_confidence = np.mean([r.get('confidence_score', 0) for r in results])
        
        print(f"   Total Images: {len(results)}")
        print(f"   Predicted Real: {real_count}")
        print(f"   Predicted Fake: {fake_count}")
        print(f"   Average Confidence: {avg_confidence:.2%}")
        print(f"   Avg Processing Time per Image: {processing_time/len(results):.3f} seconds")
        
        print(f"\n🔍 Individual Results:")
        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"   Image {i+1}: {result['prediction']} ({result['confidence_score']:.1%})")
            else:
                print(f"   Image {i+1}: Error - {result['error']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return None

def demonstrate_model_management(detector):
    """Demonstrate model saving and loading"""
    print("\n💾 MODEL MANAGEMENT DEMONSTRATION")
    print("="*50)
    
    if detector is None:
        print("❌ Cannot demonstrate model management - detector not trained")
        return
    
    try:
        # Save models
        print("Saving trained models...")
        detector.save_models('USD')
        print("✅ Models saved successfully")
        
        # Create new detector instance
        print("Creating new detector instance...")
        new_detector = CurrencyDetectionSystem()
        
        # Load models
        print("Loading saved models...")
        new_detector.load_models('USD')
        print("✅ Models loaded successfully")
        
        # Test loaded models
        print("Testing loaded models...")
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = new_detector.predict_single_image(sample_image, 'USD')
        
        print(f"✅ Loaded models working - Prediction: {result['prediction']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model management failed: {e}")
        return False

def main():
    """Main demonstration function"""
    print("🎭 CURRENCY DETECTION SYSTEM DEMONSTRATION")
    print("="*60)
    print("This demonstration shows all the capabilities of the system")
    print("using synthetic data for educational purposes.")
    print("="*60)
    
    # Step 1: Feature Extraction
    sample_image, features = demonstrate_feature_extraction()
    
    # Step 2: Training
    detector = demonstrate_training()
    
    # Step 3: Prediction
    result = demonstrate_prediction(detector, sample_image)
    
    # Step 4: Evaluation
    evaluation_results = demonstrate_evaluation(detector)
    
    # Step 5: Batch Processing
    batch_results = demonstrate_batch_processing(detector)
    
    # Step 6: Model Management
    model_mgmt_success = demonstrate_model_management(detector)
    
    # Summary
    print("\n🏁 DEMONSTRATION SUMMARY")
    print("="*50)
    print("✅ Feature Extraction: Completed")
    print(f"✅ Training: {'Completed' if detector else 'Failed'}")
    print(f"✅ Prediction: {'Completed' if result else 'Failed'}")
    print(f"✅ Evaluation: {'Completed' if evaluation_results else 'Failed'}")
    print(f"✅ Batch Processing: {'Completed' if batch_results else 'Failed'}")
    print(f"✅ Model Management: {'Completed' if model_mgmt_success else 'Failed'}")
    
    print(f"\n🎯 System Performance Summary:")
    if evaluation_results:
        ensemble_metrics = evaluation_results['ensemble']
        print(f"   Overall Accuracy: {ensemble_metrics['accuracy']:.1%}")
        print(f"   F1-Score: {ensemble_metrics['f1']:.1%}")
    
    print(f"\n📚 Next Steps:")
    print("1. Add real currency images to data folders")
    print("2. Retrain with real data for production use")
    print("3. Launch web interface: streamlit run web_app.py")
    print("4. Integrate into your application using the API")
    
    print("\n🎉 Demonstration completed successfully!")

if __name__ == "__main__":
    main()