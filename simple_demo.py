"""
Simple demonstration of the Currency Detection System
Shows core computer vision features without complex ML dependencies
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCurrencyDetector:
    """Simplified currency detector using basic computer vision"""
    
    def __init__(self):
        self.features = {}
        
    def extract_color_features(self, image):
        """Extract basic color features"""
        features = {}
        
        # RGB channel statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i]
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_min'] = np.min(channel_data)
            features[f'{channel}_max'] = np.max(channel_data)
        
        # HSV statistics
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = hsv[:, :, i]
            features[f'HSV_{channel}_mean'] = np.mean(channel_data)
            features[f'HSV_{channel}_std'] = np.std(channel_data)
        
        return features
    
    def extract_texture_features(self, image):
        """Extract basic texture features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Basic texture measures
        features['texture_variance'] = np.var(gray)
        features['texture_mean'] = np.mean(gray)
        features['texture_std'] = np.std(gray)
        
        # Calculate image entropy
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        features['texture_entropy'] = -np.sum(hist * np.log2(hist))
        
        return features
    
    def extract_edge_features(self, image):
        """Extract edge-based features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_mean'] = np.mean(magnitude)
        features['gradient_std'] = np.std(magnitude)
        
        return features
    
    def extract_geometric_features(self, image):
        """Extract geometric features"""
        features = {}
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            features['contour_area'] = cv2.contourArea(largest_contour)
            features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['aspect_ratio'] = w / h if h != 0 else 0
            
            # Number of contours
            features['num_contours'] = len(contours)
        else:
            features['contour_area'] = 0
            features['contour_perimeter'] = 0
            features['aspect_ratio'] = 0
            features['num_contours'] = 0
        
        return features
    
    def extract_all_features(self, image):
        """Extract all features from an image"""
        all_features = {}
        
        # Color features
        color_features = self.extract_color_features(image)
        all_features.update({f'color_{k}': v for k, v in color_features.items()})
        
        # Texture features
        texture_features = self.extract_texture_features(image)
        all_features.update({f'texture_{k}': v for k, v in texture_features.items()})
        
        # Edge features
        edge_features = self.extract_edge_features(image)
        all_features.update({f'edge_{k}': v for k, v in edge_features.items()})
        
        # Geometric features
        geometric_features = self.extract_geometric_features(image)
        all_features.update({f'geometric_{k}': v for k, v in geometric_features.items()})
        
        return all_features
    
    def simple_predict(self, image):
        """Simple prediction based on feature analysis"""
        features = self.extract_all_features(image)
        
        # Simple heuristic-based prediction
        # This is a demonstration - real models would be much more sophisticated
        
        # Check color consistency (real currency has consistent color patterns)
        color_consistency = features.get('color_R_std', 0) + features.get('color_G_std', 0) + features.get('color_B_std', 0)
        
        # Check edge density (real currency has more detailed edges)
        edge_density = features.get('edge_edge_density', 0)
        
        # Check texture complexity
        texture_entropy = features.get('texture_texture_entropy', 0)
        
        # Simple scoring system
        authenticity_score = 0
        
        if color_consistency > 50:  # Good color variation
            authenticity_score += 0.3
        
        if edge_density > 0.1:  # Sufficient edge details
            authenticity_score += 0.4
        
        if texture_entropy > 5:  # Complex texture
            authenticity_score += 0.3
        
        # Convert to percentage
        confidence = min(authenticity_score, 1.0)
        prediction = "Real" if confidence > 0.5 else "Fake"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'features_count': len(features),
            'key_features': {
                'color_consistency': color_consistency,
                'edge_density': edge_density,
                'texture_entropy': texture_entropy
            }
        }

def create_sample_currency_image():
    """Create a sample currency-like image for demonstration"""
    # Create base image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    
    # Add greenish tint (like USD)
    img[:, :, 1] = np.clip(img[:, :, 1] + 30, 0, 255)  # More green
    
    # Add some geometric patterns
    cv2.rectangle(img, (20, 20), (204, 204), (100, 150, 100), 3)
    cv2.rectangle(img, (40, 40), (184, 184), (80, 120, 80), 2)
    
    # Add circular pattern
    cv2.circle(img, (112, 112), 40, (60, 100, 60), 2)
    cv2.circle(img, (112, 112), 20, (40, 80, 40), 1)
    
    # Add some text-like patterns
    cv2.putText(img, "SAMPLE", (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 80, 40), 2)
    cv2.putText(img, "CURRENCY", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 80, 40), 1)
    
    # Add some noise for texture
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def demonstrate_feature_extraction():
    """Demonstrate feature extraction"""
    print("🔍 FEATURE EXTRACTION DEMONSTRATION")
    print("="*50)
    
    # Create sample image
    sample_image = create_sample_currency_image()
    
    # Initialize detector
    detector = SimpleCurrencyDetector()
    
    # Extract features
    print("Extracting features from sample currency image...")
    features = detector.extract_all_features(sample_image)
    
    print(f"✅ Extracted {len(features)} features")
    print("\n📊 Sample Features:")
    
    # Show first 15 features
    feature_items = list(features.items())[:15]
    for i, (name, value) in enumerate(feature_items, 1):
        print(f"{i:2d}. {name}: {value:.4f}")
    
    if len(features) > 15:
        print(f"... and {len(features) - 15} more features")
    
    return sample_image, features

def demonstrate_prediction():
    """Demonstrate currency prediction"""
    print("\n🔮 PREDICTION DEMONSTRATION")
    print("="*50)
    
    detector = SimpleCurrencyDetector()
    
    # Create different sample images
    print("Creating sample currency images...")
    
    # Sample 1: "Real" currency (more detailed)
    real_currency = create_sample_currency_image()
    
    # Sample 2: "Fake" currency (simpler, less detailed)
    fake_currency = np.ones((224, 224, 3), dtype=np.uint8) * 180
    cv2.rectangle(fake_currency, (50, 50), (174, 174), (100, 100, 100), 2)
    cv2.putText(fake_currency, "FAKE", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    
    samples = [
        ("Real Currency Sample", real_currency),
        ("Fake Currency Sample", fake_currency)
    ]
    
    print("\nAnalyzing samples...")
    for name, image in samples:
        print(f"\n🔍 Analyzing {name}:")
        
        start_time = time.time()
        result = detector.simple_predict(image)
        processing_time = time.time() - start_time
        
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Features Used: {result['features_count']}")
        print(f"   Processing Time: {processing_time:.3f} seconds")
        
        print(f"   Key Indicators:")
        for indicator, value in result['key_features'].items():
            print(f"     {indicator}: {value:.4f}")

def demonstrate_batch_processing():
    """Demonstrate batch processing"""
    print("\n📦 BATCH PROCESSING DEMONSTRATION")
    print("="*50)
    
    detector = SimpleCurrencyDetector()
    
    # Create batch of images
    batch_size = 5
    print(f"Creating batch of {batch_size} sample images...")
    
    batch_images = []
    expected_labels = []
    
    for i in range(batch_size):
        if i % 2 == 0:
            # Create "real" currency
            img = create_sample_currency_image()
            expected_labels.append("Real")
        else:
            # Create "fake" currency
            img = np.ones((224, 224, 3), dtype=np.uint8) * 160
            cv2.rectangle(img, (30, 30), (194, 194), (80, 80, 80), 1)
            expected_labels.append("Fake")
        
        batch_images.append(img)
    
    print("Processing batch...")
    start_time = time.time()
    
    results = []
    for i, image in enumerate(batch_images):
        result = detector.simple_predict(image)
        result['index'] = i
        result['expected'] = expected_labels[i]
        results.append(result)
    
    total_time = time.time() - start_time
    
    print(f"✅ Batch processing completed in {total_time:.3f} seconds")
    print(f"📊 Results:")
    
    correct = 0
    for result in results:
        status = "✅" if result['prediction'] == result['expected'] else "❌"
        print(f"   Image {result['index']+1}: {result['prediction']} (Expected: {result['expected']}) {status}")
        print(f"      Confidence: {result['confidence']:.1%}")
        
        if result['prediction'] == result['expected']:
            correct += 1
    
    accuracy = correct / len(results)
    print(f"\n📈 Batch Summary:")
    print(f"   Total Images: {len(results)}")
    print(f"   Correct Predictions: {correct}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Avg Processing Time: {total_time/len(results):.3f} seconds per image")

def save_sample_images():
    """Save sample images for visualization"""
    print("\n💾 SAVING SAMPLE IMAGES")
    print("="*30)
    
    # Create data directory if it doesn't exist
    os.makedirs("sample_images", exist_ok=True)
    
    # Create and save real currency sample
    real_currency = create_sample_currency_image()
    cv2.imwrite("sample_images/real_currency_sample.jpg", cv2.cvtColor(real_currency, cv2.COLOR_RGB2BGR))
    
    # Create and save fake currency sample
    fake_currency = np.ones((224, 224, 3), dtype=np.uint8) * 180
    cv2.rectangle(fake_currency, (50, 50), (174, 174), (100, 100, 100), 2)
    cv2.putText(fake_currency, "FAKE", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    cv2.imwrite("sample_images/fake_currency_sample.jpg", cv2.cvtColor(fake_currency, cv2.COLOR_RGB2BGR))
    
    print("✅ Sample images saved to 'sample_images/' folder")
    print("   - real_currency_sample.jpg")
    print("   - fake_currency_sample.jpg")

def main():
    """Main demonstration function"""
    print("🎭 SIMPLIFIED CURRENCY DETECTION DEMONSTRATION")
    print("="*60)
    print("This demonstration shows core computer vision features")
    print("for currency authentication using only basic dependencies.")
    print("="*60)
    
    try:
        # Step 1: Feature Extraction
        sample_image, features = demonstrate_feature_extraction()
        
        # Step 2: Prediction
        demonstrate_prediction()
        
        # Step 3: Batch Processing
        demonstrate_batch_processing()
        
        # Step 4: Save sample images
        save_sample_images()
        
        # Summary
        print("\n🏁 DEMONSTRATION SUMMARY")
        print("="*50)
        print("✅ Feature Extraction: Completed")
        print("   - Color features (RGB, HSV statistics)")
        print("   - Texture features (variance, entropy)")
        print("   - Edge features (Canny, Sobel)")
        print("   - Geometric features (contours, shapes)")
        
        print("✅ Prediction: Completed")
        print("   - Simple heuristic-based classification")
        print("   - Confidence scoring")
        
        print("✅ Batch Processing: Completed")
        print("   - Multiple image analysis")
        print("   - Performance metrics")
        
        print("✅ Sample Generation: Completed")
        print("   - Synthetic currency images")
        print("   - Saved for visualization")
        
        print(f"\n🎯 Key Insights:")
        print("   - Real currency detection requires sophisticated algorithms")
        print("   - Multiple features provide better accuracy")
        print("   - Color, texture, and edge analysis are fundamental")
        print("   - Machine learning would significantly improve results")
        
        print(f"\n📚 Next Steps for Production:")
        print("1. Collect real currency datasets")
        print("2. Implement machine learning models")
        print("3. Add deep learning for better accuracy")
        print("4. Include security feature detection")
        print("5. Deploy with web interface")
        
        print("\n🎉 Basic demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    main()