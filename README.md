# 💰 Advanced Currency Detection System

A comprehensive, professional-grade currency authentication system that combines traditional computer vision, machine learning, and deep learning techniques to detect counterfeit currency notes.

## 🌟 Features

### Multi-Algorithm Approach
- **Traditional Computer Vision**: Color analysis, texture analysis, edge detection, geometric features
- **Machine Learning**: Ensemble of 6+ algorithms including Random Forest, SVM, Gradient Boosting
- **Deep Learning**: Custom CNNs, Transfer Learning (EfficientNet, ResNet, MobileNet), Attention mechanisms

### Supported Currencies
- USD (US Dollar)
- SAR (Saudi Arabian Riyal)
- Easily extensible to other currencies

### Advanced Detection Capabilities
- **Security Feature Detection**: Watermarks, security threads, microprinting
- **Texture Analysis**: GLCM, Local Binary Patterns
- **Geometric Analysis**: Shape detection, contour analysis
- **Color Analysis**: Multi-colorspace analysis, histogram features

### Professional Features
- **Web Interface**: User-friendly Streamlit web application
- **Batch Processing**: Analyze multiple currency notes simultaneously
- **Model Management**: Save/load trained models
- **Performance Analytics**: Comprehensive evaluation metrics
- **Detection History**: Track and analyze past detections

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd currencyDetection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the setup**
```python
python setup.py
```

### Using the Web Interface

1. **Start the web application**
```bash
streamlit run web_app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Train the system** using the sidebar controls

4. **Upload currency images** and get instant authentication results

### Using the Python API

```python
from currency_detector import CurrencyDetectionSystem
import cv2

# Initialize the detection system
detector = CurrencyDetectionSystem()

# Train the system (this will create synthetic data for demonstration)
detector.train_system('USD')

# Load and analyze a currency image
image = cv2.imread('path/to/currency/image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get prediction
result = detector.predict_single_image(image_rgb, 'USD')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence_score']:.2%}")
print(f"Processing Time: {result['processing_time']:.3f} seconds")
```

## 📊 System Architecture

```
Currency Detection System
├── Traditional Computer Vision
│   ├── Color Analysis (RGB, HSV, LAB)
│   ├── Texture Analysis (GLCM, LBP)
│   ├── Edge Detection (Canny, Sobel)
│   ├── Geometric Features (Contours, Shapes)
│   └── Security Features (Watermarks, Threads)
├── Machine Learning
│   ├── Feature Engineering
│   ├── Model Training (6+ algorithms)
│   ├── Ensemble Methods
│   └── Cross-validation
├── Deep Learning
│   ├── Custom CNN Architecture
│   ├── Transfer Learning Models
│   ├── Attention Mechanisms
│   └── Model Ensemble
└── Integration Layer
    ├── Weighted Ensemble
    ├── Confidence Calibration
    └── Result Interpretation
```

## 🔧 Configuration

The system is highly configurable through `config.py`:

```python
# Image processing parameters
IMAGE_SIZE = (224, 224)
EDGE_DETECTION_THRESHOLD = (50, 150)

# Machine Learning parameters
ML_CONFIG = {
    'test_size': 0.2,
    'cross_validation_folds': 5,
    'feature_selection_k': 20
}

# Deep Learning parameters
DL_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high_confidence': 0.8,
    'medium_confidence': 0.6,
    'low_confidence': 0.4
}
```

## 📁 Project Structure

```
currencyDetection/
├── algorithms/
│   ├── traditional_cv.py      # Computer vision algorithms
│   ├── machine_learning.py    # ML models and training
│   └── deep_learning.py       # Neural networks
├── utils/
│   └── data_loader.py         # Data loading utilities
├── data/                      # Dataset directory
│   ├── USD/
│   │   ├── real/
│   │   └── fake/
│   └── SAR/
│       ├── real/
│       └── fake/
├── models/                    # Saved models
├── results/                   # Evaluation results
├── logs/                      # System logs
├── currency_detector.py      # Main detection system
├── web_app.py                 # Streamlit web interface
├── config.py                  # Configuration settings
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🎯 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confidence Calibration**: Reliability of confidence scores

### Expected Performance (with quality dataset):
- Accuracy: >95%
- Precision: >90%
- Recall: >90%
- F1-Score: >90%

## 📚 Technical Details

### Traditional Computer Vision Features

1. **Color Features (25+ features)**
   - RGB, HSV, LAB color statistics
   - Color histograms and peaks
   - Color coherence analysis

2. **Texture Features (30+ features)**
   - Gray Level Co-occurrence Matrix (GLCM)
   - Local Binary Patterns (LBP)
   - Texture energy and entropy

3. **Edge Features (15+ features)**
   - Canny and Sobel edge detection
   - Edge density and orientation
   - Gradient magnitude analysis

4. **Geometric Features (10+ features)**
   - Contour analysis
   - Shape properties (area, perimeter, aspect ratio)
   - Hough transforms for lines and circles

5. **Security Features (10+ features)**
   - Watermark detection
   - Microprinting analysis
   - Security thread detection

### Machine Learning Models

1. **Random Forest**: Ensemble of decision trees
2. **Gradient Boosting**: Sequential weak learner ensemble
3. **Support Vector Machine**: Kernel-based classification
4. **Logistic Regression**: Linear probabilistic model
5. **K-Nearest Neighbors**: Instance-based learning
6. **Naive Bayes**: Probabilistic classifier
7. **Voting Classifier**: Meta-ensemble of all models

### Deep Learning Architectures

1. **Custom CNN**: Purpose-built for currency analysis
2. **Transfer Learning**: Pre-trained ImageNet models
   - EfficientNetB0
   - ResNet50
   - MobileNetV2
3. **Attention CNN**: Focuses on important image regions
4. **Ensemble Network**: Combines multiple models

## 🔒 Security Features Detection

The system can detect various security features commonly found in modern currency:

### USD Security Features
- Watermarks
- Security threads
- Microprinting
- Color-changing ink
- Raised printing

### SAR Security Features
- Watermarks
- Security threads
- Microprinting
- Holographic strips
- Tactile features

## 📖 Usage Examples

### Training with Custom Data

```python
# Add your currency images to the appropriate folders
data/USD/real/     # Real USD notes
data/USD/fake/     # Fake USD notes
data/SAR/real/     # Real SAR notes
data/SAR/fake/     # Fake SAR notes

# Train the system
detector = CurrencyDetectionSystem()
detector.train_system('USD')
detector.save_models('USD')
```

### Batch Processing

```python
import glob
import cv2

# Load multiple images
image_paths = glob.glob('currency_images/*.jpg')
images = []

for path in image_paths:
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    images.append(img_resized)

# Process batch
results = detector.predict_batch(images, 'USD')

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['prediction']} ({result['confidence_score']:.2%})")
```

### Model Evaluation

```python
# Evaluate system performance
evaluation_results = detector.evaluate_system('USD')

print(f"Ensemble Accuracy: {evaluation_results['ensemble']['accuracy']:.2%}")
print(f"Ensemble F1-Score: {evaluation_results['ensemble']['f1']:.2%}")

# Plot feature importance
detector.ml_detector.plot_feature_importance('USD', top_n=20)
```

## 🚨 Important Notes

### Data Requirements
- **High-quality images**: Well-lit, clear, and focused
- **Balanced dataset**: Equal numbers of real and fake currency
- **Diverse samples**: Different denominations, conditions, and angles
- **Proper labeling**: Accurate classification of real vs fake

### Legal Considerations
- Use only legally obtained currency images
- Comply with local laws regarding currency handling
- This system is for educational and research purposes
- Consult legal experts for commercial deployment

### Performance Optimization
- GPU recommended for deep learning training
- Minimum 8GB RAM for large datasets
- SSD storage for faster data loading
- Consider model quantization for deployment

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 🆘 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
```bash
# Install CPU-only TensorFlow if GPU issues occur
pip uninstall tensorflow
pip install tensorflow-cpu
```

2. **Memory Issues**
```python
# Reduce batch size in config.py
DL_CONFIG['batch_size'] = 16  # Reduce from 32
```

3. **Missing Dependencies**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Getting Help

- Check the logs in `currency_detection.log`
- Review the configuration in `config.py`
- Ensure data is properly formatted
- Contact support for technical issues

## 🔮 Future Enhancements

- Multi-currency detection in single image
- Real-time video analysis
- Mobile application
- Blockchain integration for verification
- Advanced anti-spoofing techniques
- Integration with hardware scanners

---

**Made with ❤️ for advancing currency security technology**
