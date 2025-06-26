"""
Data loading utilities for currency detection
"""
import os
import requests
import zipfile
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from config import DATA_DIR, SUPPORTED_CURRENCIES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CurrencyDataLoader:
    """Class to handle currency dataset loading and preprocessing"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.currencies = SUPPORTED_CURRENCIES
        
    def download_sample_datasets(self):
        """
        Download sample currency datasets from publicly available sources
        Note: In a real implementation, you would use actual currency datasets
        For demonstration, we'll create a structure for loading data
        """
        logger.info("Setting up data directory structure...")
        
        for currency in self.currencies:
            # Create directories for each currency
            real_dir = os.path.join(self.data_dir, currency, 'real')
            fake_dir = os.path.join(self.data_dir, currency, 'fake')
            
            os.makedirs(real_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)
            
            logger.info(f"Created directories for {currency}")
        
        # Create a sample dataset info file
        dataset_info = {
            'USD': {
                'denominations': [1, 5, 10, 20, 50, 100],
                'sources': [
                    'https://www.treasury.gov/services/Pages/currency.aspx',
                    'Public domain currency images'
                ]
            },
            'SAR': {
                'denominations': [1, 5, 10, 50, 100, 200, 500],
                'sources': [
                    'Saudi Arabian Monetary Authority (SAMA)',
                    'Public domain currency images'
                ]
            }
        }
        
        # Save dataset info
        info_file = os.path.join(self.data_dir, 'dataset_info.txt')
        with open(info_file, 'w') as f:
            f.write("Currency Detection Dataset Information\n")
            f.write("=====================================\n\n")
            for currency, info in dataset_info.items():
                f.write(f"{currency}:\n")
                f.write(f"  Denominations: {info['denominations']}\n")
                f.write(f"  Sources: {', '.join(info['sources'])}\n\n")
        
        logger.info("Dataset structure created. Please add your currency images to the respective folders.")
        return dataset_info
    
    def load_images_from_directory(self, directory_path, label):
        """Load images from a directory and assign labels"""
        images = []
        labels = []
        filenames = []
        
        if not os.path.exists(directory_path):
            logger.warning(f"Directory {directory_path} does not exist")
            return images, labels, filenames
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(supported_formats):
                img_path = os.path.join(directory_path, filename)
                try:
                    # Load image using OpenCV
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(label)
                        filenames.append(filename)
                except Exception as e:
                    logger.error(f"Error loading image {filename}: {e}")
        
        logger.info(f"Loaded {len(images)} images from {directory_path}")
        return images, labels, filenames
    
    def load_currency_dataset(self, currency='USD'):
        """Load dataset for a specific currency"""
        if currency not in self.currencies:
            raise ValueError(f"Currency {currency} not supported. Use one of: {self.currencies}")
        
        # Load real currency images
        real_dir = os.path.join(self.data_dir, currency, 'real')
        real_images, real_labels, real_filenames = self.load_images_from_directory(real_dir, 1)  # 1 for real
        
        # Load fake currency images
        fake_dir = os.path.join(self.data_dir, currency, 'fake')
        fake_images, fake_labels, fake_filenames = self.load_images_from_directory(fake_dir, 0)  # 0 for fake
        
        # Combine datasets
        all_images = real_images + fake_images
        all_labels = real_labels + fake_labels
        all_filenames = real_filenames + fake_filenames
        
        if len(all_images) == 0:
            logger.warning(f"No images found for {currency}. Please add images to the dataset directories.")
            return None, None, None
        
        logger.info(f"Total dataset size for {currency}: {len(all_images)} images")
        logger.info(f"Real images: {len(real_images)}, Fake images: {len(fake_images)}")
        
        return np.array(all_images), np.array(all_labels), all_filenames
    
    def preprocess_images(self, images, target_size=(224, 224)):
        """Preprocess images for model training"""
        processed_images = []
        
        for img in images:
            # Resize image
            img_resized = cv2.resize(img, target_size)
            
            # Normalize pixel values
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            processed_images.append(img_normalized)
        
        return np.array(processed_images)
    
    def create_train_test_split(self, images, labels, test_size=0.2, random_state=42):
        """Create train-test split"""
        return train_test_split(images, labels, test_size=test_size, 
                               random_state=random_state, stratify=labels)
    
    def generate_synthetic_data(self, currency='USD', num_samples=100):
        """
        Generate synthetic currency data for demonstration purposes
        In a real scenario, you would use actual currency images
        """
        logger.info(f"Generating synthetic data for {currency}...")
        
        synthetic_images = []
        synthetic_labels = []
        
        # Create synthetic images with different characteristics
        for i in range(num_samples):
            # Create a base image
            if currency == 'USD':
                # Create USD-like patterns (greenish colors)
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img[:, :, 1] = np.clip(img[:, :, 1] + 50, 0, 255)  # Add green tint
            else:  # SAR
                # Create SAR-like patterns (brownish/golden colors)
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img[:, :, 0] = np.clip(img[:, :, 0] + 30, 0, 255)  # Add red tint
                img[:, :, 1] = np.clip(img[:, :, 1] + 30, 0, 255)  # Add green tint
            
            # Add some geometric patterns (simulating security features)
            cv2.rectangle(img, (50, 50), (174, 174), (255, 255, 255), 2)
            cv2.circle(img, (112, 112), 30, (0, 0, 0), 1)
            
            synthetic_images.append(img)
            synthetic_labels.append(1 if i % 2 == 0 else 0)  # Alternate between real and fake
        
        # Save synthetic data
        synthetic_dir = os.path.join(self.data_dir, currency, 'synthetic')
        os.makedirs(synthetic_dir, exist_ok=True)
        
        for i, (img, label) in enumerate(zip(synthetic_images, synthetic_labels)):
            filename = f"synthetic_{currency}_{i:03d}_{'real' if label else 'fake'}.jpg"
            filepath = os.path.join(synthetic_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Generated {num_samples} synthetic images for {currency}")
        return np.array(synthetic_images), np.array(synthetic_labels)

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    loader = CurrencyDataLoader()
    
    # Create directory structure
    loader.download_sample_datasets()
    
    # Generate synthetic data for both currencies
    usd_images, usd_labels = loader.generate_synthetic_data('USD', 200)
    sar_images, sar_labels = loader.generate_synthetic_data('SAR', 200)
    
    return {
        'USD': (usd_images, usd_labels),
        'SAR': (sar_images, sar_labels)
    }

if __name__ == "__main__":
    # Create sample dataset for demonstration
    create_sample_dataset()
    logger.info("Sample dataset created successfully!")