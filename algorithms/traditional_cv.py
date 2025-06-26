"""
Traditional Computer Vision algorithms for currency detection
Includes edge detection, color analysis, texture analysis, and geometric feature detection
"""
import cv2
import numpy as np
from skimage import feature, measure, segmentation
try:
    from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
except ImportError:
    try:
        from skimage.feature import gray_comatrix as greycomatrix, gray_coprops as greycoprops, local_binary_pattern
    except ImportError:
        # For newer versions of scikit-image
        from skimage.feature import local_binary_pattern
        
        def greycomatrix(*args, **kwargs):
            """Placeholder for greycomatrix if not available"""
            return None
            
        def greycoprops(*args, **kwargs):
            """Placeholder for greycoprops if not available"""
            return [[0.5]]
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class TraditionalCVDetector:
    """Traditional Computer Vision based currency authentication"""
    
    def __init__(self):
        self.features = {}
        
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color-based features from currency image"""
        features = {}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # RGB statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i].flatten()
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_skew'] = self._calculate_skewness(channel_data)
            features[f'{channel}_kurt'] = self._calculate_kurtosis(channel_data)
        
        # HSV statistics
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = hsv[:, :, i].flatten()
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
        
        # LAB statistics
        for i, channel in enumerate(['L', 'A', 'B']):
            channel_data = lab[:, :, i].flatten()
            features[f'LAB_{channel}_mean'] = np.mean(channel_data)
            features[f'LAB_{channel}_std'] = np.std(channel_data)
        
        # Color histogram features
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        features['hist_r_peak'] = np.argmax(hist_r)
        features['hist_g_peak'] = np.argmax(hist_g)
        features['hist_b_peak'] = np.argmax(hist_b)
        
        # Color coherence
        features['color_coherence'] = self._calculate_color_coherence(image)
        
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features using GLCM and LBP"""
        features = {}
        
        # Convert to grayscale
        gray = rgb2gray(image)
        gray_uint = img_as_ubyte(gray)
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = greycomatrix(gray_uint, distances=distances, angles=angles, 
                          levels=256, symmetric=True, normed=True)
        
        # Calculate GLCM properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in properties:
            prop_values = greycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(prop_values)
            features[f'glcm_{prop}_std'] = np.std(prop_values)
        
        # Local Binary Pattern (LBP) features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                  range=(0, n_points + 2), density=True)
        features.update({f'lbp_bin_{i}': lbp_hist[i] for i in range(len(lbp_hist))})
        
        # Additional texture measures
        features['texture_energy'] = np.sum(gray**2)
        features['texture_entropy'] = self._calculate_entropy(gray_uint)
        
        return features
    
    def extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract edge-based features"""
        features = {}
        
        # Convert to grayscale
        gray = rgb2gray(image)
        
        # Canny edge detection
        edges_canny = feature.canny(gray, sigma=1.5, low_threshold=0.1, high_threshold=0.2)
        features['canny_edge_density'] = np.sum(edges_canny) / edges_canny.size
        
        # Sobel edge detection
        edges_sobel = sobel(gray)
        features['sobel_mean'] = np.mean(edges_sobel)
        features['sobel_std'] = np.std(edges_sobel)
        
        # Edge direction histogram
        grad_x = cv2.Sobel(img_as_ubyte(gray), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_as_ubyte(gray), cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Direction histogram (8 bins for 8 directions)
        dir_hist, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist / np.sum(dir_hist)  # Normalize
        
        features.update({f'edge_dir_{i}': dir_hist[i] for i in range(8)})
        features['edge_magnitude_mean'] = np.mean(magnitude)
        features['edge_magnitude_std'] = np.std(magnitude)
        
        return features
    
    def extract_geometric_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract geometric features like shapes, contours, and patterns"""
        features = {}
        
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Largest contour (assuming it's the currency boundary)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Contour features
            features['contour_area'] = cv2.contourArea(largest_contour)
            features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['aspect_ratio'] = w / h if h != 0 else 0
            
            # Extent (ratio of contour area to bounding rectangle area)
            rect_area = w * h
            features['extent'] = features['contour_area'] / rect_area if rect_area != 0 else 0
            
            # Solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = features['contour_area'] / hull_area if hull_area != 0 else 0
            
            # Compactness
            features['compactness'] = (features['contour_perimeter']**2) / features['contour_area'] if features['contour_area'] != 0 else 0
        
        # Hough Line Transform for detecting straight lines (security threads, borders)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        features['num_lines'] = len(lines) if lines is not None else 0
        
        # Hough Circle Transform for detecting circular patterns
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        features['num_circles'] = len(circles[0]) if circles is not None else 0
        
        return features
    
    def detect_security_features(self, image: np.ndarray, currency: str = 'USD') -> Dict[str, float]:
        """Detect currency-specific security features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Watermark detection (using variance in specific regions)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        features['watermark_variance'] = np.var(center_region)
        
        # Microprinting detection (high frequency components)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # High frequency energy (indicator of microprinting)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h//2, w//2
        high_freq_region = magnitude_spectrum[center_h-h//8:center_h+h//8, 
                                           center_w-w//8:center_w+w//8]
        features['microprint_energy'] = np.sum(high_freq_region)
        
        # Security thread detection (vertical or horizontal lines)
        # Horizontal projection
        horizontal_proj = np.sum(gray, axis=1)
        features['horizontal_line_strength'] = np.max(horizontal_proj) - np.min(horizontal_proj)
        
        # Vertical projection
        vertical_proj = np.sum(gray, axis=0)
        features['vertical_line_strength'] = np.max(vertical_proj) - np.min(vertical_proj)
        
        # Raised printing detection (using gradient analysis)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['raised_print_strength'] = np.mean(gradient_magnitude)
        
        return features
    
    def extract_all_features(self, image: np.ndarray, currency: str = 'USD') -> Dict[str, float]:
        """Extract all traditional CV features from an image"""
        all_features = {}
        
        try:
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
            
            # Security features
            security_features = self.detect_security_features(image, currency)
            all_features.update({f'security_{k}': v for k, v in security_features.items()})
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            
        return all_features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_color_coherence(self, image: np.ndarray) -> float:
        """Calculate color coherence of the image"""
        # Simple implementation: variance of RGB channels
        return np.var(image)
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of the image"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))

# Example usage and testing
if __name__ == "__main__":
    # Create a sample detector
    detector = TraditionalCVDetector()
    
    # Create a sample image for testing
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Extract features
    features = detector.extract_all_features(sample_image, 'USD')
    
    print(f"Extracted {len(features)} traditional CV features:")
    for i, (feature_name, value) in enumerate(list(features.items())[:10]):
        print(f"{i+1:2d}. {feature_name}: {value:.4f}")
    
    print(f"... and {len(features) - 10} more features")