"""
Advanced Feature Extraction System for Currency Detection
Professional-grade feature extraction for USD and SAR currencies
"""
import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try importing advanced libraries, fallback if not available
try:
    from skimage import feature, measure, segmentation, filters
    from skimage.feature import local_binary_pattern, hog
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False
    logging.warning("Advanced features not available. Install scikit-image for full functionality.")

try:
    import pywt
    WAVELET_FEATURES = True
except ImportError:
    WAVELET_FEATURES = False
    logging.warning("Wavelet features not available. Install PyWavelets for full functionality.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    """Professional feature extraction for currency authentication"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        self.currency_specific_features = {
            'USD': {
                'expected_colors': {
                    'primary_green': [100, 150, 100],
                    'security_blue': [50, 80, 120],
                    'background_cream': [220, 215, 200]
                },
                'security_thread_position': 0.5,  # Relative position
                'watermark_regions': [(0.7, 0.8, 0.1, 0.3)],  # (x, y, w, h) relative
                'denomination_regions': [(0.1, 0.1, 0.3, 0.3), (0.6, 0.6, 0.3, 0.3)]
            },
            'SAR': {
                'expected_colors': {
                    'primary_brown': [160, 120, 80],
                    'security_gold': [200, 180, 100],
                    'background_tan': [200, 180, 160]
                },
                'security_thread_position': 0.45,
                'watermark_regions': [(0.2, 0.2, 0.2, 0.4)],
                'denomination_regions': [(0.3, 0.3, 0.4, 0.4)]
            }
        }
    
    def extract_comprehensive_features(self, image: np.ndarray, currency: str = 'USD') -> Dict[str, float]:
        """Extract comprehensive feature set for currency authentication"""
        
        features = {}
        
        # Basic preprocessing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 1. Color-based features
        color_features = self._extract_advanced_color_features(image, currency)
        features.update(color_features)
        
        # 2. Texture analysis
        texture_features = self._extract_advanced_texture_features(gray)
        features.update(texture_features)
        
        # 3. Edge and gradient features
        edge_features = self._extract_advanced_edge_features(gray)
        features.update(edge_features)
        
        # 4. Geometric and shape features
        geometric_features = self._extract_geometric_features(gray)
        features.update(geometric_features)
        
        # 5. Security-specific features
        security_features = self._extract_security_features(image, currency)
        features.update(security_features)
        
        # 6. Statistical features
        statistical_features = self._extract_statistical_features(gray)
        features.update(statistical_features)
        
        # 7. Frequency domain features
        frequency_features = self._extract_frequency_features(gray)
        features.update(frequency_features)
        
        # 8. Advanced texture descriptors
        if ADVANCED_FEATURES:
            advanced_features = self._extract_advanced_descriptors(gray)
            features.update(advanced_features)
        
        # 9. Wavelet features
        if WAVELET_FEATURES:
            wavelet_features = self._extract_wavelet_features(gray)
            features.update(wavelet_features)
        
        logger.debug(f"Extracted {len(features)} features for {currency}")
        return features
    
    def _extract_advanced_color_features(self, image: np.ndarray, currency: str) -> Dict[str, float]:
        """Extract advanced color-based features"""
        features = {}
        
        # Multiple color spaces
        rgb = image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        color_spaces = {'RGB': rgb, 'HSV': hsv, 'LAB': lab, 'YUV': yuv}
        
        for space_name, color_img in color_spaces.items():
            for channel in range(3):
                channel_data = color_img[:, :, channel].flatten()
                
                # Basic statistics
                features[f'{space_name}_ch{channel}_mean'] = np.mean(channel_data)
                features[f'{space_name}_ch{channel}_std'] = np.std(channel_data)
                features[f'{space_name}_ch{channel}_skew'] = skew(channel_data)
                features[f'{space_name}_ch{channel}_kurtosis'] = kurtosis(channel_data)
                
                # Percentiles
                features[f'{space_name}_ch{channel}_p25'] = np.percentile(channel_data, 25)
                features[f'{space_name}_ch{channel}_p75'] = np.percentile(channel_data, 75)
                
                # Histogram features
                hist, _ = np.histogram(channel_data, bins=32, range=(0, 256))
                hist = hist / np.sum(hist)  # Normalize
                features[f'{space_name}_ch{channel}_entropy'] = entropy(hist + 1e-10)
                features[f'{space_name}_ch{channel}_peak'] = np.argmax(hist)
        
        # Color coherence and consistency
        features['color_coherence'] = self._calculate_color_coherence(rgb)
        
        # Currency-specific color matching
        if currency in self.currency_specific_features:
            color_match_score = self._calculate_color_match_score(rgb, currency)
            features['currency_color_match'] = color_match_score
        
        return features
    
    def _extract_advanced_texture_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract advanced texture features"""
        features = {}
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        glcm_features = self._calculate_glcm_features(gray)
        features.update(glcm_features)
        
        # Local Binary Pattern (LBP)
        if ADVANCED_FEATURES:
            lbp_features = self._calculate_lbp_features(gray)
            features.update(lbp_features)
        
        # Texture energy and homogeneity
        features['texture_energy'] = np.sum(gray.astype(np.float64) ** 2) / (gray.shape[0] * gray.shape[1])
        features['texture_variance'] = np.var(gray)
        
        # Laws' texture measures
        laws_features = self._calculate_laws_texture(gray)
        features.update(laws_features)
        
        # Gabor filter responses
        gabor_features = self._calculate_gabor_features(gray)
        features.update(gabor_features)
        
        return features
    
    def _extract_advanced_edge_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract advanced edge and gradient features"""
        features = {}
        
        # Multiple edge detectors
        canny = cv2.Canny(gray, 50, 150)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Edge statistics
        features['canny_edge_density'] = np.sum(canny > 0) / canny.size
        features['canny_edge_mean'] = np.mean(canny)
        
        # Gradient magnitude and direction
        grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_direction = np.arctan2(sobel_y, sobel_x)
        
        features['gradient_magnitude_mean'] = np.mean(grad_magnitude)
        features['gradient_magnitude_std'] = np.std(grad_magnitude)
        features['gradient_direction_std'] = np.std(grad_direction)
        
        # Laplacian features
        features['laplacian_variance'] = np.var(laplacian)
        features['laplacian_energy'] = np.sum(laplacian**2)
        
        # Edge orientation histogram
        edge_hist = self._calculate_edge_orientation_histogram(sobel_x, sobel_y)
        for i, bin_val in enumerate(edge_hist):
            features[f'edge_orientation_bin_{i}'] = bin_val
        
        return features
    
    def _extract_geometric_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract geometric and shape features"""
        features = {}
        
        # Thresholding for shape analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Contour analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Largest contour analysis
            largest_contour = max(contours, key=cv2.contourArea)
            
            features['contour_area'] = cv2.contourArea(largest_contour)
            features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
            
            # Shape descriptors
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['aspect_ratio'] = w / h if h > 0 else 0
            features['extent'] = cv2.contourArea(largest_contour) / (w * h) if w * h > 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            features['solidity'] = cv2.contourArea(largest_contour) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            # Moments and shape invariants
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                features['centroid_x'] = moments['m10'] / moments['m00']
                features['centroid_y'] = moments['m01'] / moments['m00']
            
            # Hu moments (rotation invariant)
            hu_moments = cv2.HuMoments(moments)
            for i, hu in enumerate(hu_moments.flatten()):
                features[f'hu_moment_{i}'] = -np.sign(hu) * np.log10(np.abs(hu)) if hu != 0 else 0
        
        # Corner detection
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        features['corner_count'] = len(corners) if corners is not None else 0
        
        return features
    
    def _extract_security_features(self, image: np.ndarray, currency: str) -> Dict[str, float]:
        """Extract currency-specific security features"""
        features = {}
        
        # Security thread detection
        thread_score = self._detect_security_thread(image, currency)
        features['security_thread_score'] = thread_score
        
        # Watermark detection
        watermark_score = self._detect_watermark(image, currency)
        features['watermark_score'] = watermark_score
        
        # Microprinting detection
        microprint_score = self._detect_microprinting(image)
        features['microprint_score'] = microprint_score
        
        # Holographic features (for SAR)
        if currency == 'SAR':
            holo_score = self._detect_holographic_features(image)
            features['holographic_score'] = holo_score
        
        # Print quality assessment
        print_quality = self._assess_print_quality(image)
        features['print_quality_score'] = print_quality
        
        return features
    
    def _extract_statistical_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract statistical features"""
        features = {}
        
        # Basic statistics
        features['pixel_mean'] = np.mean(gray)
        features['pixel_std'] = np.std(gray)
        features['pixel_skewness'] = skew(gray.flatten())
        features['pixel_kurtosis'] = kurtosis(gray.flatten())
        
        # Range and percentiles
        features['pixel_range'] = np.ptp(gray)
        features['pixel_p10'] = np.percentile(gray, 10)
        features['pixel_p90'] = np.percentile(gray, 90)
        
        # Entropy
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        features['pixel_entropy'] = entropy(hist + 1e-10)
        
        return features
    
    def _extract_frequency_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        features = {}
        
        # FFT analysis
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        fft_phase = np.angle(fft)
        
        # Frequency domain statistics
        features['fft_magnitude_mean'] = np.mean(fft_magnitude)
        features['fft_magnitude_std'] = np.std(fft_magnitude)
        features['fft_phase_std'] = np.std(fft_phase)
        
        # Power spectral density
        psd = fft_magnitude ** 2
        features['power_spectral_density'] = np.sum(psd)
        
        # Dominant frequencies
        freq_hist, _ = np.histogram(fft_magnitude.flatten(), bins=50)
        features['dominant_freq_bin'] = np.argmax(freq_hist)
        
        return features
    
    def _extract_advanced_descriptors(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract advanced texture descriptors (requires scikit-image)"""
        features = {}
        
        try:
            # HOG (Histogram of Oriented Gradients)
            hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=False)
            
            # Statistical summary of HOG features
            features['hog_mean'] = np.mean(hog_features)
            features['hog_std'] = np.std(hog_features)
            features['hog_energy'] = np.sum(hog_features**2)
            
            # Local Binary Pattern
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
            lbp_hist = lbp_hist / np.sum(lbp_hist)
            
            for i, val in enumerate(lbp_hist):
                features[f'lbp_bin_{i}'] = val
            
        except Exception as e:
            logger.debug(f"Advanced descriptors extraction failed: {e}")
        
        return features
    
    def _extract_wavelet_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract wavelet-based features"""
        features = {}
        
        try:
            # Wavelet decomposition
            coeffs = pywt.dwt2(gray, 'db4')
            cA, (cH, cV, cD) = coeffs
            
            # Statistics for each subband
            subbands = {'LL': cA, 'LH': cH, 'HL': cV, 'HH': cD}
            
            for name, subband in subbands.items():
                features[f'wavelet_{name}_energy'] = np.sum(subband**2)
                features[f'wavelet_{name}_std'] = np.std(subband)
                features[f'wavelet_{name}_mean'] = np.mean(subband)
                
        except Exception as e:
            logger.debug(f"Wavelet features extraction failed: {e}")
        
        return features
    
    # Helper methods for specific feature calculations
    
    def _calculate_color_coherence(self, rgb: np.ndarray) -> float:
        """Calculate color coherence measure"""
        # Simple color coherence based on neighboring pixel similarity
        diff_sum = 0
        count = 0
        
        for i in range(1, rgb.shape[0]):
            for j in range(1, rgb.shape[1]):
                diff = np.linalg.norm(rgb[i,j] - rgb[i-1,j]) + np.linalg.norm(rgb[i,j] - rgb[i,j-1])
                diff_sum += diff
                count += 1
        
        return diff_sum / count if count > 0 else 0
    
    def _calculate_color_match_score(self, rgb: np.ndarray, currency: str) -> float:
        """Calculate how well colors match expected currency colors"""
        expected_colors = self.currency_specific_features[currency]['expected_colors']
        
        mean_color = np.mean(rgb.reshape(-1, 3), axis=0)
        
        min_distance = float('inf')
        for color_name, expected_rgb in expected_colors.items():
            distance = np.linalg.norm(mean_color - np.array(expected_rgb))
            min_distance = min(min_distance, distance)
        
        # Convert distance to similarity score (0-1, higher is better)
        return max(0, 1 - min_distance / 255)
    
    def _calculate_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate GLCM (Gray Level Co-occurrence Matrix) features"""
        features = {}
        
        # Simple GLCM implementation
        try:
            # Calculate co-occurrence for different directions
            glcm = self._compute_glcm(gray, distance=1, angle=0)
            
            # Calculate texture properties
            features['glcm_contrast'] = self._glcm_contrast(glcm)
            features['glcm_dissimilarity'] = self._glcm_dissimilarity(glcm)
            features['glcm_homogeneity'] = self._glcm_homogeneity(glcm)
            features['glcm_energy'] = self._glcm_energy(glcm)
            
        except Exception as e:
            logger.debug(f"GLCM calculation failed: {e}")
            # Fallback values
            features.update({
                'glcm_contrast': 0,
                'glcm_dissimilarity': 0,
                'glcm_homogeneity': 0,
                'glcm_energy': 0
            })
        
        return features
    
    def _compute_glcm(self, gray: np.ndarray, distance: int, angle: int) -> np.ndarray:
        """Compute GLCM matrix"""
        # Simplified GLCM computation
        levels = 16  # Reduce levels for efficiency
        gray_scaled = (gray // (256 // levels)).astype(int)
        
        glcm = np.zeros((levels, levels), dtype=float)
        
        rows, cols = gray_scaled.shape
        for i in range(rows - distance):
            for j in range(cols - distance):
                ref_pixel = gray_scaled[i, j]
                neighbor_pixel = gray_scaled[i + distance, j]
                if 0 <= ref_pixel < levels and 0 <= neighbor_pixel < levels:
                    glcm[ref_pixel, neighbor_pixel] += 1
        
        # Normalize
        glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
        return glcm
    
    def _glcm_contrast(self, glcm: np.ndarray) -> float:
        """Calculate GLCM contrast"""
        contrast = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i, j] * (i - j) ** 2
        return contrast
    
    def _glcm_dissimilarity(self, glcm: np.ndarray) -> float:
        """Calculate GLCM dissimilarity"""
        dissimilarity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                dissimilarity += glcm[i, j] * abs(i - j)
        return dissimilarity
    
    def _glcm_homogeneity(self, glcm: np.ndarray) -> float:
        """Calculate GLCM homogeneity"""
        homogeneity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
        return homogeneity
    
    def _glcm_energy(self, glcm: np.ndarray) -> float:
        """Calculate GLCM energy"""
        return np.sum(glcm ** 2)
    
    def _calculate_lbp_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate Local Binary Pattern features"""
        features = {}
        
        try:
            # Simple LBP implementation
            lbp = self._compute_lbp(gray)
            
            # LBP histogram
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist / np.sum(hist)
            
            # Statistical features from LBP
            features['lbp_uniformity'] = np.sum(hist ** 2)
            features['lbp_entropy'] = entropy(hist + 1e-10)
            features['lbp_mean'] = np.mean(lbp)
            features['lbp_std'] = np.std(lbp)
            
        except Exception as e:
            logger.debug(f"LBP calculation failed: {e}")
            features.update({
                'lbp_uniformity': 0,
                'lbp_entropy': 0,
                'lbp_mean': 0,
                'lbp_std': 0
            })
        
        return features
    
    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern"""
        # Simplified LBP implementation
        rows, cols = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray[i, j]
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                binary_string = ''.join(['1' if n >= center else '0' for n in neighbors])
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def _calculate_laws_texture(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate Laws' texture measures"""
        features = {}
        
        # Laws' texture kernels
        L5 = np.array([1, 4, 6, 4, 1])  # Level
        E5 = np.array([-1, -2, 0, 2, 1])  # Edge
        S5 = np.array([-1, 0, 2, 0, -1])  # Spot
        
        kernels = {
            'LL': np.outer(L5, L5),
            'LE': np.outer(L5, E5),
            'EL': np.outer(E5, L5),
            'EE': np.outer(E5, E5),
            'SS': np.outer(S5, S5)
        }
        
        for name, kernel in kernels.items():
            filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            features[f'laws_{name}_energy'] = np.mean(filtered ** 2)
            features[f'laws_{name}_mean'] = np.mean(filtered)
        
        return features
    
    def _calculate_gabor_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate Gabor filter responses"""
        features = {}
        
        # Gabor filter parameters
        orientations = [0, 45, 90, 135]
        frequencies = [0.1, 0.3, 0.5]
        
        for angle in orientations:
            for freq in frequencies:
                # Create Gabor kernel
                kernel_real = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 
                                               2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel_real)
                
                feature_name = f'gabor_{angle}deg_{freq:.1f}freq'
                features[f'{feature_name}_mean'] = np.mean(filtered)
                features[f'{feature_name}_std'] = np.std(filtered)
                features[f'{feature_name}_energy'] = np.sum(filtered ** 2)
        
        return features
    
    def _calculate_edge_orientation_histogram(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Calculate edge orientation histogram"""
        # Calculate gradient orientations
        orientations = np.arctan2(grad_y, grad_x)
        
        # Convert to degrees and make positive
        orientations_deg = np.degrees(orientations) % 180
        
        # Create histogram
        hist, _ = np.histogram(orientations_deg, bins=18, range=(0, 180))
        
        # Normalize
        return hist / np.sum(hist) if np.sum(hist) > 0 else hist
    
    def _detect_security_thread(self, image: np.ndarray, currency: str) -> float:
        """Detect security thread presence"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Expected thread position
        thread_pos = self.currency_specific_features[currency]['security_thread_position']
        thread_col = int(thread_pos * image.shape[1])
        
        # Extract vertical line around expected position
        thread_region = gray[:, max(0, thread_col-5):min(image.shape[1], thread_col+5)]
        
        # Look for vertical line patterns
        vertical_edges = cv2.Sobel(thread_region, cv2.CV_64F, 1, 0, ksize=3)
        thread_strength = np.mean(np.abs(vertical_edges))
        
        return min(thread_strength / 50, 1.0)  # Normalize to 0-1
    
    def _detect_watermark(self, image: np.ndarray, currency: str) -> float:
        """Detect watermark presence"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Get watermark regions for currency
        watermark_regions = self.currency_specific_features[currency]['watermark_regions']
        
        total_score = 0
        for region in watermark_regions:
            x, y, w, h = region
            x1 = int(x * image.shape[1])
            y1 = int(y * image.shape[0])
            x2 = int((x + w) * image.shape[1])
            y2 = int((y + h) * image.shape[0])
            
            # Extract region
            roi = gray[y1:y2, x1:x2]
            
            # Look for subtle brightness variations (watermark characteristic)
            laplacian = cv2.Laplacian(roi, cv2.CV_64F)
            watermark_strength = np.std(laplacian)
            total_score += watermark_strength
        
        return min(total_score / (len(watermark_regions) * 30), 1.0)
    
    def _detect_microprinting(self, image: np.ndarray) -> float:
        """Detect microprinting quality"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # High-frequency content indicates fine printing
        high_freq = cv2.Laplacian(gray, cv2.CV_64F)
        microprint_score = np.std(high_freq)
        
        return min(microprint_score / 40, 1.0)
    
    def _detect_holographic_features(self, image: np.ndarray) -> float:
        """Detect holographic features (for SAR)"""
        # Look for color variations and reflective properties
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Holographic features show high saturation variation
        saturation_var = np.var(hsv[:, :, 1])
        
        return min(saturation_var / 1000, 1.0)
    
    def _assess_print_quality(self, image: np.ndarray) -> float:
        """Assess overall print quality"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Sharpness measure using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Noise assessment
        noise_level = np.std(gray)
        
        # Quality score combines sharpness and low noise
        quality_score = sharpness / (1 + noise_level / 50)
        
        return min(quality_score / 100, 1.0)

# Example usage
if __name__ == "__main__":
    # Test the feature extractor
    extractor = AdvancedFeatureExtractor()
    
    # Create test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Extract features
    features = extractor.extract_comprehensive_features(test_image, 'USD')
    
    print(f"Extracted {len(features)} advanced features:")
    for i, (name, value) in enumerate(list(features.items())[:20]):  # Show first 20
        print(f"{i+1:2d}. {name}: {value:.6f}")
    
    print(f"... and {len(features)-20} more features")
    print(f"\nFeature extraction successful for professional currency detection!")