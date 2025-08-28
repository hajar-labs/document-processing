#image preprocessing for optimal OCR results
"""
Comprehensive image preprocessing pipeline optimized for multilingual documents.
Includes noise reduction, skew correction, contrast enhancement, and quality assessment.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage
from skimage import filters, morphology, segmentation
from skimage.transform import rotate
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ImageQualityAssessor:
    """Assess image quality for OCR suitability."""
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate image contrast using standard deviation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return np.std(gray)
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate average brightness."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return np.mean(gray)
    
    @staticmethod
    def detect_blur(image: np.ndarray, threshold: float = 100.0) -> bool:
        """Detect if image is blurred."""
        sharpness = ImageQualityAssessor.calculate_sharpness(image)
        return sharpness < threshold
    
    @staticmethod
    def assess_quality(image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive image quality assessment."""
        sharpness = ImageQualityAssessor.calculate_sharpness(image)
        contrast = ImageQualityAssessor.calculate_contrast(image)
        brightness = ImageQualityAssessor.calculate_brightness(image)
        
        # Quality scores (0-1 scale)
        sharpness_score = min(sharpness / 500.0, 1.0)  # Normalize
        contrast_score = min(contrast / 100.0, 1.0)    # Normalize
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5  # Optimal at 127.5
        
        # Overall quality score
        quality_score = (sharpness_score + contrast_score + brightness_score) / 3.0
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness,
            'sharpness_score': sharpness_score,
            'contrast_score': contrast_score,
            'brightness_score': brightness_score,
            'overall_quality': quality_score,
            'is_blurred': sharpness < 100.0,
            'is_low_contrast': contrast < 30.0,
            'is_too_dark': brightness < 50,
            'is_too_bright': brightness > 200
        }


class AdvancedImagePreprocessor:
    """
    Advanced image preprocessing pipeline with specialized techniques
    for government documents and multilingual text.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.quality_assessor = ImageQualityAssessor()
        
        # Preprocessing parameters
        self.target_dpi = self.config.get('target_dpi', 300)
        self.min_image_size = self.config.get('min_image_size', (300, 300))
        self.max_image_size = self.config.get('max_image_size', (4000, 4000))
        self.noise_kernel_size = self.config.get('noise_kernel_size', 3)
        self.skew_threshold = self.config.get('skew_threshold', 0.5)
        
    def resize_image(self, image: np.ndarray, target_dpi: int = None) -> np.ndarray:
        """Resize image to optimal dimensions for OCR."""
        if target_dpi is None:
            target_dpi = self.target_dpi
        
        h, w = image.shape[:2]
        min_h, min_w = self.min_image_size
        max_h, max_w = self.max_image_size
        
        # Calculate optimal size
        if h < min_h or w < min_w:
            scale_factor = max(min_h / h, min_w / w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Upscaled image from {w}x{h} to {new_w}x{new_h}")
            return resized
        
        elif h > max_h or w > max_w:
            scale_factor = min(max_h / h, max_w / w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Downscaled image from {w}x{h} to {new_w}x{new_h}")
            return resized
        
        return image
    
    def denoise_image(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """Advanced noise removal using multiple techniques."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'adaptive':
            # Use different denoising based on image characteristics
            quality = self.quality_assessor.assess_quality(gray)
            
            if quality['is_low_contrast']:
                # Use non-local means for low contrast images
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            else:
                # Use bilateral filter for normal images
                denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        elif method == 'morphological':
            # Morphological noise removal
            kernel = np.ones((self.noise_kernel_size, self.noise_kernel_size), np.uint8)
            denoised = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        elif method == 'gaussian':
            # Gaussian blur for simple noise
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        elif method == 'median':
            # Median filter for salt-and-pepper noise
            denoised = cv2.medianBlur(gray, 5)
        
        else:
            # Bilateral filter as default
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return denoised
    
    def correct_skew_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Advanced skew detection and correction using Hough transform."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return image, 0.0
        
        # Calculate angles of detected lines
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:  # Only consider reasonable skew angles
                angles.append(angle)
        
        if not angles:
            return image, 0.0
        
        # Use median angle to avoid outliers
        skew_angle = np.median(angles)
        
        # Only correct if skew is significant
        if abs(skew_angle) > self.skew_threshold:
            # Rotate image
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            
            # Calculate new image dimensions
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust rotation matrix for new dimensions
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            corrected = cv2.warpAffine(gray, M, (new_w, new_h), 
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            
            logger.info(f"Corrected skew angle: {skew_angle:.2f} degrees")
            return corrected, skew_angle
        
        return gray, skew_angle
    
    def enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Adaptive contrast enhancement based on image characteristics."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Assess current contrast
        quality = self.quality_assessor.assess_quality(gray)
        
        if quality['is_low_contrast']:
            # Use CLAHE for low contrast images
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
        else:
            # Use histogram equalization for normal images
            enhanced = cv2.equalizeHist(gray)
        
        return enhanced
    
    def binarize_adaptive(self, image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """Advanced binarization using multiple methods."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'otsu':
            # Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif method == 'adaptive_gaussian':
            # Adaptive Gaussian thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        elif method == 'adaptive_mean':
            # Adaptive mean thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        elif method == 'sauvola':
            # Sauvola thresholding (better for documents with varying illumination)
            try:
                from skimage.filters import threshold_sauvola
                threshold = threshold_sauvola(gray, window_size=15)
                binary = gray > threshold
                binary = (binary * 255).astype(np.uint8)
            except ImportError:
                logger.warning("Skimage not available for Sauvola thresholding, using Otsu instead")
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        else:
            # Default to Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def remove_borders_smart(self, image: np.ndarray, border_ratio: float = 0.05) -> np.ndarray:
        """Smart border removal that detects actual content boundaries."""
        h, w = image.shape[:2]
        
        # Calculate initial border size
        border_h = int(h * border_ratio)
        border_w = int(w * border_ratio)
        
        # Analyze edge regions for content
        top_region = image[:border_h, :]
        bottom_region = image[h-border_h:, :]
        left_region = image[:, :border_w]
        right_region = image[:, w-border_w:]
        
        # Check if borders contain significant content
        def has_content(region, threshold=10):
            return np.std(region) > threshold
        
        # Adjust borders based on content analysis
        top_cut = border_h if not has_content(top_region) else 0
        bottom_cut = border_h if not has_content(bottom_region) else 0
        left_cut = border_w if not has_content(left_region) else 0
        right_cut = border_w if not has_content(right_region) else 0
        
        # Apply border removal
        cropped = image[top_cut:h-bottom_cut, left_cut:w-right_cut]
        
        if cropped.size > 0:
            return cropped
        else:
            return image
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions using MSER (Maximally Stable Extremal Regions)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        try:
            # Create MSER detector
            mser = cv2.MSER_create()
            
            # Detect regions
            regions, _ = mser.detectRegions(gray)
            
            # Convert regions to bounding boxes
            text_regions = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                
                # Filter regions by size (remove too small or too large regions)
                if 10 < w < gray.shape[1]//2 and 5 < h < gray.shape[0]//2:
                    text_regions.append((x, y, w, h))
            
            return text_regions
        
        except Exception as e:
            logger.warning(f"MSER text region detection failed: {e}")
            return []
    
    def enhance_for_arabic_text(self, image: np.ndarray) -> np.ndarray:
        """Specific enhancements for Arabic text recognition."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Arabic text often requires different processing
        # 1. Stronger noise reduction
        denoised = cv2.bilateralFilter(gray, 15, 80, 80)
        
        # 2. Gentle contrast enhancement (Arabic diacritics are sensitive)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Morphological operations to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def enhance_for_french_text(self, image: np.ndarray) -> np.ndarray:
        """Specific enhancements for French text with accents."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # French accents require careful processing
        # 1. Preserve fine details
        denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
        
        # 2. Moderate contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Sharpening to enhance accent marks
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def preprocess_image(self, image_path: Union[str, Path], 
                        preprocessing_steps: List[str] = None) -> np.ndarray:
        """
        Simple preprocessing method for backward compatibility.
        
        Args:
            image_path: Path to the image file
            preprocessing_steps: List of preprocessing steps to apply
            
        Returns:
            Preprocessed image as numpy array
        """
        if preprocessing_steps is None:
            preprocessing_steps = ['denoise', 'skew_correction', 'enhance_contrast', 
                                 'binarize', 'remove_borders']
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing steps
        processed = gray.copy()
        
        for step in preprocessing_steps:
            try:
                if step == 'denoise':
                    processed = self.denoise_image(processed)
                elif step == 'skew_correction':
                    processed, _ = self.correct_skew_advanced(processed)
                elif step == 'enhance_contrast':
                    processed = self.enhance_contrast_adaptive(processed)
                elif step == 'binarize':
                    processed = self.binarize_adaptive(processed)
                elif step == 'remove_borders':
                    processed = self.remove_borders_smart(processed)
                else:
                    logger.warning(f"Unknown preprocessing step: {step}")
            except Exception as e:
                logger.warning(f"Preprocessing step '{step}' failed: {e}")
                continue
        
        return processed
    
    def preprocess_pipeline(self, image_path: Union[str, Path], 
                          language_hint: str = None,
                          custom_steps: List[str] = None) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline with language-specific optimizations.
        
        Args:
            image_path: Path to the image file
            language_hint: Expected language ('french', 'arabic', 'mixed', None)
            custom_steps: Custom preprocessing steps
            
        Returns:
            Dictionary containing processed image and metadata
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            original_shape = image.shape[:2]
            
            # Initial quality assessment
            initial_quality = self.quality_assessor.assess_quality(image)
            
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            processing_steps = custom_steps or [
                'resize', 'denoise', 'skew_correction', 'contrast_enhancement',
                'language_specific', 'binarization', 'border_removal'
            ]
            
            processed = gray.copy()
            applied_steps = []
            skew_angle = 0.0
            
            # Apply preprocessing steps
            for step in processing_steps:
                try:
                    if step == 'resize':
                        processed = self.resize_image(processed)
                        applied_steps.append('resize')
                    
                    elif step == 'denoise':
                        method = 'adaptive' if initial_quality['is_low_contrast'] else 'bilateral'
                        processed = self.denoise_image(processed, method)
                        applied_steps.append(f'denoise_{method}')
                    
                    elif step == 'skew_correction':
                        processed, skew_angle = self.correct_skew_advanced(processed)
                        applied_steps.append('skew_correction')
                    
                    elif step == 'contrast_enhancement':
                        processed = self.enhance_contrast_adaptive(processed)
                        applied_steps.append('contrast_enhancement')
                    
                    elif step == 'language_specific' and language_hint:
                        if language_hint == 'arabic':
                            processed = self.enhance_for_arabic_text(processed)
                            applied_steps.append('arabic_enhancement')
                        elif language_hint == 'french':
                            processed = self.enhance_for_french_text(processed)
                            applied_steps.append('french_enhancement')
                    
                    elif step == 'binarization':
                        # Choose binarization method based on image quality
                        if initial_quality['is_low_contrast']:
                            binary_method = 'sauvola'
                        else:
                            binary_method = 'otsu'
                        processed = self.binarize_adaptive(processed, binary_method)
                        applied_steps.append(f'binarization_{binary_method}')
                    
                    elif step == 'border_removal':
                        processed = self.remove_borders_smart(processed)
                        applied_steps.append('border_removal')
                    
                except Exception as e:
                    logger.warning(f"Preprocessing step '{step}' failed: {e}")
                    continue
            
            # Final quality assessment
            final_quality = self.quality_assessor.assess_quality(processed)
            
            # Detect text regions
            text_regions = self.detect_text_regions(processed)
            
            return {
                'processed_image': processed,
                'original_shape': original_shape,
                'final_shape': processed.shape[:2],
                'initial_quality': initial_quality,
                'final_quality': final_quality,
                'applied_steps': applied_steps,
                'skew_angle': skew_angle,
                'text_regions_count': len(text_regions),
                'text_regions': text_regions,
                'quality_improved': final_quality['overall_quality'] > initial_quality['overall_quality'],
                'processing_metadata': {
                    'language_hint': language_hint,
                    'custom_steps': custom_steps is not None,
                    'total_steps': len(applied_steps)
                }
            }
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed for {image_path}: {e}")
            return {
                'processed_image': None,
                'error': str(e),
                'original_shape': None,
                'final_shape': None,
                'applied_steps': [],
                'quality_improved': False
            }
    
    def create_preprocessing_variants(self, image: np.ndarray, 
                                    count: int = 3) -> List[np.ndarray]:
        """
        Create multiple preprocessing variants for ensemble OCR.
        
        Args:
            image: Input image
            count: Number of variants to create
            
        Returns:
            List of preprocessed image variants
        """
        variants = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Variant 1: Conservative preprocessing
        variant1 = self.denoise_image(gray, 'bilateral')
        variant1 = self.enhance_contrast_adaptive(variant1)
        variant1 = self.binarize_adaptive(variant1, 'otsu')
        variants.append(variant1)
        
        if count > 1:
            # Variant 2: Aggressive preprocessing
            variant2 = self.denoise_image(gray, 'adaptive')
            variant2, _ = self.correct_skew_advanced(variant2)
            variant2 = self.enhance_contrast_adaptive(variant2)
            variant2 = self.binarize_adaptive(variant2, 'sauvola')
            variants.append(variant2)
        
        if count > 2:
            # Variant 3: Minimal preprocessing
            variant3 = self.denoise_image(gray, 'gaussian')
            variant3 = self.binarize_adaptive(variant3, 'adaptive_gaussian')
            variants.append(variant3)
        
        return variants[:count]