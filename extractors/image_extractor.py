# Images with OCR (Tesseract, OpenCV)
"""
Image text extraction using OCR with support for Arabic and French.
Implements sophisticated preprocessing and multi-engine OCR for maximum accuracy.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import re
import concurrent.futures
from dataclasses import dataclass

from .base_extractor import BaseExtractor, ExtractionResult, ExtractionStatus, DocumentType
from preprocessors.image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    languages: List[str]
    psm: int  # Page segmentation mode
    oem: int  # OCR Engine mode
    config_string: str
    confidence_threshold: float = 60.0



class ImageExtractor(BaseExtractor):
    """
    image text extractor with multilingual OCR support.
    Optimized for French and Arabic text extraction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Supported file types
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        self.supported_mime_types = {
            'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 
            'image/tiff', 'image/gif'
        }
        
        # OCR configurations for different languages
        self.ocr_configs = {
            'french': OCRConfig(
                languages=['fra'],
                psm=3,  # Fully automatic page segmentation
                oem=1,  # Neural nets LSTM engine
                config_string='--psm 3 --oem 1 -c preserve_interword_spaces=1',
                confidence_threshold=60.0
            ),
            'arabic': OCRConfig(
                languages=['ara'],
                psm=6,  # Single uniform block of text
                oem=1,
                config_string='--psm 6 --oem 1 -c preserve_interword_spaces=1',
                confidence_threshold=50.0  # Lower threshold for Arabic due to complexity
            ),
            'multilingual': OCRConfig(
                languages=['fra', 'ara', 'eng'],
                psm=3,
                oem=1,
                config_string='--psm 3 --oem 1 -c preserve_interword_spaces=1',
                confidence_threshold=55.0
            )
        }
        
        self.preprocessor = ImagePreprocessor()
        self.default_config = self.config.get('default_ocr', 'multilingual')
        self.enable_parallel_processing = self.config.get('parallel_processing', True)
        self.max_image_size = self.config.get('max_image_size', (4000, 4000))
    
    def _resize_image_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it exceeds maximum dimensions."""
        h, w = image.shape[:2]
        max_h, max_w = self.max_image_size
        
        if h > max_h or w > max_w:
            scale_factor = min(max_h / h, max_w / w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
            return resized
        
        return image
    
    def _extract_with_config(self, image: np.ndarray, config: OCRConfig) -> Tuple[str, float]:
        """Extract text using specific OCR configuration."""
        try:
            # Configure tesseract
            lang_string = '+'.join(config.languages)
            custom_config = config.config_string
            
            # Get OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                image, 
                lang=lang_string,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text with confidence filtering
            text_parts = []
            confidences = []
            
            for i in range(len(ocr_data['text'])):
                word = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                if word and confidence >= config.confidence_threshold:
                    text_parts.append(word)
                    confidences.append(confidence)
            
            text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return text, avg_confidence
            
        except Exception as e:
            logger.error(f"OCR extraction failed with config {config.languages}: {e}")
            return "", 0.0
    
    def _detect_text_language(self, text: str) -> str:
        """Simple language detection based on character analysis."""
        if not text:
            return "unknown"
        
        # Count Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        
        # Count French/Latin characters
        latin_chars = len(re.findall(r'[a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
        
        total_chars = len(re.findall(r'\S', text))  # Non-whitespace characters
        
        if total_chars == 0:
            return "unknown"
        
        arabic_ratio = arabic_chars / total_chars
        latin_ratio = latin_chars / total_chars
        
        if arabic_ratio > 0.3:
            return "arabic"
        elif latin_ratio > 0.5:
            return "french"
        else:
            return "mixed"
    
    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text from image using advanced OCR with preprocessing.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ExtractionResult containing extracted text and metadata
        """
        try:
            file_path = Path(file_path)
            
            # Preprocess image
            processed_image = self.preprocessor.preprocess_image(file_path)
            processed_image = self._resize_image_if_needed(processed_image)
            
            # Try different OCR configurations
            best_text = ""
            best_confidence = 0.0
            best_config_name = "multilingual"
            extraction_results = {}
            
            if self.enable_parallel_processing:
                # Parallel extraction with different configs
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_config = {
                        executor.submit(self._extract_with_config, processed_image, config): name
                        for name, config in self.ocr_configs.items()
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_config):
                        config_name = future_to_config[future]
                        try:
                            text, confidence = future.result()
                            extraction_results[config_name] = (text, confidence)
                            
                            if confidence > best_confidence and len(text.strip()) > 0:
                                best_text = text
                                best_confidence = confidence
                                best_config_name = config_name
                                
                        except Exception as e:
                            logger.error(f"OCR failed for config {config_name}: {e}")
                            extraction_results[config_name] = ("", 0.0)
            else:
                # Sequential extraction
                for config_name, config in self.ocr_configs.items():
                    text, confidence = self._extract_with_config(processed_image, config)
                    extraction_results[config_name] = (text, confidence)
                    
                    if confidence > best_confidence and len(text.strip()) > 0:
                        best_text = text
                        best_confidence = confidence
                        best_config_name = config_name
            
            # Post-process text
            cleaned_text = self._clean_extracted_text(best_text)
            detected_language = self._detect_text_language(cleaned_text)
            
            # Determine extraction status
            if best_confidence < 30:
                status = ExtractionStatus.FAILED
            elif best_confidence < 60:
                status = ExtractionStatus.PARTIAL
            else:
                status = ExtractionStatus.SUCCESS
            
            # Create metadata
            metadata = {
                'ocr_confidence': round(best_confidence, 2),
                'best_config': best_config_name,
                'detected_language': detected_language,
                'extraction_results': {
                    name: {'confidence': round(conf, 2), 'text_length': len(text)}
                    for name, (text, conf) in extraction_results.items()
                },
                'image_dimensions': processed_image.shape[:2],
                'preprocessing_applied': True
            }
            
            return ExtractionResult(
                text=cleaned_text,
                metadata=metadata,
                status=status,
                document_type=DocumentType.IMAGE,
                language=detected_language,
                confidence=best_confidence / 100.0  # Normalize to 0-1
            )
            
        except Exception as e:
            logger.error(f"Image extraction failed for {file_path}: {e}")
            return ExtractionResult(
                text="",
                metadata={'error': str(e)},
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.IMAGE,
                errors=[str(e)]
            )
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove isolated single characters (common OCR artifacts)
        cleaned = re.sub(r'\b[a-zA-Z]\b', '', cleaned)
        
        # Clean up punctuation spacing
        cleaned = re.sub(r'\s+([.!?;:,])', r'\1', cleaned)
        cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)
        
        # Remove lines with too few characters (likely artifacts)
        lines = cleaned.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if len(line) >= 3:  # Keep lines with at least 3 characters
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines).strip()
    
    def extract_regions(self, file_path: Union[str, Path], 
                       regions: List[Tuple[int, int, int, int]]) -> List[ExtractionResult]:
        """
        Extract text from specific regions of an image.
        
        Args:
            file_path: Path to the image file
            regions: List of (x, y, width, height) tuples defining regions
            
        Returns:
            List of ExtractionResult for each region
        """
        results = []
        
        try:
            # Load and preprocess full image
            processed_image = self.preprocessor.preprocess_image(file_path)
            
            for i, (x, y, w, h) in enumerate(regions):
                try:
                    # Extract region
                    region = processed_image[y:y+h, x:x+w]
                    
                    if region.size == 0:
                        continue
                    
                    # Extract text from region
                    text, confidence = self._extract_with_config(
                        region, self.ocr_configs[self.default_config]
                    )
                    
                    cleaned_text = self._clean_extracted_text(text)
                    detected_language = self._detect_text_language(cleaned_text)
                    
                    status = ExtractionStatus.SUCCESS if confidence >= 50 else ExtractionStatus.PARTIAL
                    
                    result = ExtractionResult(
                        text=cleaned_text,
                        metadata={
                            'region_index': i,
                            'region_coordinates': (x, y, w, h),
                            'ocr_confidence': round(confidence, 2),
                            'detected_language': detected_language
                        },
                        status=status,
                        document_type=DocumentType.IMAGE,
                        language=detected_language,
                        confidence=confidence / 100.0
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Region extraction failed for region {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Region-based extraction failed for {file_path}: {e}")
        
        return results
