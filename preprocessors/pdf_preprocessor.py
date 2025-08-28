# PDF preprocessing for enhanced text extraction
"""
PDF preprocessing to optimize text extraction quality.
Handles complex layouts, multilingual content, and scanned documents.
"""

import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import cv2
from PIL import Image
import io
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PDFPageAnalysis:
    """Analysis results for a PDF page."""
    page_number: int
    is_scanned: bool
    text_density: float
    image_coverage: float
    layout_complexity: float
    needs_ocr: bool
    quality_score: float


class PDFPreprocessor:
    """
    Advanced PDF preprocessing for optimal text extraction.
    Analyzes document structure and applies appropriate processing strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.min_text_threshold = self.config.get('min_text_threshold', 100)
        self.image_coverage_threshold = self.config.get('image_coverage_threshold', 0.3)
        self.quality_threshold = self.config.get('quality_threshold', 0.6)
        self.max_pages_analyze = self.config.get('max_pages_analyze', 50)
        
    def analyze_pdf_structure(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze PDF structure to determine optimal processing strategy.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Analysis results with processing recommendations
        """
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = doc.page_count
            
            # Limit analysis for large documents
            pages_to_analyze = min(total_pages, self.max_pages_analyze)
            
            page_analyses = []
            total_text_length = 0
            total_images = 0
            scanned_pages = 0
            
            for page_num in range(pages_to_analyze):
                try:
                    page = doc[page_num]
                    analysis = self._analyze_page(page, page_num + 1)
                    page_analyses.append(analysis)
                    
                    # Accumulate statistics
                    if analysis.is_scanned:
                        scanned_pages += 1
                    
                    # Get basic text length for overall assessment
                    text = page.get_text().strip()
                    total_text_length += len(text)
                    total_images += len(page.get_images())
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
            # Calculate overall document characteristics
            avg_quality = np.mean([p.quality_score for p in page_analyses]) if page_analyses else 0
            scanned_ratio = scanned_pages / len(page_analyses) if page_analyses else 0
            avg_text_density = np.mean([p.text_density for p in page_analyses]) if page_analyses else 0
            
            # Determine document type and processing strategy
            document_type = self._classify_document_type(
                scanned_ratio, avg_text_density, avg_quality
            )
            
            processing_strategy = self._recommend_processing_strategy(
                document_type, page_analyses
            )
            
            return {
                'total_pages': total_pages,
                'analyzed_pages': len(page_analyses),
                'document_type': document_type,
                'scanned_pages': scanned_pages,
                'scanned_ratio': scanned_ratio,
                'average_quality': avg_quality,
                'average_text_density': avg_text_density,
                'total_images': total_images,
                'processing_strategy': processing_strategy,
                'page_analyses': page_analyses[:10],  # Store first 10 for detailed review
                'recommendations': self._generate_recommendations(
                    document_type, processing_strategy, page_analyses
                )
            }
            
        except Exception as e:
            logger.error(f"PDF structure analysis failed: {e}")
            return {
                'error': str(e),
                'document_type': 'unknown',
                'processing_strategy': 'default'
            }
    
    def _analyze_page(self, page, page_num: int) -> PDFPageAnalysis:
        """Analyze individual PDF page characteristics."""
        try:
            # Get text content
            text = page.get_text().strip()
            text_length = len(text)
            
            # Get images
            images = page.get_images()
            image_count = len(images)
            
            # Calculate page dimensions
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            
            # Calculate image coverage
            total_image_area = 0
            for img in images:
                try:
                    img_rects = page.get_image_rects(img[0])
                    for rect in img_rects:
                        total_image_area += rect.width * rect.height
                except:
                    continue
            
            image_coverage = total_image_area / page_area if page_area > 0 else 0
            
            # Calculate text density (characters per area unit)
            text_density = text_length / (page_area / 10000) if page_area > 0 else 0
            
            # Analyze layout complexity
            layout_complexity = self._calculate_layout_complexity(page)
            
            # Determine if page is scanned
            is_scanned = (text_length < self.min_text_threshold and 
                         (image_coverage > self.image_coverage_threshold or image_count > 0))
            
            # Determine if OCR is needed
            needs_ocr = is_scanned or text_length < 50
            
            # Calculate overall quality score
            quality_score = self._calculate_page_quality_score(
                text_length, image_coverage, layout_complexity, is_scanned
            )
            
            return PDFPageAnalysis(
                page_number=page_num,
                is_scanned=is_scanned,
                text_density=text_density,
                image_coverage=image_coverage,
                layout_complexity=layout_complexity,
                needs_ocr=needs_ocr,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Page analysis failed for page {page_num}: {e}")
            return PDFPageAnalysis(
                page_number=page_num,
                is_scanned=True,
                text_density=0.0,
                image_coverage=0.0,
                layout_complexity=0.5,
                needs_ocr=True,
                quality_score=0.0
            )
    
    def _calculate_layout_complexity(self, page) -> float:
        """Calculate layout complexity based on text blocks and structure."""
        try:
            # Get text in dictionary format for structure analysis
            text_dict = page.get_text("dict")
            
            if not text_dict or 'blocks' not in text_dict:
                return 0.0
            
            blocks = text_dict['blocks']
            text_blocks = [b for b in blocks if 'lines' in b]
            
            if not text_blocks:
                return 0.0
            
            # Factors contributing to complexity
            complexity_factors = []
            
            # 1. Number of text blocks (more blocks = more complex)
            complexity_factors.append(min(len(text_blocks) / 10.0, 1.0))
            
            # 2. Font variation (more fonts = more complex)
            fonts = set()
            font_sizes = []
            
            for block in text_blocks:
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        fonts.add(span.get('font', ''))
                        font_sizes.append(span.get('size', 12))
            
            font_complexity = min(len(fonts) / 5.0, 1.0)
            complexity_factors.append(font_complexity)
            
            # 3. Font size variation
            if font_sizes:
                size_std = np.std(font_sizes)
                size_complexity = min(size_std / 10.0, 1.0)
                complexity_factors.append(size_complexity)
            
            # 4. Spatial distribution of text blocks
            if len(text_blocks) > 1:
                y_positions = [block['bbox'][1] for block in text_blocks]
                y_std = np.std(y_positions)
                spatial_complexity = min(y_std / 100.0, 1.0)
                complexity_factors.append(spatial_complexity)
            
            return np.mean(complexity_factors)
            
        except Exception as e:
            logger.warning(f"Layout complexity calculation failed: {e}")
            return 0.5  # Default medium complexity
    
    def _calculate_page_quality_score(self, text_length: int, image_coverage: float,
                                    layout_complexity: float, is_scanned: bool) -> float:
        """Calculate overall page quality score for text extraction."""
        quality_factors = []
        
        # Text availability (higher text length = better quality)
        text_quality = min(text_length / 1000.0, 1.0)
        quality_factors.append(text_quality)
        
        # Image coverage (moderate coverage is best for mixed documents)
        if image_coverage < 0.1:
            image_quality = 1.0  # Pure text document
        elif image_coverage > 0.8:
            image_quality = 0.2  # Mostly images, likely scanned
        else:
            image_quality = 0.6  # Mixed content
        quality_factors.append(image_quality)
        
        # Layout complexity (moderate complexity is good)
        if layout_complexity < 0.3:
            layout_quality = 0.8  # Simple layout
        elif layout_complexity > 0.7:
            layout_quality = 0.4  # Very complex layout
        else:
            layout_quality = 1.0  # Optimal complexity
        quality_factors.append(layout_quality)
        
        # Scanned document penalty
        if is_scanned:
            quality_factors.append(0.3)
        else:
            quality_factors.append(1.0)
        
        return np.mean(quality_factors)
    
    def _classify_document_type(self, scanned_ratio: float, 
                               avg_text_density: float, avg_quality: float) -> str:
        """Classify document type based on analysis."""
        if scanned_ratio > 0.8:
            return 'fully_scanned'
        elif scanned_ratio > 0.3:
            return 'mixed_scanned_native'
        elif avg_text_density > 50 and avg_quality > 0.7:
            return 'high_quality_native'
        elif avg_quality > 0.5:
            return 'standard_native'
        else:
            return 'low_quality_mixed'
    
    def _recommend_processing_strategy(self, document_type: str, 
                                     page_analyses: List[PDFPageAnalysis]) -> Dict[str, Any]:
        """Recommend processing strategy based on document analysis."""
        strategy = {
            'primary_method': 'native_text',
            'use_ocr': False,
            'parallel_processing': True,
            'preprocessing_level': 'standard',
            'ocr_confidence_threshold': 60,
            'quality_filter': False
        }
        
        if document_type == 'fully_scanned':
            strategy.update({
                'primary_method': 'ocr_only',
                'use_ocr': True,
                'preprocessing_level': 'aggressive',
                'ocr_confidence_threshold': 50,
                'quality_filter': True
            })
        
        elif document_type == 'mixed_scanned_native':
            strategy.update({
                'primary_method': 'hybrid',
                'use_ocr': True,
                'preprocessing_level': 'adaptive',
                'ocr_confidence_threshold': 55,
                'quality_filter': True
            })
        
        elif document_type == 'low_quality_mixed':
            strategy.update({
                'primary_method': 'hybrid',
                'use_ocr': True,
                'preprocessing_level': 'enhanced',
                'ocr_confidence_threshold': 40,
                'quality_filter': True
            })
        
        # Adjust for page-specific characteristics
        high_complexity_pages = len([p for p in page_analyses if p.layout_complexity > 0.7])
        if high_complexity_pages > len(page_analyses) * 0.3:
            strategy['preprocessing_level'] = 'enhanced'
            strategy['parallel_processing'] = False  # More careful processing
        
        return strategy
    
    def _generate_recommendations(self, document_type: str, 
                                processing_strategy: Dict[str, Any],
                                page_analyses: List[PDFPageAnalysis]) -> List[str]:
        """Generate human-readable processing recommendations."""
        recommendations = []
        
        # Document type recommendations
        if document_type == 'fully_scanned':
            recommendations.append(
                "Document appears to be fully scanned. OCR processing is required for all pages."
            )
        elif document_type == 'mixed_scanned_native':
            recommendations.append(
                "Document contains both native text and scanned pages. Hybrid processing recommended."
            )
        elif document_type == 'high_quality_native':
            recommendations.append(
                "High-quality native PDF. Standard text extraction should work well."
            )
        
        # Quality-based recommendations
        low_quality_pages = len([p for p in page_analyses if p.quality_score < 0.3])
        if low_quality_pages > 0:
            recommendations.append(
                f"{low_quality_pages} pages have low quality scores and may require special processing."
            )
        
        # OCR recommendations
        if processing_strategy['use_ocr']:
            recommendations.append(
                "OCR processing enabled. Consider preprocessing images for better accuracy."
            )
            
            confidence_threshold = processing_strategy['ocr_confidence_threshold']
            recommendations.append(
                f"OCR confidence threshold set to {confidence_threshold}%. "
                "Lower thresholds may include more text but with potential errors."
            )
        
        # Performance recommendations
        if processing_strategy['parallel_processing']:
            recommendations.append(
                "Parallel processing enabled for faster extraction."
            )
        else:
            recommendations.append(
                "Sequential processing recommended due to document complexity."
            )
        
        return recommendations
    
    def optimize_extraction_parameters(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate optimized extraction parameters based on analysis.
        
        Args:
            analysis_results: Results from analyze_pdf_structure
            
        Returns:
            Optimized parameters for PDF extraction
        """
        document_type = analysis_results.get('document_type', 'unknown')
        strategy = analysis_results.get('processing_strategy', {})
        
        # Base parameters
        params = {
            'preserve_formatting': True,
            'extract_tables': True,
            'extract_images': True,
            'extract_metadata': True,
            'parallel_processing': strategy.get('parallel_processing', True),
            'max_pages': None
        }
        
        # OCR parameters
        if strategy.get('use_ocr', False):
            params.update({
                'enable_ocr': True,
                'ocr_threshold': 50,  # Lower threshold for documents needing OCR
                'ocr_config': {
                    'default_ocr': 'multilingual',
                    'parallel_processing': True,
                    'max_image_size': (3000, 3000)
                }
            })
        else:
            params.update({
                'enable_ocr': False,
                'ocr_threshold': 100
            })
        
        # Quality-based adjustments
        avg_quality = analysis_results.get('average_quality', 0.5)
        if avg_quality < 0.3:
            params.update({
                'preserve_formatting': False,  # Focus on content over formatting
                'extract_tables': False,      # Tables likely corrupted
                'enable_ocr': True,
                'ocr_threshold': 25
            })
        
        # Document size adjustments
        total_pages = analysis_results.get('total_pages', 0)
        if total_pages > 100:
            params.update({
                'parallel_processing': True,
                'max_pages': 200  # Limit for very large documents
            })
        
        return params
    
    def preprocess_for_multilingual(self, pdf_path: Union[str, Path],
                                  target_languages: List[str] = None) -> Dict[str, Any]:
        """
        Specialized preprocessing for multilingual documents (French/Arabic).
        
        Args:
            pdf_path: Path to PDF file
            target_languages: Expected languages in document
            
        Returns:
            Preprocessing results with language-specific optimizations
        """
        if target_languages is None:
            target_languages = ['french', 'arabic']
        
        try:
            # First, analyze document structure
            analysis = self.analyze_pdf_structure(pdf_path)
            
            # Get optimized parameters
            base_params = self.optimize_extraction_parameters(analysis)
            
            # Language-specific adjustments
            multilingual_params = base_params.copy()
            
            if 'arabic' in target_languages:
                # Arabic text requires special handling
                multilingual_params.update({
                    'ocr_config': {
                        **multilingual_params.get('ocr_config', {}),
                        'default_ocr': 'arabic' if len(target_languages) == 1 else 'multilingual',
                        'arabic_optimization': True
                    }
                })
            
            if 'french' in target_languages:
                # French accents need careful processing
                multilingual_params.update({
                    'preserve_formatting': True,  # Important for French punctuation
                    'ocr_config': {
                        **multilingual_params.get('ocr_config', {}),
                        'french_optimization': True
                    }
                })
            
            # Mixed language documents need hybrid approach
            if len(target_languages) > 1:
                multilingual_params.update({
                    'ocr_config': {
                        **multilingual_params.get('ocr_config', {}),
                        'default_ocr': 'multilingual',
                        'enable_language_detection': True
                    }
                })
            
            return {
                'analysis': analysis,
                'extraction_parameters': multilingual_params,
                'target_languages': target_languages,
                'optimization_applied': True,
                'recommendations': analysis.get('recommendations', []) + [
                    f"Optimized for {', '.join(target_languages)} languages",
                    "Use multilingual OCR configuration for best results"
                ]
            }
            
        except Exception as e:
            logger.error(f"Multilingual preprocessing failed: {e}")
            return {
                'error': str(e),
                'extraction_parameters': {
                    'enable_ocr': True,
                    'ocr_config': {'default_ocr': 'multilingual'}
                },
                'optimization_applied': False
            }
