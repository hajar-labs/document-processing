"""
Base extractor class providing common interface and utilities for all document extractors.
Implements abstract methods for consistent extraction across different file types.
"""

import logging
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import hashlib
import time
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Enumeration of supported document types."""
    PDF = "pdf"
    WORD = "word" 
    IMAGE = "image"
    TEXT = "text"
    UNKNOWN = "unknown"


class ExtractionStatus(Enum):
    """Enumeration of extraction status codes."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass
class ExtractionResult:
    """Container for extraction results with metadata."""
    text: str
    metadata: Dict[str, Any]
    status: ExtractionStatus
    document_type: DocumentType
    page_count: Optional[int] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    file_hash: Optional[str] = None
    errors: Optional[List[str]] = None


class BaseExtractor(ABC):
    """
    Abstract base class for all document extractors.
    Provides common functionality and enforces consistent interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base extractor with configuration.
        
        Args:
            config: Configuration dictionary with extractor-specific settings
        """
        self.config = config or {}
        self.supported_extensions = set()
        self.supported_mime_types = set()
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB default
        self.encoding_fallbacks = ['utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        
        # Performance tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_processing_time': 0.0
        }
    
    @abstractmethod
    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text from document. Must be implemented by subclasses.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ExtractionResult containing extracted text and metadata
        """
        pass
    
    def can_handle(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file can be handled, False otherwise
        """
        file_path = Path(file_path)
        
        # Check file extension
        extension = file_path.suffix.lower()
        if extension in self.supported_extensions:
            return True
        
        # Check MIME type
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and mime_type in self.supported_mime_types:
                return True
        except Exception as e:
            logger.warning(f"Error checking MIME type for {file_path}: {e}")
        
        return False
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate file before extraction.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary containing validation results
        """
        file_path = Path(file_path)
        validation_result = {
            'is_valid': False,
            'file_exists': False,
            'file_size': 0,
            'is_readable': False,
            'mime_type': None,
            'errors': []
        }
        
        try:
            # Check if file exists
            if not file_path.exists():
                validation_result['errors'].append(f"File does not exist: {file_path}")
                return validation_result
            
            validation_result['file_exists'] = True
            
            # Check file size
            file_size = file_path.stat().st_size
            validation_result['file_size'] = file_size
            
            if file_size == 0:
                validation_result['errors'].append("File is empty")
                return validation_result
            
            if file_size > self.max_file_size:
                validation_result['errors'].append(
                    f"File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)"
                )
                return validation_result
            
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)
                validation_result['is_readable'] = True
            except PermissionError:
                validation_result['errors'].append("Permission denied: cannot read file")
                return validation_result
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            validation_result['mime_type'] = mime_type
            
            # Check if we can handle this file type
            if not self.can_handle(file_path):
                validation_result['errors'].append(
                    f"Unsupported file type: {file_path.suffix} (MIME: {mime_type})"
                )
                return validation_result
            
            validation_result['is_valid'] = True
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"File validation failed for {file_path}: {e}")
        
        return validation_result
    
    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Calculate MD5 hash of file for deduplication and caching.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def extract_with_validation(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text with full validation and error handling.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ExtractionResult with comprehensive metadata
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Update statistics
        self.extraction_stats['total_extractions'] += 1
        
        # Validate file
        validation = self.validate_file(file_path)
        if not validation['is_valid']:
            result = ExtractionResult(
                text="",
                metadata={
                    'file_path': str(file_path),
                    'file_size': validation['file_size'],
                    'mime_type': validation['mime_type'],
                    'validation_errors': validation['errors']
                },
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.UNKNOWN,
                errors=validation['errors'],
                processing_time=time.time() - start_time
            )
            self.extraction_stats['failed_extractions'] += 1
            return result
        
        try:
            # Calculate file hash for caching/deduplication
            file_hash = self.calculate_file_hash(file_path)
            
            # Perform extraction
            result = self.extract(file_path)
            
            # Enhance result with additional metadata
            result.metadata.update({
                'file_path': str(file_path),
                'file_size': validation['file_size'],
                'mime_type': validation['mime_type'],
                'extraction_timestamp': time.time(),
                'extractor_type': self.__class__.__name__
            })
            
            result.file_hash = file_hash
            result.processing_time = time.time() - start_time
            
            # Update statistics
            if result.status == ExtractionStatus.SUCCESS:
                self.extraction_stats['successful_extractions'] += 1
            else:
                self.extraction_stats['failed_extractions'] += 1
            
            # Update average processing time
            total_time = (self.extraction_stats['average_processing_time'] * 
                         (self.extraction_stats['total_extractions'] - 1) + 
                         result.processing_time)
            self.extraction_stats['average_processing_time'] = total_time / self.extraction_stats['total_extractions']
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed for {file_path}: {e}")
            self.extraction_stats['failed_extractions'] += 1
            
            return ExtractionResult(
                text="",
                metadata={
                    'file_path': str(file_path),
                    'file_size': validation['file_size'],
                    'mime_type': validation['mime_type'],
                    'extraction_error': str(e)
                },
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.UNKNOWN,
                errors=[str(e)],
                processing_time=time.time() - start_time
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics for performance monitoring.
        
        Returns:
            Dictionary containing extraction statistics
        """
        success_rate = 0.0
        if self.extraction_stats['total_extractions'] > 0:
            success_rate = (self.extraction_stats['successful_extractions'] / 
                          self.extraction_stats['total_extractions'] * 100)
        
        return {
            **self.extraction_stats,
            'success_rate_percent': round(success_rate, 2),
            'supported_extensions': list(self.supported_extensions),
            'supported_mime_types': list(self.supported_mime_types)
        }
    
    def reset_statistics(self):
        """Reset extraction statistics."""
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_processing_time': 0.0
        }