"""
Extractors package for document text extraction.
"""

from .base_extractor import BaseExtractor, ExtractionResult, ExtractionStatus, DocumentType
from .pdf_extractor import PDFExtractor, PageInfo, TableInfo

try:
    from .image_extractor import ImageExtractor
except ImportError:
    ImageExtractor = None

try:
    from .word_extractor import WordExtractor
except ImportError:
    WordExtractor = None

__all__ = [
    'BaseExtractor',
    'ExtractionResult', 
    'ExtractionStatus',
    'DocumentType',
    'PDFExtractor',
    'PageInfo',
    'TableInfo',
    'ImageExtractor',
    'WordExtractor'
]

