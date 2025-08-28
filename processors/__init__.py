"""
Processors package for document processing.
"""

try:
    from .metadata_extractor import MetadataExtractor
except ImportError:
    MetadataExtractor = None

try:
    from .structure_analyzer import StructureAnalyzer
except ImportError:
    StructureAnalyzer = None

__all__ = [
    'MetadataExtractor',
    'StructureAnalyzer'
]