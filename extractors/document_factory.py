# document_factory.py
"""
Factory pattern pour la sélection automatique d'extracteurs
avec gestion d'erreurs robuste et fallback strategies
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import mimetypes
import logging
from enum import Enum

from .base_extractor import BaseExtractor, ExtractionResult, ExtractionStatus
from .pdf_extractor import PDFExtractor
from .word_extractor import WordExtractor
from .image_extractor import ImageExtractor

logger = logging.getLogger(__name__)


class ExtractorType(Enum):
    """Types d'extracteurs disponibles"""
    PDF = "pdf"
    WORD = "word"
    IMAGE = "image"
    TEXT = "text"


class DocumentProcessorFactory:
    """
    Factory pour la sélection automatique et la gestion des extracteurs
    avec stratégies de fallback et validation robuste
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.extractors = self._initialize_extractors()
        self.fallback_enabled = self.config.get('enable_fallback', True)
        
    def _initialize_extractors(self) -> Dict[ExtractorType, BaseExtractor]:
        """Initialise tous les extracteurs avec leur configuration"""
        extractors = {}
        
        try:
            extractors[ExtractorType.PDF] = PDFExtractor(
                self.config.get('pdf_config', {})
            )
        except Exception as e:
            logger.error(f"Failed to initialize PDF extractor: {e}")
            
        try:
            extractors[ExtractorType.WORD] = WordExtractor(
                self.config.get('word_config', {})
            )
        except Exception as e:
            logger.error(f"Failed to initialize Word extractor: {e}")
            
        try:
            extractors[ExtractorType.IMAGE] = ImageExtractor(
                self.config.get('image_config', {})
            )
        except Exception as e:
            logger.error(f"Failed to initialize Image extractor: {e}")
            
        return extractors
    
    def get_extractor_for_file(self, file_path: Union[str, Path]) -> Optional[BaseExtractor]:
        """
        Sélectionne l'extracteur approprié pour un fichier
        """
        file_path = Path(file_path)
        
        # Priorité 1: Extension de fichier
        extension = file_path.suffix.lower()
        
        if extension == '.pdf' and ExtractorType.PDF in self.extractors:
            return self.extractors[ExtractorType.PDF]
        elif extension in {'.docx', '.doc', '.docm'} and ExtractorType.WORD in self.extractors:
            return self.extractors[ExtractorType.WORD]
        elif extension in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'} and ExtractorType.IMAGE in self.extractors:
            return self.extractors[ExtractorType.IMAGE]
        
        # Priorité 2: Type MIME
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                for extractor_type, extractor in self.extractors.items():
                    if extractor.can_handle(file_path):
                        return extractor
        except Exception as e:
            logger.warning(f"MIME type detection failed: {e}")
        
        return None
    
    def extract_with_fallback(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extraction avec stratégies de fallback multiples
        """
        file_path = Path(file_path)
        primary_extractor = self.get_extractor_for_file(file_path)
        
        if not primary_extractor:
            return ExtractionResult(
                text="",
                metadata={'error': 'No suitable extractor found'},
                status=ExtractionStatus.UNSUPPORTED,
                document_type=DocumentType.UNKNOWN,
                errors=['Unsupported file type']
            )
        
        # Tentative d'extraction primaire
        try:
            result = primary_extractor.extract_with_validation(file_path)
            
            # Si succès complet, retourner le résultat
            if result.status == ExtractionStatus.SUCCESS:
                return result
            
            # Si échec et fallback activé, essayer les alternatives
            if (result.status == ExtractionStatus.FAILED and 
                self.fallback_enabled and 
                len(result.text.strip()) < 50):
                
                return self._try_fallback_extraction(file_path, primary_extractor)
            
            return result
            
        except Exception as e:
            logger.error(f"Primary extraction failed: {e}")
            
            if self.fallback_enabled:
                return self._try_fallback_extraction(file_path, primary_extractor)
            
            return ExtractionResult(
                text="",
                metadata={'error': str(e)},
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.UNKNOWN,
                errors=[str(e)]
            )
    
    def _try_fallback_extraction(self, file_path: Path, 
                               excluded_extractor: BaseExtractor) -> ExtractionResult:
        """
        Essaie les extracteurs alternatifs en cas d'échec
        """
        fallback_results = []
        
        # Essayer tous les autres extracteurs
        for extractor_type, extractor in self.extractors.items():
            if extractor == excluded_extractor:
                continue
                
            try:
                if extractor.can_handle(file_path):
                    result = extractor.extract_with_validation(file_path)
                    fallback_results.append((extractor_type.value, result))
                    
                    # Si on trouve un résultat acceptable, l'utiliser
                    if (result.status in [ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL] 
                        and len(result.text.strip()) > 50):
                        
                        # Marquer comme extraction de fallback
                        result.metadata['extraction_method'] = 'fallback'
                        result.metadata['fallback_extractor'] = extractor_type.value
                        return result
                        
            except Exception as e:
                logger.warning(f"Fallback extraction failed with {extractor_type.value}: {e}")
                continue
        
        # Si aucun fallback n'a fonctionné, retourner le meilleur résultat partiel
        if fallback_results:
            best_result = max(
                fallback_results, 
                key=lambda x: len(x[1].text) if x[1].text else 0
            )
            
            result = best_result[1]
            result.metadata['extraction_method'] = 'fallback'
            result.metadata['fallback_extractor'] = best_result[0]
            result.status = ExtractionStatus.PARTIAL
            return result
        
        # Échec complet
        return ExtractionResult(
            text="",
            metadata={'error': 'All extraction methods failed'},
            status=ExtractionStatus.FAILED,
            document_type=DocumentType.UNKNOWN,
            errors=['All extraction methods failed']
        )
    
    def batch_extract(self, file_paths: List[Union[str, Path]], 
                     parallel: bool = True) -> List[ExtractionResult]:
        """
        Extraction en lot avec parallélisation optionnelle
        """
        if parallel:
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.extract_with_fallback, path): path 
                    for path in file_paths
                }
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        file_path = futures[future]
                        logger.error(f"Batch extraction failed for {file_path}: {e}")
                        results.append(ExtractionResult(
                            text="",
                            metadata={'error': str(e), 'file_path': str(file_path)},
                            status=ExtractionStatus.FAILED,
                            document_type=DocumentType.UNKNOWN,
                            errors=[str(e)]
                        ))
                
                return results
        else:
            # Traitement séquentiel
            return [self.extract_with_fallback(path) for path in file_paths]
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Retourne les formats supportés par extracteur"""
        supported = {}
        
        for extractor_type, extractor in self.extractors.items():
            supported[extractor_type.value] = {
                'extensions': list(extractor.supported_extensions),
                'mime_types': list(extractor.supported_mime_types)
            }
            
        return supported
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statistiques globales de tous les extracteurs"""
        stats = {}
        
        for extractor_type, extractor in self.extractors.items():
            stats[extractor_type.value] = extractor.get_statistics()
            
        return {
            'extractors': stats,
            'factory_config': {
                'fallback_enabled': self.fallback_enabled,
                'available_extractors': list(self.extractors.keys())
            }
        }


# Exemple d'utilisation et tests
if __name__ == "__main__":
    import asyncio
    
    async def test_factory():
        # Configuration pour tests
        config = {
            'pdf_config': {'enable_ocr': True, 'parallel_processing': True},
            'word_config': {'extract_images': True, 'preserve_formatting': True},
            'image_config': {'enable_parallel_processing': True},
            'enable_fallback': True
        }
        
        factory = DocumentProcessorFactory(config)
        
        # Test des formats supportés
        print("=== Formats supportés ===")
        supported = factory.get_supported_formats()
        for extractor, formats in supported.items():
            print(f"{extractor}: {formats['extensions']}")
        
        # Test d'extraction simple
        test_files = [
            Path("test.pdf"),
            Path("test.docx"),
            Path("test.jpg")
        ]
        
        for test_file in test_files:
            if test_file.exists():
                print(f"\n=== Test {test_file} ===")
                result = factory.extract_with_fallback(test_file)
                print(f"Status: {result.status}")
                print(f"Text length: {len(result.text)}")
                print(f"Method: {result.metadata.get('extraction_method', 'primary')}")
    
    # Exécuter les tests
    asyncio.run(test_factory())