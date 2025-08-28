# Word documents (.docx, .doc)
# Word document processing (python-docx, python-docx2txt)
"""
Advanced Word document text extraction with support for complex formatting,
tables, images, and multilingual content (French/Arabic).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import re
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from PIL import Image
import tempfile
import os
import numpy as np

from .base_extractor import BaseExtractor, ExtractionResult, ExtractionStatus, DocumentType
from .image_extractor import ImageExtractor

logger = logging.getLogger(__name__)


class WordExtractor(BaseExtractor):
    """
    Advanced Word document extractor with comprehensive formatting preservation
    and multilingual support for transport ministry documents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Supported file types
        self.supported_extensions = {'.docx', '.doc', '.docm', '.dotx', '.dotm'}
        self.supported_mime_types = {
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/vnd.ms-word.document.macroEnabled.12',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.template',
            'application/vnd.ms-word.template.macroEnabled.12'
        }
        
        # Configuration
        self.preserve_formatting = self.config.get('preserve_formatting', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_images = self.config.get('extract_images', True)
        self.extract_headers_footers = self.config.get('extract_headers_footers', True)
        self.extract_metadata = self.config.get('extract_metadata', True)
        self.include_comments = self.config.get('include_comments', False)
        self.include_track_changes = self.config.get('include_track_changes', False)
        
        # Initialize OCR extractor for embedded images
        if self.extract_images:
            try:
                self.ocr_extractor = ImageExtractor(config.get('ocr_config', {}))
            except Exception as e:
                logger.warning(f"OCR extractor initialization failed: {e}")
                self.ocr_extractor = None
        
        # Style patterns for document structure
        self.heading_patterns = [
            r'^TITRE\s+[IVXLCDM]+',  # French titles
            r'^CHAPITRE\s+\d+',      # French chapters
            r'^SECTION\s+\d+',       # French sections
            r'^ARTICLE\s+\d+',       # French articles
            r'^العنوان\s+\d+',        # Arabic titles
            r'^الفصل\s+\d+',          # Arabic chapters
            r'^القسم\s+\d+',          # Arabic sections
            r'^المادة\s+\d+',         # Arabic articles
        ]
    
    def _extract_document_properties(self, doc: DocxDocument) -> Dict[str, Any]:
        """Extract document metadata and properties."""
        properties = {}
        
        try:
            core_props = doc.core_properties
            
            properties.update({
                'title': core_props.title or '',
                'author': core_props.author or '',
                'subject': core_props.subject or '',
                'keywords': core_props.keywords or '',
                'comments': core_props.comments or '',
                'category': core_props.category or '',
                'created': str(core_props.created) if core_props.created else '',
                'modified': str(core_props.modified) if core_props.modified else '',
                'last_modified_by': core_props.last_modified_by or '',
                'revision': str(core_props.revision) if core_props.revision else '',
                'language': core_props.language or ''
            })
            
        except Exception as e:
            logger.warning(f"Error extracting document properties: {e}")
        
        return properties
    
    def _extract_paragraph_text(self, paragraph: Paragraph) -> Dict[str, Any]:
        """Extract text from paragraph with formatting information."""
        try:
            text = paragraph.text.strip()
            
            if not text:
                return None
            
            # Analyze paragraph style
            style_name = paragraph.style.name if paragraph.style else 'Normal'
            
            # Detect heading level
            heading_level = 0
            if 'heading' in style_name.lower() or 'titre' in style_name.lower():
                # Extract heading level from style name
                heading_match = re.search(r'(\d+)', style_name)
                if heading_match:
                    heading_level = int(heading_match.group(1))
                else:
                    heading_level = 1
            
            # Check for list items
            is_list_item = paragraph.style.name.startswith('List') if paragraph.style else False
            
            # Analyze formatting
            has_bold = any(run.bold for run in paragraph.runs if run.bold)
            has_italic = any(run.italic for run in paragraph.runs if run.italic)
            has_underline = any(run.underline for run in paragraph.runs if run.underline)
            
            # Detect paragraph type based on content and formatting
            paragraph_type = self._classify_paragraph(text, style_name, heading_level)
            
            return {
                'text': text,
                'style': style_name,
                'type': paragraph_type,
                'heading_level': heading_level,
                'is_list_item': is_list_item,
                'formatting': {
                    'bold': has_bold,
                    'italic': has_italic,
                    'underline': has_underline
                },
                'alignment': str(paragraph.alignment) if paragraph.alignment else 'left'
            }
            
        except Exception as e:
            logger.error(f"Error extracting paragraph: {e}")
            return None
    
    def _classify_paragraph(self, text: str, style: str, heading_level: int) -> str:
        """Classify paragraph type based on content and style."""
        text_lower = text.lower().strip()
        
        # Check for headings
        if heading_level > 0 or 'heading' in style.lower() or 'titre' in style.lower():
            return 'heading'
        
        # Check for specific patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return 'heading'
        
        # Check for lists
        if (re.match(r'^\s*[-•▪▫◦‣⁃*]\s+', text) or 
            re.match(r'^\s*\d+[.)]\s+', text) or
            re.match(r'^\s*[a-zA-Z][.)]\s+', text)):
            return 'list_item'
        
        # Check for table of contents
        if ('table' in text_lower and ('contents' in text_lower or 'matières' in text_lower or
                                      'محتويات' in text_lower)):
            return 'toc'
        
        # Check for references/citations
        if text_lower.startswith(('ref:', 'référence:', 'source:', 'مرجع:')):
            return 'reference'
        
        # Default to body text
        return 'body'
    
    def _extract_table_text(self, table: Table) -> Dict[str, Any]:
        """Extract text from table with structure preservation."""
        try:
            rows_data = []
            max_cols = 0
            
            for i, row in enumerate(table.rows):
                row_data = []
                for j, cell in enumerate(row.cells):
                    cell_text = []
                    for paragraph in cell.paragraphs:
                        if paragraph.text.strip():
                            cell_text.append(paragraph.text.strip())
                    
                    cell_content = ' '.join(cell_text)
                    row_data.append(cell_content)
                
                rows_data.append(row_data)
                max_cols = max(max_cols, len(row_data))
            
            # Normalize row lengths
            for row in rows_data:
                while len(row) < max_cols:
                    row.append('')
            
            # Create formatted table text
            if self.preserve_formatting:
                formatted_text = "[TABLE]\n"
                for i, row in enumerate(rows_data):
                    if i == 0:  # Header row
                        formatted_text += "| " + " | ".join(row) + " |\n"
                        formatted_text += "|" + "|".join([" --- " for _ in row]) + "|\n"
                    else:
                        formatted_text += "| " + " | ".join(row) + " |\n"
                formatted_text += "[/TABLE]\n"
            else:
                formatted_text = '\n'.join(['\t'.join(row) for row in rows_data])
            
            return {
                'text': formatted_text,
                'rows': len(rows_data),
                'columns': max_cols,
                'data': rows_data
            }
            
        except Exception as e:
            logger.error(f"Error extracting table: {e}")
            return {'text': '', 'rows': 0, 'columns': 0, 'data': []}
    
    def _extract_images_from_zip(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract embedded images from Word document."""
        images = []
        
        if not self.extract_images or not self.ocr_extractor:
            return images
        
        try:
            with zipfile.ZipFile(file_path, 'r') as docx_zip:
                # Look for image files in the media folder
                media_files = [f for f in docx_zip.namelist() if f.startswith('word/media/')]
                
                for media_file in media_files:
                    try:
                        # Extract image
                        image_data = docx_zip.read(media_file)
                        
                        # Create temporary file for OCR processing
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                            temp_file.write(image_data)
                            temp_path = temp_file.name
                        
                        # Extract text from image using OCR
                        ocr_result = self.ocr_extractor.extract(temp_path)
                        
                        images.append({
                            'filename': media_file,
                            'text': ocr_result.text,
                            'confidence': ocr_result.confidence,
                            'language': ocr_result.language,
                            'status': ocr_result.status.value
                        })
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                            
                    except Exception as e:
                        logger.warning(f"Failed to process image {media_file}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error extracting images from Word document: {e}")
        
        return images
    
    def _extract_headers_footers(self, doc_path: Path) -> Dict[str, str]:
        """Extract headers and footers from Word document."""
        headers_footers = {'headers': '', 'footers': ''}
        
        if not self.extract_headers_footers:
            return headers_footers
        
        try:
            # Parse the document XML to extract headers/footers
            with zipfile.ZipFile(doc_path, 'r') as docx_zip:
                # Look for header and footer files
                header_files = [f for f in docx_zip.namelist() if 'header' in f.lower()]
                footer_files = [f for f in docx_zip.namelist() if 'footer' in f.lower()]
                
                # Extract headers
                header_texts = []
                for header_file in header_files:
                    try:
                        header_xml = docx_zip.read(header_file).decode('utf-8')
                        root = ET.fromstring(header_xml)
                        
                        # Extract text from header XML
                        for text_elem in root.iter():
                            if text_elem.tag.endswith('}t') and text_elem.text:
                                header_texts.append(text_elem.text.strip())
                    except Exception as e:
                        logger.warning(f"Failed to extract header from {header_file}: {e}")
                
                # Extract footers
                footer_texts = []
                for footer_file in footer_files:
                    try:
                        footer_xml = docx_zip.read(footer_file).decode('utf-8')
                        root = ET.fromstring(footer_xml)
                        
                        # Extract text from footer XML
                        for text_elem in root.iter():
                            if text_elem.tag.endswith('}t') and text_elem.text:
                                footer_texts.append(text_elem.text.strip())
                    except Exception as e:
                        logger.warning(f"Failed to extract footer from {footer_file}: {e}")
                
                headers_footers['headers'] = ' '.join(header_texts)
                headers_footers['footers'] = ' '.join(footer_texts)
                
        except Exception as e:
            logger.error(f"Error extracting headers/footers: {e}")
        
        return headers_footers
    
    def _detect_document_language(self, text: str) -> str:
        """Detect primary document language."""
        if not text:
            return "unknown"
        
        # Count Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        
        # Count French-specific characters
        french_chars = len(re.findall(r'[àâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
        
        # Count general Latin characters
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(re.findall(r'\S', text))
        
        if total_chars == 0:
            return "unknown"
        
        arabic_ratio = arabic_chars / total_chars
        french_ratio = (french_chars + latin_chars) / total_chars
        
        if arabic_ratio > 0.1:
            if french_ratio > 0.1:
                return "mixed"
            return "arabic"
        elif french_chars > 0 or french_ratio > 0.7:
            return "french"
        else:
            return "unknown"
    
    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text from Word document with comprehensive formatting preservation.
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            ExtractionResult containing extracted text and metadata
        """
        try:
            file_path = Path(file_path)
            
            # Check if file is .doc (older format)
            if file_path.suffix.lower() == '.doc':
                return self._extract_legacy_doc(file_path)
            
            # Open Word document
            doc = Document(str(file_path))
            
            # Extract document properties
            doc_properties = {}
            if self.extract_metadata:
                doc_properties = self._extract_document_properties(doc)
            
            # Extract headers and footers
            headers_footers = self._extract_headers_footers(file_path)
            
            # Extract embedded images
            embedded_images = self._extract_images_from_zip(file_path)
            
            # Process document elements
            extracted_elements = []
            tables_extracted = 0
            images_with_text = 0
            
            # Iterate through document elements (paragraphs and tables)
            for element in doc.element.body:
                if isinstance(element, CT_P):  # Paragraph
                    paragraph = Paragraph(element, doc)
                    para_info = self._extract_paragraph_text(paragraph)
                    if para_info:
                        extracted_elements.append(para_info)
                
                elif isinstance(element, CT_Tbl):  # Table
                    if self.extract_tables:
                        table = Table(element, doc)
                        table_info = self._extract_table_text(table)
                        if table_info['text']:
                            extracted_elements.append({
                                'text': table_info['text'],
                                'type': 'table',
                                'rows': table_info['rows'],
                                'columns': table_info['columns']
                            })
                            tables_extracted += 1
            
            # Count images with extracted text
            images_with_text = len([img for img in embedded_images if img['text'].strip()])
            
            # Combine all text
            full_text = ""
            
            # Add headers if available
            if headers_footers['headers']:
                full_text += f"[HEADER]\n{headers_footers['headers']}\n[/HEADER]\n\n"
            
            # Process extracted elements
            for element in extracted_elements:
                if self.preserve_formatting:
                    element_type = element.get('type', 'body')
                    text = element['text']
                    
                    if element_type == 'heading':
                        level = element.get('heading_level', 1)
                        full_text += f"{'#' * level} {text}\n\n"
                    elif element_type == 'list_item':
                        full_text += f"• {text}\n"
                    elif element_type == 'table':
                        full_text += f"{text}\n"
                    else:
                        full_text += f"{text}\n\n"
                else:
                    full_text += f"{element['text']}\n"
            
            # Add image text if extracted
            if embedded_images and images_with_text > 0:
                full_text += "\n[EXTRACTED FROM IMAGES]\n"
                for img in embedded_images:
                    if img['text'].strip():
                        full_text += f"{img['text']}\n"
                full_text += "[/EXTRACTED FROM IMAGES]\n"
            
            # Add footers if available
            if headers_footers['footers']:
                full_text += f"\n[FOOTER]\n{headers_footers['footers']}\n[/FOOTER]\n"
            
            # Detect language
            detected_language = self._detect_document_language(full_text)
            
            # Calculate confidence based on successful extraction
            confidence = 0.9  # High confidence for native Word extraction
            if embedded_images:
                avg_ocr_confidence = np.mean([img['confidence'] for img in embedded_images 
                                            if img['confidence'] is not None]) if embedded_images else 0
                confidence = (confidence + avg_ocr_confidence) / 2
            
            # Determine extraction status
            if len(full_text.strip()) > 100:
                status = ExtractionStatus.SUCCESS
            elif len(full_text.strip()) > 0:
                status = ExtractionStatus.PARTIAL
            else:
                status = ExtractionStatus.FAILED
            
            # Create comprehensive metadata
            metadata = {
                'document_properties': doc_properties,
                'total_paragraphs': len([e for e in extracted_elements if e.get('type') != 'table']),
                'total_tables': tables_extracted,
                'total_images': len(embedded_images),
                'images_with_text': images_with_text,
                'has_headers': bool(headers_footers['headers']),
                'has_footers': bool(headers_footers['footers']),
                'detected_language': detected_language,
                'extraction_method': 'native',
                'formatting_preserved': self.preserve_formatting,
                'structure_analysis': {
                    'headings': len([e for e in extracted_elements if e.get('type') == 'heading']),
                    'lists': len([e for e in extracted_elements if e.get('type') == 'list_item']),
                    'body_paragraphs': len([e for e in extracted_elements if e.get('type') == 'body'])
                }
            }
            
            if embedded_images:
                metadata['image_details'] = [
                    {
                        'filename': img['filename'],
                        'has_text': bool(img['text'].strip()),
                        'confidence': img['confidence'],
                        'language': img['language']
                    }
                    for img in embedded_images
                ]
            
            return ExtractionResult(
                text=full_text.strip(),
                metadata=metadata,
                status=status,
                document_type=DocumentType.WORD,
                language=detected_language,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Word document extraction failed for {file_path}: {e}")
            return ExtractionResult(
                text="",
                metadata={'error': str(e)},
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.WORD,
                errors=[str(e)]
            )
    
    def _extract_legacy_doc(self, file_path: Path) -> ExtractionResult:
        """Handle legacy .doc files (requires additional tools like antiword or LibreOffice)."""
        try:
            # Try using python-docx2txt as fallback for .doc files
            import docx2txt
            
            # Extract text
            text = docx2txt.process(str(file_path))
            
            if text:
                # Detect language
                detected_language = self._detect_document_language(text)
                
                return ExtractionResult(
                    text=text.strip(),
                    metadata={
                        'extraction_method': 'docx2txt',
                        'detected_language': detected_language,
                        'file_format': 'legacy_doc'
                    },
                    status=ExtractionStatus.SUCCESS,
                    document_type=DocumentType.WORD,
                    language=detected_language,
                    confidence=0.8  # Slightly lower confidence for legacy extraction
                )
            else:
                return ExtractionResult(
                    text="",
                    metadata={'error': 'No text extracted from legacy document'},
                    status=ExtractionStatus.FAILED,
                    document_type=DocumentType.WORD,
                    errors=['Failed to extract text from .doc file']
                )
                
        except ImportError:
            logger.error("docx2txt not available for legacy .doc file processing")
            return ExtractionResult(
                text="",
                metadata={'error': 'docx2txt not available for .doc files'},
                status=ExtractionStatus.UNSUPPORTED,
                document_type=DocumentType.WORD,
                errors=['Legacy .doc format requires docx2txt library']
            )
        except Exception as e:
            logger.error(f"Legacy .doc extraction failed: {e}")
            return ExtractionResult(
                text="",
                metadata={'error': str(e)},
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.WORD,
                errors=[str(e)]
            )