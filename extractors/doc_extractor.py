import logging
import sys
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
from enum import Enum
from dataclasses import dataclass
import time

from base_extractor import BaseExtractor, ExtractionResult, DocumentType, ExtractionStatus
from image_extractor import ImageExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WordExtractor(BaseExtractor):
    """
    Advanced Word document extractor with comprehensive formatting preservation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Supported file types
        self.supported_extensions = {'.docx', '.doc', '.docm', '.dotx', '.dotm'}

        # Configuration
        self.preserve_formatting = self.config.get('preserve_formatting', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_images = self.config.get('extract_images', True)
        self.extract_headers_footers = self.config.get('extract_headers_footers', True)
        self.extract_metadata = self.config.get('extract_metadata', True)

        # Initialize OCR extractor for embedded images
        if self.extract_images:
            try:
                self.ocr_extractor = ImageExtractor(config.get('ocr_config', {}) if config else {})
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
                'created': str(core_props.created) if core_props.created else '',
                'modified': str(core_props.modified) if core_props.modified else '',
                'last_modified_by': core_props.last_modified_by or '',
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

            # Detect paragraph type based on content and formatting
            paragraph_type = self._classify_paragraph(text, style_name, heading_level)

            return {
                'text': text,
                'style': style_name,
                'type': paragraph_type,
                'heading_level': heading_level,
                'is_list_item': paragraph.style.name.startswith('List') if paragraph.style else False,
                'formatting': {
                    'bold': any(run.bold for run in paragraph.runs if run.bold),
                    'italic': any(run.italic for run in paragraph.runs if run.italic),
                    'underline': any(run.underline for run in paragraph.runs if run.underline)
                }
            }

        except Exception as e:
            logger.error(f"Error extracting paragraph: {e}")
            return None

    def _classify_paragraph(self, text: str, style: str, heading_level: int) -> str:
        """Classify paragraph type based on content and style."""
        text_lower = text.lower().strip()

        # Check for headings
        if heading_level > 0 or 'heading' in style.lower():
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
        if ('table' in text_lower and ('contents' in text_lower or 'matières' in text_lower)):
            return 'toc'

        # Check for references/citations
        if text_lower.startswith(('ref:', 'référence:', 'source:')):
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
        """Extract text from Word document with comprehensive formatting preservation."""
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

            # Process document elements
            extracted_elements = []
            tables_extracted = 0

            # Iterate through document elements
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

            # Combine all text
            full_text = ""

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

            # Detect language
            detected_language = self._detect_document_language(full_text)

            # Calculate confidence
            confidence = 0.9  # High confidence for native Word extraction

            # Determine extraction status
            if len(full_text.strip()) > 100:
                status = ExtractionStatus.SUCCESS
            elif len(full_text.strip()) > 0:
                status = ExtractionStatus.PARTIAL
            else:
                status = ExtractionStatus.FAILED

            # Create metadata
            metadata = {
                'document_properties': doc_properties,
                'total_paragraphs': len([e for e in extracted_elements if e.get('type') != 'table']),
                'total_tables': tables_extracted,
                'total_images': 0,  # Simplified for this version
                'detected_language': detected_language,
                'extraction_method': 'native',
                'formatting_preserved': self.preserve_formatting,
                'structure_analysis': {
                    'headings': len([e for e in extracted_elements if e.get('type') == 'heading']),
                    'lists': len([e for e in extracted_elements if e.get('type') == 'list_item']),
                    'body_paragraphs': len([e for e in extracted_elements if e.get('type') == 'body'])
                }
            }

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
        """Handle legacy .doc files."""
        try:
            # Try using python-docx2txt if available
            try:
                import docx2txt
                text = docx2txt.process(str(file_path))
            except ImportError:
                # Fallback method
                text = "Extraction de fichiers .doc nécessite docx2txt"

            if text:
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
                    confidence=0.8
                )
            else:
                return ExtractionResult(
                    text="",
                    metadata={'error': 'No text extracted from legacy document'},
                    status=ExtractionStatus.FAILED,
                    document_type=DocumentType.WORD,
                    errors=['Failed to extract text from .doc file']
                )
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={'error': str(e)},
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.WORD,
                errors=[str(e)]
            )
