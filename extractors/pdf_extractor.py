import sys
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import re
import concurrent.futures
import time
import numpy as np
import cv2
from PIL import Image
import pytesseract
import fitz #PyMuPDF
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from base_extractor import BaseExtractor, ExtractionResult, ExtractionStatus, DocumentType
from image_extractor import ImageExtractor

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class PageInfo:
    number: int
    text: str
    images: List[Dict[str, Any]]
    has_text: bool
    is_scanned: bool
    confidence: float
    language: Optional[str] = None

@dataclass
class TableInfo:
    page_number: int
    bbox: Tuple[float, float, float, float]
    rows: int
    cols: int
    text_content: str

class PDFExtractor(BaseExtractor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.supported_extensions = {'.pdf'}
        self.supported_mime_types = {'application/pdf'}

        # Configuration
        self.ocr_threshold = self.config.get('ocr_threshold', 100)
        self.enable_ocr = self.config.get('enable_ocr', True)
        self.preserve_formatting = self.config.get('preserve_formatting', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_metadata = self.config.get('extract_metadata', True)
        self.max_pages = self.config.get('max_pages', None)
        self.parallel_processing = self.config.get('parallel_processing', False)

        if self.enable_ocr:
            self.ocr_extractor = ImageExtractor(self.config.get('ocr_config', {}))

        # Patterns
        self.header_patterns = [
            r'^[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ\u0600-\u06FF\s]+$',
            r'^\d+\.\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ\u0600-\u06FF]',
            r'^CHAPITRE\s+\d+',
            r'^الفصل\s+\d+'
        ]

        self.footer_patterns = [
            r'^\d+$',
            r'^p\s+\d+',
            r'^ص\s+\d+',
            r'^\d{2}/\d{2}/\d{4}'
        ]

    def _extract_text_blocks(self, page) -> List[Dict[str, Any]]:
        """Extract text blocks with positioning and formatting."""
        blocks = []

        try:
            # Vérifier que la page est encore valide
            if page.parent is None:
                logger.error("Page has no parent document")
                return blocks

            text_dict = page.get_text("dict")

            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue

                block_text = ""
                block_fonts = []
                block_sizes = []

                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            block_fonts.append(span.get("font", ""))
                            block_sizes.append(span.get("size", 0))

                    if line_text.strip():
                        block_text += line_text.strip() + "\n"

                if block_text.strip():
                    avg_size = np.mean(block_sizes) if block_sizes else 12
                    most_common_font = max(set(block_fonts), key=block_fonts.count) if block_fonts else ""
                    block_type = self._classify_text_block(block_text, avg_size, most_common_font)

                    blocks.append({
                        'text': block_text.strip(),
                        'bbox': block.get('bbox', [0, 0, 0, 0]),
                        'type': block_type,
                        'font_size': avg_size,
                        'font': most_common_font
                    })

        except Exception as e:
            logger.error(f"Error extracting text blocks: {e}")
            try:
                # Fallback simple
                if page.parent is not None:
                    simple_text = page.get_text()
                    if simple_text.strip():
                        blocks.append({
                            'text': simple_text.strip(),
                            'bbox': [0, 0, page.rect.width, page.rect.height],
                            'type': 'body',
                            'font_size': 12,
                            'font': 'unknown'
                        })
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")

        return blocks

    def _classify_text_block(self, text: str, font_size: float, font: str) -> str:
        """Classify text block type."""
        text_clean = text.strip()

        if font_size > 14 or any(re.match(pattern, text_clean, re.MULTILINE) for pattern in self.header_patterns):
            return 'header'

        if any(re.match(pattern, text_clean, re.MULTILINE) for pattern in self.footer_patterns):
            return 'footer'

        if re.match(r'^\s*[-•▪▫◦‣⁃]\s+', text_clean) or re.match(r'^\s*\d+\.\s+', text_clean):
            return 'list'

        lines = text_clean.split('\n')
        if len(lines) > 2 and all(len(line.split()) > 3 for line in lines[:3]):
            return 'table'

        return 'body'

    def _detect_scanned_page(self, page) -> Tuple[bool, float]:
        """Detect if page is scanned."""
        try:
            # Vérifier que la page est encore valide
            if page.parent is None:
                return False, 0.5

            text = page.get_text().strip()
            text_length = len(text)
            images = page.get_images()
            image_count = len(images)

            # Heuristics
            is_scanned = (text_length < self.ocr_threshold and image_count > 0)
            confidence = 0.8 if not is_scanned else 0.9

            return is_scanned, confidence

        except Exception as e:
            logger.error(f"Error detecting scanned page: {e}")
            return False, 0.5

    def _process_page_with_doc(self, doc, page_num: int) -> PageInfo:
        """Process a single PDF page - version corrigée avec document passé en paramètre."""
        try:
            # Obtenir la page depuis le document
            page = doc[page_num - 1]  # page_num est 1-based

            text_blocks = self._extract_text_blocks(page)
            extracted_text = ""

            if text_blocks:
                if self.preserve_formatting:
                    headers = [block for block in text_blocks if block['type'] == 'header']
                    body_blocks = [block for block in text_blocks if block['type'] == 'body']
                    lists = [block for block in text_blocks if block['type'] == 'list']
                    tables = [block for block in text_blocks if block['type'] == 'table']

                    formatted_parts = []
                    for header in headers:
                        formatted_parts.append(f"# {header['text']}\n")
                    for body in body_blocks:
                        formatted_parts.append(f"{body['text']}\n")
                    for list_block in lists:
                        formatted_parts.append(f"{list_block['text']}\n")
                    for table in tables:
                        formatted_parts.append(f"[TABLE]\n{table['text']}\n[/TABLE]\n")

                    extracted_text = '\n'.join(formatted_parts)
                else:
                    extracted_text = '\n'.join([block['text'] for block in text_blocks])

            is_scanned, confidence = self._detect_scanned_page(page)

            # OCR si nécessaire
            if is_scanned and self.enable_ocr and len(extracted_text.strip()) < self.ocr_threshold:
                try:
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")

                    temp_path = f"/tmp/page_{page_num}.png"
                    with open(temp_path, "wb") as f:
                        f.write(img_data)

                    ocr_result = self.ocr_extractor.extract(temp_path)

                    if ocr_result.status in [ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL]:
                        if len(ocr_result.text) > len(extracted_text):
                            extracted_text = ocr_result.text
                            confidence = ocr_result.confidence or 0.7

                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except:
                        pass

                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")

            language = self._detect_text_language(extracted_text)

            # Images info
            images_info = []
            try:
                for img in page.get_images():
                    images_info.append({
                        'xref': img[0],
                        'width': img[2] if len(img) > 2 else None,
                        'height': img[3] if len(img) > 3 else None
                    })
            except:
                pass

            return PageInfo(
                number=page_num,
                text=extracted_text.strip(),
                images=images_info,
                has_text=len(extracted_text.strip()) > 0,
                is_scanned=is_scanned,
                confidence=confidence,
                language=language
            )

        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return PageInfo(
                number=page_num,
                text="",
                images=[],
                has_text=False,
                is_scanned=False,
                confidence=0.0
            )

    def _detect_text_language(self, text: str) -> str:
        """Detect text language."""
        if not text:
            return "unknown"

        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        french_chars = len(re.findall(r'[àâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'\S', text))

        if total_chars == 0:
            return "unknown"

        arabic_ratio = arabic_chars / total_chars
        french_ratio = (french_chars + latin_chars) / total_chars

        if arabic_ratio > 0.1:
            return "mixed" if french_ratio > 0.1 else "arabic"
        elif french_chars > 0 or (latin_chars / total_chars) > 0.7:
            return "french"
        else:
            return "unknown"

    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Extract text from PDF - VERSION CORRIGÉE."""
        doc = None
        try:
            file_path = Path(file_path)
            doc = fitz.open(str(file_path))

            if doc.page_count == 0:
                doc.close()
                return ExtractionResult(
                    text="",
                    metadata={'error': 'PDF has no pages'},
                    status=ExtractionStatus.FAILED,
                    document_type=DocumentType.PDF,
                    page_count=0
                )

            max_pages = min(doc.page_count, self.max_pages) if self.max_pages else doc.page_count
            pages_info = []

            # Traitement séquentiel avec document maintenu ouvert
            for i in range(max_pages):
                page_info = self._process_page_with_doc(doc, i + 1)
                pages_info.append(page_info)
                print(f"📄 Page {i+1}/{max_pages} traitée")

            # Metadata du document
            doc_metadata = {}
            if self.extract_metadata:
                try:
                    doc_metadata = {
                        'title': doc.metadata.get('title', ''),
                        'author': doc.metadata.get('author', ''),
                        'subject': doc.metadata.get('subject', ''),
                        'creator': doc.metadata.get('creator', ''),
                        'producer': doc.metadata.get('producer', '')
                    }
                except:
                    pass

            # Stocker le nombre de pages avant de fermer le document
            page_count = doc.page_count

            # Fermer le document maintenant que nous avons terminé
            doc.close()
            doc = None

            # Combiner le texte
            full_text = ""
            total_confidence = 0
            scanned_pages = 0
            languages = []

            for page_info in pages_info:
                if page_info.text:
                    full_text += f"\n--- Page {page_info.number} ---\n"
                    full_text += page_info.text + "\n"

                total_confidence += page_info.confidence
                if page_info.is_scanned:
                    scanned_pages += 1
                if page_info.language and page_info.language != "unknown":
                    languages.append(page_info.language)

            avg_confidence = total_confidence / len(pages_info) if pages_info else 0

            # Langue principale
            primary_language = "unknown"
            if languages:
                from collections import Counter
                lang_counts = Counter(languages)
                primary_language = lang_counts.most_common(1)[0][0]

            # Status
            if avg_confidence >= 0.8:
                status = ExtractionStatus.SUCCESS
            elif avg_confidence >= 0.5:
                status = ExtractionStatus.PARTIAL
            else:
                status = ExtractionStatus.FAILED

            # Metadata complète
            extraction_metadata = {
                'total_pages': page_count,
                'processed_pages': len(pages_info),
                'scanned_pages': scanned_pages,
                'average_confidence': round(avg_confidence, 3),
                'primary_language': primary_language,
                'languages_detected': list(set(languages)),
                'has_images': sum(len(p.images) for p in pages_info) > 0,
                'total_images': sum(len(p.images) for p in pages_info),
                'ocr_used': scanned_pages > 0,
                'document_metadata': doc_metadata,
                'page_details': [
                    {
                        'page': p.number,
                        'has_text': p.has_text,
                        'is_scanned': p.is_scanned,
                        'confidence': round(p.confidence, 3),
                        'language': p.language,
                        'image_count': len(p.images),
                        'text_length': len(p.text)
                    }
                    for p in pages_info
                ]
            }

            return ExtractionResult(
                text=full_text.strip(),
                metadata=extraction_metadata,
                status=status,
                document_type=DocumentType.PDF,
                page_count=page_count,
                language=primary_language,
                confidence=avg_confidence
            )

        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
            return ExtractionResult(
                text="",
                metadata={'error': str(e)},
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.PDF,
                errors=[str(e)]
            )
