# PDF processing (PyPDF2, pdfplumber, pdfminer)
"""
High-performance PDF text extraction with OCR fallback and advanced formatting preservation.
Handles complex layouts, multilingual content (French/Arabic), and scanned documents.
"""

import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import re
import cv2
import numpy as np
from PIL import Image
import io
import concurrent.futures
from dataclasses import dataclass

from .base_extractor import BaseExtractor, ExtractionResult, ExtractionStatus, DocumentType
from .image_extractor import ImageExtractor

logger = logging.getLogger(__name__)


@dataclass
class PageInfo:
    """Information about a PDF page."""
    number: int
    text: str
    images: List[Dict[str, Any]]
    has_text: bool
    is_scanned: bool
    confidence: float
    language: Optional[str] = None


@dataclass
class TableInfo:
    """Information about detected tables."""
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    rows: int
    cols: int
    text_content: str


class PDFExtractor(BaseExtractor):
    """
    Advanced PDF text extractor with intelligent text/OCR hybrid approach.
    Optimized for government documents with complex layouts and multilingual content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Supported file types
        self.supported_extensions = {'.pdf'}
        self.supported_mime_types = {'application/pdf'}
        
        # Configuration
        self.ocr_threshold = self.config.get('ocr_threshold', 100)  # Min chars before OCR
        self.enable_ocr = self.config.get('enable_ocr', True)
        self.preserve_formatting = self.config.get('preserve_formatting', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_metadata = self.config.get('extract_metadata', True)
        self.max_pages = self.config.get('max_pages', None)
        self.parallel_processing = self.config.get('parallel_processing', True)
        
        # Initialize OCR extractor
        if self.enable_ocr:
            self.ocr_extractor = ImageExtractor(config.get('ocr_config', {}))
        
        # Text extraction patterns
        self.header_patterns = [
            r'^[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ\u0600-\u06FF\s]+$',  # All caps headers
            r'^\d+\.\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ\u0600-\u06FF]',  # Numbered headers
            r'^CHAPITRE\s+\d+',  # French chapter headers
            r'^الفصل\s+\d+',  # Arabic chapter headers
        ]
        
        self.footer_patterns = [
            r'^\d+',  # Page numbers
            r'^p\s+\d+',  # Page indicators
            r'^ص\s+\d+',  # Arabic page indicators
            r'^\d{2}/\d{2}/\d{4}',  # Date patterns
        ]
    
    def _extract_text_blocks(self, page) -> List[Dict[str, Any]]:
        """Extract text blocks with positioning and formatting information."""
        blocks = []
        
        try:
            # Get text blocks with formatting
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
                    # Determine block type based on formatting
                    avg_size = np.mean(block_sizes) if block_sizes else 12
                    most_common_font = max(set(block_fonts), key=block_fonts.count) if block_fonts else ""
                    
                    block_type = self._classify_text_block(block_text, avg_size, most_common_font)
                    
                    blocks.append({
                        'text': block_text.strip(),
                        'bbox': block.get('bbox', [0, 0, 0, 0]),
                        'type': block_type,
                        'font_size': avg_size,
                        'font': most_common_font,
                        'line_count': len([line for line in block_text.split('\n') if line.strip()])
                    })
        
        except Exception as e:
            logger.error(f"Error extracting text blocks: {e}")
            # Fallback to simple text extraction
            try:
                simple_text = page.get_text()
                if simple_text.strip():
                    blocks.append({
                        'text': simple_text.strip(),
                        'bbox': [0, 0, page.rect.width, page.rect.height],
                        'type': 'body',
                        'font_size': 12,
                        'font': 'unknown',
                        'line_count': len(simple_text.split('\n'))
                    })
            except Exception as fallback_error:
                logger.error(f"Fallback text extraction failed: {fallback_error}")
        
        return blocks
    
    def _classify_text_block(self, text: str, font_size: float, font: str) -> str:
        """Classify text block type based on content and formatting."""
        text_clean = text.strip()
        
        # Check for headers
        if font_size > 14 or any(re.match(pattern, text_clean, re.MULTILINE) for pattern in self.header_patterns):
            return 'header'
        
        # Check for footers
        if any(re.match(pattern, text_clean, re.MULTILINE) for pattern in self.footer_patterns):
            return 'footer'
        
        # Check for lists
        if re.match(r'^\s*[-•▪▫◦‣⁃]\s+', text_clean) or re.match(r'^\s*\d+\.\s+', text_clean):
            return 'list'
        
        # Check for tables (basic heuristic)
        lines = text_clean.split('\n')
        if len(lines) > 2 and all(len(line.split()) > 3 for line in lines[:3]):
            return 'table'
        
        return 'body'
    
    def _detect_scanned_page(self, page) -> Tuple[bool, float]:
        """Detect if page is scanned based on text/image ratio."""
        try:
            # Get text content
            text = page.get_text().strip()
            text_length = len(text)
            
            # Get images
            images = page.get_images()
            image_count = len(images)
            
            # Calculate page area covered by images
            total_image_area = 0
            page_area = page.rect.width * page.rect.height
            
            for img in images:
                try:
                    img_dict = page.get_image_rects(img[0])
                    for rect in img_dict:
                        total_image_area += rect.width * rect.height
                except:
                    continue
            
            image_coverage = total_image_area / page_area if page_area > 0 else 0
            
            # Heuristic: likely scanned if low text and high image coverage
            is_scanned = (text_length < self.ocr_threshold and 
                         (image_coverage > 0.3 or image_count > 0))
            
            confidence = 0.8 if is_scanned else 0.9
            
            return is_scanned, confidence
            
        except Exception as e:
            logger.error(f"Error detecting scanned page: {e}")
            return False, 0.5
    
    def _extract_page_images(self, page, page_num: int) -> List[np.ndarray]:
        """Extract images from PDF page for OCR processing."""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to RGB if necessary
                    if base_image.n - base_image.alpha < 4:
                        image_data = base_image.tobytes("ppm")
                    else:
                        # Convert CMYK to RGB
                        rgb_image = fitz.Pixmap(fitz.csRGB, base_image)
                        image_data = rgb_image.tobytes("ppm")
                        rgb_image = None
                    
                    # Convert to numpy array
                    pil_image = Image.open(io.BytesIO(image_data))
                    np_image = np.array(pil_image)
                    
                    # Convert RGB to BGR for OpenCV
                    if len(np_image.shape) == 3:
                        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                    
                    images.append(np_image)
                    base_image = None
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")
        
        return images
    
    def _process_page(self, page, page_num: int) -> PageInfo:
        """Process a single PDF page with text extraction and OCR fallback."""
        try:
            # Extract text blocks
            text_blocks = self._extract_text_blocks(page)
            extracted_text = ""
            
            if text_blocks:
                if self.preserve_formatting:
                    # Preserve structure based on block types
                    headers = [block for block in text_blocks if block['type'] == 'header']
                    body_blocks = [block for block in text_blocks if block['type'] == 'body']
                    lists = [block for block in text_blocks if block['type'] == 'list']
                    tables = [block for block in text_blocks if block['type'] == 'table']
                    
                    # Reconstruct text with formatting
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
            
            # Detect if page needs OCR
            is_scanned, confidence = self._detect_scanned_page(page)
            
            # Apply OCR if needed
            if is_scanned and self.enable_ocr and len(extracted_text.strip()) < self.ocr_threshold:
                try:
                    # Render page as image for OCR
                    mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Save temporarily and process with OCR
                    temp_path = f"/tmp/page_{page_num}.png"
                    with open(temp_path, "wb") as f:
                        f.write(img_data)
                    
                    # Extract text using OCR
                    ocr_result = self.ocr_extractor.extract(temp_path)
                    
                    if ocr_result.status in [ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL]:
                        if len(ocr_result.text) > len(extracted_text):
                            extracted_text = ocr_result.text
                            confidence = ocr_result.confidence or 0.7
                    
                    # Clean up temp file
                    try:
                        Path(temp_path).unlink()
                    except:
                        pass
                        
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")
            
            # Detect language
            language = self._detect_text_language(extracted_text)
            
            # Extract images info
            images_info = []
            try:
                for img in page.get_images():
                    images_info.append({
                        'xref': img[0],
                        'smask': img[1],
                        'width': img[2],
                        'height': img[3],
                        'bpc': img[4],
                        'colorspace': img[5],
                        'alt': img[6] if len(img) > 6 else None,
                        'name': img[7] if len(img) > 7 else None,
                        'filter': img[8] if len(img) > 8 else None
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
        """Detect text language based on character analysis."""
        if not text:
            return "unknown"
        
        # Count Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        
        # Count French/Latin characters
        french_chars = len(re.findall(r'[àâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
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
        elif french_chars > 0 or (latin_chars / total_chars) > 0.7:
            return "french"
        else:
            return "unknown"
    
    def _extract_tables(self, page) -> List[TableInfo]:
        """Extract table information from PDF page."""
        tables = []
        
        try:
            # Simple table detection based on text alignment
            text_dict = page.get_text("dict")
            
            # Group text by vertical position to find table-like structures
            lines_by_y = {}
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    y_pos = round(line["bbox"][1])  # Top y coordinate
                    
                    if y_pos not in lines_by_y:
                        lines_by_y[y_pos] = []
                    
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    
                    if line_text.strip():
                        lines_by_y[y_pos].append({
                            'text': line_text.strip(),
                            'bbox': line["bbox"]
                        })
            
            # Look for table patterns (multiple aligned columns)
            potential_tables = []
            current_table_lines = []
            
            for y_pos in sorted(lines_by_y.keys()):
                lines = lines_by_y[y_pos]
                
                # Check if this line could be part of a table (multiple columns)
                if len(lines) >= 2:
                    # Check alignment with previous lines
                    if current_table_lines:
                        # Simple heuristic: similar number of columns
                        if abs(len(lines) - len(current_table_lines[-1])) <= 1:
                            current_table_lines.append(lines)
                        else:
                            # End current table
                            if len(current_table_lines) >= 2:
                                potential_tables.append(current_table_lines)
                            current_table_lines = [lines]
                    else:
                        current_table_lines = [lines]
                else:
                    # End current table if we have enough lines
                    if len(current_table_lines) >= 2:
                        potential_tables.append(current_table_lines)
                    current_table_lines = []
            
            # Process potential tables
            for i, table_lines in enumerate(potential_tables):
                if len(table_lines) < 2:
                    continue
                
                # Calculate bounding box
                min_x = min(min(line['bbox'][0] for line in row) for row in table_lines)
                min_y = min(min(line['bbox'][1] for line in row) for row in table_lines)
                max_x = max(max(line['bbox'][2] for line in row) for row in table_lines)
                max_y = max(max(line['bbox'][3] for line in row) for row in table_lines)
                
                # Extract text content
                table_text = ""
                for row in table_lines:
                    row_text = " | ".join([line['text'] for line in row])
                    table_text += row_text + "\n"
                
                tables.append(TableInfo(
                    page_number=page.number,
                    bbox=(min_x, min_y, max_x, max_y),
                    rows=len(table_lines),
                    cols=max(len(row) for row in table_lines),
                    text_content=table_text.strip()
                ))
                
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
        
        return tables
    
    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text from PDF with intelligent text/OCR hybrid approach.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ExtractionResult containing extracted text and comprehensive metadata
        """
        try:
            file_path = Path(file_path)
            
            # Open PDF document
            doc = fitz.open(str(file_path))
            
            if doc.page_count == 0:
                return ExtractionResult(
                    text="",
                    metadata={'error': 'PDF has no pages'},
                    status=ExtractionStatus.FAILED,
                    document_type=DocumentType.PDF,
                    page_count=0
                )
            
            # Limit pages if specified
            max_pages = min(doc.page_count, self.max_pages) if self.max_pages else doc.page_count
            
            # Process pages
            pages_info = []
            
            if self.parallel_processing and max_pages > 1:
                # Parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_page = {
                        executor.submit(self._process_page, doc[i], i + 1): i
                        for i in range(max_pages)
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_page):
                        page_info = future.result()
                        pages_info.append(page_info)
            else:
                # Sequential processing
                for i in range(max_pages):
                    page_info = self._process_page(doc[i], i + 1)
                    pages_info.append(page_info)
            
            # Sort pages by number
            pages_info.sort(key=lambda x: x.number)
            
            # Extract document metadata
            doc_metadata = {}
            if self.extract_metadata:
                try:
                    doc_metadata = {
                        'title': doc.metadata.get('title', ''),
                        'author': doc.metadata.get('author', ''),
                        'subject': doc.metadata.get('subject', ''),
                        'creator': doc.metadata.get('creator', ''),
                        'producer': doc.metadata.get('producer', ''),
                        'creation_date': doc.metadata.get('creationDate', ''),
                        'modification_date': doc.metadata.get('modDate', ''),
                        'keywords': doc.metadata.get('keywords', '')
                    }
                except:
                    pass
            
            # Extract tables if enabled
            all_tables = []
            if self.extract_tables:
                for i in range(max_pages):
                    try:
                        page_tables = self._extract_tables(doc[i])
                        all_tables.extend(page_tables)
                    except Exception as e:
                        logger.warning(f"Table extraction failed for page {i+1}: {e}")
            
            # Combine text from all pages
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
            
            # Calculate overall statistics
            avg_confidence = total_confidence / len(pages_info) if pages_info else 0
            
            # Determine primary language
            primary_language = "unknown"
            if languages:
                from collections import Counter
                lang_counts = Counter(languages)
                primary_language = lang_counts.most_common(1)[0][0]
            
            # Determine extraction status
            if avg_confidence >= 0.8:
                status = ExtractionStatus.SUCCESS
            elif avg_confidence >= 0.5:
                status = ExtractionStatus.PARTIAL
            else:
                status = ExtractionStatus.FAILED
            
            # Create comprehensive metadata
            extraction_metadata = {
                'total_pages': doc.page_count,
                'processed_pages': len(pages_info),
                'scanned_pages': scanned_pages,
                'average_confidence': round(avg_confidence, 3),
                'primary_language': primary_language,
                'languages_detected': list(set(languages)),
                'has_images': sum(len(p.images) for p in pages_info) > 0,
                'total_images': sum(len(p.images) for p in pages_info),
                'tables_extracted': len(all_tables),
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
            
            if all_tables:
                extraction_metadata['tables'] = [
                    {
                        'page': t.page_number,
                        'rows': t.rows,
                        'cols': t.cols,
                        'bbox': t.bbox
                    }
                    for t in all_tables
                ]
            
            # Close document
            doc.close()
            
            return ExtractionResult(
                text=full_text.strip(),
                metadata=extraction_metadata,
                status=status,
                document_type=DocumentType.PDF,
                page_count=doc.page_count,
                language=primary_language,
                confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            return ExtractionResult(
                text="",
                metadata={'error': str(e)},
                status=ExtractionStatus.FAILED,
                document_type=DocumentType.PDF,
                errors=[str(e)]
            )