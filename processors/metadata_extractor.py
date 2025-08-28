# Document metadata extraction and analysis
"""
Comprehensive metadata extraction from various document types.
Extracts structural, content, and administrative metadata for transport ministry documents.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re
from collections import Counter, defaultdict
import numpy as np
from dateutil import parser as date_parser
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Comprehensive document metadata container."""
    # Basic file information
    file_path: str
    file_name: str
    file_size: int
    file_extension: str
    file_hash: str
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    
    # Document properties
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: List[str] = None
    language: str = ""
    page_count: int = 0
    word_count: int = 0
    character_count: int = 0
    
    # Content analysis
    document_type: str = ""  # report, regulation, memo, etc.
    document_category: str = ""  # legal, technical, administrative
    ministry_department: str = ""
    document_classification: str = ""  # public, restricted, etc.
    
    # Structural metadata
    has_table_of_contents: bool = False
    section_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    reference_count: int = 0
    
    # Language and content metadata
    primary_language: str = ""
    secondary_languages: List[str] = None
    text_quality_score: float = 0.0
    readability_score: float = 0.0
    
    # Administrative metadata
    document_version: str = ""
    document_status: str = ""
    approval_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    
    # Processing metadata
    extraction_timestamp: datetime = None
    processing_time: float = 0.0
    extraction_method: str = ""
    confidence_score: float = 0.0


class DocumentClassifier:
    """Classify documents based on content and structure."""
    
    def __init__(self):
        # Classification patterns for transport ministry documents
        self.document_types = {
            'regulation': [
                r'arrêté|décret|circulaire|directive',
                r'قرار|مرسوم|منشور|توجيه',
                r'article\s+\d+|chapitre\s+\d+',
                r'المادة\s+\d+|الفصل\s+\d+'
            ],
            'report': [
                r'rapport|étude|analyse|bilan',
                r'تقرير|دراسة|تحليل|حصيلة',
                r'résultats|conclusions|recommandations',
                r'النتائج|الخلاصات|التوصيات'
            ],
            'memo': [
                r'note|mémo|correspondance',
                r'مذكرة|مراسلة',
                r'objet|référence',
                r'الموضوع|المرجع'
            ],
            'procedure': [
                r'procédure|manuel|guide|instruction',
                r'إجراء|دليل|تعليمة',
                r'étape|processus|méthode',
                r'خطوة|عملية|طريقة'
            ],
            'contract': [
                r'contrat|accord|convention|marché',
                r'عقد|اتفاق|اتفاقية|صفقة',
                r'partie contractante|signataire',
                r'الطرف المتعاقد|الموقع'
            ],
            'budget': [
                r'budget|financement|coût|prix',
                r'ميزانية|تمويل|كلفة|سعر',
                r'dépense|recette|allocation',
                r'نفقة|إيراد|مخصص'
            ]
        }
        
        self.document_categories = {
            'legal': [
                r'loi|juridique|légal|réglementation',
                r'قانون|قانوني|تشريع',
                r'tribunal|cour|justice',
                r'محكمة|عدالة'
            ],
            'technical': [
                r'technique|ingénierie|spécification',
                r'تقني|هندسة|مواصفات',
                r'norme|standard|qualité',
                r'معيار|جودة'
            ],
            'administrative': [
                r'administratif|gestion|organisation',
                r'إداري|تسيير|تنظيم',
                r'personnel|ressources humaines',
                r'موظفين|موارد بشرية'
            ],
            'financial': [
                r'financier|comptable|fiscal',
                r'مالي|محاسبي|جبائي',
                r'trésorerie|comptabilité',
                r'خزينة|محاسبة'
            ]
        }
        
        self.ministry_departments = {
            'transport_routier': [
                r'transport routier|route|autoroute',
                r'النقل البري|طريق|طريق سيار'
            ],
            'transport_ferroviaire': [
                r'transport ferroviaire|train|chemin de fer',
                r'النقل بالسكك الحديدية|قطار'
            ],
            'transport_aerien': [
                r'transport aérien|aviation|aéroport',
                r'النقل الجوي|طيران|مطار'
            ],
            'transport_maritime': [
                r'transport maritime|port|navigation',
                r'النقل البحري|ميناء|ملاحة'
            ],
            'logistique': [
                r'logistique|chaîne d\'approvisionnement',
                r'اللوجستيك|سلسلة التموين'
            ]
        }


class MetadataExtractor:
    """Comprehensive metadata extractor for transport ministry documents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.classifier = DocumentClassifier()
        
        # Date extraction patterns
        self.date_patterns = [
            # French date patterns
            r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})',
            r'(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})',
            r'le\s+(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})',
            
            # Arabic date patterns
            r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})',
            r'(\d{1,2})\s+(يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+(\d{4})',
            r'في\s+(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})'
        ]
        
        # Reference patterns
        self.reference_patterns = [
            r'n°\s*(\d+[/\-]\d+)',  # French reference numbers
            r'réf[éerence]*\s*:?\s*([^\n]+)',
            r'رقم\s*(\d+[/\-]\d+)',  # Arabic reference numbers
            r'مرجع\s*:?\s*([^\n]+)'
        ]
        
        # Version patterns
        self.version_patterns = [
            r'version\s+(\d+\.?\d*)',
            r'v\.?\s*(\d+\.?\d*)',
            r'révision\s+(\d+)',
            r'النسخة\s+(\d+\.?\d*)'
        ]
    
    def extract_from_text(self, text: str, file_info: Dict[str, Any]) -> DocumentMetadata:
        """
        Extract metadata from document text content.
        
        Args:
            text: Document text content
            file_info: Basic file information
            
        Returns:
            DocumentMetadata object
        """
        start_time = datetime.now()
        
        # Initialize metadata with file information
        metadata = DocumentMetadata(
            file_path=file_info.get('file_path', ''),
            file_name=file_info.get('file_name', ''),
            file_size=file_info.get('file_size', 0),
            file_extension=file_info.get('file_extension', ''),
            file_hash=file_info.get('file_hash', ''),
            creation_date=file_info.get('creation_date'),
            modification_date=file_info.get('modification_date'),
            extraction_timestamp=start_time,
            extraction_method='text_analysis'
        )
        
        if not text:
            metadata.processing_time = (datetime.now() - start_time).total_seconds()
            return metadata
        
        # Extract basic content metrics
        metadata.word_count = len(text.split())
        metadata.character_count = len(text)
        
        # Extract document title
        metadata.title = self._extract_title(text)
        
        # Extract dates
        dates_info = self._extract_dates(text)
        if 'approval_date' in dates_info:
            metadata.approval_date = dates_info['approval_date']
        if 'effective_date' in dates_info:
            metadata.effective_date = dates_info['effective_date']
        
        # Extract document classification
        metadata.document_type = self._classify_document_type(text)
        metadata.document_category = self._classify_document_category(text)
        metadata.ministry_department = self._identify_ministry_department(text)
        
        # Extract structural elements
        structure_info = self._analyze_document_structure(text)
        metadata.has_table_of_contents = structure_info['has_toc']
        metadata.section_count = structure_info['section_count']
        metadata.table_count = structure_info['table_count']
        metadata.figure_count = structure_info['figure_count']
        metadata.reference_count = structure_info['reference_count']
        
        # Extract language information
        language_info = self._analyze_language_content(text)
        metadata.primary_language = language_info['primary_language']
        metadata.secondary_languages = language_info['secondary_languages']
        
        # Extract document version and status
        metadata.document_version = self._extract_version(text)
        metadata.document_status = self._extract_status(text)
        
        # Extract keywords
        metadata.keywords = self._extract_keywords(text)
        
        # Calculate quality scores
        metadata.text_quality_score = self._calculate_quality_score(text)
        metadata.readability_score = self._calculate_readability_score(text)
        
        # Calculate confidence score
        metadata.confidence_score = self._calculate_confidence_score(metadata, text)
        
        # Processing time
        metadata.processing_time = (datetime.now() - start_time).total_seconds()
        
        return metadata
    
    def _extract_title(self, text: str) -> str:
        """Extract document title from text."""
        lines = text.split('\n')
        
        # Look for title in first few lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            # Skip very short lines
            if len(line) < 10:
                continue
            
            # Check if line looks like a title
            if self._is_likely_title(line):
                return line
        
        # Fallback: use first substantial line
        for line in lines[:5]:
            line = line.strip()
            if 20 <= len(line) <= 200:  # Reasonable title length
                return line
        
        return ""
    
    def _is_likely_title(self, line: str) -> bool:
        """Determine if a line is likely to be a title."""
        line = line.strip()
        
        # Check for title indicators
        title_indicators = [
            line.isupper(),  # All caps
            len(line.split()) < 15,  # Not too long
            not line.endswith('.'),  # Doesn't end with period
            not re.match(r'^\d+\.', line),  # Doesn't start with number
            not line.startswith('Page '),  # Not a page indicator
            not line.startswith('صفحة ')  # Arabic page indicator
        ]
        
        return sum(title_indicators) >= 3
    
    def _extract_dates(self, text: str) -> Dict[str, datetime]:
        """Extract various dates from document text."""
        dates = {}
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    date_str = match.group()
                    
                    # Try to parse the date
                    parsed_date = self._parse_date_string(date_str)
                    
                    if parsed_date:
                        # Determine date type based on context
                        context = text[max(0, match.start()-50):match.end()+50].lower()
                        
                        if any(word in context for word in ['approuvé', 'adopté', 'موافق عليه']):
                            dates['approval_date'] = parsed_date
                        elif any(word in context for word in ['effectif', 'entrée en vigueur', 'ساري المفعول']):
                            dates['effective_date'] = parsed_date
                        elif 'creation_date' not in dates:
                            dates['creation_date'] = parsed_date
                
                except Exception as e:
                    logger.warning(f"Failed to parse date: {date_str}, error: {e}")
                    continue
        
        return dates
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse a date string into datetime object."""
        try:
            # French month names
            french_months = {
                'janvier': 'january', 'février': 'february', 'mars': 'march',
                'avril': 'april', 'mai': 'may', 'juin': 'june',
                'juillet': 'july', 'août': 'august', 'septembre': 'september',
                'octobre': 'october', 'novembre': 'november', 'décembre': 'december'
            }
            
            # Arabic month names
            arabic_months = {
                'يناير': 'january', 'فبراير': 'february', 'مارس': 'march',
                'أبريل': 'april', 'مايو': 'may', 'يونيو': 'june',
                'يوليو': 'july', 'أغسطس': 'august', 'سبتمبر': 'september',
                'أكتوبر': 'october', 'نوفمبر': 'november', 'ديسمبر': 'december'
            }
            
            # Replace month names
            normalized_date = date_str.lower()
            for french, english in french_months.items():
                normalized_date = normalized_date.replace(french, english)
            for arabic, english in arabic_months.items():
                normalized_date = normalized_date.replace(arabic, english)
            
            # Try to parse
            return date_parser.parse(normalized_date, fuzzy=True)
            
        except Exception:
            return None
    
    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on content patterns."""
        text_lower = text.lower()
        type_scores = {}
        
        for doc_type, patterns in self.classifier.document_types.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            type_scores[doc_type] = score
        
        # Return type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)
        return 'unknown'
    
    def _classify_document_category(self, text: str) -> str:
        """Classify document category based on content patterns."""
        text_lower = text.lower()
        category_scores = {}
        
        for category, patterns in self.classifier.document_categories.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'unknown'
    
    def _identify_ministry_department(self, text: str) -> str:
        """Identify which ministry department the document belongs to."""
        text_lower = text.lower()
        dept_scores = {}
        
        for department, patterns in self.classifier.ministry_departments.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            dept_scores[department] = score
        
        if dept_scores:
            return max(dept_scores, key=dept_scores.get)
        return 'unknown'
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and count elements."""
        structure = {
            'has_toc': False,
            'section_count': 0,
            'table_count': 0,
            'figure_count': 0,
            'reference_count': 0
        }
        
        # Check for table of contents
        toc_patterns = [
            r'table\s+des?\s+matières?',
            r'sommaire',
            r'index',
            r'فهرس\s+المحتويات?',
            r'المحتويات'
        ]
        
        for pattern in toc_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure['has_toc'] = True
                break
        
        # Count sections
        section_patterns = [
            r'chapitre\s+\d+',
            r'section\s+\d+',
            r'article\s+\d+',
            r'الفصل\s+\d+',
            r'القسم\s+\d+',
            r'المادة\s+\d+'
        ]
        
        for pattern in section_patterns:
            structure['section_count'] += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Count tables
        table_patterns = [
            r'tableau\s+\d+',
            r'\[table\]',
            r'جدول\s+\d+',
            r'الجدول\s+\d+'
        ]
        
        for pattern in table_patterns:
            structure['table_count'] += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Count figures
        figure_patterns = [
            r'figure\s+\d+',
            r'fig\.\s+\d+',
            r'شكل\s+\d+',
            r'الشكل\s+\d+'
        ]
        
        for pattern in figure_patterns:
            structure['figure_count'] += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Count references
        reference_patterns = [
            r'référence\s*:',
            r'voir\s+aussi',
            r'cf\.',
            r'مرجع\s*:',
            r'انظر\s+أيضا'
        ]
        
        for pattern in reference_patterns:
            structure['reference_count'] += len(re.findall(pattern, text, re.IGNORECASE))
        
        return structure
    
    def _analyze_language_content(self, text: str) -> Dict[str, Any]:
        """Analyze language content and distribution."""
        # Count different character types
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        french_chars = len(re.findall(r'[àâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(re.findall(r'\S', text))
        
        if total_chars == 0:
            return {
                'primary_language': 'unknown',
                'secondary_languages': [],
                'language_distribution': {}
            }
        
        # Calculate language ratios
        arabic_ratio = arabic_chars / total_chars
        french_ratio = (french_chars + latin_chars) / total_chars
        
        # Determine primary language
        if arabic_ratio > 0.3:
            primary_language = 'arabic'
        elif french_chars > 0 or french_ratio > 0.7:
            primary_language = 'french'
        else:
            primary_language = 'unknown'
        
        # Determine secondary languages
        secondary_languages = []
        if primary_language != 'arabic' and arabic_ratio > 0.1:
            secondary_languages.append('arabic')
        if primary_language != 'french' and french_ratio > 0.1:
            secondary_languages.append('french')
        
        return {
            'primary_language': primary_language,
            'secondary_languages': secondary_languages,
            'language_distribution': {
                'arabic_ratio': arabic_ratio,
                'french_ratio': french_ratio,
                'other_ratio': 1 - arabic_ratio - french_ratio
            }
        }
    
    def _extract_version(self, text: str) -> str:
        """Extract document version information."""
        for pattern in self.version_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""
    
    def _extract_status(self, text: str) -> str:
        """Extract document status information."""
        status_patterns = [
            (r'brouillon|draft|مسودة', 'draft'),
            (r'final|définitif|نهائي', 'final'),
            (r'approuvé|approved|موافق عليه', 'approved'),
            (r'en révision|under review|قيد المراجعة', 'under_review'),
            (r'obsolète|deprecated|ملغى', 'obsolete')
        ]
        
        for pattern, status in status_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return status
        
        return 'unknown'
    
    def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract key terms and phrases from the document."""
        # Transport domain keywords
        transport_terms = [
            # French terms
            r'transport\w*', r'route\w*', r'circulation', r'traffic', r'véhicule\w*',
            r'autoroute\w*', r'infrastructure\w*', r'sécurité', r'réglementation',
            r'logistique', r'port\w*', r'aéroport\w*', r'gare\w*',
            
            # Arabic terms
            r'نقل', r'طريق', r'مرور', r'حركة', r'مركبة', r'طريق سيار',
            r'بنية تحتية', r'أمن', r'تنظيم', r'لوجستيك', r'ميناء', r'مطار', r'محطة'
        ]
        
        keywords = []
        text_lower = text.lower()
        
        # Extract transport-specific terms
        for term_pattern in transport_terms:
            matches = re.findall(term_pattern, text_lower)
            keywords.extend(matches)
        
        # Extract frequent important words
        words = re.findall(r'\b[a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ\u0600-\u06FF]{4,}\b', text)
        word_freq = Counter(word.lower() for word in words)
        
        # Add most frequent words
        for word, freq in word_freq.most_common(max_keywords):
            if freq > 2 and word not in keywords:  # Appear at least 3 times
                keywords.append(word)
        
        return keywords[:max_keywords]
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate overall text quality score."""
        if not text:
            return 0.0
        
        factors = []
        
        # Length factor (reasonable document length)
        length_score = min(len(text) / 10000, 1.0)  # Normalize to 10k chars
        factors.append(length_score)
        
        # Structure factor (has sections, paragraphs)
        has_structure = bool(re.search(r'(chapitre|section|article|\n\n)', text, re.IGNORECASE))
        structure_score = 1.0 if has_structure else 0.5
        factors.append(structure_score)
        
        # Sentence completeness
        sentences = re.split(r'[.!?؟۔]', text)
        complete_sentences = len([s for s in sentences if len(s.strip()) > 10])
        sentence_score = min(complete_sentences / 10, 1.0)
        factors.append(sentence_score)
        
        # Language consistency
        words = text.split()
        if words:
            # Check for mixed scripts (might indicate OCR errors)
            mixed_script_ratio = len([w for w in words if re.search(r'[a-zA-Z]', w) and re.search(r'[\u0600-\u06FF]', w)]) / len(words)
            language_score = 1.0 - mixed_script_ratio
            factors.append(language_score)
        
        return np.mean(factors)
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate document readability score."""
        if not text:
            return 0.0
        
        # Basic readability metrics
        sentences = re.split(r'[.!?؟۔]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Simple readability formula (adapted for multilingual)
        # Lower scores for very long sentences and words
        sentence_penalty = max(0, (avg_sentence_length - 20) / 30)
        word_penalty = max(0, (avg_word_length - 6) / 4)
        
        readability = max(0, 1.0 - sentence_penalty - word_penalty)
        
        return readability
    
    def _calculate_confidence_score(self, metadata: DocumentMetadata, text: str) -> float:
        """Calculate confidence score for extracted metadata."""
        confidence_factors = []
        
        # Text length factor
        if metadata.character_count > 1000:
            confidence_factors.append(1.0)
        elif metadata.character_count > 100:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # Classification confidence
        if metadata.document_type != 'unknown':
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Language detection confidence
        if metadata.primary_language != 'unknown':
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)
        
        # Structure confidence
        if metadata.section_count > 0:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Title extraction confidence
        if metadata.title:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        return np.mean(confidence_factors)
    
    def extract_from_document_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from document properties (from extractors)."""
        metadata_dict = {}
        
        # Map common properties
        property_mapping = {
            'title': 'title',
            'author': 'author',
            'subject': 'subject',
            'keywords': 'keywords',
            'creator': 'author',
            'creation_date': 'creation_date',
            'modification_date': 'modification_date',
            'creationDate': 'creation_date',
            'modDate': 'modification_date',
            'language': 'language'
        }
        
        for prop_key, meta_key in property_mapping.items():
            if prop_key in properties and properties[prop_key]:
                value = properties[prop_key]
                
                # Handle date strings
                if 'date' in meta_key and isinstance(value, str):
                    try:
                        value = date_parser.parse(value)
                    except:
                        pass
                
                metadata_dict[meta_key] = value
        
        return metadata_dict
    
    def merge_metadata(self, *metadata_sources: Dict[str, Any]) -> DocumentMetadata:
        """Merge metadata from multiple sources."""
        merged = {}
        
        # Merge all sources
        for source in metadata_sources:
            if source:
                merged.update(source)
        
        # Convert to DocumentMetadata object
        # Handle missing required fields
        required_fields = {
            'file_path': '',
            'file_name': '',
            'file_size': 0,
            'file_extension': '',
            'file_hash': ''
        }
        
        for field, default in required_fields.items():
            if field not in merged:
                merged[field] = default
        
        # Initialize secondary_languages as empty list if None
        if merged.get('secondary_languages') is None:
            merged['secondary_languages'] = []
        
        if merged.get('keywords') is None:
            merged['keywords'] = []
        
        # Create DocumentMetadata object
        try:
            return DocumentMetadata(**merged)
        except Exception as e:
            logger.error(f"Failed to create DocumentMetadata object: {e}")
            # Return minimal metadata
            return DocumentMetadata(**required_fields)
    
    def export_metadata(self, metadata: DocumentMetadata, format: str = 'json') -> str:
        """Export metadata in specified format."""
        metadata_dict = asdict(metadata)
        
        # Convert datetime objects to strings
        for key, value in metadata_dict.items():
            if isinstance(value, datetime):
                metadata_dict[key] = value.isoformat()
        
        if format.lower() == 'json':
            return json.dumps(metadata_dict, indent=2, ensure_ascii=False)
        elif format.lower() == 'xml':
            return self._to_xml(metadata_dict)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _to_xml(self, data: Dict[str, Any], root_name: str = 'document_metadata') -> str:
        """Convert metadata dictionary to XML format."""
        def dict_to_xml(d, parent_name):
            xml_str = f"<{parent_name}>"
            
            for key, value in d.items():
                if isinstance(value, dict):
                    xml_str += dict_to_xml(value, key)
                elif isinstance(value, list):
                    xml_str += f"<{key}>"
                    for item in value:
                        if isinstance(item, dict):
                            xml_str += dict_to_xml(item, 'item')
                        else:
                            xml_str += f"<item>{item}</item>"
                    xml_str += f"</{key}>"
                else:
                    xml_str += f"<{key}>{value}</{key}>"
            
            xml_str += f"</{parent_name}>"
            return xml_str
        
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{dict_to_xml(data, root_name)}'