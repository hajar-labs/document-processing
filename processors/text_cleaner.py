# Advanced text cleaning and normalization
"""
Comprehensive text cleaning pipeline for extracted documents.
Handles noise removal, formatting correction, and text normalization.
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Union, Any, Tuple
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Could not download NLTK data")


class TextCleaner:
    """
    Advanced text cleaner for multilingual government documents.
    Specialized for French and Arabic text processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Cleaning configuration
        self.aggressive_cleaning = self.config.get('aggressive_cleaning', False)
        self.preserve_structure = self.config.get('preserve_structure', True)
        self.fix_encoding = self.config.get('fix_encoding', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        self.remove_redundant = self.config.get('remove_redundant', True)
        
        # Language-specific settings
        self.french_normalization = self.config.get('french_normalization', True)
        self.arabic_normalization = self.config.get('arabic_normalization', True)
        
        # OCR error patterns (common mistakes)
        self.ocr_corrections = {
            # French OCR corrections
            r'\br\s+n\b': 'rn',
            r'\bm\s+e\b': 'me',
            r'\bt\s+e\b': 'te',
            r'\bl\s+e\b': 'le',
            r'\bd\s+e\b': 'de',
            r'\bc\s+e\b': 'ce',
            r'\bq\s+u\s+e\b': 'que',
            r'\bp\s+o\s+u\s+r\b': 'pour',
            
            # Common OCR character mistakes
            r'(?<!\w)rn(?!\w)': 'm',
            r'(?<!\w)cl(?=\s|$)': 'd',
            r'(?<!\w)ri(?=\s|$)': 'n',
            r'(?<!\w)vv(?!\w)': 'w',
            
            # Arabic OCR corrections
            r'(?<!\w)ه\s+ا(?!\w)': 'ها',
            r'(?<!\w)ا\s+ل(?!\w)': 'ال',
            r'(?<!\w)م\s+ن(?!\w)': 'من',
            r'(?<!\w)ف\s+ي(?!\w)': 'في'
        }
        
        # Headers and footers patterns for government documents
        self.header_footer_patterns = [
            r'^Page\s+\d+(\s+(sur|of|من)\s+\d+)?$',
            r'^MINISTÈRE.*$',
            r'^وزارة.*$',
            r'^ROYAUME\s+DU\s+MAROC.*$',
            r'^المملكة\s+المغربية.*$',
            r'^\d{2}/\d{2}/\d{4}.*$',
            r'^.*@.*\.ma$',
            r'^Tél\.?\s*:.*$',
            r'^هاتف\s*:.*$',
            r'^B\.P\.?\s*\d+.*$',
            r'^ص\.ب\.?\s*\d+.*$',
            r'^.*CONFIDENTIEL.*$',
            r'^.*سري.*$'
        ]
        
        # Noise patterns to remove
        self.noise_patterns = [
            r'^[^\w\s]*$',  # Lines with only punctuation
            r'^\s*[.]{3,}\s*$',  # Lines with only dots
            r'^\s*[-]{3,}\s*$',  # Lines with only dashes
            r'^\s*[=]{3,}\s*$',  # Lines with only equals
            r'^\s*[_]{3,}\s*$',  # Lines with only underscores
            r'^\s*\|\s*\|\s*\|\s*$',  # Empty table separators
            r'^\s*[\+\-\|]{3,}\s*$'  # Table borders
        ]
        
        # Word boundary patterns for different languages
        self.word_patterns = {
            'french': r'\b[a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]+\b',
            'arabic': r'\b[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+\b',
            'mixed': r'\b[\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+\b'
        }
    
    def clean_text(self, text: str, language: str = 'mixed', 
                   cleaning_level: str = 'standard') -> str:
        """
        Main text cleaning function with configurable intensity.
        
        Args:
            text: Input text to clean
            language: Primary language ('french', 'arabic', 'mixed')
            cleaning_level: 'light', 'standard', 'aggressive'
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        cleaned = text
        
        # Step 1: Fix encoding issues
        if self.fix_encoding:
            cleaned = self._fix_encoding_issues(cleaned)
        
        # Step 2: Unicode normalization
        if self.normalize_unicode:
            cleaned = self._normalize_unicode(cleaned, language)
        
        # Step 3: Remove headers and footers
        cleaned = self._remove_headers_footers(cleaned)
        
        # Step 4: Fix OCR errors
        cleaned = self._fix_ocr_errors(cleaned, language)
        
        # Step 5: Clean based on intensity level
        if cleaning_level == 'light':
            cleaned = self._light_cleaning(cleaned)
        elif cleaning_level == 'aggressive':
            cleaned = self._aggressive_cleaning(cleaned, language)
        else:  # standard
            cleaned = self._standard_cleaning(cleaned, language)
        
        # Step 6: Language-specific cleaning
        if language == 'french' or (language == 'mixed' and self._detect_french(cleaned)):
            cleaned = self._clean_french_text(cleaned)
        if language == 'arabic' or (language == 'mixed' and self._detect_arabic(cleaned)):
            cleaned = self._clean_arabic_text(cleaned)
        
        # Step 7: Final normalization
        cleaned = self._final_normalization(cleaned)
        
        return cleaned.strip()
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues."""
        # Common encoding fixes
        fixes = {
            'Ã ': 'à',
            'Ã¡': 'á',
            'Ã¢': 'â',
            'Ã¨': 'è',
            'Ã©': 'é',
            'Ãª': 'ê',
            'Ã¬': 'ì',
            'Ã­': 'í',
            'Ã®': 'î',
            'Ã²': 'ò',
            'Ã³': 'ó',
            'Ã´': 'ô',
            'Ã¹': 'ù',
            'Ãº': 'ú',
            'Ã»': 'û',
            'Ã§': 'ç',
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '–',
            'â€"': '—'
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        # Try to decode common encoding issues
        try:
            # Handle UTF-8 encoded as Latin-1
            if 'Ã' in text:
                encoded = text.encode('latin-1')
                text = encoded.decode('utf-8')
        except:
            pass  # Keep original if decoding fails
        
        return text
    
    def _normalize_unicode(self, text: str, language: str) -> str:
        """Normalize Unicode characters."""
        # Basic Unicode normalization
        normalized = unicodedata.normalize('NFKC', text)
        
        if language == 'arabic':
            # Arabic-specific normalizations
            # Normalize Alef variants
            normalized = re.sub(r'[إأآا]', 'ا', normalized)
            
            # Normalize Yeh variants
            normalized = re.sub(r'[يىئ]', 'ي', normalized)
            
            # Normalize Teh Marbuta
            normalized = re.sub(r'[ةه]', 'ة', normalized)
            
            # Remove Arabic tatweel (kashida)
            normalized = re.sub(r'ـ+', '', normalized)
            
            # Normalize Arabic punctuation
            normalized = re.sub(r'،', ',', normalized)
            normalized = re.sub(r'؛', ';', normalized)
            normalized = re.sub(r'؟', '?', normalized)
        
        return normalized
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove headers and footers from government documents."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                cleaned_lines.append('')
                continue
            
            # Check against header/footer patterns
            is_header_footer = any(re.match(pattern, line, re.IGNORECASE) 
                                 for pattern in self.header_footer_patterns)
            
            if not is_header_footer:
                cleaned_lines.append(line)
            else:
                logger.debug(f"Removed header/footer: {line[:50]}")
        
        return '\n'.join(cleaned_lines)
    
    def _fix_ocr_errors(self, text: str, language: str) -> str:
        """Fix common OCR errors."""
        fixed = text
        
        # Apply OCR corrections
        for error_pattern, correction in self.ocr_corrections.items():
            fixed = re.sub(error_pattern, correction, fixed, flags=re.IGNORECASE)
        
        # Fix broken words (single letters followed by space)
        if language in ['french', 'mixed']:
            # Fix French broken words
            fixed = re.sub(r'\b([a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ])\s+([a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ])\b', 
                          r'\1\2', fixed)
        
        if language in ['arabic', 'mixed']:
            # Fix Arabic broken words
            fixed = re.sub(r'\b([\u0600-\u06FF])\s+([\u0600-\u06FF])\b', r'\1\2', fixed)
        
        return fixed
    
    def _light_cleaning(self, text: str) -> str:
        """Light cleaning - minimal processing."""
        # Just remove excessive whitespace and basic noise
        cleaned = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple newlines to double
        
        return cleaned
    
    def _standard_cleaning(self, text: str, language: str) -> str:
        """Standard cleaning - balanced approach."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                cleaned_lines.append('')
                continue
            
            # Remove noise patterns
            is_noise = any(re.match(pattern, line) for pattern in self.noise_patterns)
            if is_noise:
                continue
            
            # Clean the line
            line = self._clean_line_standard(line, language)
            
            if line:  # Only add non-empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _aggressive_cleaning(self, text: str, language: str) -> str:
        """Aggressive cleaning - maximum noise removal."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove noise patterns
            is_noise = any(re.match(pattern, line) for pattern in self.noise_patterns)
            if is_noise:
                continue
            
            # Skip very short lines (likely artifacts)
            if len(line) < 5:
                continue
            
            # Skip lines with too many special characters
            special_char_ratio = sum(1 for c in line if c in string.punctuation) / len(line)
            if special_char_ratio > 0.5:
                continue
            
            # Clean the line aggressively
            line = self._clean_line_aggressive(line, language)
            
            # Skip if cleaning removed too much content
            if len(line) >= 3:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_line_standard(self, line: str, language: str) -> str:
        """Standard line cleaning."""
        # Fix punctuation spacing
        line = re.sub(r'\s+([.!?;:,])', r'\1', line)
        line = re.sub(r'([.!?])\s*([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ\u0600-\u06FF])', r'\1 \2', line)
        
        # Remove excessive punctuation
        line = re.sub(r'[.]{3,}', '...', line)
        line = re.sub(r'[-]{3,}', '---', line)
        
        # Fix quotation marks
        line = re.sub(r'[""„]', '"', line)
        line = re.sub(r'[''`´]', "'", line)
        
        # Remove standalone special characters (OCR artifacts)
        line = re.sub(r'\s[^\w\s]{1}\s', ' ', line)
        
        # Normalize whitespace
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()
    
    def _clean_line_aggressive(self, line: str, language: str) -> str:
        """Aggressive line cleaning."""
        # Start with standard cleaning
        line = self._clean_line_standard(line, language)
        
        # Remove isolated single characters (except common single-letter words)
        if language == 'french':
            # Keep French single-letter words: à, y, etc.
            line = re.sub(r'\b(?![àyAÀY])[a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]\b', '', line)
        else:
            line = re.sub(r'\b[a-zA-Z]\b', '', line)
        
        # Remove words with mixed scripts (likely OCR errors)
        if language != 'mixed':
            line = re.sub(r'\b\w*[a-zA-Z][\u0600-\u06FF]\w*\b', '', line)
            line = re.sub(r'\b\w*[\u0600-\u06FF][a-zA-Z]\w*\b', '', line)
        
        # Remove words with too many consecutive identical characters
        line = re.sub(r'\b\w*(.)\1{3,}\w*\b', '', line)
        
        # Clean up multiple spaces created by removals
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()
    
    def _clean_french_text(self, text: str) -> str:
        """French-specific text cleaning."""
        if not self.french_normalization:
            return text
        
        # Fix French apostrophes and contractions
        text = re.sub(r"(\w)\s*['`´']\s*(\w)", r"\1'\2", text)
        
        # Fix French quotation marks
        text = re.sub(r'<<\s*', '« ', text)
        text = re.sub(r'\s*>>', ' »', text)
        
        # Fix French punctuation spacing (French typography rules)
        text = re.sub(r'\s*([;:!?])', r' \1', text)
        text = re.sub(r'([;:!?])\s*', r'\1 ', text)
        
        # Fix French abbreviations
        text = re.sub(r'\b([A-Z])\.\s*([A-Z])\.\s*([A-Z])', r'\1.\2.\3', text)
        
        # Fix common French word breaks
        french_fixes = {
            r"\bd'\s*": "d'",
            r"\bl'\s*": "l'",
            r"\bqu'\s*": "qu'",
            r"\bn'\s*": "n'",
            r"\bs'\s*": "s'",
            r"\bc'\s*": "c'",
            r"\bj'\s*": "j'",
            r"\bt'\s*": "t'",
            r"\bm'\s*": "m'"
        }
        
        for pattern, replacement in french_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _clean_arabic_text(self, text: str) -> str:
        """Arabic-specific text cleaning."""
        if not self.arabic_normalization:
            return text
        
        # Fix Arabic punctuation spacing
        text = re.sub(r'\s*([،؛؟])', r'\1', text)
        text = re.sub(r'([،؛؟])\s*', r'\1 ', text)
        
        # Fix parentheses direction for Arabic (common OCR issue)
        # In Arabic text, parentheses are often reversed
        arabic_ratio = len(re.findall(r'[\u0600-\u06FF]', text)) / max(len(text), 1)
        if arabic_ratio > 0.3:  # Mostly Arabic text
            text = re.sub(r'\)', '(temp)', text)
            text = re.sub(r'\(', ')', text)
            text = re.sub(r'\(temp\)', '(', text)
        
        # Fix Arabic word boundaries
        text = re.sub(r'(\w)\s+(ال)', r'\1\2', text)  # Fix broken "ال" (the)
        text = re.sub(r'(و)\s+(\w)', r'\1\2', text)   # Fix broken "و" (and)
        
        # Remove excessive Arabic diacritics if requested
        if self.config.get('remove_arabic_diacritics', False):
            # Remove most diacritics but keep some important ones
            text = re.sub(r'[\u064B-\u0650\u0652]', '', text)  # Remove some diacritics
        
        return text
    
    def _final_normalization(self, text: str) -> str:
        """Final text normalization."""
        # Normalize line breaks
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        
        # Remove trailing spaces from lines
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        
        # Remove excessive empty lines
        cleaned_lines = []
        empty_count = 0
        
        for line in lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= 2:  # Allow max 2 consecutive empty lines
                    cleaned_lines.append(line)
            else:
                empty_count = 0
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final whitespace normalization
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        return text
    
    def _detect_french(self, text: str) -> bool:
        """Detect if text contains French content."""
        french_chars = len(re.findall(r'[àâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'\S', text))
        
        if total_chars == 0:
            return False
        
        return (french_chars + latin_chars) / total_chars > 0.3
    
    def _detect_arabic(self, text: str) -> bool:
        """Detect if text contains Arabic content."""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        total_chars = len(re.findall(r'\S', text))
        
        if total_chars == 0:
            return False
        
        return arabic_chars / total_chars > 0.1
    
    def remove_redundant_content(self, text: str, similarity_threshold: float = 0.8) -> str:
        """
        Remove redundant or repeated content from text.
        
        Args:
            text: Input text
            similarity_threshold: Threshold for considering content redundant
            
        Returns:
            Text with redundant content removed
        """
        if not self.remove_redundant:
            return text
        
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            is_redundant = False
            
            # Check against existing sentences
            for existing in unique_sentences:
                similarity = self._calculate_sentence_similarity(sentence, existing)
                if similarity > similarity_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                unique_sentences.append(sentence)
        
        return ' '.join(unique_sentences)
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences."""
        # Simple word-based similarity
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def clean_for_summarization(self, text: str, language: str = 'mixed') -> str:
        """
        Clean text specifically for summarization tasks.
        
        Args:
            text: Input text
            language: Primary language
            
        Returns:
            Cleaned text optimized for summarization
        """
        # Use standard cleaning as base
        cleaned = self.clean_text(text, language, 'standard')
        
        # Additional summarization-specific cleaning
        cleaned = self.remove_redundant_content(cleaned, 0.7)
        
        # Remove very short sentences that don't add value
        sentences = sent_tokenize(cleaned)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Keep sentences with at least 5 words
            if len(sentence.split()) >= 5:
                meaningful_sentences.append(sentence)
        
        return ' '.join(meaningful_sentences)
    
    def extract_clean_sentences(self, text: str, language: str = 'mixed',
                               min_length: int = 10, max_length: int = 500) -> List[str]:
        """
        Extract clean, well-formed sentences from text.
        
        Args:
            text: Input text
            language: Primary language
            min_length: Minimum sentence length
            max_length: Maximum sentence length
            
        Returns:
            List of clean sentences
        """
        cleaned_text = self.clean_text(text, language, 'standard')
        sentences = sent_tokenize(cleaned_text)
        
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter by length
            if min_length <= len(sentence) <= max_length:
                # Additional quality checks
                if self._is_well_formed_sentence(sentence, language):
                    clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _is_well_formed_sentence(self, sentence: str, language: str) -> bool:
        """Check if a sentence is well-formed."""
        # Must have at least 3 words
        words = sentence.split()
        if len(words) < 3:
            return False
        
        # Must end with appropriate punctuation
        if not re.search(r'[.!?؟]', sentence.strip()):
            return False
        
        # Must have reasonable character diversity
        unique_chars = len(set(sentence.lower()))
        if unique_chars < len(sentence) * 0.3:
            return False
        
        # Language-specific checks
        if language == 'arabic':
            # Must contain Arabic characters
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', sentence))
            if arabic_chars < len(sentence) * 0.1:
                return False
        elif language == 'french':
            # Must contain Latin characters
            latin_chars = len(re.findall(r'[a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', sentence))
            if latin_chars < len(sentence) * 0.5:
                return False
        
        return True
    
    def assess_text_quality(self, text: str, language: str = 'mixed') -> Dict[str, Any]:
        """
        Assess the quality of text for downstream processing.
        
        Args:
            text: Input text
            language: Primary language
            
        Returns:
            Quality assessment metrics
        """
        if not text or not text.strip():
            return {'overall_quality': 0.0, 'issues': ['empty_text']}
        
        metrics = {}
        issues = []
        
        # Basic statistics
        total_chars = len(text)
        total_words = len(text.split())
        unique_chars = len(set(text.lower()))
        
        metrics['character_count'] = total_chars
        metrics['word_count'] = total_words
        metrics['unique_character_ratio'] = unique_chars / total_chars if total_chars > 0 else 0
        
        # Sentence analysis
        sentences = sent_tokenize(text)
        well_formed_sentences = len([s for s in sentences if self._is_well_formed_sentence(s, language)])
        metrics['sentence_count'] = len(sentences)
        metrics['well_formed_ratio'] = well_formed_sentences / len(sentences) if sentences else 0
        
        # Language consistency
        if language != 'mixed':
            consistency = self._calculate_language_consistency(text, language)
            metrics['language_consistency'] = consistency
            if consistency < 0.7:
                issues.append('inconsistent_language')
        
        # Noise level assessment
        noise_indicators = [
            len(re.findall(r'[^\w\s\u0600-\u06FF.!?،؛؟\'"()-]', text)),  # Special characters
            len(re.findall(r'\b\w{1}\b', text)),  # Single character "words"
            len(re.findall(r'\b\w*(.)\1{3,}\w*\b', text))  # Repeated characters
        ]
        
        total_noise = sum(noise_indicators)
        noise_ratio = total_noise / total_words if total_words > 0 else 1
        metrics['noise_ratio'] = noise_ratio
        
        if noise_ratio > 0.1:
            issues.append('high_noise_level')
        
        # Overall quality score
        quality_factors = []
        
        # Length factor (moderate length is good)
        if 100 <= total_chars <= 10000:
            quality_factors.append(1.0)
        elif total_chars < 100:
            quality_factors.append(0.3)
        else:
            quality_factors.append(0.8)
        
        # Character diversity
        quality_factors.append(min(metrics['unique_character_ratio'] * 2, 1.0))
        
        # Sentence quality
        quality_factors.append(metrics['well_formed_ratio'])
        
        # Language consistency (if applicable)
        if 'language_consistency' in metrics:
            quality_factors.append(metrics['language_consistency'])
        
        # Noise penalty
        quality_factors.append(max(0.0, 1.0 - noise_ratio * 5))
        
        overall_quality = sum(quality_factors) / len(quality_factors)
        metrics['overall_quality'] = round(overall_quality, 3)
        metrics['issues'] = issues
        
        # Quality classification
        if overall_quality >= 0.8:
            metrics['quality_level'] = 'high'
        elif overall_quality >= 0.6:
            metrics['quality_level'] = 'medium'
        elif overall_quality >= 0.4:
            metrics['quality_level'] = 'low'
        else:
            metrics['quality_level'] = 'very_low'
        
        return metrics
    
    def _calculate_language_consistency(self, text: str, expected_language: str) -> float:
        """Calculate language consistency score."""
        if expected_language == 'french':
            # Count French/Latin characters
            french_chars = len(re.findall(r'[àâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
            latin_chars = len(re.findall(r'[a-zA-Z]', text))
            total_letters = len(re.findall(r'[a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
            
            if total_letters == 0:
                return 0.0
            
            return (french_chars + latin_chars) / total_letters
        
        elif expected_language == 'arabic':
            # Count Arabic characters
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
            total_chars = len(re.findall(r'\S', text))
            
            if total_chars == 0:
                return 0.0
            
            return arabic_chars / total_chars
        
        return 0.5  # Default for unknown languages