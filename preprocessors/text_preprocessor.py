# Text preprocessing and normalization
"""
Advanced text preprocessing pipeline for multilingual documents.
Handles cleaning, normalization, segmentation, and quality assessment.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import unicodedata
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import numpy as np
from collections import Counter, defaultdict
import spacy
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Failed to download NLTK data")


@dataclass
class TextQualityMetrics:
    """Text quality assessment metrics."""
    readability_score: float
    coherence_score: float
    completeness_score: float
    language_consistency: float
    overall_quality: float
    issues: List[str]


@dataclass
class ProcessedText:
    """Container for processed text with metadata."""
    original_text: str
    cleaned_text: str
    normalized_text: str
    sentences: List[str]
    paragraphs: List[str]
    tokens: List[str]
    language: str
    language_confidence: float
    quality_metrics: TextQualityMetrics
    preprocessing_steps: List[str]
    statistics: Dict[str, Any]


class MultilingualTextProcessor:
    """Advanced text processor with multilingual support for French and Arabic."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Language-specific configurations
        self.supported_languages = ['french', 'arabic', 'english']
        self.primary_languages = self.config.get('primary_languages', ['fr', 'ar'])
        
        # Initialize language resources
        self._initialize_language_resources()
        
        # Text processing parameters
        self.min_sentence_length = self.config.get('min_sentence_length', 10)
        self.max_sentence_length = self.config.get('max_sentence_length', 1000)
        self.remove_short_paragraphs = self.config.get('remove_short_paragraphs', True)
        self.min_paragraph_length = self.config.get('min_paragraph_length', 50)
        
        # Cleaning patterns
        self._setup_cleaning_patterns()
    
    def _initialize_language_resources(self):
        """Initialize language-specific processing resources."""
        self.stemmers = {}
        self.stop_words = {}
        self.spacy_models = {}
        
        try:
            # French resources
            self.stemmers['french'] = SnowballStemmer('french')
            try:
                self.stop_words['french'] = set(stopwords.words('french'))
            except:
                self.stop_words['french'] = set()
            
            # Arabic resources
            try:
                self.stemmers['arabic'] = SnowballStemmer('arabic')
            except:
                logger.warning("Arabic stemmer not available")
            
            try:
                self.stop_words['arabic'] = set(stopwords.words('arabic'))
            except:
                self.stop_words['arabic'] = set()
                
            # English resources (fallback)
            self.stemmers['english'] = SnowballStemmer('english')
            try:
                self.stop_words['english'] = set(stopwords.words('english'))
            except:
                self.stop_words['english'] = set()
            
            # SpaCy models for advanced processing
            try:
                import spacy
                for lang_code in ['fr', 'en']:  # Arabic model might not be available
                    try:
                        model_name = f"{lang_code}_core_news_sm"
                        self.spacy_models[lang_code] = spacy.load(model_name)
                    except OSError:
                        logger.warning(f"SpaCy model {model_name} not available")
            except ImportError:
                logger.warning("SpaCy not available for advanced text processing")
                
        except Exception as e:
            logger.error(f"Error initializing language resources: {e}")
    
    def _setup_cleaning_patterns(self):
        """Setup regex patterns for text cleaning."""
        # Common cleaning patterns
        self.patterns = {
            # Remove excessive whitespace
            'excess_whitespace': re.compile(r'\s+'),
            
            # Remove page numbers and references
            'page_numbers': re.compile(r'\b[Pp]age\s+\d+\b|\b\d+\s*/\s*\d+\b'),
            'arabic_page_numbers': re.compile(r'\bصفحة\s+\d+\b'),
            
            # Remove headers/footers patterns
            'headers_footers': re.compile(r'^[-=_]{3,}.*[-=_]{3,}$|^\d{1,3}\s*$', re.MULTILINE),
            
            # Remove table markers
            'table_markers': re.compile(r'\[TABLE\]|\[/TABLE\]|\[HEADER\]|\[/HEADER\]|\[FOOTER\]|\[/FOOTER\]'),
            
            # Remove excessive punctuation
            'excess_punctuation': re.compile(r'[.!?]{3,}'),
            
            # Remove isolated numbers and single characters
            'isolated_chars': re.compile(r'\b[a-zA-Z0-9]\b'),
            
            # Fix punctuation spacing
            'punctuation_spacing': re.compile(r'\s+([,.!?;:])'),
            
            # Remove URLs and email addresses
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Remove special characters but preserve Arabic and French
            'special_chars': re.compile(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0590-\u05FF\u00C0-\u017F\u1E00-\u1EFF,.!?;:()\[\]{}"\'`''""«»‹›-]'),
            
            # French-specific patterns
            'french_artifacts': re.compile(r'\b(?:cf|vs|etc|fig|tableau|annexe)\b\.?', re.IGNORECASE),
            
            # Arabic-specific patterns
            'arabic_artifacts': re.compile(r'\b(?:انظر|مقابل|الخ|شكل|جدول|ملحق)\b'),
            
            # Remove repeated lines
            'repeated_lines': re.compile(r'^(.+)$(?:\n\1)+', re.MULTILINE),
        }
        
        # Normalization patterns
        self.normalization_patterns = {
            # Normalize French accented characters for processing
            'french_quotes': re.compile(r'[''`]'),
            'french_dashes': re.compile(r'[‒–—―]'),
            
            # Normalize Arabic characters
            'arabic_alef': re.compile(r'[إأآا]'),
            'arabic_yeh': re.compile(r'[يى]'),
            'arabic_teh': re.compile(r'[ةه]'),
            
            # Normalize numbers
            'arabic_numbers': re.compile(r'[٠-٩]'),
            'roman_numbers': re.compile(r'\b[IVXLCDM]+\b'),
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect text language with confidence score.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or len(text.strip()) < 10:
            return 'unknown', 0.0
        
        try:
            # Use langdetect for primary detection
            detected_langs = detect_langs(text)
            
            if detected_langs:
                primary_lang = detected_langs[0]
                
                # Map to our supported languages
                lang_mapping = {
                    'fr': 'french',
                    'ar': 'arabic',
                    'en': 'english'
                }
                
                detected_code = primary_lang.lang
                confidence = primary_lang.prob
                
                if detected_code in lang_mapping:
                    return lang_mapping[detected_code], confidence
                else:
                    # Fallback to character-based detection
                    return self._detect_by_characters(text)
            
        except (LangDetectError, Exception) as e:
            logger.warning(f"Language detection failed: {e}")
        
        # Fallback to character-based detection
        return self._detect_by_characters(text)
    
    def _detect_by_characters(self, text: str) -> Tuple[str, float]:
        """Fallback language detection based on character analysis."""
        if not text:
            return 'unknown', 0.0
        
        # Count character types
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        french_chars = len(re.findall(r'[àâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(re.findall(r'\S', text))
        
        if total_chars == 0:
            return 'unknown', 0.0
        
        arabic_ratio = arabic_chars / total_chars
        french_ratio = french_chars / total_chars
        latin_ratio = latin_chars / total_chars
        
        # Determine language based on character ratios
        if arabic_ratio > 0.1:
            confidence = min(arabic_ratio * 2, 1.0)  # Scale confidence
            return 'arabic', confidence
        elif french_ratio > 0.02 or (latin_ratio > 0.7 and french_ratio > 0):
            confidence = min((french_ratio + latin_ratio), 1.0)
            return 'french', confidence
        elif latin_ratio > 0.5:
            confidence = min(latin_ratio, 1.0)
            return 'english', confidence
        else:
            return 'unknown', 0.3
    
    def clean_text(self, text: str, language: str = None) -> str:
        """
        Comprehensive text cleaning pipeline.
        
        Args:
            text: Raw text to clean
            language: Detected or known language
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove table and document markers
        cleaned = self.patterns['table_markers'].sub('', cleaned)
        
        # Remove URLs and emails
        cleaned = self.patterns['urls'].sub('', cleaned)
        cleaned = self.patterns['emails'].sub('', cleaned)
        
        # Remove page numbers
        cleaned = self.patterns['page_numbers'].sub('', cleaned)
        cleaned = self.patterns['arabic_page_numbers'].sub('', cleaned)
        
        # Remove headers/footers patterns
        cleaned = self.patterns['headers_footers'].sub('', cleaned)
        
        # Language-specific cleaning
        if language == 'french':
            cleaned = self.patterns['french_artifacts'].sub('', cleaned)
        elif language == 'arabic':
            cleaned = self.patterns['arabic_artifacts'].sub('', cleaned)
        
        # Remove excessive punctuation
        cleaned = self.patterns['excess_punctuation'].sub('...', cleaned)
        
        # Fix punctuation spacing
        cleaned = self.patterns['punctuation_spacing'].sub(r'\1', cleaned)
        
        # Remove special characters (preserve language-specific)
        cleaned = self.patterns['special_chars'].sub(' ', cleaned)
        
        # Remove isolated characters
        cleaned = self.patterns['isolated_chars'].sub('', cleaned)
        
        # Remove repeated lines
        cleaned = self.patterns['repeated_lines'].sub(r'\1', cleaned)
        
        # Normalize whitespace
        cleaned = self.patterns['excess_whitespace'].sub(' ', cleaned)
        
        # Remove empty lines and excessive line breaks
        lines = cleaned.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        cleaned = '\n'.join(non_empty_lines)
        
        return cleaned.strip()
    
    def normalize_text(self, text: str, language: str = None) -> str:
        """
        Normalize text for consistent processing.
        
        Args:
            text: Text to normalize
            language: Detected language for specific normalization
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        normalized = text
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKC', normalized)
        
        # Language-specific normalization
        if language == 'french':
            # Normalize French quotes and dashes
            normalized = self.normalization_patterns['french_quotes'].sub("'", normalized)
            normalized = self.normalization_patterns['french_dashes'].sub('-', normalized)
        
        elif language == 'arabic':
            # Normalize Arabic characters
            normalized = self.normalization_patterns['arabic_alef'].sub('ا', normalized)
            normalized = self.normalization_patterns['arabic_yeh'].sub('ي', normalized)
            
            # Convert Arabic-Indic numbers to regular numbers
            arabic_to_western = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
            normalized = normalized.translate(arabic_to_western)
        
        # Remove extra spaces around punctuation
        normalized = re.sub(r'\s+([,.!?;:])', r'\1', normalized)
        normalized = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1\2', normalized)
        
        # Normalize multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def segment_text(self, text: str, language: str = None) -> Dict[str, List[str]]:
        """
        Segment text into sentences and paragraphs.
        
        Args:
            text: Input text
            language: Language for language-specific segmentation
            
        Returns:
            Dictionary with sentences and paragraphs
        """
        if not text:
            return {'sentences': [], 'paragraphs': []}
        
        # Paragraph segmentation
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Filter short paragraphs if enabled
        if self.remove_short_paragraphs:
            paragraphs = [p for p in paragraphs if len(p) >= self.min_paragraph_length]
        
        # Sentence segmentation
        sentences = []
        
        try:
            # Use NLTK for sentence tokenization
            if language == 'french':
                # Use French-specific sentence tokenization
                for paragraph in paragraphs:
                    para_sentences = sent_tokenize(paragraph, language='french')
                    sentences.extend(para_sentences)
            elif language == 'arabic':
                # Arabic sentence segmentation (NLTK might not work well)
                for paragraph in paragraphs:
                    # Use simple rule-based approach for Arabic
                    para_sentences = self._segment_arabic_sentences(paragraph)
                    sentences.extend(para_sentences)
            else:
                # Default English tokenization
                for paragraph in paragraphs:
                    para_sentences = sent_tokenize(paragraph)
                    sentences.extend(para_sentences)
        
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}")
            # Fallback: simple rule-based segmentation
            sentences = self._simple_sentence_segmentation(text)
        
        # Filter sentences by length
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (self.min_sentence_length <= len(sentence) <= self.max_sentence_length):
                filtered_sentences.append(sentence)
        
        return {
            'sentences': filtered_sentences,
            'paragraphs': paragraphs
        }
    
    def _segment_arabic_sentences(self, text: str) -> List[str]:
        """Arabic-specific sentence segmentation."""
        # Arabic sentence endings
        sentence_endings = r'[.!?؟۔]'
        
        # Split by sentence endings
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                result.append(sentence)
        
        return result
    
    def _simple_sentence_segmentation(self, text: str) -> List[str]:
        """Simple rule-based sentence segmentation fallback."""
        # Split by common sentence endings
        sentence_pattern = r'[.!?؟۔]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def tokenize_text(self, text: str, language: str = None) -> List[str]:
        """
        Tokenize text into words with language-specific handling.
        
        Args:
            text: Input text
            language: Language for tokenization
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            # Use SpaCy if available for better tokenization
            if language == 'french' and 'fr' in self.spacy_models:
                doc = self.spacy_models['fr'](text)
                tokens = [token.text.lower() for token in doc if not token.is_space]
            elif language == 'english' and 'en' in self.spacy_models:
                doc = self.spacy_models['en'](text)
                tokens = [token.text.lower() for token in doc if not token.is_space]
            else:
                # Fallback to NLTK
                tokens = word_tokenize(text.lower())
        
        except Exception as e:
            logger.warning(f"Advanced tokenization failed: {e}")
            # Simple tokenization fallback
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove very short tokens and stopwords
        if language in self.stop_words:
            stopwords_set = self.stop_words[language]
            tokens = [token for token in tokens 
                     if len(token) > 2 and token not in stopwords_set]
        
        return tokens
    
    def assess_text_quality(self, text: str, language: str = None) -> TextQualityMetrics:
        """
        Assess text quality with multiple metrics.
        
        Args:
            text: Text to assess
            language: Text language
            
        Returns:
            TextQualityMetrics object
        """
        if not text:
            return TextQualityMetrics(0, 0, 0, 0, 0, ['Empty text'])
        
        issues = []
        
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?؟۔]+', text))
        
        # Readability score (simple approximation)
        if sentence_count > 0 and word_count > 0:
            avg_words_per_sentence = word_count / sentence_count
            avg_chars_per_word = char_count / word_count
            
            # Simple readability formula (adapted for multilingual)
            readability_score = max(0, min(1, 1 - (avg_words_per_sentence - 15) / 20))
        else:
            readability_score = 0
            issues.append('Cannot calculate readability')
        
        # Coherence score (based on repetition and structure)
        tokens = self.tokenize_text(text, language)
        if tokens:
            token_freq = Counter(tokens)
            # Coherence based on vocabulary richness
            vocabulary_richness = len(set(tokens)) / len(tokens)
            coherence_score = min(vocabulary_richness * 2, 1.0)
        else:
            coherence_score = 0
            issues.append('No valid tokens found')
        
        # Completeness score (based on sentence structure)
        complete_sentences = len(re.findall(r'[.!?؟۔]', text))
        if sentence_count > 0:
            completeness_score = min(complete_sentences / sentence_count, 1.0)
        else:
            completeness_score = 0
            issues.append('No sentences detected')
        
        # Language consistency (detect mixed languages)
        if language:
            language_consistency = self._assess_language_consistency(text, language)
        else:
            language_consistency = 0.5  # Neutral if language unknown
        
        # Check for common issues
        if char_count < 100:
            issues.append('Text too short')
        
        if word_count < 10:
            issues.append('Very few words')
        
        if len(re.findall(r'[0-9]', text)) / char_count > 0.3:
            issues.append('High number density')
        
        if len(re.findall(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u00C0-\u017F]', text)) / char_count > 0.2:
            issues.append('High special character density')
        
        # Overall quality score
        quality_scores = [readability_score, coherence_score, completeness_score, language_consistency]
        overall_quality = np.mean(quality_scores)
        
        return TextQualityMetrics(
            readability_score=readability_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            language_consistency=language_consistency,
            overall_quality=overall_quality,
            issues=issues
        )
    
    def _assess_language_consistency(self, text: str, expected_language: str) -> float:
        """Assess consistency of language throughout the text."""
        # Split text into chunks
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        
        consistent_chunks = 0
        total_chunks = len(chunks)
        
        for chunk in chunks:
            detected_lang, confidence = self.detect_language(chunk)
            if detected_lang == expected_language and confidence > 0.5:
                consistent_chunks += 1
        
        return consistent_chunks / total_chunks if total_chunks > 0 else 0.0
    
    def calculate_text_statistics(self, text: str, tokens: List[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive text statistics."""
        if not text:
            return {}
        
        if tokens is None:
            tokens = self.tokenize_text(text)
        
        # Basic counts
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?؟۔]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Advanced statistics
        token_freq = Counter(tokens)
        vocabulary_size = len(set(tokens))
        
        # Calculate type-token ratio (vocabulary richness)
        ttr = vocabulary_size / len(tokens) if tokens else 0
        
        # Most common words
        most_common_words = token_freq.most_common(10)
        
        # Average lengths
        avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Language distribution (character-based)
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z\u00C0-\u017F]', text))
        numeric_chars = len(re.findall(r'[0-9]', text))
        
        return {
            'character_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'vocabulary_size': vocabulary_size,
            'type_token_ratio': ttr,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'most_common_words': most_common_words,
            'character_distribution': {
                'arabic_chars': arabic_chars,
                'latin_chars': latin_chars,
                'numeric_chars': numeric_chars,
                'other_chars': char_count - arabic_chars - latin_chars - numeric_chars
            },
            'density_metrics': {
                'arabic_density': arabic_chars / char_count if char_count > 0 else 0,
                'latin_density': latin_chars / char_count if char_count > 0 else 0,
                'numeric_density': numeric_chars / char_count if char_count > 0 else 0
            }
        }
    
    def process_text(self, text: str, 
                    custom_steps: List[str] = None) -> ProcessedText:
        """
        Complete text processing pipeline.
        
        Args:
            text: Raw input text
            custom_steps: Custom processing steps
            
        Returns:
            ProcessedText object with all processing results
        """
        if not text:
            return ProcessedText(
                original_text="",
                cleaned_text="",
                normalized_text="",
                sentences=[],
                paragraphs=[],
                tokens=[],
                language="unknown",
                language_confidence=0.0,
                quality_metrics=TextQualityMetrics(0, 0, 0, 0, 0, ['Empty text']),
                preprocessing_steps=[],
                statistics={}
            )
        
        processing_steps = custom_steps or [
            'language_detection', 'cleaning', 'normalization', 
            'segmentation', 'tokenization', 'quality_assessment'
        ]
        
        applied_steps = []
        original_text = text
        
        # Step 1: Language Detection
        if 'language_detection' in processing_steps:
            language, lang_confidence = self.detect_language(text)
            applied_steps.append('language_detection')
        else:
            language, lang_confidence = 'unknown', 0.0
        
        # Step 2: Text Cleaning
        if 'cleaning' in processing_steps:
            cleaned_text = self.clean_text(text, language)
            applied_steps.append('cleaning')
        else:
            cleaned_text = text
        
        # Step 3: Text Normalization
        if 'normalization' in processing_steps:
            normalized_text = self.normalize_text(cleaned_text, language)
            applied_steps.append('normalization')
        else:
            normalized_text = cleaned_text
        
        # Step 4: Text Segmentation
        if 'segmentation' in processing_steps:
            segmentation_result = self.segment_text(normalized_text, language)
            sentences = segmentation_result['sentences']
            paragraphs = segmentation_result['paragraphs']
            applied_steps.append('segmentation')
        else:
            sentences = [normalized_text]
            paragraphs = [normalized_text]
        
        # Step 5: Tokenization
        if 'tokenization' in processing_steps:
            tokens = self.tokenize_text(normalized_text, language)
            applied_steps.append('tokenization')
        else:
            tokens = normalized_text.split()
        
        # Step 6: Quality Assessment
        if 'quality_assessment' in processing_steps:
            quality_metrics = self.assess_text_quality(normalized_text, language)
            applied_steps.append('quality_assessment')
        else:
            quality_metrics = TextQualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, [])
        
        # Calculate statistics
        statistics = self.calculate_text_statistics(normalized_text, tokens)
        
        return ProcessedText(
            original_text=original_text,
            cleaned_text=cleaned_text,
            normalized_text=normalized_text,
            sentences=sentences,
            paragraphs=paragraphs,
            tokens=tokens,
            language=language,
            language_confidence=lang_confidence,
            quality_metrics=quality_metrics,
            preprocessing_steps=applied_steps,
            statistics=statistics
        )
