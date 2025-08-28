"""
Document Quality Checker for French and Arabic Documents
Evaluates text quality, readability, and extraction accuracy
"""

import re
import string
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import unicodedata


class QualityChecker:
    """
    Comprehensive quality checker for extracted document content
    Supports both French and Arabic text quality assessment
    """
    
    def __init__(self):
        # French stop words for readability analysis
        self.french_stop_words = {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 
            'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas',
            'tout', 'plus', 'par', 'grand', 'en', 'une', 'être', 'et', 'à', 'il',
            'avoir', 'ne', 'je', 'son', 'que', 'se', 'qui', 'ce', 'dans', 'en',
            'du', 'elle', 'au', 'de', 'ce', 'le', 'pour', 'sont', 'avec', 'ils',
            'être', 'à', 'un', 'avoir', 'tout', 'mais', 'nous', 'vous', 'votre'
        }
        
        # Arabic stop words
        self.arabic_stop_words = {
            'في', 'من', 'إلى', 'على', 'أن', 'هذا', 'هذه', 'التي', 'الذي', 'كان',
            'كانت', 'يكون', 'تكون', 'وقد', 'قد', 'لقد', 'كل', 'بعض', 'جميع',
            'عند', 'عندما', 'حيث', 'أين', 'كيف', 'ماذا', 'متى', 'لماذا', 'لكن',
            'لكي', 'حتى', 'إذا', 'لو', 'أم', 'أما', 'إما', 'بل', 'لا', 'ليس'
        }
        
        # Quality thresholds
        self.thresholds = {
            'minimum_length': 10,
            'maximum_gibberish_ratio': 0.3,
            'minimum_word_avg_length': 2.0,
            'maximum_word_avg_length': 15.0,
            'minimum_sentence_avg_length': 3,
            'maximum_sentence_avg_length': 50,
            'maximum_repetition_ratio': 0.4,
            'minimum_vocabulary_diversity': 0.3,
            'maximum_special_char_ratio': 0.2,
            'minimum_alpha_ratio': 0.7
        }
    
    def check_basic_quality(self, text: str) -> Dict[str, Any]:
        """
        Perform basic quality checks on text
        """
        if not text or not text.strip():
            return {
                'valid': False,
                'score': 0.0,
                'issues': ['empty_text'],
                'stats': {'length': 0}
            }
        
        text = text.strip()
        issues = []
        stats = self._calculate_text_statistics(text)
        
        # Length check
        if stats['char_count'] < self.thresholds['minimum_length']:
            issues.append('text_too_short')
        
        # Character composition check
        if stats['alpha_ratio'] < self.thresholds['minimum_alpha_ratio']:
            issues.append('low_alphabetic_content')
        
        if stats['special_char_ratio'] > self.thresholds['maximum_special_char_ratio']:
            issues.append('high_special_character_ratio')
        
        # Word length analysis
        if stats['avg_word_length'] < self.thresholds['minimum_word_avg_length']:
            issues.append('words_too_short')
        elif stats['avg_word_length'] > self.thresholds['maximum_word_avg_length']:
            issues.append('words_too_long')
        
        # Sentence length analysis
        if stats['avg_sentence_length'] < self.thresholds['minimum_sentence_avg_length']:
            issues.append('sentences_too_short')
        elif stats['avg_sentence_length'] > self.thresholds['maximum_sentence_avg_length']:
            issues.append('sentences_too_long')
        
        # Calculate basic quality score
        score = self._calculate_basic_score(stats, issues)
        
        return {
            'valid': score > 0.5 and len(issues) < 3,
            'score': score,
            'issues': issues,
            'stats': stats
        }
    
    def check_extraction_quality(self, text: str, source_type: str = 'unknown') -> Dict[str, Any]:
        """
        Check quality specific to extracted text from documents
        """
        basic_check = self.check_basic_quality(text)
        
        if not basic_check['valid']:
            return {
                **basic_check,
                'extraction_issues': ['failed_basic_quality']
            }
        
        extraction_issues = []
        
        # Check for common extraction artifacts
        extraction_issues.extend(self._check_extraction_artifacts(text))
        
        # Check for encoding issues
        if self._has_encoding_issues(text):
            extraction_issues.append('encoding_issues')
        
        # Check for OCR-specific issues if likely from OCR
        if source_type in ['pdf', 'image']:
            extraction_issues.extend(self._check_ocr_issues(text))
        
        # Check text structure
        structure_score = self._assess_text_structure(text)
        
        # Calculate extraction quality score
        extraction_score = basic_check['score'] * (1 - len(extraction_issues) * 0.15) * structure_score
        
        return {
            **basic_check,
            'extraction_valid': extraction_score > 0.6 and len(extraction_issues) < 3,
            'extraction_score': max(0.0, extraction_score),
            'extraction_issues': extraction_issues,
            'structure_score': structure_score
        }
    
    def check_readability(self, text: str, language: str = 'auto') -> Dict[str, Any]:
        """
        Assess text readability and coherence
        """
        if language == 'auto':
            language = self._detect_language_simple(text)
        
        readability_metrics = {}
        
        # Basic readability metrics
        stats = self._calculate_text_statistics(text)
        
        # Vocabulary diversity
        vocabulary_diversity = self._calculate_vocabulary_diversity(text)
        readability_metrics['vocabulary_diversity'] = vocabulary_diversity
        
        # Sentence complexity
        sentence_complexity = self._calculate_sentence_complexity(text)
        readability_metrics['sentence_complexity'] = sentence_complexity
        
        # Repetition analysis
        repetition_score = self._analyze_repetition(text)
        readability_metrics['repetition_score'] = repetition_score
        
        # Language-specific readability
        if language == 'french':
            readability_metrics.update(self._french_readability_metrics(text))
        elif language == 'arabic':
            readability_metrics.update(self._arabic_readability_metrics(text))
        
        # Overall readability score
        overall_score = self._calculate_readability_score(readability_metrics)
        
        return {
            'language': language,
            'readability_score': overall_score,
            'metrics': readability_metrics,
            'readable': overall_score > 0.6
        }
    
    def comprehensive_check(self, text: str, source_type: str = 'unknown', 
                          language: str = 'auto') -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment
        """
        # Run all checks
        basic = self.check_basic_quality(text)
        extraction = self.check_extraction_quality(text, source_type)
        readability = self.check_readability(text, language)
        
        # Calculate overall quality score
        weights = {'basic': 0.4, 'extraction': 0.3, 'readability': 0.3}
        overall_score = (
            basic['score'] * weights['basic'] +
            extraction['extraction_score'] * weights['extraction'] +
            readability['readability_score'] * weights['readability']
        )
        
        # Collect all issues
        all_issues = (basic.get('issues', []) + 
                     extraction.get('extraction_issues', []))
        
        # Quality rating
        if overall_score >= 0.8:
            rating = 'excellent'
        elif overall_score >= 0.6:
            rating = 'good'
        elif overall_score >= 0.4:
            rating = 'fair'
        else:
            rating = 'poor'
        
        return {
            'overall_score': overall_score,
            'rating': rating,
            'usable': overall_score > 0.4,
            'recommended_for_summarization': overall_score > 0.5,
            'recommended_for_chat': overall_score > 0.6,
            'basic_check': basic,
            'extraction_check': extraction,
            'readability_check': readability,
            'all_issues': list(set(all_issues)),
            'recommendations': self._generate_recommendations(all_issues, overall_score)
        }
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive text statistics"""
        if not text:
            return {'char_count': 0}
        
        # Basic counts
        char_count = len(text)
        alpha_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        space_count = sum(1 for c in text if c.isspace())
        punct_count = sum(1 for c in text if c in string.punctuation + '،؛؟!""''«»()[]{}')
        special_count = char_count - alpha_count - digit_count - space_count - punct_count
        
        # Word analysis
        words = self._tokenize_words(text)
        word_count = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        # Sentence analysis
        sentences = self._tokenize_sentences(text)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        return {
            'char_count': char_count,
            'alpha_count': alpha_count,
            'digit_count': digit_count,
            'space_count': space_count,
            'punct_count': punct_count,
            'special_count': special_count,
            'word_count': word_count,
            'unique_words': unique_words,
            'sentence_count': sentence_count,
            'alpha_ratio': alpha_count / max(char_count, 1),
            'special_char_ratio': special_count / max(char_count, 1),
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'vocabulary_ratio': unique_words / max(word_count, 1)
        }
    
    def _calculate_basic_score(self, stats: Dict, issues: List[str]) -> float:
        """Calculate basic quality score based on statistics and issues"""
        base_score = 1.0
        
        # Penalize issues
        base_score -= len(issues) * 0.15
        
        # Adjust based on ratios
        if stats['alpha_ratio'] < 0.5:
            base_score -= 0.2
        elif stats['alpha_ratio'] > 0.9:
            base_score += 0.1
        
        if stats['vocabulary_ratio'] > 0.5:
            base_score += 0.1
        elif stats['vocabulary_ratio'] < 0.2:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _check_extraction_artifacts(self, text: str) -> List[str]:
        """Check for common extraction artifacts"""
        issues = []
        
        # Check for excessive whitespace
        if re.search(r'\s{5,}', text):
            issues.append('excessive_whitespace')
        
        # Check for broken words (common in PDF extraction)
        broken_words = re.findall(r'\b\w{1,2}\s+\w{1,2}\b', text)
        if len(broken_words) > len(text.split()) * 0.1:
            issues.append('broken_words')
        
        # Check for header/footer artifacts
        if re.search(r'page\s+\d+|©|\d+/\d+', text, re.IGNORECASE):
            issues.append('header_footer_artifacts')
        
        # Check for table artifacts
        if re.search(r'\|\s*\||\t{3,}', text):
            issues.append('table_artifacts')
        
        # Check for bullet point artifacts
        if re.search(r'^[\s•\-\*]{2,}', text, re.MULTILINE):
            issues.append('formatting_artifacts')
        
        return issues
    
    def _has_encoding_issues(self, text: str) -> bool:
        """Check for encoding issues"""
        # Check for replacement characters
        if '�' in text or '\ufffd' in text:
            return True
        
        # Check for mojibake patterns
        mojibake_patterns = [
            r'Ã¡|Ã©|Ã­|Ã³|Ãº',  # Common mojibake for accented characters
            r'â€™|â€œ|â€\u009d',      # Smart quotes mojibake
        ]
        
        for pattern in mojibake_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _check_ocr_issues(self, text: str) -> List[str]:
        """Check for OCR-specific issues"""
        issues = []
        
        # Common OCR character substitutions
        ocr_errors = [
            (r'\b[Il1|]{2,}\b', 'character_confusion'),  # I, l, 1, | confusion
            (r'\brn\b', 'rn_m_confusion'),               # rn mistaken for m
            (r'\bvv\b', 'vv_w_confusion'),               # vv mistaken for w
            (r'\b[O0]{2,}\b', 'o_zero_confusion'),       # O and 0 confusion
        ]
        
        for pattern, issue in ocr_errors:
            if re.search(pattern, text):
                issues.append(issue)
        
        # Check for excessive single characters
        single_chars = len(re.findall(r'\b\w\b', text))
        if single_chars > len(text.split()) * 0.15:
            issues.append('excessive_single_characters')
        
        # Check for non-dictionary words (potential OCR errors)
        # This is a simplified check
        words = self._tokenize_words(text.lower())
        if words:
            suspicious_ratio = sum(1 for word in words if len(word) > 2 and 
                                 not word.isalpha()) / len(words)
            if suspicious_ratio > 0.2:
                issues.append('suspicious_character_sequences')
        
        return issues
    
    def _assess_text_structure(self, text: str) -> float:
        """Assess the structural quality of text"""
        # Check for proper sentence structure
        sentences = self._tokenize_sentences(text)
        if not sentences:
            return 0.0
        
        structure_score = 1.0
        
        # Check sentence endings
        properly_ended = sum(1 for s in sentences if s.strip().endswith(('.', '!', '?', '؟')))
        if sentences:
            ending_ratio = properly_ended / len(sentences)
            structure_score *= ending_ratio
        
        # Check for paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            structure_score += 0.1  # Bonus for paragraph structure
        
        # Check for consistent formatting
        if not re.search(r'[A-Z][^.!?]*[a-z][^.!?]*[.!?]', text):  # Basic sentence pattern
            structure_score -= 0.2
        
        return max(0.0, min(1.0, structure_score))
    
    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (Type-Token Ratio)"""
        words = self._tokenize_words(text.lower())
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Use modified TTR for longer texts
        if total_words > 100:
            # Root TTR
            return unique_words / (total_words ** 0.5)
        else:
            return unique_words / total_words
    
    def _calculate_sentence_complexity(self, text: str) -> float:
        """Calculate average sentence complexity"""
        sentences = self._tokenize_sentences(text)
        if not sentences:
            return 0.0
        
        complexities = []
        for sentence in sentences:
            words = self._tokenize_words(sentence)
            word_count = len(words)
            
            # Simple complexity measure based on word count and punctuation
            punct_count = sum(1 for c in sentence if c in ',:;()[]{}')
            complexity = word_count + (punct_count * 0.5)
            complexities.append(complexity)
        
        avg_complexity = sum(complexities) / len(complexities)
        # Normalize to 0-1 scale (assuming max reasonable complexity is 30)
        return min(1.0, avg_complexity / 30.0)
    
    def _analyze_repetition(self, text: str) -> float:
        """Analyze text for excessive repetition"""
        words = self._tokenize_words(text.lower())
        if len(words) < 10:
            return 1.0  # Too short to analyze meaningfully
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Calculate repetition ratio
        most_common_count = word_freq.most_common(1)[0][1] if word_freq else 0
        repetition_ratio = most_common_count / len(words)
        
        # Convert to quality score (lower repetition = higher score)
        return max(0.0, 1.0 - (repetition_ratio - 0.1))  # Allow some repetition
    
    def _french_readability_metrics(self, text: str) -> Dict[str, float]:
        """French-specific readability metrics"""
        words = self._tokenize_words(text.lower())
        
        if not words:
            return {'french_stop_word_ratio': 0.0, 'french_accent_ratio': 0.0}
        
        # Check stop word usage
        stop_words_count = sum(1 for word in words if word in self.french_stop_words)
        stop_word_ratio = stop_words_count / len(words)
        
        # Check for French accented characters
        accented_chars = len(re.findall(r'[àâäéèêëïîôöùûüÿñç]', text))
        total_alpha = sum(1 for c in text if c.isalpha())
        accent_ratio = accented_chars / max(total_alpha, 1)
        
        return {
            'french_stop_word_ratio': stop_word_ratio,
            'french_accent_ratio': accent_ratio
        }
    
    def _arabic_readability_metrics(self, text: str) -> Dict[str, float]:
        """Arabic-specific readability metrics"""
        words = self._tokenize_words(text)
        
        if not words:
            return {'arabic_stop_word_ratio': 0.0, 'arabic_diacritic_ratio': 0.0}
        
        # Check stop word usage
        stop_words_count = sum(1 for word in words if word in self.arabic_stop_words)
        stop_word_ratio = stop_words_count / len(words)
        
        # Check for Arabic diacritics
        diacritics = len(re.findall(r'[\u064B-\u065F\u0670]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        diacritic_ratio = diacritics / max(arabic_chars, 1)
        
        return {
            'arabic_stop_word_ratio': stop_word_ratio,
            'arabic_diacritic_ratio': diacritic_ratio
        }
    
    def _calculate_readability_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall readability score from metrics"""
        base_score = 0.7  # Start with moderate score
        
        # Vocabulary diversity (higher is better up to a point)
        vocab_diversity = metrics.get('vocabulary_diversity', 0.5)
        if 0.3 <= vocab_diversity <= 0.7:
            base_score += 0.1
        elif vocab_diversity < 0.2 or vocab_diversity > 0.8:
            base_score -= 0.1
        
        # Sentence complexity (moderate complexity is good)
        complexity = metrics.get('sentence_complexity', 0.5)
        if 0.3 <= complexity <= 0.7:
            base_score += 0.1
        elif complexity < 0.2 or complexity > 0.8:
            base_score -= 0.1
        
        # Repetition (lower is better)
        repetition = metrics.get('repetition_score', 0.8)
        base_score += (repetition - 0.5) * 0.2
        
        # Language-specific adjustments
        if 'french_stop_word_ratio' in metrics:
            if metrics['french_stop_word_ratio'] > 0.15:  # Good French text should have stop words
                base_score += 0.1
            if metrics['french_accent_ratio'] > 0.02:  # Presence of French accents is good
                base_score += 0.05
        
        if 'arabic_stop_word_ratio' in metrics:
            if metrics['arabic_stop_word_ratio'] > 0.15:
                base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation and split
        import string
        arabic_punctuation = '،؛؟!""''«»()[]{}؍'
        all_punctuation = string.punctuation + arabic_punctuation
        translator = str.maketrans('', '', all_punctuation)
        clean_text = text.translate(translator)
        return [word.strip() for word in clean_text.split() if word.strip()]
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        # Split on sentence endings
        sentences = re.split(r'[.!?؟]+\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_language_simple(self, text: str) -> str:
        """Simple language detection for internal use"""
        # Basic detection based on character sets
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[A-Za-z]', text))
        
        if arabic_chars > latin_chars:
            return 'arabic'
        elif latin_chars > 0:
            return 'french'  # Assume French for Latin script
        else:
            return 'unknown'
    
    def _generate_recommendations(self, issues: List[str], overall_score: float) -> List[str]:
        """Generate recommendations based on identified issues"""
        recommendations = []
        
        if overall_score < 0.4:
            recommendations.append("Text quality is poor. Consider re-extracting from source or using a different extraction method.")
        
        if 'text_too_short' in issues:
            recommendations.append("Text is too short for reliable analysis. Ensure complete extraction.")
        
        if 'encoding_issues' in issues:
            recommendations.append("Encoding issues detected. Check source encoding and extraction settings.")
        
        if 'excessive_whitespace' in issues or 'formatting_artifacts' in issues:
            recommendations.append("Clean up formatting artifacts and excessive whitespace.")
        
        if 'broken_words' in issues:
            recommendations.append("Text appears to have broken words. Consider OCR post-processing or manual review.")
        
        if 'character_confusion' in issues or 'rn_m_confusion' in issues:
            recommendations.append("OCR character confusion detected. Manual review recommended for critical content.")
        
        if 'low_alphabetic_content' in issues:
            recommendations.append("Text has low alphabetic content. May contain too much formatting or special characters.")
        
        if overall_score > 0.7:
            recommendations.append("Text quality is good and suitable for all processing tasks.")
        elif overall_score > 0.5:
            recommendations.append("Text quality is adequate for most processing tasks.")
        else:
            recommendations.append("Text quality is marginal. Proceed with caution for important applications.")
        
        return recommendations


# Quality assessment classes for specific document types
class PDFQualityChecker(QualityChecker):
    """Specialized quality checker for PDF-extracted text"""
    
    def __init__(self):
        super().__init__()
        # PDF-specific thresholds
        self.thresholds.update({
            'maximum_single_char_ratio': 0.1,  # PDFs often have character spacing issues
            'minimum_line_length': 20,          # PDF lines should have reasonable length
        })
    
    def check_pdf_specific_issues(self, text: str) -> Dict[str, Any]:
        """Check for PDF-specific extraction issues"""
        issues = []
        
        # Check for column merge issues
        if re.search(r'\w+[A-Z][a-z]+[A-Z]', text):
            issues.append('column_merge_issues')
        
        # Check for font size artifacts
        if re.search(r'\b[A-Z]{3,}\b.*\b[A-Z]{3,}\b', text):
            issues.append('font_size_artifacts')
        
        # Check for hyperlink artifacts
        if text.count('http') > len(text.split()) * 0.05:
            issues.append('excessive_hyperlinks')
        
        return {
            'pdf_issues': issues,
            'pdf_specific_score': max(0.0, 1.0 - len(issues) * 0.2)
        }


class ImageQualityChecker(QualityChecker):
    """Specialized quality checker for image-extracted text (OCR)"""
    
    def __init__(self):
        super().__init__()
        # OCR-specific thresholds
        self.thresholds.update({
            'maximum_ocr_error_ratio': 0.15,    # Expected OCR error rate
            'minimum_confidence_words': 0.7,     # Minimum ratio of confident words
        })
    
    def check_ocr_confidence(self, text: str, confidence_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Check OCR confidence if available"""
        if confidence_data is None:
            # Estimate confidence based on text patterns
            words = self._tokenize_words(text)
            if not words:
                return {'confidence_score': 0.0}
            
            # Simple heuristic: fewer single characters and strange patterns = higher confidence
            suspicious_patterns = [
                r'\b\w\b',           # Single characters
                r'[Il1|]{2,}',       # Character confusion
                r'[^\w\s\.,!?؟،؛]',  # Unusual characters
            ]
            
            suspicious_count = 0
            for pattern in suspicious_patterns:
                suspicious_count += len(re.findall(pattern, text))
            
            confidence = max(0.0, 1.0 - (suspicious_count / len(words)))
            return {'confidence_score': confidence}
        
        # Use provided confidence data
        return {'confidence_score': confidence_data.get('average_confidence', 0.5)}


# Utility functions
def quick_quality_check(text: str) -> bool:
    """Quick quality check - returns True if text is usable"""
    checker = QualityChecker()
    result = checker.check_basic_quality(text)
    return result['valid']

def assess_document_quality(text: str, source_type: str = 'unknown', 
                          language: str = 'auto') -> Dict[str, Any]:
    """Complete document quality assessment"""
    if source_type.lower() == 'pdf':
        checker = PDFQualityChecker()
    elif source_type.lower() in ['image', 'scan', 'ocr']:
        checker = ImageQualityChecker()
    else:
        checker = QualityChecker()
    
    return checker.comprehensive_check(text, source_type, language)

def get_quality_score(text: str) -> float:
    """Get a simple quality score (0-1) for text"""
    checker = QualityChecker()
    result = checker.comprehensive_check(text)
    return result['overall_score']

def is_text_suitable_for_processing(text: str, task: str = 'general') -> bool:
    """Check if text is suitable for specific processing tasks"""
    result = assess_document_quality(text)
    
    if task == 'summarization':
        return result['recommended_for_summarization']
    elif task == 'chat':
        return result['recommended_for_chat']
    else:
        return result['usable']

# Quality improvement suggestions
class QualityImprover:
    """Suggest improvements for poor quality text"""
    
    @staticmethod
    def suggest_improvements(text: str, issues: List[str]) -> List[str]:
        """Suggest specific improvements based on detected issues"""
        suggestions = []
        
        if 'excessive_whitespace' in issues:
            suggestions.append("Remove excessive whitespace using regex: re.sub(r'\\s+', ' ', text)")
        
        if 'broken_words' in issues:
            suggestions.append("Consider post-processing to rejoin broken words")
        
        if 'encoding_issues' in issues:
            suggestions.append("Re-extract with proper encoding (UTF-8) or use encoding detection")
        
        if 'header_footer_artifacts' in issues:
            suggestions.append("Remove headers/footers using pattern matching")
        
        if 'character_confusion' in issues:
            suggestions.append("Apply OCR post-correction using character substitution rules")
        
        return suggestions
    
    @staticmethod
    def clean_text_basic(text: str) -> str:
        """Apply basic text cleaning"""
        if not text:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'\|\s*\|', ' ', text)  # Table separators
        text = re.sub(r'_{3,}', '', text)     # Underscores
        text = re.sub(r'-{3,}', '', text)     # Dashes
        
        # Clean up line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple line breaks
        
        return text.strip()