"""
Language Detection Utility for French and Arabic Documents
Supports multiple detection methods and confidence scoring
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import unicodedata


class LanguageDetector:
    """
    Language detector optimized for French and Arabic text detection
    Uses multiple detection methods for improved accuracy
    """
    
    def __init__(self):
        # French common words (high frequency)
        self.french_words = {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
            'par', 'grand', 'en', 'une', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je',
            'son', 'que', 'se', 'qui', 'ce', 'dans', 'en', 'du', 'elle', 'au', 'de',
            'ce', 'le', 'pour', 'sont', 'avec', 'ils', 'être', 'à', 'un', 'avoir',
            'tout', 'mais', 'nous', 'vous', 'votre', 'cette', 'faire', 'leur', 'si',
            'peut', 'dire', 'elle', 'ses', 'temps', 'très', 'lui', 'vont', 'voir',
            'en', 'fait', 'celle', 'voire', 'être', 'deux', 'même', 'autre', 'après',
            'seulement', 'nom', 'aussi', 'voici', 'bien', 'où', 'voilà', 'vers', 'tout',
            'pendant', 'contre', 'tous', 'homme', 'ici', 'moins', 'maintenant', 'sans',
            'moi', 'année', 'monde', 'jour', 'monsieur', 'demander', 'autre', 'entre',
            'première', 'venir', 'pendant', 'passer', 'peu', 'lequel', 'suite', 'bon',
            'comprendre', 'depuis', 'point', 'ainsi', 'heure', 'rester', 'savoir',
            'aller', 'devant', 'ville', 'chaque', 'même', 'donner', 'rien', 'être',
            'chose', 'pourquoi', 'notre', 'sans', 'grand', 'donc', 'alors', 'fin',
            'nouveau', 'cas', 'route', 'pont', 'voiture', 'train', 'bus', 'avion',
            'europe', 'afrique', 'pays', 'gouvernement', 'partir', 'place', 
            'groupe', 'vers', 'partie', 'prendre', 'eau', 'de',
            'travail', 'devoir', 'vie', 'système', 'mettre', 'important',
            'mot', 'chercher', 'premier', 'temps', 'personne', 'année', 'semaine',
            'mois', 'cause', 'loi', 'programme', 'société',
            'marché', 'politique',
            'suite', 'peuple', 'Maroc', 'exemple', 'penser', 'situation',
            'différent', 'mer', 'cours', 'contre', 'publique',
            'plan', 'étude', 'question',
            'suivant', 'organisé', 'développer', 'président', 'communauté', 'pour', 'moins',
            'culture', 'changer', 'école', 'cinquante', 'région',
            'souvent', 'exister', 'alors', 'résultat', 'action', 'répondre', 'organization',
            'activité', 'kilomètre',
            'information', 'entreprise'
        }
        
        self.arabic_words = {
            # Mots fonctionnels de base (stopwords)
            'في', 'من', 'إلى', 'على', 'عن', 'أن', 'هذا', 'هذه', 'ذلك', 'كانت', 'قد', 'كل', 'بعض',
            'عند', 'حيث', 'متى', 'لكن', 'حتى', 'إذا', 'بل', 'لا', 'ليس', 'مع', 'قبل', 'بعد', 'خلال',

            # Transport de base
            'طريق', 'شارع', 'محطة', 'مطار', 'ميناء', 'رصيف', 'جسر', 'نفق',
            'سيارة', 'حافلة', 'قطار', 'شاحنة', 'دراجة', 'دراجة_نارية', 'مترو', 'ترامواي', 'سفينة', 'طائرة',
            
            # Logistique et mobilité
            'مرور', 'ازدحام', 'إشارة', 'توقف', 'سرعة', 'ممنوع', 'مسموح', 'خطر', 'سلامة', 'حزام',
            'وقود', 'طاقة', 'ديزل', 'بنزين', 'كهرباء', 'بطارية', 'صيانة', 'إصلاح', 'حجز', 'تذكرة',
            'تحميل', 'تفريغ', 'شحن', 'نقل', 'مستودع', 'مخزن', 'بضاعة', 'حاوية', 'طرد', 'طرود',

            # Institutions
            'وزارة', 'حكومة', 'شرطة', 'مراقبة', 'قانون', 'مكتب', 'إدارة',

            # Maroc et villes principales
            'المغرب', 'الرباط', 'الدار_البيضاء', 'مراكش', 'فاس', 'طنجة', 'أكادير',
            'وجدة', 'مكناس', 'تطوان', 'الصويرة', 'ورزازات', 'الناظور', 'العيون', 'الداخلة',

            # Documents administratifs
            'عقد', 'اتفاقية', 'مذكرة', 'قرار', 'قانون', 'مرسوم',
            'إعلان', 'طلب', 'استمارة', 'وثيقة', 'شهادة', 'ترخيص',
            'فاتورة', 'وصل', 'مذكرة_خدمة', 'محضر', 'ملف',
            'تقرير', 'مراسلة', 'بريد', 'ختم', 'تأشيرة',
            'دفتر_التحملات', 'كراسة_الشروط', 'عطاء', 'مناقصة'
}
        
        # French character patterns
        self.french_patterns = [
            r'[àâäéèêëïîôöùûüÿñç]',  # French accented characters
            r'\b(le|la|les|un|une|des|du|de|ce|cette|ces|et|ou|mais|donc|ni|car)\b',
            r'\b(je|tu|il|elle|nous|vous|ils|elles)\b',
            r'\b(avoir|être|faire|aller|dire|venir|voir|savoir|vouloir)\b'
        ]
        
        # Arabic character patterns
        self.arabic_patterns = [
            r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]',  # Arabic Unicode ranges
            r'[ابتثجحخدذرزسشصضطظعغفقكلمنهوي]',  # Arabic letters
            r'[أإآؤئءة]',  # Arabic hamza forms
            r'[\u064B-\u065F\u0670\u0640]',  # Arabic diacritics and tatweel
        ]
        
    def detect_script(self, text: str) -> Dict[str, float]:
        """
        Detect script based on Unicode character ranges
        """
        if not text.strip():
            return {"unknown": 1.0}
            
        char_count = {"latin": 0, "arabic": 0, "other": 0}
        
        for char in text:
            # Latin script (including French accented characters)
            if ('\u0041' <= char <= '\u005A' or  # A-Z
                '\u0061' <= char <= '\u007A' or  # a-z
                '\u00C0' <= char <= '\u017F'):   # Latin Extended-A (includes French accents)
                char_count["latin"] += 1
            # Arabic script
            elif ('\u0600' <= char <= '\u06FF' or  # Arabic
                  '\u0750' <= char <= '\u077F' or  # Arabic Supplement
                  '\u08A0' <= char <= '\u08FF' or  # Arabic Extended-A
                  '\uFB50' <= char <= '\uFDFF' or  # Arabic Presentation Forms-A
                  '\uFE70' <= char <= '\uFEFF'):   # Arabic Presentation Forms-B
                char_count["arabic"] += 1
            elif char.isalpha():
                char_count["other"] += 1
                
        total_chars = sum(char_count.values())
        if total_chars == 0:
            return {"unknown": 1.0}
            
        return {script: count/total_chars for script, count in char_count.items()}
    
    def detect_by_common_words(self, text: str) -> Dict[str, float]:
        """
        Detect language based on common words frequency
        """
        if not text.strip():
            return {"unknown": 1.0}
            
        # Normalize and tokenize
        words = self._tokenize(text.lower())
        if not words:
            return {"unknown": 1.0}
            
        french_score = sum(1 for word in words if word in self.french_words)
        arabic_score = sum(1 for word in words if word in self.arabic_words)
        
        total_words = len(words)
        french_ratio = french_score / total_words
        arabic_ratio = arabic_score / total_words
        
        # Normalize scores
        total_score = french_ratio + arabic_ratio
        if total_score == 0:
            return {"unknown": 1.0}
            
        return {
            "french": french_ratio / total_score,
            "arabic": arabic_ratio / total_score
        }
    
    def detect_by_patterns(self, text: str) -> Dict[str, float]:
        """
        Detect language using regex patterns
        """
        if not text.strip():
            return {"unknown": 1.0}
            
        french_matches = 0
        for pattern in self.french_patterns:
            french_matches += len(re.findall(pattern, text, re.IGNORECASE))
            
        arabic_matches = 0
        for pattern in self.arabic_patterns:
            arabic_matches += len(re.findall(pattern, text))
            
        total_matches = french_matches + arabic_matches
        if total_matches == 0:
            return {"unknown": 1.0}
            
        return {
            "french": french_matches / total_matches,
            "arabic": arabic_matches / total_matches
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for word-based analysis
        """
        # Remove punctuation and split on whitespace
        import string
        translator = str.maketrans('', '', string.punctuation + '،؛؟!""''«»()[]{}')
        clean_text = text.translate(translator)
        return [word.strip() for word in clean_text.split() if word.strip()]
    
    def _calculate_text_stats(self, text: str) -> Dict[str, int]:
        """
        Calculate basic text statistics
        """
        return {
            "total_chars": len(text),
            "alpha_chars": sum(1 for c in text if c.isalpha()),
            "words": len(self._tokenize(text)),
            "sentences": len(re.split(r'[.!?؟]', text)),
            "arabic_chars": len(re.findall(r'[\u0600-\u06FF]', text)),
            "latin_chars": len(re.findall(r'[A-Za-z]', text)),
            "accented_chars": len(re.findall(r'[àâäéèêëïîôöùûüÿñç]', text))
        }
    
    def detect_language(self, text: str, min_confidence: float = 0.3) -> Dict[str, any]:
        """
        Main language detection method with confidence scoring
        
        Args:
            text: Input text to analyze
            min_confidence: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            Dictionary with detected language, confidence, and analysis details
        """
        if not text or not text.strip():
            return {
                "language": "unknown",
                "confidence": 0.0,
                "details": {
                    "error": "Empty or whitespace-only text"
                }
            }
            
        # Get text statistics
        stats = self._calculate_text_stats(text)
        
        # If text is too short, use script detection only
        if stats["words"] < 3:
            script_scores = self.detect_script(text)
            if script_scores.get("arabic", 0) > 0.6:
                return {
                    "language": "arabic",
                    "confidence": script_scores["arabic"],
                    "details": {
                        "method": "script_only",
                        "reason": "text_too_short",
                        "stats": stats
                    }
                }
            elif script_scores.get("latin", 0) > 0.6:
                return {
                    "language": "french",
                    "confidence": script_scores["latin"],
                    "details": {
                        "method": "script_only",
                        "reason": "text_too_short",
                        "stats": stats
                    }
                }
        
        # Multi-method detection for longer texts
        script_scores = self.detect_script(text)
        word_scores = self.detect_by_common_words(text)
        pattern_scores = self.detect_by_patterns(text)
        
        # Weighted combination of methods
        weights = {"script": 0.3, "words": 0.5, "patterns": 0.2}
        
        final_scores = {"french": 0.0, "arabic": 0.0, "unknown": 0.0}
        
        # Script-based scoring
        if "latin" in script_scores:
            final_scores["french"] += weights["script"] * script_scores["latin"]
        if "arabic" in script_scores:
            final_scores["arabic"] += weights["script"] * script_scores["arabic"]
            
        # Word-based scoring
        if "french" in word_scores:
            final_scores["french"] += weights["words"] * word_scores["french"]
        if "arabic" in word_scores:
            final_scores["arabic"] += weights["words"] * word_scores["arabic"]
            
        # Pattern-based scoring
        if "french" in pattern_scores:
            final_scores["french"] += weights["patterns"] * pattern_scores["french"]
        if "arabic" in pattern_scores:
            final_scores["arabic"] += weights["patterns"] * pattern_scores["arabic"]
        
        # Determine best language
        best_lang = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_lang]
        
        # Apply minimum confidence threshold
        if confidence < min_confidence:
            best_lang = "unknown"
            confidence = 0.0
            
        return {
            "language": best_lang,
            "confidence": confidence,
            "details": {
                "method": "multi_method",
                "individual_scores": {
                    "script": script_scores,
                    "words": word_scores,
                    "patterns": pattern_scores
                },
                "final_scores": final_scores,
                "stats": stats
            }
        }
    
    def detect_language_simple(self, text: str) -> str:
        """
        Simple interface that returns just the language name
        """
        result = self.detect_language(text)
        return result["language"]
    
    def is_mixed_language(self, text: str, threshold: float = 0.3) -> bool:
        """
        Detect if text contains mixed languages
        """
        result = self.detect_language(text)
        details = result.get("details", {})
        final_scores = details.get("final_scores", {})
        
        # Check if multiple languages have significant scores
        significant_langs = [lang for lang, score in final_scores.items() 
                           if score > threshold and lang != "unknown"]
        
        return len(significant_langs) > 1
    
    def get_language_distribution(self, text: str) -> Dict[str, float]:
        """
        Get the distribution of languages in the text
        """
        result = self.detect_language(text)
        details = result.get("details", {})
        return details.get("final_scores", {"unknown": 1.0})


# Utility functions for easy integration
def detect_language(text: str) -> str:
    """Quick language detection function"""
    detector = LanguageDetector()
    return detector.detect_language_simple(text)

def detect_with_confidence(text: str) -> Tuple[str, float]:
    """Language detection with confidence score"""
    detector = LanguageDetector()
    result = detector.detect_language(text)
    return result["language"], result["confidence"]

def is_arabic(text: str) -> bool:
    """Check if text is Arabic"""
    return detect_language(text) == "arabic"

def is_french(text: str) -> bool:
    """Check if text is French"""
    return detect_language(text) == "french"