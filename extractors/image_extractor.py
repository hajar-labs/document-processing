import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import re
from typing import List, Dict, Optional, Tuple
import logging

# Configuration de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageExtractor:
    """Extracteur de texte à partir d'images avec support français et arabe"""
    
    def __init__(self, config=None):
        self.config = config or {}
        # Configuration Tesseract pour français et arabe
        self.tesseract_config = {
            'french': '--oem 3 --psm 6 -l fra',
            'arabic': '--oem 3 --psm 6 -l ara',
            'mixed': '--oem 3 --psm 6 -l fra+ara'
        }
    
    def detect_language(self, image: np.ndarray) -> str:
        """Détecte la langue principale dans l'image"""
        try:
            # Test avec français
            text_fr = pytesseract.image_to_string(image, config=self.tesseract_config['french'])
            # Test avec arabe  
            text_ar = pytesseract.image_to_string(image, config=self.tesseract_config['arabic'])
            
            # Compter les caractères français vs arabes
            french_chars = len(re.findall(r'[a-zA-ZÀ-ÿ]', text_fr))
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text_ar))
            
            if arabic_chars > french_chars:
                return 'arabic'
            elif french_chars > arabic_chars:
                return 'french'
            else:
                return 'mixed'
        except Exception as e:
            logger.error(f"Erreur détection langue: {e}")
            return 'mixed'
    
    def extract_text_from_image(self, image_path: str, language: Optional[str] = None) -> Dict:
        """Extrait le texte d'une image avec détection automatique de langue"""
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            # Préprocessing de base
            processed_image = self._preprocess_image(image)
            
            # Détection de langue si non spécifiée
            if language is None:
                language = self.detect_language(processed_image)
            
            # Extraction selon la langue
            config = self.tesseract_config.get(language, self.tesseract_config['mixed'])
            text = pytesseract.image_to_string(processed_image, config=config)
            
            # Extraction des informations structurées
            data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
            
            return {
                'text': text.strip(),
                'language': language,
                'confidence': self._calculate_confidence(data),
                'word_count': len(text.split()),
                'structured_data': data
            }
            
        except Exception as e:
            logger.error(f"Erreur extraction image {image_path}: {e}")
            return {'text': '', 'language': 'unknown', 'confidence': 0, 'word_count': 0}
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Préprocessing d'image pour améliorer l'OCR"""
        # Conversion en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Débruitage
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Seuillage adaptatif
        thresh = cv2.adaptiveThreshold(
            contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calcule la confiance moyenne de l'OCR"""
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        return sum(confidences) / len(confidences) if confidences else 0


class ImagePreprocessor:
    """Préprocesseur pour documents images français et arabes"""
    
    def __init__(self):
        self.extractor = ImageExtractor()
        
    def clean_french_text(self, text: str) -> str:
        """Nettoyage spécifique au français"""
        # Correction des caractères mal reconnus
        replacements = {
            'â': 'a', 'ä': 'a', 'à': 'a', 'á': 'a',
            'ê': 'e', 'ë': 'e', 'è': 'e', 'é': 'e',
            'î': 'i', 'ï': 'i', 'ì': 'i', 'í': 'i',
            'ô': 'o', 'ö': 'o', 'ò': 'o', 'ó': 'o',
            'û': 'u', 'ü': 'u', 'ù': 'u', 'ú': 'u',
            'ç': 'c', 'ñ': 'n'
        }
        
        cleaned_text = text
        # Supprimer caractères non-français/non-arabes
        cleaned_text = re.sub(r'[^\w\s\u0600-\u06FFÀ-ÿ.,;:!?()\-\'\"]+', ' ', cleaned_text)
        
        # Normaliser les espaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def clean_arabic_text(self, text: str) -> str:
        """Nettoyage spécifique à l'arabe"""
        # Normalisation des caractères arabes
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه')
        text = text.replace('ى', 'ي')
        
        # Supprimer les diacritiques
        text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
        
        # Nettoyer les caractères non-arabes (garder ponctuation de base)
        text = re.sub(r'[^\u0600-\u06FF\w\s.,;:!?()\-\'\"]+', ' ', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def normalize_text(self, text: str, language: str) -> str:
        """Normalise le texte selon la langue"""
        if language == 'arabic':
            return self.clean_arabic_text(text)
        elif language == 'french':
            return self.clean_french_text(text)
        else:
            # Traitement mixte
            text = self.clean_french_text(text)
            text = self.clean_arabic_text(text)
            return text
    
    def remove_noise(self, text: str) -> str:
        """Supprime le bruit commun dans l'OCR"""
        # Supprimer lignes très courtes (probablement du bruit)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 2]
        
        # Supprimer caractères isolés
        text = ' '.join(cleaned_lines)
        text = re.sub(r'\b\w\b', ' ', text)
        
        # Supprimer répétitions de caractères
        text = re.sub(r'(.)\1{3,}', r'\1', text)
        
        return text
    
    def process_image(self, image_path: str, language: Optional[str] = None) -> Dict:
        """Traite une image complète: extraction + préprocessing"""
        logger.info(f"Traitement de l'image: {image_path}")
        
        # Extraction
        extraction_result = self.extractor.extract_text_from_image(image_path, language)
        
        if not extraction_result['text']:
            return extraction_result
        
        # Préprocessing
        raw_text = extraction_result['text']
        detected_language = extraction_result['language']
        
        # Nettoyage selon la langue
        normalized_text = self.normalize_text(raw_text, detected_language)
        denoised_text = self.remove_noise(normalized_text)
        
        # Segmentation en phrases
        sentences = self._segment_sentences(denoised_text, detected_language)
        
        # Résultat final
        result = {
            'raw_text': raw_text,
            'processed_text': denoised_text,
            'language': detected_language,
            'confidence': extraction_result['confidence'],
            'word_count': len(denoised_text.split()),
            'sentence_count': len(sentences),
            'sentences': sentences,
            'metadata': {
                'source_type': 'image',
                'source_path': image_path,
                'processing_status': 'success' if denoised_text else 'failed'
            }
        }
        
        logger.info(f"Image traitée: {len(denoised_text)} caractères, {len(sentences)} phrases")
        return result
    
    def _segment_sentences(self, text: str, language: str) -> List[str]:
        """Segmente le texte en phrases selon la langue"""
        if language == 'arabic':
            # Segmentation pour l'arabe
            sentences = re.split(r'[.!?؟।]+', text)
        else:
            # Segmentation pour le français
            sentences = re.split(r'[.!?]+', text)
        
        # Nettoyer et filtrer
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        return sentences
    
    def batch_process(self, image_paths: List[str]) -> List[Dict]:
        """Traite plusieurs images en lot"""
        results = []
        for path in image_paths:
            try:
                result = self.process_image(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur traitement {path}: {e}")
                results.append({
                    'processed_text': '',
                    'language': 'unknown',
                    'metadata': {'source_path': path, 'processing_status': 'failed', 'error': str(e)}
                })
        return results
