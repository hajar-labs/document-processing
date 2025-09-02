import re
import spacy
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from collections import Counter
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentAnalysis:
    """Classe pour structurer les résultats d'analyse"""
    type_document: str
    confidence_score: float
    language: str  # 'fr', 'ar', 'mixed'
    personnes: List[str]
    lieux: List[str] 
    dates: List[str]
    secteur: str
    mots_cles: List[str]
    organisations: List[str]
    numeros_reference: List[str]
    
class BilingualDocumentAnalyzer:
    """Analyseur intelligent de documents ministériels français et arabes"""
    
    def __init__(self):
        # Chargement des modèles spaCy
        try:
            self.nlp_fr = spacy.load("fr_core_news_sm")
        except OSError:
            logger.warning("Modèle français non trouvé. Installez avec: python -m spacy download fr_core_news_sm")
            self.nlp_fr = None
            
        # Dictionnaires de classification - Français
        self.types_documents_fr = {
            "cahier_charges": {
                "keywords": [
                    "cahier des charges", "appel d'offres", "marché public", "consultation",
                    "soumissionnaire", "candidat", "prestation", "fourniture", "travaux",
                    "spécifications techniques", "critères d'évaluation", "délai d'exécution"
                ],
                "patterns": [
                    r"cahier\s+des?\s+charges?",
                    r"appel\s+d[''']offres?",
                    r"marché\s+public",
                    r"consultation\s+publique"
                ]
            },
            "rapport": {
                "keywords": [
                    "rapport", "bilan", "évaluation", "analyse", "étude", "diagnostic",
                    "recommandations", "conclusions", "synthèse", "état des lieux",
                    "performance", "résultats", "indicateurs"
                ],
                "patterns": [
                    r"rapport\s+(annuel|mensuel|trimestriel|d[''']activité)",
                    r"bilan\s+d[''']activité",
                    r"évaluation\s+des?\s+performances?"
                ]
            },
            "arrete_decret": {
                "keywords": [
                    "arrêté", "décret", "ministre", "secrétaire d'état", "dispositions",
                    "article", "vu le", "considérant", "arrête", "décrète",
                    "journal officiel", "publication", "abrogé", "modifié"
                ],
                "patterns": [
                    r"arrêté\s+(ministériel|n°)",
                    r"décret\s+(n°|du)",
                    r"vu\s+le\s+décret",
                    r"journal\s+officiel"
                ]
            },
            "circulaire": {
                "keywords": [
                    "circulaire", "instruction", "directive", "note de service",
                    "application", "mise en œuvre", "procédure", "modalités"
                ],
                "patterns": [
                    r"circulaire\s+n°",
                    r"note\s+de\s+service",
                    r"instruction\s+ministérielle"
                ]
            },
            "contrat": {
                "keywords": [
                    "contrat", "convention", "accord", "partenariat", "engagement",
                    "parties", "obligations", "durée", "résiliation", "clause"
                ],
                "patterns": [
                    r"contrat\s+de",
                    r"convention\s+de",
                    r"accord\s+de\s+partenariat"
                ]
            }
        }
        
        # Dictionnaires de classification - Arabe
        self.types_documents_ar = {
            "cahier_charges": {
                "keywords": [
                    "دفتر التحملات", "طلب عروض", "صفقة عمومية", "استشارة",
                    "مترشح", "مقدم العرض", "خدمة", "توريد", "أشغال",
                    "المواصفات التقنية", "معايير التقييم", "آجال التنفيذ", "المناقصة"
                ],
                "patterns": [
                    r"دفتر\s+التحملات",
                    r"طلب\s+عروض",
                    r"صفقة\s+عمومية",
                    r"استشارة\s+عمومية"
                ]
            },
            "rapport": {
                "keywords": [
                    "تقرير", "حصيلة", "تقييم", "تحليل", "دراسة", "تشخيص",
                    "توصيات", "خلاصات", "ملخص", "وضعية", "أداء", "نتائج", "مؤشرات"
                ],
                "patterns": [
                    r"تقرير\s+(سنوي|شهري|فصلي|النشاط)",
                    r"حصيلة\s+النشاط",
                    r"تقييم\s+الأداء"
                ]
            },
            "arrete_decret": {
                "keywords": [
                    "قرار", "مرسوم", "وزير", "كاتب الدولة", "مقتضيات",
                    "مادة", "بناء على", "اعتبارا", "يقرر", "يرسم",
                    "الجريدة الرسمية", "نشر", "ملغى", "معدل"
                ],
                "patterns": [
                    r"قرار\s+(وزاري|رقم)",
                    r"مرسوم\s+(رقم|بتاريخ)",
                    r"بناء\s+على\s+المرسوم",
                    r"الجريدة\s+الرسمية"
                ]
            },
            "circulaire": {
                "keywords": [
                    "منشور", "تعليمة", "توجيه", "مذكرة خدمة",
                    "تطبيق", "تفعيل", "إجراء", "كيفيات"
                ],
                "patterns": [
                    r"منشور\s+رقم",
                    r"مذكرة\s+خدمة",
                    r"تعليمة\s+وزارية"
                ]
            },
            "contrat": {
                "keywords": [
                    "عقد", "اتفاقية", "اتفاق", "شراكة", "التزام",
                    "أطراف", "التزامات", "مدة", "فسخ", "بند"
                ],
                "patterns": [
                    r"عقد\s+",
                    r"اتفاقية\s+",
                    r"اتفاق\s+شراكة"
                ]
            }
        }
        
        # Secteurs - Français et Arabe
        self.secteurs_fr = {
            "transports_routiers": [
                "transport routier", "autoroute", "route nationale", "circulation routière",
                "permis de conduire", "véhicule", "camion", "bus", "autocar",
                "sécurité routière", "code de la route", "signalisation", "voiture", "NARSA"
            ],
            "logistique": [
                "logistique", "chaîne d'approvisionnement", "entreposage", "stockage",
                "distribution", "supply chain", "flux de marchandises", "plateforme logistique", "AMDL", "SNTL"
            ],
            "transport_aerien": [
                "transport aérien", "aviation", "aéroport", "compagnie aérienne",
                "avion", "vol", "navigation aérienne", "contrôle aérien",
                "sécurité aérienne", "aviation civile", "AIAC", "ONDA", "RAM"
            ],
            "marine_marchande": [
                "marine marchande", "transport maritime", "port", "navire", "bateau",
                "maritime", "shipping", "fret maritime", "armateur", "capitainerie", "ISEM"
            ],
            "transport_ferroviaire": [
                "transport ferroviaire", "chemin de fer", "train", "ONCF", "gare",
                "voie ferrée", "locomotive", "wagon", "rail", "infrastructure ferroviaire"
            ],
            "gouvernance": [
                "gouvernance", "politique publique", "administration", "ministère",
                "direction générale", "réforme", "stratégie", "pilotage", "coordination"
            ]
        }
        
        self.secteurs_ar = {
            "transports_routiers": [
                "النقل الطرقي", "الطريق السيار", "الطريق الوطنية", "حركة المرور",
                "رخصة السياقة", "مركبة", "شاحنة", "حافلة", "أوتوكار",
                "السلامة الطرقية", "مدونة السير", "الإشارات المرورية"
            ],
            "logistique": [
                "اللوجستيك", "سلسلة التموين", "التخزين", "المستودعات",
                "التوزيع", "تدفق البضائع", "المنصة اللوجستيكية"
            ],
            "transport_aerien": [
                "النقل الجوي", "الطيران", "مطار", "شركة طيران",
                "طائرة", "رحلة", "الملاحة الجوية", "المراقبة الجوية",
                "السلامة الجوية", "الطيران المدني"
            ],
            "marine_marchande": [
                "الأسطول التجاري", "النقل البحري", "ميناء", "سفينة", "باخرة",
                "بحري", "الشحن البحري", "مالك السفينة", "قبطانية الميناء"
            ],
            "transport_ferroviaire": [
                "النقل بالسكك الحديدية", "السكة الحديدية", "قطار", "المكتب الوطني للسكك الحديدية",
                "محطة", "خط السكة الحديدية", "قاطرة", "عربة", "سكة", "البنية التحتية للسكك الحديدية"
            ],
            "gouvernance": [
                "الحكامة", "السياسة العمومية", "الإدارة", "وزارة",
                "المديرية العامة", "إصلاح", "استراتيجية", "قيادة", "تنسيق"
            ]
        }
        
        # Patterns pour les entités - Français et Arabe
        self.patterns_fr = {
            "dates": [
                r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}\b",
                r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b",
                r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b",
                r"\b\d{4}\b"
            ],
            "references": [
                r"\bn°\s*\d+[\-\/]\d+",
                r"\b(arrêté|décret|circulaire)\s+n°\s*[\d\-\/A-Z]+",
                r"\bréférence\s*:\s*[\w\-\/]+",
                r"\b[A-Z]{2,}\s*\d+[\-\/]\d+"
            ]
        }
        
        self.patterns_ar = {
            "dates": [
                r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}\b",  # Format numérique
                r"\b\d{1,2}\s+(يناير|فبراير|مارس|أبريل|ماي|يونيو|يوليوز|غشت|شتنبر|أكتوبر|نونبر|دجنبر)\s+\d{4}\b",
                r"\b(يناير|فبراير|مارس|أبريل|ماي|يونيو|يوليوز|غشت|شتنبر|أكتوبر|نونبر|دجنبر)\s+\d{4}\b",
                r"\bسنة\s+\d{4}\b",
                r"\b\d{4}\s+هـ\b"  # Année hégirienne
            ],
            "references": [
                r"\bرقم\s*\d+[\-\/]\d+",
                r"\b(قرار|مرسوم|منشور)\s+رقم\s*[\d\-\/أ-ي]+",
                r"\bمرجع\s*:\s*[\w\-\/]+",
                r"\b[أ-ي]{2,}\s*\d+[\-\/]\d+"
            ]
        }
        
        # Lieux du Maroc en français et arabe
        self.lieux_maroc = {
            "fr": [
                r"\b(Rabat|Casablanca|Marrakech|Fès|Tanger|Agadir|Meknès|Oujda|Kenitra|Tétouan|Salé)\b",
                r"\b(région|province|préfecture)\s+de\s+\w+",
                r"\baéroport\s+(Mohammed V|de Rabat-Salé|de Marrakech)",
                r"\bport\s+de\s+(Casablanca|Tanger|Agadir)"
            ],
            "ar": [
                r"\b(الرباط|الدار البيضاء|مراكش|فاس|طنجة|أكادير|مكناس|وجدة|القنيطرة|تطوان|سلا)\b",
                r"\b(جهة|إقليم|عمالة)\s+\w+",
                r"\bmطار\s+(محمد الخامس|الرباط سلا|مراكش)",
                r"\bميناء\s+(الدار البيضاء|طنجة|أكادير)"
            ]
        }
    
    def detect_language(self, text: str) -> str:
        """Détection de la langue du document"""
        # Comptage des caractères arabes
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        # Comptage des caractères latins
        latin_chars = len(re.findall(r'[a-zA-ZàâäéèêëïîôöùûüçÀÂÄÉÈÊËÏÎÔÖÙÛÜÇ]', text))
        
        total_chars = arabic_chars + latin_chars
        
        if total_chars == 0:
            return "unknown"
        
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio > 0.6:
            return "ar"
        elif arabic_ratio < 0.2:
            return "fr"
        else:
            return "mixed"
    
    def extract_entities_with_spacy(self, text: str, language: str) -> Tuple[List[str], List[str], List[str]]:
        """Extraction d'entités avec spaCy (pour le français principalement)"""
        personnes, lieux, organisations = [], [], []
        
        if language in ["fr", "mixed"] and self.nlp_fr:
            doc = self.nlp_fr(text)
            for ent in doc.ents:
                if ent.label_ == "PER":
                    personnes.append(ent.text.strip())
                elif ent.label_ in ["LOC", "GPE"]:
                    lieux.append(ent.text.strip())
                elif ent.label_ == "ORG":
                    organisations.append(ent.text.strip())
        
        # Pour l'arabe, utilisation de patterns manuels (en l'absence de spaCy arabe)
        if language in ["ar", "mixed"]:
            # Extraction des noms propres arabes (patterns simples)
            arabic_names = re.findall(r'\b[أ-ي]{3,}\s+[أ-ي]{3,}\b', text)
            personnes.extend(arabic_names)
            
            # Extraction des organisations gouvernementales arabes
            org_patterns = [
                r'وزارة\s+[أ-ي\s]+',
                r'المديرية\s+العامة\s+[أ-ي\s]+',
                r'المكتب\s+الوطني\s+[أ-ي\s]+'
            ]
            for pattern in org_patterns:
                matches = re.findall(pattern, text)
                organisations.extend(matches)
        
        return personnes, lieux, organisations
    
    def extract_dates(self, text: str, language: str) -> List[str]:
        """Extraction des dates selon la langue"""
        dates = []
        
        if language in ["fr", "mixed"]:
            for pattern in self.patterns_fr["dates"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                dates.extend(matches)
        
        if language in ["ar", "mixed"]:
            for pattern in self.patterns_ar["dates"]:
                matches = re.findall(pattern, text)
                dates.extend(matches)
        
        return list(set([date.strip() for date in dates if len(date.strip()) > 3]))
    
    def extract_references(self, text: str, language: str) -> List[str]:
        """Extraction des numéros de référence selon la langue"""
        references = []
        
        if language in ["fr", "mixed"]:
            for pattern in self.patterns_fr["references"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                references.extend(matches)
        
        if language in ["ar", "mixed"]:
            for pattern in self.patterns_ar["references"]:
                matches = re.findall(pattern, text)
                references.extend(matches)
        
        return list(set(references))
    
    def extract_moroccan_locations(self, text: str, language: str) -> List[str]:
        """Extraction des lieux marocains selon la langue"""
        lieux = []
        
        if language in ["fr", "mixed"]:
            for pattern in self.lieux_maroc["fr"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                lieux.extend(matches)
        
        if language in ["ar", "mixed"]:
            for pattern in self.lieux_maroc["ar"]:
                matches = re.findall(pattern, text)
                lieux.extend(matches)
        
        return list(set(lieux))
    
    def classify_document_type(self, text: str, language: str) -> Tuple[str, float]:
        """Classification du type de document selon la langue"""
        text_lower = text.lower()
        scores = {}
        
        # Sélection du dictionnaire selon la langue
        if language == "ar":
            types_dict = self.types_documents_ar
        elif language == "mixed":
            # Fusion des deux dictionnaires pour documents mixtes
            types_dict = {}
            for doc_type in self.types_documents_fr:
                types_dict[doc_type] = {
                    "keywords": self.types_documents_fr[doc_type]["keywords"] + self.types_documents_ar[doc_type]["keywords"],
                    "patterns": self.types_documents_fr[doc_type]["patterns"] + self.types_documents_ar[doc_type]["patterns"]
                }
        else:  # français
            types_dict = self.types_documents_fr
        
        for doc_type, config in types_dict.items():
            score = 0
            
            # Score basé sur les mots-clés
            for keyword in config["keywords"]:
                if language == "ar" or language == "mixed":
                    count = text.count(keyword)  # Pas de lower() pour l'arabe
                else:
                    count = text_lower.count(keyword.lower())
                score += count * 2
            
            # Score basé sur les patterns regex
            for pattern in config["patterns"]:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 3
            
            scores[doc_type] = score
        
        if scores:
            best_type = max(scores, key=scores.get)
            max_score = scores[best_type]
            
            total_score = sum(scores.values())
            confidence = (max_score / total_score * 100) if total_score > 0 else 0
            
            return best_type, min(confidence, 100.0)
        
        return "document_general", 0.0
    
    def identify_sector(self, text: str, language: str) -> str:
        """Identification du secteur selon la langue"""
        text_lower = text.lower()
        sector_scores = {}
        
        # Sélection du dictionnaire des secteurs
        if language == "ar":
            secteurs_dict = self.secteurs_ar
        elif language == "mixed":
            secteurs_dict = {}
            for sector in self.secteurs_fr:
                secteurs_dict[sector] = self.secteurs_fr[sector] + self.secteurs_ar[sector]
        else:
            secteurs_dict = self.secteurs_fr
        
        for sector, keywords in secteurs_dict.items():
            score = 0
            for keyword in keywords:
                if language == "ar" or language == "mixed":
                    count = text.count(keyword)
                else:
                    count = text_lower.count(keyword.lower())
                score += count
            sector_scores[sector] = score
        
        if sector_scores:
            best_sector = max(sector_scores, key=sector_scores.get)
            if sector_scores[best_sector] > 0:
                return best_sector
        
        return "gouvernance"
    
    def extract_keywords(self, text: str, language: str, top_n: int = 10) -> List[str]:
        """Extraction des mots-clés selon la langue"""
        if language == "ar" or language == "mixed":
            # Pour l'arabe : extraction des mots de plus de 3 caractères arabes
            words = re.findall(r'[\u0600-\u06FF]{3,}', text)
            
            # Mots vides arabes basiques
            arabic_stop_words = {
                'في', 'من', 'إلى', 'على', 'عن', 'مع', 'كل', 'هذا', 'هذه',
                'ذلك', 'التي', 'الذي', 'وهي', 'وهو', 'كما', 'لكن', 'أن'
            }
            words = [word for word in words if word not in arabic_stop_words]
        
        if language == "fr" or language == "mixed":
            # Pour le français
            stop_words_fr = {
                'le', 'de', 'un', 'à', 'être', 'et', 'en', 'avoir', 'que', 'pour',
                'dans', 'ce', 'il', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout',
                'plus', 'par', 'grand', 'mais', 'si', 'ou', 'son', 'du', 'les', 'des'
            }
            
            fr_words = re.findall(r'\b[a-zA-Zàâäéèêëïîôöùûüç]{3,}\b', text.lower())
            fr_words = [word for word in fr_words if word not in stop_words_fr]
            
            if language == "mixed":
                words.extend(fr_words)
            else:
                words = fr_words
        
        # Comptage et sélection des plus fréquents
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(top_n)]
    
    def analyze_document(self, text: str) -> DocumentAnalysis:
        """Analyse complète d'un document bilingue"""
        logger.info("Début de l'analyse du document...")
        
        # Détection de la langue
        language = self.detect_language(text)
        logger.info(f"Langue détectée: {language}")
        
        # Classification du type de document
        doc_type, confidence = self.classify_document_type(text, language)
        logger.info(f"Type de document identifié: {doc_type} (confiance: {confidence:.1f}%)")
        
        # Extraction des entités
        personnes_spacy, lieux_spacy, organisations = self.extract_entities_with_spacy(text, language)
        
        # Extraction des lieux spécifiques au Maroc
        lieux_maroc = self.extract_moroccan_locations(text, language)
        
        # Combinaison des lieux
        lieux = list(set(lieux_spacy + lieux_maroc))
        
        # Autres extractions
        dates = self.extract_dates(text, language)
        references = self.extract_references(text, language)
        secteur = self.identify_sector(text, language)
        mots_cles = self.extract_keywords(text, language)
        
        # Création du résultat structuré
        analysis = DocumentAnalysis(
            type_document=doc_type,
            confidence_score=confidence,
            language=language,
            personnes=list(set(personnes_spacy)),
            lieux=lieux,
            dates=dates,
            secteur=secteur,
            mots_cles=mots_cles,
            organisations=list(set(organisations)),
            numeros_reference=references
        )
        
        logger.info("Analyse terminée avec succès")
        return analysis
    
    def export_analysis_json(self, analysis: DocumentAnalysis, filename: str = None) -> str:
        """Export de l'analyse en format JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analyse_document_{timestamp}.json"
        
        analysis_dict = asdict(analysis)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def print_analysis_report(self, analysis: DocumentAnalysis):
        """Affichage d'un rapport d'analyse formaté"""
        print("="*60)
        print("📄 RAPPORT D'ANALYSE DOCUMENTAIRE BILINGUE")
        print("="*60)
        
        # Mapping des langues pour l'affichage
        lang_display = {"fr": "Français", "ar": "العربية", "mixed": "Mixte (Fr/Ar)", "unknown": "Inconnue"}
        
        print(f"\n🌐 Langue: {lang_display.get(analysis.language, analysis.language)}")
        print(f"🏷️  Type de document: {analysis.type_document.replace('_', ' ').title()}")
        print(f"📊 Confiance: {analysis.confidence_score:.1f}%")
        print(f"🏢 Secteur: {analysis.secteur.replace('_', ' ').title()}")
        
        print(f"\n👥 Personnes identifiées ({len(analysis.personnes)}):")
        for personne in analysis.personnes[:5]:
            print(f"   • {personne}")
        
        print(f"\n📍 Lieux mentionnés ({len(analysis.lieux)}):")
        for lieu in analysis.lieux[:5]:
            print(f"   • {lieu}")
        
        print(f"\n📅 Dates trouvées ({len(analysis.dates)}):")
        for date in analysis.dates[:5]:
            print(f"   • {date}")
        
        print(f"\n🏛️  Organisations ({len(analysis.organisations)}):")
        for org in analysis.organisations[:5]:
            print(f"   • {org}")
        
        print(f"\n🔢 Références ({len(analysis.numeros_reference)}):")
        for ref in analysis.numeros_reference[:3]:
            print(f"   • {ref}")
        
        print(f"\n🔑 Mots-clés principaux:")
        for mot in analysis.mots_cles[:8]:
            print(f"   • {mot}")
        
        print("\n" + "="*60)


# Fonction utilitaire pour l'intégration dans votre pipeline
def integrate_with_extractors(text_extrait: str, analyzer: BilingualDocumentAnalyzer = None) -> Dict:
    """
    Fonction d'intégration avec vos extractors existants
    
    Args:
        text_extrait: Texte extrait par vos extractors
        analyzer: Instance de l'analyseur (optionnel)
    
    Returns:
        Dict: Résultats de l'analyse au format dictionnaire
    """
    if analyzer is None:
        analyzer = BilingualDocumentAnalyzer()
    
    # Analyse du document
    analysis = analyzer.analyze_document(text_extrait)
    
    # Conversion en dictionnaire pour faciliter l'intégration
    result = asdict(analysis)
    
    # Ajout de métadonnées utiles
    result['metadata'] = {
        'analysis_timestamp': datetime.now().isoformat(),
        'text_length': len(text_extrait),
        'entities_total': (len(analysis.personnes) + len(analysis.lieux) + 
                          len(analysis.organisations) + len(analysis.dates)),
        'processing_success': True
    }
    
    return result

# Classe utilitaire pour le batch processing
class BatchDocumentProcessor:
    """Traitement par lot de documents"""
    
    def __init__(self):
        self.analyzer = BilingualDocumentAnalyzer()
        self.results = []
    
    def process_documents(self, documents: List[Dict[str, str]]) -> List[Dict]:
        """
        Traite une liste de documents
        
        Args:
            documents: Liste de dictionnaires avec 'id', 'text', 'filename' (optionnel)
        
        Returns:
            Liste des analyses
        """
        results = []
        
        for i, doc in enumerate(documents):
            try:
                logger.info(f"Traitement du document {i+1}/{len(documents)}")
                
                analysis = self.analyzer.analyze_document(doc['text'])
                result = asdict(analysis)
                result['document_id'] = doc.get('id', f'doc_{i+1}')
                result['filename'] = doc.get('filename', f'document_{i+1}.txt')
                result['processing_status'] = 'success'
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement du document {i+1}: {e}")
                results.append({
                    'document_id': doc.get('id', f'doc_{i+1}'),
                    'filename': doc.get('filename', f'document_{i+1}.txt'),
                    'processing_status': 'error',
                    'error_message': str(e)
                })
        
        return results
    
    def export_batch_results(self, results: List[Dict], output_file: str = None) -> str:
        """Export des résultats de traitement par lot"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"batch_analysis_{timestamp}.json"
        
        # Calcul des statistiques
        total_docs = len(results)
        successful = len([r for r in results if r.get('processing_status') == 'success'])
        failed = total_docs - successful
        
        batch_summary = {
            'batch_info': {
                'total_documents': total_docs,
                'successful_analyses': successful,
                'failed_analyses': failed,
                'success_rate': (successful / total_docs * 100) if total_docs > 0 else 0,
                'processing_timestamp': datetime.now().isoformat()
            },
            'document_analyses': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Résultats du traitement par lot exportés vers: {output_file}")
        return output_file

    