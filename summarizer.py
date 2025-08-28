from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List
import re
import asyncio
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

class SummarizationService:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Download the missing 'punkt_tab' resource
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')


        # Modèle pour les embeddings de phrases
        self.sentence_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Modèle T5 pour la génération de résumé abstrait
        self.abstractive_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.abstractive_tokenizer = T5Tokenizer.from_pretrained('t5-base')

        # Pipeline de résumé extractif
        self.extractive_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )

    async def generate_summary(
        self,
        text: str,
        summary_type: str = "hybrid",
        max_length: int = 150,
        num_sentences: int = 3
    ) -> Dict[str, Any]:
        """Génération de résumé selon la stratégie choisie"""

        if summary_type == "extractive":
            summary = await self._extractive_summary(text, num_sentences)
            method_used = "extractive"
        elif summary_type == "abstractive":
            summary = await self._abstractive_summary(text, max_length)
            method_used = "abstractive"
        else:  # hybrid
            extractive_sum = await self._extractive_summary(text, num_sentences)
            abstractive_sum = await self._abstractive_summary(text, max_length)

            # Sélection du meilleur résumé basé sur cohérence et couverture
            summary = self._select_best_summary(text, extractive_sum, abstractive_sum)
            method_used = "hybrid"

        # Calcul de métriques de qualité
        quality_score = self._compute_summary_quality(text, summary)

        return {
            "summary": summary,
            "method_used": method_used,
            "quality_score": quality_score,
            "compression_ratio": len(summary) / len(text),
            "original_length": len(text),
            "summary_length": len(summary)
        }

    async def _extractive_summary(self, text: str, num_sentences: int) -> str:
        """Résumé extractif basé sur TextRank optimisé"""
        sentences = self._segment_sentences(text)

        if len(sentences) <= num_sentences:
            return text

        # Calcul des embeddings pour chaque phrase
        sentence_embeddings = self.sentence_encoder.encode(sentences)

        # Construction de la matrice de similarité
        similarity_matrix = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(
                        sentence_embeddings[i].reshape(1, -1),
                        sentence_embeddings[j].reshape(1, -1)
                    )[0][0]

        # Application de TextRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, max_iter=50, tol=1e-4)

        # Sélection des meilleures phrases
        ranked_sentences = sorted(
            [(scores[i], i, sentences[i]) for i in range(len(sentences))],
            key=lambda x: x[0],
            reverse=True
        )

        # Maintenir l'ordre original des phrases sélectionnées
        selected_indices = sorted([sent[1] for sent in ranked_sentences[:num_sentences]])
        selected_sentences = [sentences[i] for i in selected_indices]

        return ' '.join(selected_sentences)

    async def _abstractive_summary(self, text: str, max_length: int) -> str:
        """Résumé abstractif avec T5"""
        # Préparation du texte pour T5
        input_text = f"summarize: {text}"

        # Tokenisation
        inputs = self.abstractive_tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,  # Limite T5
            truncation=True
        )

        # Génération du résumé
        summary_ids = self.abstractive_model.generate(
            inputs,
            max_length=max_length,
            min_length=max_length // 4,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        summary = self.abstractive_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary

    def _segment_sentences(self, text: str) -> List[str]:
        """Segmentation du texte en phrases"""
        # Nettoyage du texte
        cleaned_text = re.sub(r'\s+', ' ', text.strip())

        # Tokenisation en phrases
        sentences = sent_tokenize(cleaned_text)

        # Filtrage des phrases trop courtes
        sentences = [s for s in sentences if len(s.split()) > 3]

        return sentences

    def _select_best_summary(
        self,
        original_text: str,
        extractive_summary: str,
        abstractive_summary: str
    ) -> str:
        """Sélection du meilleur résumé entre extractif et abstractif"""

        # Calcul de la cohérence avec le texte original
        original_embedding = self.sentence_encoder.encode([original_text])
        extractive_embedding = self.sentence_encoder.encode([extractive_summary])
        abstractive_embedding = self.sentence_encoder.encode([abstractive_summary])

        extractive_similarity = cosine_similarity(
            original_embedding, extractive_embedding
        )[0][0]

        abstractive_similarity = cosine_similarity(
            original_embedding, abstractive_embedding
        )[0][0]

        # Facteurs de décision
        extractive_score = extractive_similarity * 0.7  # Poids pour la fidélité
        abstractive_score = abstractive_similarity * 0.6  # Poids pour la créativité

        # Bonus pour la longueur appropriée
        target_length = len(original_text) * 0.2  # 20% du texte original

        extractive_length_penalty = abs(len(extractive_summary) - target_length) / target_length
        abstractive_length_penalty = abs(len(abstractive_summary) - target_length) / target_length

        extractive_score -= extractive_length_penalty * 0.1
        abstractive_score -= abstractive_length_penalty * 0.1

        return extractive_summary if extractive_score > abstractive_score else abstractive_summary

    def _compute_summary_quality(self, original_text: str, summary: str) -> float:
        """Calcul d'un score de qualité du résumé"""

        # Métriques de base
        compression_ratio = len(summary) / len(original_text)

        # Cohérence sémantique
        original_embedding = self.sentence_encoder.encode([original_text])
        summary_embedding = self.sentence_encoder.encode([summary])
        semantic_similarity = cosine_similarity(original_embedding, summary_embedding)[0][0]

        # Score de diversité lexicale
        original_words = set(original_text.lower().split())
        summary_words = set(summary.lower().split())
        lexical_diversity = len(summary_words.intersection(original_words)) / len(summary_words) if summary_words else 0

        # Score composite (0-1)
        quality_score = (
            semantic_similarity * 0.5 +  # Cohérence sémantique
            lexical_diversity * 0.3 +     # Diversité lexicale
            min(1.0, 0.3 / compression_ratio) * 0.2  # Compression appropriée
        )

        return round(quality_score, 3)

    def get_model_info(self) -> Dict[str, Any]:
        """Informations sur les modèles utilisés"""
        return {
            "sentence_encoder": "paraphrase-multilingual-MiniLM-L12-v2",
            "abstractive_model": "t5-base",
            "extractive_pipeline": "facebook/bart-large-cnn",
            "supported_languages": ["fr", "ar", "en", "es", "it", "pt", "nl"],
            "max_input_length": 512,
            "default_summary_types": ["extractive", "abstractive", "hybrid"]
        }

# Exemple d'utilisation
async def main():
    # Initialisation du service
    summarizer = SummarizationService()

    # Texte d'exemple
    sample_text = """
    L'intelligence artificielle (IA) est une technologie révolutionnaire qui transforme notre monde.
    Elle permet aux machines d'apprendre, de raisonner et de prendre des décisions comme les humains.
    Les applications de l'IA sont vastes : de la reconnaissance vocale aux voitures autonomes,
    en passant par la médecine personnalisée et la finance. Cependant, cette technologie soulève
    aussi des questions éthiques importantes concernant l'emploi, la vie privée et la sécurité.
    Il est crucial de développer l'IA de manière responsable pour maximiser ses bénéfices
    tout en minimisant les risques potentiels pour la société.
    """

    # Test des différents types de résumé
    print("=== Test du Service de Résumé ===\n")

    for summary_type in ["extractive", "abstractive", "hybrid"]:
        print(f"--- Résumé {summary_type.upper()} ---")
        result = await summarizer.generate_summary(
            sample_text,
            summary_type=summary_type,
            num_sentences=2,
            max_length=100
        )

        print(f"Résumé: {result['summary']}")
        print(f"Méthode: {result['method_used']}")
        print(f"Score qualité: {result['quality_score']}")
        print(f"Taux compression: {result['compression_ratio']:.2f}")
        print(f"Longueur: {result['original_length']} → {result['summary_length']}")
        print()

if __name__ == "__main__":
    asyncio.run(main())