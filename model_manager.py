# model_manager.py
"""
Gestionnaire de modèles optimisé avec lazy loading, cache intelligent
et gestion mémoire pour le système MTL
"""

import logging
import threading
import time
import gc
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path
import psutil
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
import pickle
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types de modèles supportés"""
    NER = "ner"
    SUMMARIZATION = "summarization"
    EMBEDDINGS = "embeddings"
    CHAT = "chat"
    OCR = "ocr"


@dataclass
class ModelConfig:
    """Configuration d'un modèle"""
    name: str
    model_path: str
    model_type: ModelType
    max_memory_mb: int = 1024
    lazy_load: bool = True
    cache_ttl_seconds: int = 3600
    quantization: bool = False
    device: str = "auto"


class ModelManager:
    """
    Gestionnaire centralisé des modèles avec optimisations mémoire
    et chargement à la demande
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models = {}  # Modèles chargés en mémoire
        self.model_configs = {}  # Configurations des modèles
        self.model_usage = {}  # Statistiques d'usage
        self.locks = {}  # Verrous pour thread safety
        
        # Configuration globale
        self.max_memory_usage_mb = self.config.get('max_memory_mb', 4096)
        self.enable_model_caching = self.config.get('enable_caching', True)
        self.cache_dir = Path(self.config.get('cache_dir', './model_cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        # Monitoring mémoire
        self.memory_check_interval = 60  # secondes
        self.memory_monitor_thread = None
        self.monitoring_active = False
        
        # Initialiser les modèles par défaut
        self._setup_default_models()
        
        # Démarrer le monitoring mémoire
        self._start_memory_monitoring()
    
    def _setup_default_models(self):
        """Configure les modèles par défaut pour MTL"""
        default_models = {
            'french_ner': ModelConfig(
                name='french_ner',
                model_path='Jean-Baptiste/camembert-ner',
                model_type=ModelType.NER,
                max_memory_mb=512,
                lazy_load=True,
                quantization=True
            ),
            'multilingual_embeddings': ModelConfig(
                name='multilingual_embeddings',
                model_path='sentence-transformers/distiluse-base-multilingual-cased',
                model_type=ModelType.EMBEDDINGS,
                max_memory_mb=256,
                lazy_load=True
            ),
            'summarization': ModelConfig(
                name='summarization',
                model_path='facebook/bart-large-cnn',
                model_type=ModelType.SUMMARIZATION,
                max_memory_mb=1024,
                lazy_load=True,
                cache_ttl_seconds=7200  # 2 heures
            ),
            'chat': ModelConfig(
                name='chat',
                model_path='microsoft/DialoGPT-small',  # Version plus légère
                model_type=ModelType.CHAT,
                max_memory_mb=512,
                lazy_load=True,
                quantization=True
            )
        }
        
        for model_id, config in default_models.items():
            self.register_model(model_id, config)
    
    def register_model(self, model_id: str, config: ModelConfig):
        """Enregistre un modèle dans le gestionnaire"""
        self.model_configs[model_id] = config
        self.model_usage[model_id] = {
            'load_count': 0,
            'last_used': None,
            'total_inference_time': 0.0,
            'memory_peak_mb': 0
        }
        self.locks[model_id] = threading.Lock()
        logger.info(f"Registered model: {model_id}")
    
    def _get_memory_usage(self) -> float:
        """Retourne l'usage mémoire actuel en MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _should_unload_model(self, model_id: str) -> bool:
        """Détermine si un modèle doit être déchargé"""
        if model_id not in self.models:
            return False
        
        usage = self.model_usage[model_id]
        config = self.model_configs[model_id]
        
        # Vérifier TTL du cache
        if usage['last_used']:
            time_since_use = time.time() - usage['last_used']
            if time_since_use > config.cache_ttl_seconds:
                return True
        
        # Vérifier pression mémoire
        current_memory = self._get_memory_usage()
        if current_memory > self.max_memory_usage_mb * 0.8:  # 80% du maximum
            return True
        
        return False
    
    def _cleanup_models(self):
        """Nettoie les modèles inutilisés"""
        models_to_unload = []
        
        for model_id in list(self.models.keys()):
            if self._should_unload_model(model_id):
                models_to_unload.append(model_id)
        
        for model_id in models_to_unload:
            self._unload_model(model_id)
        
        if models_to_unload:
            gc.collect()  # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _unload_model(self, model_id: str):
        """Décharge un modèle de la mémoire"""
        if model_id in self.models:
            with self.locks[model_id]:
                if model_id in self.models:
                    del self.models[model_id]
                    logger.info(f"Unloaded model: {model_id}")
    
    def _load_model(self, model_id: str) -> Any:
        """Charge un modèle en mémoire avec optimisations"""
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not registered")
        
        config = self.model_configs[model_id]
        
        # Vérifier le cache sur disque
        cache_path = self.cache_dir / f"{model_id}.pkl"
        if self.enable_model_caching and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    model = pickle.load(f)
                    logger.info(f"Loaded model from cache: {model_id}")
                    return model
            except Exception as e:
                logger.warning(f"Cache loading failed for {model_id}: {e}")
        
        # Charger le modèle selon son type
        memory_before = self._get_memory_usage()
        start_time = time.time()
        
        try:
            if config.model_type == ModelType.EMBEDDINGS:
                model = SentenceTransformer(config.model_path)
            elif config.model_type in [ModelType.NER, ModelType.SUMMARIZATION, ModelType.CHAT]:
                from transformers import pipeline
                
                # Configuration pour quantification
                model_kwargs = {}
                if config.quantization and torch.cuda.is_available():
                    model_kwargs['torch_dtype'] = torch.float16
                
                # Déterminer le device
                device = -1  # CPU par défaut
                if config.device == "auto" and torch.cuda.is_available():
                    device = 0
                elif config.device.startswith("cuda"):
                    device = int(config.device.split(":")[-1])
                
                # Créer le pipeline selon le type
                task_mapping = {
                    ModelType.NER: "ner",
                    ModelType.SUMMARIZATION: "summarization",
                    ModelType.CHAT: "text-generation"
                }
                
                model = pipeline(
                    task_mapping[config.model_type],
                    model=config.model_path,
                    device=device,
                    model_kwargs=model_kwargs
                )
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Mesurer l'usage mémoire
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            # Mettre à jour les statistiques
            self.model_usage[model_id].update({
                'load_count': self.model_usage[model_id]['load_count'] + 1,
                'last_used': time.time(),
                'memory_peak_mb': max(self.model_usage[model_id]['memory_peak_mb'], memory_used)
            })
            
            # Sauvegarder en cache
            if self.enable_model_caching:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(model, f)
                except Exception as e:
                    logger.warning(f"Cache saving failed for {model_id}: {e}")
            
            logger.info(f"Loaded model {model_id} in {time.time() - start_time:.2f}s, "
                       f"memory: {memory_used:.1f}MB")
            
            return model
            
        except Exception as e:
            logger.error(f"Model loading failed for {model_id}: {e}")
            raise
    
    def get_model(self, model_id: str) -> Any:
        """Récupère un modèle, le charge si nécessaire"""
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not registered")
        
        # Thread safety
        with self.locks[model_id]:
            # Vérifier si le modèle est déjà chargé
            if model_id in self.models:
                self.model_usage[model_id]['last_used'] = time.time()
                return self.models[model_id]
            
            # Nettoyer la mémoire si nécessaire
            current_memory = self._get_memory_usage()
            config = self.model_configs[model_id]
            
            if (current_memory + config.max_memory_mb > self.max_memory_usage_mb):
                self._cleanup_models()
            
            # Charger le modèle
            model = self._load_model(model_id)
            self.models[model_id] = model
            
            return model
    
    def _start_memory_monitoring(self):
        """Démarre le monitoring mémoire en arrière-plan"""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    current_memory = self._get_memory_usage()
                    
                    if current_memory > self.max_memory_usage_mb * 0.9:  # 90% du maximum
                        logger.warning(f"High memory usage: {current_memory:.1f}MB")
                        self._cleanup_models()
                    
                    time.sleep(self.memory_check_interval)
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
        
        self.monitoring_active = True
        self.memory_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.memory_monitor_thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire"""
        current_memory = self._get_memory_usage()
        
        return {
            'total_memory_mb': current_memory,
            'max_memory_mb': self.max_memory_usage_mb,
            'memory_usage_percent': (current_memory / self.max_memory_usage_mb) * 100,
            'loaded_models': list(self.models.keys()),
            'registered_models': len(self.model_configs),
            'model_usage': dict(self.model_usage),
            'cache_enabled': self.enable_model_caching
        }
    
    def preload_models(self, model_ids: list):
        """Précharge des modèles spécifiés"""
        for model_id in model_ids:
            try:
                self.get_model(model_id)
                logger.info(f"Preloaded model: {model_id}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_id}: {e}")
    
    def shutdown(self):
        """Arrête le gestionnaire et nettoie les ressources"""
        self.monitoring_active = False
        
        if self.memory_monitor_thread:
            self.memory_monitor_thread.join(timeout=5)
        
        # Décharger tous les modèles
        for model_id in list(self.models.keys()):
            self._unload_model(model_id)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model manager shutdown complete")


# Singleton global pour le gestionnaire de modèles
_model_manager_instance = None
_model_manager_lock = threading.Lock()


def get_model_manager(config: Optional[Dict[str, Any]] = None) -> ModelManager:
    """Récupère l'instance singleton du gestionnaire de modèles"""
    global _model_manager_instance
    
    if _model_manager_instance is None:
        with _model_manager_lock:
            if _model_manager_instance is None:
                _model_manager_instance = ModelManager(config)
    
    return _model_manager_instance


# Décorateur pour injection automatique de modèles
def inject_model(model_id: str):
    """Décorateur qui injecte automatiquement un modèle dans une fonction"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            model_manager = get_model_manager()
            model = model_manager.get_model(model_id)
            return func(*args, model=model, **kwargs)
        return wrapper
    return decorator


# Classes optimisées pour remplacer vos services existants
class OptimizedExtractionService:
    """Service d'extraction optimisé utilisant le gestionnaire de modèles"""
    
    def __init__(self):
        self.model_manager = get_model_manager()
    
    @inject_model('french_ner')
    def extract_entities(self, text: str, model=None):
        """Extraction d'entités avec modèle injecté automatiquement"""
        return model(text)
    
    @inject_model('multilingual_embeddings')
    def get_embeddings(self, texts: list, model=None):
        """Génération d'embeddings avec modèle injecté"""
        return model.encode(texts)


class OptimizedSummarizationService:
    """Service de résumé optimisé"""
    
    def __init__(self):
        self.model_manager = get_model_manager()
    
    @inject_model('summarization')
    async def generate_summary(self, text: str, model=None):
        """Génération de résumé optimisée"""
        return model(text, max_length=150, min_length=50, do_sample=False)


class OptimizedChatService:
    """Service de chat optimisé"""
    
    def __init__(self):
        self.model_manager = get_model_manager()
    
    @inject_model('chat')
    def generate_response(self, query: str, context: str, model=None):
        """Génération de réponse optimisée"""
        prompt = f"Contexte: {context}\nQuestion: {query}\nRéponse:"
        return model(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']


# Exemple d'usage et tests
if __name__ == "__main__":
    import asyncio
    
    async def test_optimized_system():
        # Initialiser avec configuration personnalisée
        config = {
            'max_memory_mb': 2048,
            'enable_caching': True,
            'cache_dir': './test_cache'
        }
        
        manager = get_model_manager(config)
        
        # Précharger les modèles critiques
        manager.preload_models(['multilingual_embeddings', 'french_ner'])
        
        # Tester les services optimisés
        extraction_service = OptimizedExtractionService()
        summarization_service = OptimizedSummarizationService()
        
        # Tests
        print("=== Test Extraction ===")
        entities = extraction_service.extract_entities(
            "Le ministre Ahmed Benali visitera Casablanca demain."
        )
        print(f"Entities: {entities}")
        
        print("\n=== Test Embeddings ===")
        embeddings = extraction_service.get_embeddings([
            "Transport maritime au Maroc",
            "Infrastructure routière"
        ])
        print(f"Embeddings shape: {embeddings.shape}")
        
        print("\n=== Statistiques ===")
        stats = manager.get_stats()
        print(f"Mémoire utilisée: {stats['memory_usage_percent']:.1f}%")
        print(f"Modèles chargés: {stats['loaded_models']}")
        
        # Nettoyer
        manager.shutdown()
    
    asyncio.run(test_optimized_system())