# metrics_system.py
"""
Système de métriques et monitoring avancé pour le système MTL
avec alertes automatiques et tableau de bord temps réel
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta
import json
from pathlib import Path
import psutil
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types de métriques collectées"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Niveaux d'alerte"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Structure d'une métrique"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """Structure d'une alerte"""
    metric_name: str
    level: AlertLevel
    message: str
    threshold: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False


class MetricsCollector:
    """
    Collecteur de métriques centralisé pour le système MTL
    avec capacités d'alerting et de persistance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_store = defaultdict(lambda: deque(maxlen=1000))
        self.alerts_store = []
        self.alert_rules = {}
        self.running = False
        self.collection_thread = None
        
        # Configuration
        self.collection_interval = self.config.get('collection_interval', 30)  # secondes
        self.retention_days = self.config.get('retention_days', 30)
        self.enable_system_metrics = self.config.get('enable_system_metrics', True)
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # 5 minutes
        
        # Cache d'alertes pour éviter le spam
        self.alert_cache = {}
        
        # Métriques prédéfinies pour le système MTL
        self._setup_default_alert_rules()
        
        # Démarrer la collecte automatique
        self.start_collection()
    
    def _setup_default_alert_rules(self):
        """Configure les règles d'alerte par défaut"""
        self.alert_rules = {
            'extraction_failure_rate': {
                'threshold': 0.1,  # 10% de taux d'échec
                'level': AlertLevel.WARNING,
                'message': 'Taux d\'échec d\'extraction élevé: {actual:.2%}'
            },
            'response_time_p95': {
                'threshold': 5.0,  # 5 secondes
                'level': AlertLevel.WARNING,
                'message': 'Temps de réponse P95 élevé: {actual:.2f}s'
            },
            'memory_usage': {
                'threshold': 0.85,  # 85% de mémoire utilisée
                'level': AlertLevel.CRITICAL,
                'message': 'Utilisation mémoire critique: {actual:.1%}'
            },
            'cpu_usage': {
                'threshold': 0.90,  # 90% de CPU
                'level': AlertLevel.WARNING,
                'message': 'Utilisation CPU élevée: {actual:.1%}'
            },
            'disk_usage': {
                'threshold': 0.90,  # 90% d'espace disque
                'level': AlertLevel.CRITICAL,
                'message': 'Espace disque critique: {actual:.1%}'
            }
        }
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Enregistre une métrique"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        self.metrics_store[name].append(metric)
        
        # Vérifier les seuils d'alerte
        self._check_alert_rules(metric)
        
        logger.debug(f"Recorded metric: {name}={value} {unit}")
    
    def record_timer(self, name: str, start_time: float, tags: Optional[Dict[str, str]] = None):
        """Enregistre une métrique de temps"""
        duration = time.time() - start_time
        self.record_metric(name, duration, MetricType.TIMER, tags, "seconds")
        return duration
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Incrémente un compteur"""
        # Pour les compteurs, on additionne la valeur précédente
        current_metrics = list(self.metrics_store[name])
        current_value = current_metrics[-1].value if current_metrics else 0
        
        self.record_metric(name, current_value + value, MetricType.COUNTER, tags)
    
    def _check_alert_rules(self, metric: Metric):
        """Vérifie les règles d'alerte pour une métrique"""
        if metric.name not in self.alert_rules:
            return
        
        rule = self.alert_rules[metric.name]
        threshold = rule['threshold']
        level = rule['level']
        
        # Vérifier si le seuil est dépassé
        should_alert = False
        if metric.name.endswith('_rate') or metric.name.endswith('_usage'):
            should_alert = metric.value > threshold
        else:
            should_alert = metric.value > threshold
        
        if should_alert:
            # Vérifier le cooldown
            cache_key = f"{metric.name}_{level.value}"
            last_alert = self.alert_cache.get(cache_key)
            
            if (not last_alert or 
                (datetime.now() - last_alert).seconds > self.alert_cooldown):
                
                alert = Alert(
                    metric_name=metric.name,
                    level=level,
                    message=rule['message'].format(actual=metric.value),
                    threshold=threshold,
                    actual_value=metric.value,
                    timestamp=datetime.now()
                )
                
                self.alerts_store.append(alert)
                self.alert_cache[cache_key] = datetime.now()
                
                logger.warning(f"ALERT [{level.value.upper()}]: {alert.message}")
                
                # Déclencher les callbacks d'alerte si configurés
                self._trigger_alert_callbacks(alert)
    
    def _trigger_alert_callbacks(self, alert: Alert):
        """Déclenche les callbacks d'alerte configurés"""
        callbacks = self.config.get('alert_callbacks', [])
        for callback in callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def collect_system_metrics(self):
        """Collecte les métriques système"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('cpu_usage', cpu_percent / 100, unit="%")
            
            # Mémoire
            memory = psutil.virtual_memory()
            self.record_metric('memory_usage', memory.percent / 100, unit="%")
            self.record_metric('memory_available', memory.available, unit="bytes")
            
            # Disque
            disk = psutil.disk_usage('/')
            self.record_metric('disk_usage', disk.percent / 100, unit="%")
            self.record_metric('disk_free', disk.free, unit="bytes")
            
            # Processus
            process = psutil.Process()
            self.record_metric('process_memory', process.memory_info().rss, unit="bytes")
            self.record_metric('process_cpu', process.cpu_percent(), unit="%")
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def _collection_loop(self):
        """Boucle de collecte des métriques système"""
        while self.running:
            try:
                if self.enable_system_metrics:
                    self.collect_system_metrics()
                
                # Nettoyer les anciennes métriques
                self._cleanup_old_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _cleanup_old_metrics(self):
        """Nettoie les métriques anciennes selon la rétention configurée"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for metric_name, metrics in self.metrics_store.items():
            # Filtrer les métriques récentes
            recent_metrics = deque([m for m in metrics if m.timestamp > cutoff_date], 
                                 maxlen=metrics.maxlen)
            self.metrics_store[metric_name] = recent_metrics
    
    def start_collection(self):
        """Démarre la collecte automatique des métriques"""
        if not self.running:
            self.running = True
            self.collection_thread = threading.Thread(target=self._collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Arrête la collecte des métriques"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def get_metric_summary(self, name: str, period_minutes: int = 60) -> Dict[str, Any]:
        """Obtient un résumé statistique d'une métrique"""
        if name not in self.metrics_store:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
        recent_metrics = [
            m for m in self.metrics_store[name] 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'name': name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'last_value': values[-1],
            'unit': recent_metrics[-1].unit,
            'period_minutes': period_minutes
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Génère les données pour le tableau de bord"""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'alerts': [],
            'system_health': 'healthy'
        }
        
        # Métriques principales
        key_metrics = [
            'cpu_usage', 'memory_usage', 'disk_usage',
            'extraction_success_rate', 'response_time_p95',
            'active_users', 'documents_processed'
        ]
        
        for metric_name in key_metrics:
            summary = self.get_metric_summary(metric_name, 30)  # Dernières 30 minutes
            if summary:
                dashboard['metrics'][metric_name] = summary
        
        # Alertes actives (non résolues)
        active_alerts = [
            {
                'metric': alert.metric_name,
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'threshold': alert.threshold,
                'actual': alert.actual_value
            }
            for alert in self.alerts_store[-20:]  # 20 dernières alertes
            if not alert.resolved
        ]
        
        dashboard['alerts'] = active_alerts
        
        # Déterminer l'état global du système
        critical_alerts = [a for a in active_alerts if a['level'] == 'critical']
        warning_alerts = [a for a in active_alerts if a['level'] == 'warning']
        
        if critical_alerts:
            dashboard['system_health'] = 'critical'
        elif warning_alerts:
            dashboard['system_health'] = 'warning'
        
        return dashboard
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Exporte les métriques vers un fichier"""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics': {}
            }
            
            for metric_name, metrics in self.metrics_store.items():
                data['metrics'][metric_name] = [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'value': m.value,
                        'tags': m.tags,
                        'unit': m.unit
                    }
                    for m in metrics
                ]
            
            with open(filepath, 'w') as f:
                if format == 'json':
                    json.dump(data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            raise


# Décorateur pour mesurer automatiquement les performances
def measure_performance(metric_name: str, collector: MetricsCollector):
    """Décorateur pour mesurer automatiquement la performance des fonctions"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Enregistrer le succès
                collector.record_timer(f"{metric_name}_duration", start_time)
                collector.increment_counter(f"{metric_name}_success")
                
                return result
                
            except Exception as e:
                # Enregistrer l'échec
                collector.record_timer(f"{metric_name}_duration", start_time)
                collector.increment_counter(f"{metric_name}_failure")
                raise
        
        return wrapper
    return decorator


# Context manager pour mesurer des blocs de code
class MetricsContext:
    """Context manager pour mesurer des blocs de code"""
    
    def __init__(self, metric_name: str, collector: MetricsCollector):
        self.metric_name = metric_name
        self.collector = collector
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.collector.record_timer(self.metric_name, self.start_time)
        
        if exc_type is None:
            self.collector.increment_counter(f"{self.metric_name}_success")
        else:
            self.collector.increment_counter(f"{self.metric_name}_failure")
        
        return False  # Ne pas supprimer les exceptions


# Exemple d'intégration avec le système MTL
class MTLMetricsIntegration:
    """Intégration des métriques avec le système MTL"""
    
    def __init__(self):
        self.collector = MetricsCollector({
            'collection_interval': 30,
            'enable_system_metrics': True,
            'alert_callbacks': [self._send_email_alert, self._log_alert]
        })
    
    def _send_email_alert(self, alert: Alert):
        """Envoie une alerte par email (à implémenter)"""
        logger.info(f"Would send email alert: {alert.message}")
    
    def _log_alert(self, alert: Alert):
        """Log l'alerte dans les fichiers de log"""
        logger.warning(f"ALERT: {alert.message}")
    
    @measure_performance("document_extraction", collector)
    def extract_document(self, file_path: str):
        """Exemple d'intégration avec l'extraction de documents"""
        # Logique d'extraction ici
        pass
    
    def track_user_activity(self, user_id: str, action: str):
        """Suit l'activité des utilisateurs"""
        self.collector.increment_counter('user_actions_total', 
                                        tags={'user_id': user_id, 'action': action})
        
        # Utilisateurs actifs uniques (approximation)
        self.collector.record_metric('active_users', 1, 
                                   tags={'user_id': user_id})


# Exemple d'usage
if __name__ == "__main__":
    # Initialisation
    metrics = MTLMetricsIntegration()
    
    # Simulation d'activité
    with MetricsContext("test_operation", metrics.collector):
        time.sleep(0.1)  # Simule une opération
    
    # Attendre et afficher le tableau de bord
    time.sleep(2)
    dashboard = metrics.collector.get_dashboard_data()
    print(json.dumps(dashboard, indent=2))