from prometheus_client import Counter, Histogram, Gauge, Info
import time
import functools

# Define metrics
REQUEST_COUNT = Counter('iris_api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('iris_api_request_duration_seconds', 'Request duration')
MODEL_PREDICTIONS = Counter('iris_model_predictions_total', 'Total model predictions', ['prediction'])
MODEL_ACCURACY = Gauge('iris_model_accuracy', 'Current model accuracy')
MODEL_INFO = Info('iris_model_info', 'Model information')

def track_requests(func):
    """Decorator to track API requests"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
            return result
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    return wrapper

def track_prediction(prediction_name):
    """Track individual predictions"""
    MODEL_PREDICTIONS.labels(prediction=prediction_name).inc()

def update_model_info(version, accuracy):
    """Update model information"""
    MODEL_INFO.info({
        'version': version,
        'algorithm': 'RandomForest',
        'features': '4',
        'classes': '3'
    })
    MODEL_ACCURACY.set(accuracy)