from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import logging
import os
import sqlite3
from datetime import datetime
import json
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import uvicorn

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions')
PREDICTION_HISTOGRAM = Histogram('prediction_duration_seconds', 'Prediction request duration')

app = FastAPI(
    title="Iris Classification API",
    description="ML API for Iris flower classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None
target_names = ['setosa', 'versicolor', 'virginica']

def init_database():
    """Initialize SQLite database for logging"""
    conn = sqlite3.connect('logs/predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            features TEXT,
            prediction INTEGER,
            prediction_name TEXT,
            confidence TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_models():
    """Load trained model and scaler"""
    global model, scaler
    try:
        model = joblib.load('models/random_forest.pkl')
        scaler = joblib.load('models/scaler.pkl')
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def log_prediction(features, prediction, prediction_name, confidence):
    """Log prediction to database"""
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (timestamp, features, prediction, prediction_name, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(features),
            prediction,
            prediction_name,
            json.dumps(confidence.tolist())
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    os.makedirs('logs', exist_ok=True)
    init_database()
    load_models()

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on iris data"""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with PREDICTION_HISTOGRAM.time():
            # Preprocess features
            features_array = np.array(request.features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            confidence = model.predict_proba(features_scaled)[0]
            
            prediction_name = target_names[prediction]
            
            # Log prediction
            log_prediction(request.features, int(prediction), prediction_name, confidence)
            
            # Update metrics
            PREDICTION_COUNTER.inc()
            
            logger.info(f"Prediction made: {prediction_name} with confidence {max(confidence):.4f}")
            
            return PredictionResponse(
                prediction=int(prediction),
                prediction_name=prediction_name,
                confidence=confidence.tolist(),
                model_version="1.0.0"
            )
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/predictions/history")
async def get_prediction_history(limit: int = 10):
    """Get recent prediction history"""
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'timestamp': row[1],
                'features': json.loads(row[2]),
                'prediction': row[3],
                'prediction_name': row[4],
                'confidence': json.loads(row[5])
            })
        
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching history")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)