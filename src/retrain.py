import os
import pandas as pd
import logging
from datetime import datetime
from data_loader import IrisDataLoader
from train import ModelTrainer
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self, accuracy_threshold=0.95, data_drift_threshold=0.1):
        self.accuracy_threshold = accuracy_threshold
        self.data_drift_threshold = data_drift_threshold
        
    def check_model_performance(self):
        """Check if current model performance is below threshold"""
        try:
            # Load current model
            model = joblib.load('models/random_forest.pkl')
            scaler = joblib.load('models/scaler.pkl')
            
            # Load test data
            loader = IrisDataLoader()
            data = loader.get_preprocessed_data()
            
            # Test current model
            predictions = model.predict(data['X_test'])
            accuracy = (predictions == data['y_test']).mean()
            
            logger.info(f"Current model accuracy: {accuracy:.4f}")
            
            return accuracy < self.accuracy_threshold
            
        except Exception as e:
            logger.error(f"Error checking model performance: {str(e)}")
            return True  # Retrain if can't check
    
    def detect_data_drift(self, new_data_path):
        """Simple data drift detection"""
        try:
            # Load original training data statistics
            original_data = pd.read_csv('data/iris.csv')
            original_means = original_data[['sepal length (cm)', 'sepal width (cm)', 
                                         'petal length (cm)', 'petal width (cm)']].mean()
            
            # Load new data
            new_data = pd.read_csv(new_data_path)
            new_means = new_data[['sepal length (cm)', 'sepal width (cm)', 
                                'petal length (cm)', 'petal width (cm)']].mean()
            
            # Calculate drift
            drift = abs((new_means - original_means) / original_means).mean()
            
            logger.info(f"Data drift detected: {drift:.4f}")
            
            return drift > self.data_drift_threshold
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            return False
    
    def retrain_model(self):
        """Retrain the model with new data"""
        try:
            logger.info("Starting model retraining...")
            
            # Train new models
            trainer = ModelTrainer()
            best_model, best_model_name, best_accuracy = trainer.train_all_models()
            
            # Create backup of old model
            backup_dir = f"models/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir, exist_ok=True)
            
            if os.path.exists('models/random_forest.pkl'):
                os.rename('models/random_forest.pkl', f'{backup_dir}/random_forest.pkl')
                os.rename('models/scaler.pkl', f'{backup_dir}/scaler.pkl')
            
            logger.info(f"Model retrained successfully. New accuracy: {best_accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
            return False
    
    def should_retrain(self, new_data_path=None):
        """Determine if model should be retrained"""
        performance_issue = self.check_model_performance()
        
        data_drift = False
        if new_data_path and os.path.exists(new_data_path):
            data_drift = self.detect_data_drift(new_data_path)
        
        return performance_issue or data_drift

if __name__ == "__main__":
    retrainer = ModelRetrainer()
    if retrainer.should_retrain():
        retrainer.retrain_model()
    else:
        logger.info("No retraining needed")