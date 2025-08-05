import pytest
import numpy as np
from src.data_loader import IrisDataLoader
from src.train import ModelTrainer
import joblib
import os

class TestModel:
    
    def test_data_loader(self):
        """Test data loading and preprocessing"""
        loader = IrisDataLoader()
        data = loader.get_preprocessed_data()
        
        assert data['X_train'].shape[0] > 0
        assert data['X_test'].shape[0] > 0
        assert data['X_train'].shape[1] == 4
        assert len(data['target_names']) == 3
        assert len(np.unique(data['y_train'])) <= 3
    
    def test_model_training(self):
        """Test model training"""
        trainer = ModelTrainer()
        best_model, best_model_name, best_accuracy = trainer.train_all_models()
        
        assert best_model is not None
        assert best_accuracy > 0.8  # Expect good accuracy on Iris
        assert best_model_name in trainer.models.keys()
    
    def test_model_prediction(self):
        """Test model prediction"""
        if os.path.exists('models/random_forest.pkl'):
            model = joblib.load('models/random_forest.pkl')
            scaler = joblib.load('models/scaler.pkl')
            
            # Test prediction
            test_features = np.array([[5.1, 3.5, 1.4, 0.2]])
            test_features_scaled = scaler.transform(test_features)
            prediction = model.predict(test_features_scaled)
            
            assert len(prediction) == 1
            assert prediction[0] in [0, 1, 2]