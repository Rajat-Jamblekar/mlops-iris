import pytest
from fastapi.testclient import TestClient
from api.app import app
import json

client = TestClient(app)

class TestAPI:
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in ["healthy", "unhealthy"]
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        test_features = [5.1, 3.5, 1.4, 0.2]
        response = client.post(
            "/predict",
            json={"features": test_features}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "prediction_name" in data
            assert "confidence" in data
            assert data["prediction"] in [0, 1, 2]
            assert data["prediction_name"] in ["setosa", "versicolor", "virginica"]
            assert len(data["confidence"]) == 3
    
    def test_predict_invalid_input(self):
        """Test prediction with invalid input"""
        # Too few features
        response = client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4]}
        )
        assert response.status_code == 422
        
        # Too many features
        response = client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2, 1.0]}
        )
        assert response.status_code == 422
        
        # Non-numeric features
        response = client.post(
            "/predict",
            json={"features": ["a", "b", "c", "d"]}
        )
        assert response.status_code == 422
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]