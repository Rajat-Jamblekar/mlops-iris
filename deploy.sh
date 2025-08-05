#!/bin/bash

# Deployment script for Iris MLOps Pipeline

set -e

echo "Starting deployment of Iris MLOps Pipeline..."

# Create necessary directories
mkdir -p logs models data

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Train models
echo "Training models..."
python src/data_loader.py
python src/train.py

# Build Docker image
echo "Building Docker image..."
docker build -t iris-api:latest .

# Run tests
echo "Running tests..."
pytest tests/ -v

# Start services
echo "Starting services..."
docker-compose up -d

# Health check
echo "Performing health check..."
sleep 10
curl -f http://localhost:8000/health || {
    echo "Health check failed!"
    exit 1
}

echo "Deployment completed successfully!"
echo "API is available at: http://localhost:8000"
echo "MLflow UI: http://localhost:5000"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"