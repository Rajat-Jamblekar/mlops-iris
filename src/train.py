import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import logging
from src.data_loader import IrisDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        
    def train_model(self, model_name, X_train, X_test, y_train, y_test, target_names):
        """Train a single model with MLflow tracking"""
        
        with mlflow.start_run(run_name=f"iris_{model_name}"):
            # Get model
            model = self.models[model_name]
            
            # Log parameters
            mlflow.log_params(model.get_params())
            
            # Train model
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Log classification report
            report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{class_name}_{metric_name}", value)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                model_name,
                registered_model_name=f"iris_{model_name}"
            )
            
            # Save model locally
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, f'models/{model_name}.pkl')
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
            
            return model, accuracy
    
    def train_all_models(self):
        """Train all models and return the best one"""
        # Load data
        loader = IrisDataLoader()
        data = loader.get_preprocessed_data()
        
        # Set MLflow experiment
        mlflow.set_experiment("iris_classification")
        
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        # Train all models
        for model_name in self.models.keys():
            model, accuracy = self.train_model(
                model_name,
                data['X_train'],
                data['X_test'],
                data['y_train'],
                data['y_test'],
                data['target_names']
            )
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
        
        logger.info(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # Save best model info
        with open('models/best_model_info.txt', 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Accuracy: {best_accuracy:.4f}\n")
        
        # Save scaler
        joblib.dump(data['scaler'], 'models/scaler.pkl')
        
        return best_model, best_model_name, best_accuracy

if __name__ == "__main__":
    trainer = ModelTrainer()
    best_model, best_model_name, best_accuracy = trainer.train_all_models()
    print(f"Training completed! Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")