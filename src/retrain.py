import os
import pandas as pd
import logging
from datetime import datetime
import hashlib
import joblib
import mlflow

from src.data_loader import IrisDataLoader
from src.train import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def file_hash(filepath):
    """Generate an MD5 hash for the file"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class ModelRetrainer:
    def __init__(self, accuracy_threshold=0.95, data_drift_threshold=0.1):
        self.accuracy_threshold = accuracy_threshold
        self.data_drift_threshold = data_drift_threshold

    def check_model_performance(self):
        """Check if current model performance is below threshold"""
        try:
            model = joblib.load('models/random_forest.pkl')
            scaler = joblib.load('models/scaler.pkl')

            loader = IrisDataLoader()
            data = loader.get_preprocessed_data()

            predictions = model.predict(data['X_test'])
            accuracy = (predictions == data['y_test']).mean()

            logger.info(f"Current model accuracy: {accuracy:.4f}")
            return accuracy < self.accuracy_threshold

        except Exception as e:
            logger.error(f"Error checking model performance: {str(e)}")
            return True  # Retrain if model or data can't be loaded

    def detect_data_drift(self, new_data_path):
        """Detect data drift between original and new data"""
        try:
            original_data = pd.read_csv('data/iris.csv')
            original_means = original_data[['sepal length (cm)', 'sepal width (cm)',
                                            'petal length (cm)', 'petal width (cm)']].mean()

            new_data = pd.read_csv(new_data_path)
            new_means = new_data[['sepal length (cm)', 'sepal width (cm)',
                                  'petal length (cm)', 'petal width (cm)']].mean()

            drift = abs((new_means - original_means) / original_means).mean()
            logger.info(f"Data drift detected: {drift:.4f}")

            return drift > self.data_drift_threshold

        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            return False

    def is_new_data_truly_new(self, new_data_path):
        """Check if new data was already used"""
        hash_file = 'data/last_used_data.hash'
        new_hash = file_hash(new_data_path)

        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                last_hash = f.read()
            if last_hash == new_hash:
                logger.info("New data file has already been used previously.")
                return False

        with open(hash_file, 'w') as f:
            f.write(new_hash)

        return True

    def retrain_model(self):
        """Retrain the model and backup the old one with MLflow logging"""
        try:
            logger.info("Starting model retraining...")

            # Set or create MLflow experiment
            mlflow.set_experiment("iris_classification")

            # Tag the run to mark it as a retraining
            # with mlflow.start_run(run_name="model_retraining",nested=True) as run:
            #     mlflow.set_tag("stage", "retrain")
            #     mlflow.set_tag("triggered_by", "retrain.py")
            #     mlflow.set_tag("retrain_timestamp", datetime.now().isoformat())

            # Train new models (already logs models + metrics)
            trainer = ModelTrainer()
            best_model, best_model_name, best_accuracy = trainer.train_all_models()

            # Backup current model before overwrite
            backup_dir = f"models/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir, exist_ok=True)

            if os.path.exists('models/random_forest.pkl'):
                os.rename('models/random_forest.pkl', f'{backup_dir}/random_forest.pkl')
            if os.path.exists('models/scaler.pkl'):
                os.rename('models/scaler.pkl', f'{backup_dir}/scaler.pkl')

            logger.info(f"Model retrained successfully. New accuracy: {best_accuracy:.4f}")
            mlflow.log_param("retrain_success", True)

            return True

        except Exception as e:
            logger.error(f"Error during retraining: {str(e)}")
            mlflow.log_param("retrain_success", False)
            return False


if __name__ == "__main__":
    retrainer = ModelRetrainer()
    NEW_DATA_FLAG_FILE = 'data/new_data_available.flag'
    NEW_DATA_CSV_PATH = 'data/new_iris_data.csv'

    if os.path.exists(NEW_DATA_FLAG_FILE):
        logger.info(f"'{NEW_DATA_FLAG_FILE}' detected. Checking new data...")

        if os.path.exists(NEW_DATA_CSV_PATH):
            if retrainer.is_new_data_truly_new(NEW_DATA_CSV_PATH):
                performance_issue = retrainer.check_model_performance()
                data_drift = retrainer.detect_data_drift(NEW_DATA_CSV_PATH)

                if performance_issue or data_drift:
                    retrainer.retrain_model()
                    os.remove(NEW_DATA_FLAG_FILE)
                    logger.info("Retraining complete.")
                else:
                    logger.info("No retraining needed. Model is performing well and no data drift detected.")
                    os.remove(NEW_DATA_FLAG_FILE)
            else:
                logger.info("New data already used before. No retraining needed.")
                os.remove(NEW_DATA_FLAG_FILE)
        else:
            logger.warning(f"'{NEW_DATA_CSV_PATH}' not found. Skipping retraining.")
    else:
        logger.info("No new data flag found. Checking model performance.")
        if retrainer.check_model_performance():
            retrainer.retrain_model()
        else:
            logger.info("Model is performing well. No retraining needed.")
