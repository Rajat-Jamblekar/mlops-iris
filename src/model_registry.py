import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.client = MlflowClient()
        
    def get_best_model(self, experiment_name="iris_classification"):
        """Get the best performing model from MLflow"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.error(f"Experiment {experiment_name} not found")
                return None
                
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC"],
                max_results=1
            )
            
            if runs.empty:
                logger.error("No runs found in experiment")
                return None
                
            best_run = runs.iloc[0]
            model_uri = f"runs:/{best_run.run_id}/random_forest"  # Adjust based on your model name
            
            logger.info(f"Loading best model from run: {best_run.run_id}")
            model = mlflow.sklearn.load_model(model_uri)
            
            return model, best_run
            
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            return None, None
    
    def register_best_model(self, model_name="iris_best_model"):
        """Register the best model in MLflow Model Registry"""
        try:
            model, run = self.get_best_model()
            if model is None:
                return None
                
            # Register model
            model_version = mlflow.register_model(
                f"runs:/{run.run_id}/random_forest",
                model_name
            )
            
            # Transition to Production
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            
            logger.info(f"Model {model_name} version {model_version.version} registered and moved to Production")
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            return None

if __name__ == "__main__":
    registry = ModelRegistry()
    model_version = registry.register_best_model()
    if model_version:
        print(f"Model registered successfully: {model_version.name} v{model_version.version}")