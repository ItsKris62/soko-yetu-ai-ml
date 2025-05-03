import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from datetime import datetime
import os
from app.utils.logger import logger

class MLflowTracker:
    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_name = "soko_yetu_ai"
        
        try:
            mlflow.create_experiment(self.experiment_name)
        except:
            pass
        mlflow.set_experiment(self.experiment_name)

    def log_training(self, model, model_type, params, metrics, artifacts=None):
        """Log model training session to MLflow"""
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                if model_type == "sklearn":
                    mlflow.sklearn.log_model(model, "model")
                elif model_type == "tensorflow":
                    mlflow.tensorflow.log_model(model, "model")
                
                # Log artifacts if provided
                if artifacts:
                    for artifact in artifacts:
                        mlflow.log_artifact(artifact)
                
                # Add tags
                mlflow.set_tag("training_date", datetime.now().isoformat())
                mlflow.set_tag("project", "Soko Yetu AI")
                
            logger.info(f"Training logged to MLflow: {params['model_name']}")
            return True
        except Exception as e:
            logger.error(f"MLflow logging failed: {str(e)}", exc_info=True)
            return False

    def log_prediction(self, model_name, input_data, prediction, run_id=None):
        """Log prediction details to MLflow"""
        try:
            with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
                mlflow.log_dict(input_data, "prediction_input.json")
                mlflow.log_dict(prediction, "prediction_output.json")
                mlflow.set_tag("prediction_time", datetime.now().isoformat())
            return True
        except Exception as e:
            logger.error(f"Prediction logging failed: {str(e)}")
            return False

    def get_model(self, model_name, stage="Production"):
        """Retrieve a model from MLflow model registry"""
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model {model_name} from MLflow")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {str(e)}")
            raise