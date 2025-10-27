# tests/test_model.py (Modified Logic)
import pytest
import pandas as pd
import mlflow.sklearn

# Set the tracking URI (must be the same as the one used for logging)
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

# Define the Model Registry URI for the latest Production model
# This automatically fetches the latest version marked as 'Production'
MODEL_URI = "models:/IRIS_Classifier_RF/Production" 

@pytest.fixture(scope="session")
def model():
    """Loads the Production model from MLflow Model Registry."""
    try:
        # Load the model directly from the registry
        loaded_model = mlflow.sklearn.load_model(MODEL_URI)
        return loaded_model
    except Exception as e:
        pytest.fail(f"Failed to load model from MLflow Registry at {MODEL_URI}: {e}")

# ... rest of your data and performance test functions remain the same
# e.g., test_model_performance(model, data)
