# tests/test_model.py
import os
import pytest
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn

# Use environment variable if set; otherwise fallback to a local store.
# In CI you should set the secret MLFLOW_TRACKING_URI; if not set, tests will skip.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
MODEL_URI = "models:/IRIS_Classifier_RF/Production"

@pytest.fixture(scope="session")
def mlflow_available():
    """Return True if MLflow tracking URI is reachable; otherwise False."""
    if not MLFLOW_TRACKING_URI:
        pytest.skip("MLFLOW_TRACKING_URI not provided; skipping MLflow model tests.")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # quick health check: try getting the root endpoint or experiments list
        # This will raise if server is not reachable.
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        # a lightweight call to confirm connectivity
        _ = client.list_experiments()
        return True
    except Exception as e:
        pytest.skip(f"MLflow server not reachable at {MLFLOW_TRACKING_URI}: {e}")

@pytest.fixture(scope="session")
def model(mlflow_available):
    """Load the Production model from MLflow Model Registry. Skips on failure."""
    try:
        loaded_model = mlflow.sklearn.load_model(MODEL_URI)
        return loaded_model
    except Exception as e:
        pytest.skip(f"Failed to load model from Model Registry ({MODEL_URI}): {e}")

def test_model_predicts_consistently(model):
    """Basic sanity check: model predicts a single sample correctly (shape and type)."""
    # Use a tiny sample from Iris dataset
    iris = load_iris()
    sample = iris.data[0].reshape(1, -1)
    preds = model.predict(sample)
    assert len(preds) == 1
    assert preds.dtype.kind in ("i", "u") or isinstance(preds[0], (int,))  # integer labels

def test_model_accuracy_threshold(model):
    """Optional: load a small test set and assert accuracy > 0.7 (example)."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    preds = model.predict(X)
    # compute accuracy simply
    acc = (preds == y).mean()
    assert acc >= 0.7, f"Model accuracy too low: {acc:.3f}"
