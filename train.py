# train.py
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- MLFLOW CONFIGURATION ---
# IMPORTANT: Replace with your actual MLflow Tracking URI (e.g., your remote server)
# For local testing, ensure your MLflow server is running on http://127.0.0.1:5000
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_REGISTRY_NAME = "IRIS_Classifier_RF"

def train_and_log_model(X_train, X_test, y_train, y_test, params, run_name):
    """
    Trains a model with given hyperparameters, evaluates it, and logs everything to MLflow.
    """
    with mlflow.start_run(run_name=run_name) as run:
        # 1. LOG PARAMETERS
        mlflow.log_params(params)

        # Train the model
        clf = RandomForestClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate the model
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # 2. LOG METRICS
        mlflow.log_metric("accuracy", accuracy)

        # 3. LOG MODEL TO REGISTRY (Removing DVC dependency)
        # The model is now tracked and registered in MLflow
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME
        )
        print(f"Run '{run_name}' logged. Accuracy: {accuracy:.4f}")
        print(f"Model logged to MLflow Registry: {MODEL_REGISTRY_NAME}")
        return accuracy, run.info.run_id

def main():
    # --- SETUP AND DATA PREPARATION ---
    # Directories are still created, but models directory will be largely ignored by Git/DVC for the model file
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load and process data (Kept from original script)
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target.rename("species")], axis=1)
    df.to_csv("data/iris_local.csv", index=False)

    X = iris.data
    y = iris.target.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- MLFLOW EXPERIMENT SETUP ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("IRIS_MLflow_Hyperparameter_Tuning")

    # --- HYPERPARAMETER TUNING LOOP ---
    # Define the search space (Simulating a tuning process)
    hyperparameter_sets = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
    ]

    best_accuracy = 0.0
    best_run_id = None

    for i, params in enumerate(hyperparameter_sets):
        run_name = f"RF_Run_{i}_N{params['n_estimators']}_D{params['max_depth']}"
        accuracy, run_id = train_and_log_model(
            X_train, X_test, y_train, y_test, params, run_name
        )
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run_id
            
    print("\n--- TUNING SUMMARY ---")
    print(f"Tuning completed. Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Run ID: {best_run_id}")

if __name__ == "__main__":
    main()
