# train.py
import os
import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from requests.exceptions import ConnectionError

# Config
MODEL_REGISTRY_NAME = "IRIS_Classifier_RF"
EXPERIMENT_NAME = "IRIS_MLflow_Hyperparameter_Tuning"

def get_mlflow_uri():
    """
    Prefer MLFLOW_TRACKING_URI from environment; otherwise use a file-backed store
    (good fallback for CI or local dev when no server is available).
    """
    return os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")


def init_mlflow():
    uri = get_mlflow_uri()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    try:
        # set_experiment will create the experiment if it doesn't exist
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        # If we cannot contact a remote server, fall back to local file store
        print(f"Warning: unable to set experiment at {uri}: {e}", file=sys.stderr)
        fallback = "file:///tmp/mlruns"
        if uri != fallback:
            print(f"Falling back to local MLflow store: {fallback}", file=sys.stderr)
            mlflow.set_tracking_uri(fallback)
            mlflow.set_experiment(EXPERIMENT_NAME)
        else:
            raise
    return mlflow.get_tracking_uri()


def train_and_log_model(X_train, X_test, y_train, y_test, params, run_name):
    """
    Trains a model with given hyperparameters, evaluates it, and logs everything to MLflow.
    Attempts to register the model name in the MLflow Model Registry if available.
    Returns (accuracy, run_id).
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params(params)

        # Train
        clf = RandomForestClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        predictions = clf.predict(X_test)
        accuracy = float(accuracy_score(y_test, predictions))

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model artifact
        # If the tracking server supports the model registry and registration succeeds,
        # the model will be registered under MODEL_REGISTRY_NAME.
        try:
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                registered_model_name=MODEL_REGISTRY_NAME,
            )
            registered = True
        except Exception as e:
            # If registration (or remote uploading) fails, still log the model artifact locally
            print(f"Warning: model registration/logging to remote failed: {e}", file=sys.stderr)
            # Fallback: log model as artifact without registry name (ensures artifact exists)
            try:
                mlflow.sklearn.log_model(sk_model=clf, artifact_path="model")
            except Exception as e2:
                # If even this fails (e.g., remote unreachable), save locally to keep a record
                local_model_path = os.path.join("models", f"{run.info.run_id}.pkl")
                joblib.dump(clf, local_model_path)
                print(f"Saved model locally to {local_model_path}", file=sys.stderr)
            registered = False

        print(f"Run '{run_name}' logged. Accuracy: {accuracy:.4f} (run_id={run.info.run_id})")
        return accuracy, run.info.run_id, registered


def promote_best_run_to_production(best_run_id):
    """
    If a registered model version exists for best_run_id, find it and transition to 'Production'.
    """
    try:
        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        # Search model versions that came from this run
        filter_str = f"run_id = '{best_run_id}'"
        versions = client.search_model_versions(filter_str)
        if not versions:
            print(f"No registered model versions found for run_id={best_run_id}. Skipping promotion.")
            return None
        # If multiple versions found, pick the first (should normally be one)
        mv = versions[0]
        name = mv.name
        version = mv.version
        print(f"Found registered model version {name} v{version} for run {best_run_id}")
        # Transition to Production
        client.transition_model_version_stage(
            name=name, version=version, stage="Production", archive_existing_versions=True
        )
        print(f"Model {name} v{version} promoted to Production.")
        return (name, version)
    except Exception as e:
        print(f"Warning: could not promote model to Production: {e}", file=sys.stderr)
        return None


def main():
    # prepare directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # load iris data
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target.rename("species")], axis=1)
    df.to_csv("data/iris_local.csv", index=False)

    X = iris.data
    y = iris.target.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # initialize mlflow (uses env var or fallback)
    uri = init_mlflow()
    print(f"MLflow tracking URI: {uri}")

    # hyperparameter search (example)
    hyperparameter_sets = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
    ]

    best_accuracy = -1.0
    best_run_id = None
    best_registered = False

    for i, params in enumerate(hyperparameter_sets):
        run_name = f"RF_Run_{i}_N{params['n_estimators']}_D{params['max_depth']}"
        accuracy, run_id, registered = train_and_log_model(
            X_train, X_test, y_train, y_test, params, run_name
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run_id
            best_registered = registered

    print("\n--- TUNING SUMMARY ---")
    print(f"Tuning completed. Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Run ID: {best_run_id}")

    # If the best run was registered, attempt to find its registered version and promote to Production
    if best_run_id and best_registered:
        promoted = promote_best_run_to_production(best_run_id)
        if promoted:
            name, version = promoted
            print(f"Promoted {name} version {version} to Production.")
        else:
            print("Promotion to Production was not completed (see warnings).")
    else:
        print("Best run not registered (or registration not supported). Skipping model promotion.")

    print("Done.")


if __name__ == "__main__":
    main()
