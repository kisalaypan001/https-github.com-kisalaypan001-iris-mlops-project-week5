# train.py
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def main():
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target.rename("species")], axis=1)
    df.to_csv("data/iris_local.csv", index=False)

    X = iris.data.values
    y = iris.target.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    model_path = "models/iris_model.joblib"
    joblib.dump(clf, model_path)

    acc = clf.score(X_test, y_test)
    pd.DataFrame({"metric": ["accuracy"], "value": [acc]}).to_csv("reports/metrics.csv", index=False)
    print(f"Saved model to {model_path} with test accuracy {acc:.4f}")

if __name__ == "__main__":
    main()
