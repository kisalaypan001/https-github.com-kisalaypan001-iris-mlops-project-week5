# test.py
import os
import unittest
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "models/iris_model.joblib"
EVAL_DATA_PATH = "data/iris_eval.csv"  # DVC will import this in CI
MIN_ACCEPTABLE_ACC = 0.80


class TestIrisPipeline(unittest.TestCase):

    def test_eval_file_exists_and_valid(self):
        self.assertTrue(os.path.exists(EVAL_DATA_PATH), f"Eval file missing: {EVAL_DATA_PATH}")
        df = pd.read_csv(EVAL_DATA_PATH)
        self.assertFalse(df.empty, "Eval dataset is empty")

        has_expected = (
            set(["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]).issubset(df.columns)
            or set(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "species"]).issubset(df.columns)
        )
        self.assertTrue(has_expected, "Expected columns not found")

    def test_model_exists_and_predicts(self):
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model missing at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred = model.predict(sample)
        self.assertEqual(pred.shape[0], 1, "Model prediction shape mismatch")

    def test_eval_accuracy_threshold(self):
        self.assertTrue(os.path.exists(EVAL_DATA_PATH), f"Eval file missing: {EVAL_DATA_PATH}")
        df = pd.read_csv(EVAL_DATA_PATH)

        cols = df.columns
        if "sepal_length" in cols:
            X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
        else:
            X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
        y = df["species"].values

        split = int(len(X) * 0.8)
        X_test, y_test = X[split:], y[split:]

        model = joblib.load(MODEL_PATH)
        acc = (model.predict(X_test) == y_test).mean()
        self.assertGreaterEqual(acc, MIN_ACCEPTABLE_ACC, f"accuracy {acc:.3f} below threshold {MIN_ACCEPTABLE_ACC}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

