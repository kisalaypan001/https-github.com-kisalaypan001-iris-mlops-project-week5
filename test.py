import os
import unittest
from typing import Any, List, Sequence

import joblib
import numpy as np
import pandas as pd


class TestIrisModel(unittest.TestCase):
    """
    Unit tests for the Iris classifier saved with joblib.

    Improvements made:
    - Model path configurable via MODEL_PATH env var (defaults to 'models/iris_model.joblib')
    - Better, clearer failure messages
    - Handles models that predict numeric labels by mapping via `model.classes_` when available
    - Parameterized-style tests using subTest for easy extension
    """

    model: Any = None
    model_path: str = os.getenv("MODEL_PATH", "models/iris_model.joblib")

    @classmethod
    def setUpClass(cls) -> None:
        """Load the model once for all tests; fail loudly if model not present or load fails."""
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{cls.model_path}'. "
                "Ensure the CI pipeline ran 'dvc pull' (or otherwise placed the model) before tests."
            )

        try:
            cls.model = joblib.load(cls.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from '{cls.model_path}': {exc}") from exc

    def _predict_to_label(self, X: pd.DataFrame) -> List[str]:
        """
        Run model.predict and convert predictions to string labels.

        If the model returns numeric labels but exposes `classes_`, map numeric predictions
        to the corresponding class name via that attribute. Otherwise return str(pred).
        """
        raw_pred = self.model.predict(X)
        # ensure numpy array for consistent handling
        raw = np.asarray(raw_pred)

        # if predictions are strings already, just cast to str and return
        if raw.dtype.kind in {"U", "S", "O"}:
            return [str(x) for x in raw]

        # If numeric and model has classes_, try mapping indices to class names
        classes_attr = getattr(self.model, "classes_", None)
        if classes_attr is not None:
            # classes_ may contain string class names (e.g., ['setosa', ...])
            # and raw might be numeric indices (0,1,2) or actual labels matching classes_.
            mapped = []
            for p in raw:
                # if p is an integer index and within range, map it
                if isinstance(p, (np.integer, int)) and 0 <= int(p) < len(classes_attr):
                    mapped.append(str(classes_attr[int(p)]))
                else:
                    # otherwise, try to find p inside classes_ or fall back to str(p)
                    try:
                        # If p itself matches an entry in classes_, use that
                        idx = list(classes_attr).index(p)
                        mapped.append(str(classes_attr[idx]))
                    except ValueError:
                        mapped.append(str(p))
            return mapped

        # Fallback: convert each prediction to string
        return [str(x) for x in raw]

    def _make_df(self, features: Sequence[float]) -> pd.DataFrame:
        """Utility to create DataFrame with expected column order."""
        columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        return pd.DataFrame([features], columns=columns)

    def test_model_is_loaded(self):
        """Sanity check: model object should be loaded and not None."""
        self.assertIsNotNone(self.model, "Expected the model to be loaded into TestIrisModel.model")

    def test_known_samples(self):
        """
        Evaluate a small set of known samples (species) using subTest so failures are isolated.

        Add new cases to the `cases` list to expand coverage.
        """
        cases = [
            # (features, expected_label)
            ([5.1, 3.5, 1.4, 0.2], "setosa"),
            ([5.9, 3.0, 4.2, 1.5], "versicolor"),
            ([6.9, 3.1, 5.4, 2.1], "virginica"),
        ]

        for features, expected in cases:
            with self.subTest(features=features, expected=expected):
                df = self._make_df(features)
                preds = self._predict_to_label(df)
                # ensure we got at least one prediction back
                self.assertGreaterEqual(
                    len(preds), 1, f"No prediction returned for input {features}"
                )
                predicted_label = preds[0]
                self.assertEqual(
                    predicted_label,
                    expected,
                    f"Input {features} -> predicted '{predicted_label}' but expected '{expected}'. "
                    "If your model uses numeric labels, ensure it exposes `classes_` or adapt this test."
                )


if __name__ == "__main__":
    # Show verbose output in CI logs
    unittest.main(verbosity=2)
