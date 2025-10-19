import os
import unittest
from typing import Any, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class TestIrisModel(unittest.TestCase):
    """
    
    this is for Unit tests for the Iris classifier saved with joblib.

    This test accepts models that predict:
    - string labels ('setosa', 'versicolor', 'virginica'), OR
    - numeric labels (0,1,2) — in which case we map them using sklearn's iris target_names.
    """

    model: Any = None
    model_path: str = os.getenv("MODEL_PATH", "models/iris_model.joblib")

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{cls.model_path}'. Ensure the CI pipeline ran 'dvc pull' or the model is present."
            )
        try:
            cls.model = joblib.load(cls.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from '{cls.model_path}': {exc}") from exc

        # canonical iris target names for mapping 0->'setosa', etc.
        cls.iris_target_names = list(load_iris().target_names)

    def _make_df(self, features: Sequence[float]) -> pd.DataFrame:
        columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        return pd.DataFrame([features], columns=columns)

    def _pred_to_name(self, raw_pred: np.ndarray) -> List[str]:
        """
        Convert raw predictions to species names (strings).

        - If predictions are strings, return them.
        - If numeric, map 0/1/2 using iris target names.
        - Otherwise fall back to str() on prediction.
        """
        arr = np.asarray(raw_pred)
        # string-like predictions
        if arr.dtype.kind in {"U", "S", "O"}:
            return [str(x) for x in arr]

        # numeric predictions -> map through iris target names if they look like 0/1/2
        mapped = []
        for p in arr:
            # try integer-like mapping
            try:
                p_int = int(p)
                if 0 <= p_int < len(self.iris_target_names):
                    mapped.append(self.iris_target_names[p_int])
                    continue
            except Exception:
                pass
            # fallback: try to use classes_ if available and map index->class (less common situation)
            classes_attr = getattr(self.model, "classes_", None)
            if classes_attr is not None:
                try:
                    # if prediction matches one of classes_, use that
                    classes_list = list(classes_attr)
                    if p in classes_list:
                        mapped.append(str(p))
                        continue
                except Exception:
                    pass
            # final fallback
            mapped.append(str(p))
        return mapped

    def test_model_is_loaded(self):
        self.assertIsNotNone(self.model, "Expected the model to be loaded into TestIrisModel.model")

    def test_known_samples(self):
        cases = [
            ([5.1, 3.5, 1.4, 0.2], "setosa"),
            ([5.9, 3.0, 4.2, 1.5], "versicolor"),
            ([6.9, 3.1, 5.4, 2.1], "virginica"),
        ]

        for features, expected in cases:
            with self.subTest(features=features, expected=expected):
                df = self._make_df(features)
                raw_pred = self.model.predict(df)
                # convert prediction(s) to names
                preds = self._pred_to_name(np.asarray(raw_pred))
                self.assertGreaterEqual(len(preds), 1, f"No prediction returned for input {features}")
                predicted_label = preds[0]
                self.assertEqual(
                    predicted_label,
                    expected,
                    f"Input {features} -> predicted '{predicted_label}' but expected '{expected}'. "
                    "This test accepts numeric outputs 0/1/2 mapped to iris names as a fallback."
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
