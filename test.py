import unittest
import joblib
import pandas as pd
import os

class TestIrisModel(unittest.TestCase):

    model = None

    @classmethod
    def setUpClass(cls):
        """
        This method runs once before any tests.
        It loads the model from the file downloaded by DVC.
        """
        model_path = 'models/iris_model.joblib'
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Make sure 'dvc pull' ran successfully in the CI pipeline."
            )
        
        # Load the model
        cls.model = joblib.load(model_path)

    def test_model_is_loaded(self):
        """
        A simple sanity check to ensure the model was loaded.
        """
        self.assertIsNotNone(self.model, "Model object should not be None.")

    def test_setosa_prediction(self):
        """
        This is the main evaluation test.
        It predicts a known 'setosa' sample and asserts the prediction is correct.
        """
        # Create a sample DataFrame for a known 'setosa' flower
        # Features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        test_data = [[5.1, 3.5, 1.4, 0.2]]
        columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        test_sample = pd.DataFrame(test_data, columns=columns)
        
        known_species = 'setosa'
        
        # Get the model's prediction
        prediction = self.model.predict(test_sample)
        
        # Check the prediction
        self.assertEqual(prediction[0], known_species,
                         f"Model predicted {prediction[0]} but expected {known_species}")

    def test_versicolor_prediction(self):
        """
        A second evaluation test for 'versicolor'.
        """
        # Features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        test_data = [[5.9, 3.0, 4.2, 1.5]]
        columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        test_sample = pd.DataFrame(test_data, columns=columns)
        
        known_species = 'versicolor'
        
        # Get the model's prediction
        prediction = self.model.predict(test_sample)
        
        # Check the prediction
        self.assertEqual(prediction[0], known_species,
                         f"Model predicted {prediction[0]} but expected {known_species}")


if __name__ == '__main__':
    unittest.main()
