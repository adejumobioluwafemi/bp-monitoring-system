import sys
import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from ml.model import Model
from data_layer.object_connector import ObjectConnector
from data_layer.data_connector import DataConnector

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))


class Inference:
    """
    Application for making predictions with trained hypertension model
    """

    def __init__(self, model_path='models_chpt/hypertension_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model pipeline"""
        try:
            self.model = ObjectConnector.get_object(self.model_path)
            print("✅ Model loaded successfully")
        except FileNotFoundError:
            print(f"❌ Model file not found at {self.model_path}")
            print("Please run training first or check the model path")
            sys.exit(1)

    def get_feature_requirements(self):
        """
        Return the expected features and their types
        """
        return {
            'numerical_features': ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate'],
            'categorical_features': ['Gender', 'Medical_History', 'Smoking', 'Sporting'],
            'expected_dtypes': {
                'Age': 'numeric',
                'BMI': 'numeric',
                'Systolic_BP': 'numeric',
                'Diastolic_BP': 'numeric',
                'Heart_Rate': 'numeric',
                'Gender': 'categorical',
                'Medical_History': 'categorical',
                'Smoking': 'categorical',
                'Sporting': 'categorical'
            }
        }

    def prediction_single(self, input_data):
        """
        Make prediction for a single sample

        Args:
            input_data (dict): Dictionary containing feature values

        Returns:
            dict: Prediction results
        """

        try:
            input_df = pd.DataFrame([input_data])

            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0]

            if len(probability) == 2:
                confidence = probability[1] if prediction == 1 else probability[0]
            else:
                confidence = np.max(probability)

            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'probabilities': probability.tolist(),
                'status': 'success'
            }
        except Exception as e:
            return {
                'prediction': None,
                'confidence': 0.0,
                'probabilities': [],
                'status': f'error: {str(e)}'
            }

    def predict_batch(self, input_filepath):
        """
        Make predictions for a batch of samples from a file

        Args:
            input_filepath (str): Path to input CSV file

        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        try:
            data = DataConnector.get_data(input_filepath)
            print(f"📊 Loaded {len(data)} samples for prediction")

            predictions = self.model.predict(data)
            probabilities = self.model.predict_proba(data)

            results = data.copy()
            results['prediction'] = predictions
            results['confidence'] = np.max(probabilities, axis=1)

            for i in range(probabilities.shape[1]):
                results[f'probability_class_{i}'] = probabilities[:, i]
            return results

        except Exception as e:
            print(f"❌ Error during batch prediction: {str(e)}")
            return None

    def save_predictions(self, predictions, output_path):
        """
        Save prediction results to file

        Args:
            predictions (pd.DataFrame): Prediction results
            output_path (str): Path to save results        
        """
        try:
            DataConnector.put_data(predictions, output_path)
            print(f"✅ Predictions saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving predictions: {str(e)}")


def run_single_prediction():
    """

    """
    return


def run_batch_prediction(input_file, output_file=None):
    """

    """
    return


def main():
    """Main application entry point"""

    return


if __name__ == '__main__':
    main()
