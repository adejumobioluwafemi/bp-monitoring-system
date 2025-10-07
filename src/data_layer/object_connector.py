# src/data/data_connector.py
import pandas as pd
import joblib


class ObjectConnector:
    @staticmethod
    def get_object(filepath):
        """Load a model binary or transformer."""
        return joblib.load(filepath)

    @staticmethod
    def put_object(obj, filepath):
        """Save a model binary or transformer."""
        joblib.dump(obj, filepath)
