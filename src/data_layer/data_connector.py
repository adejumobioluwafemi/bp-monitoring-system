import pandas as pd
import joblib


class DataConnector:
    @staticmethod
    def get_data(filepath):
        """Read data from a source (e.g., CSV)."""
        return pd.read_csv(filepath)

    @staticmethod
    def put_data(data, filepath):
        """Save data to a destination."""
        data.to_csv(filepath, index=False)
