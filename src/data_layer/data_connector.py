import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class DataConnector:
    @staticmethod
    def get_data(filepath, **kwargs):
        """Load data from file (alias for load_data for compatibility)"""
        return DataConnector.load_data(filepath, **kwargs)
    
    @staticmethod
    def load_data(filepath, **kwargs):
        """Load data from CSV file with proper type handling"""
        try:
            data = pd.read_csv(filepath, **kwargs)
            print(f"✅ Loaded data from {filepath} - Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            raise
        except Exception as e:
            print(f"❌ Error loading data from {filepath}: {str(e)}")
            raise

    @staticmethod
    def save_data(data, filepath, **kwargs):
        """Save data to CSV file"""
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(filepath, index=False, **kwargs)
            print(f"✅ Saved data to {filepath} - Shape: {data.shape}")
        except Exception as e:
            print(f"❌ Error saving data to {filepath}: {str(e)}")
            raise
