#!/usr/bin/env python3
"""
Data validation script for DVC pipeline
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data_layer.data_connector import DataConnector

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy and pandas data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def convert_to_serializable(obj):
    """Recursively convert numpy/pandas types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

def validate_data():
    """Validate processed data and generate report"""
    
    try:
        # Load processed data
        print("📥 Loading processed data...")
        train_data = DataConnector.get_data('data/processed/train.csv')
        val_data = DataConnector.get_data('data/processed/val.csv') 
        test_data = DataConnector.get_data('data/processed/test.csv')
        
        print(f"   Train samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        print(f"   Test samples: {len(test_data)}")
        
        # Data quality checks - convert to Python native types immediately
        validation_report = {
            "data_validation": {
                "train_samples": int(len(train_data)),
                "val_samples": int(len(val_data)),
                "test_samples": int(len(test_data)),
                "total_samples": int(len(train_data) + len(val_data) + len(test_data)),
                "train_missing_values": int(train_data.isnull().sum().sum()),
                "val_missing_values": int(val_data.isnull().sum().sum()),
                "test_missing_values": int(test_data.isnull().sum().sum()),
                "class_balance_train": convert_to_serializable(
                    train_data['Hypertension_Tests'].value_counts().to_dict()
                ),
                "class_balance_val": convert_to_serializable(
                    val_data['Hypertension_Tests'].value_counts().to_dict()
                ),
                "class_balance_test": convert_to_serializable(
                    test_data['Hypertension_Tests'].value_counts().to_dict()
                ),
                "feature_summary": {
                    "numerical_features": convert_to_serializable(
                        train_data.select_dtypes(include=[np.number]).columns.tolist()
                    ),
                    "categorical_features": convert_to_serializable(
                        train_data.select_dtypes(include=['object']).columns.tolist()
                    )
                },
                "validation_timestamp": datetime.now().isoformat()
            }
        }
        
        # Save validation report with custom encoder
        print("💾 Saving validation report...")
        with open('reports/data_validation_report.json', 'w') as f:
            json.dump(validation_report, f, indent=2, cls=NumpyEncoder)
        
        # Clean data (remove any remaining nulls)
        print("🧹 Cleaning data...")
        train_clean = train_data.dropna()
        val_clean = val_data.dropna()
        test_clean = test_data.dropna()
        
        print(f"   After cleaning - Train: {len(train_clean)}, Val: {len(val_clean)}, Test: {len(test_clean)}")
        
        # Save cleaned data
        DataConnector.put_data(train_clean, 'data/processed/train_clean.csv')
        DataConnector.put_data(val_clean, 'data/processed/val_clean.csv')
        DataConnector.put_data(test_clean, 'data/processed/test_clean.csv')
        
        # Print summary
        print("\n✅ Data validation completed successfully!")
        print(f"   Original total: {validation_report['data_validation']['total_samples']}")
        print(f"   Cleaned total: {len(train_clean) + len(val_clean) + len(test_clean)}")
        print(f"   Removed due to missing values: {validation_report['data_validation']['total_samples'] - (len(train_clean) + len(val_clean) + len(test_clean))}")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please run the prepare_data stage first")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Column not found: {e}")
        print("Please check if 'Hypertension_Test' column exists in your data")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during data validation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    validate_data()