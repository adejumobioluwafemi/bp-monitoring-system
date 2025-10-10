#!/usr/bin/env python3
"""
Data validation script for DVC pipeline
"""

from data_layer.data_connector import DataConnector
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def validate_data():
    """Validate processed data and generate report"""

    # Load processed data
    train_data = DataConnector.load_data('data/processed/train.csv')
    val_data = DataConnector.load_data('data/processed/val.csv')
    test_data = DataConnector.load_data('data/processed/test.csv')

    # Data quality checks
    validation_report = {
        "data_validation": {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "total_samples": len(train_data) + len(val_data) + len(test_data),
            "train_missing_values": train_data.isnull().sum().sum(),
            "val_missing_values": val_data.isnull().sum().sum(),
            "test_missing_values": test_data.isnull().sum().sum(),
            "class_balance_train": train_data['Hypertension_Tests'].value_counts().to_dict(),
            "class_balance_val": val_data['Hypertension_Tests'].value_counts().to_dict(),
            "class_balance_test": test_data['Hypertension_Tests'].value_counts().to_dict()
        }
    }

    # Save validation report
    with open('reports/data_validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)

    # Clean data (remove any remaining nulls)
    train_clean = train_data.dropna()
    val_clean = val_data.dropna()
    test_clean = test_data.dropna()

    # Save cleaned data
    DataConnector.save_data(train_clean, 'data/processed/train_clean.csv')
    DataConnector.save_data(val_clean, 'data/processed/val_clean.csv')
    DataConnector.save_data(test_clean, 'data/processed/test_clean.csv')

    print("✅ Data validation completed and cleaned data saved")


if __name__ == '__main__':
    validate_data()
