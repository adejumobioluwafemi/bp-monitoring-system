#!/usr/bin/env python3
"""
Model evaluation script for DVC pipeline
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_layer.data_connector import DataConnector, ObjectConnector
from ml.cross_validator import CrossValidator

def evaluate_model():
    """Evaluate trained model on test set"""
    
    # Load model and test data
    model = ObjectConnector.load_object('models_chpt/hypertension_model.pkl')
    test_data = DataConnector.load_data('data/processed/test_clean.csv')
    
    # Prepare features and target
    X_test = test_data.drop(columns=['Hypertension_Tests'])
    y_test = test_data['Hypertension_Tests']
    
    # Initialize cross-validator for evaluation
    cv = CrossValidator(pipeline=model)
    
    # Comprehensive evaluation
    metrics = cv.evaluate_model(
        X_test, y_test, 
        save_plots=True, 
        plot_path='reports/plots/'
    )
    
    # Save evaluation metrics
    evaluation_report = {
        "model_evaluation": metrics,
        "test_set_size": len(X_test),
        "positive_class_ratio": y_test.mean()
    }
    
    with open('reports/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print("✅ Model evaluation completed")

if __name__ == '__main__':
    evaluate_model()