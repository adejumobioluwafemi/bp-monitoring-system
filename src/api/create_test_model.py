# src/api/create_test_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os

def create_test_model():
    """Create a simple test model if no models exist"""
    print("Creating test model...")
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.normal(45, 15, n_samples),
        'BMI': np.random.normal(25, 5, n_samples),
        'Systolic_BP': np.random.normal(120, 20, n_samples),
        'Diastolic_BP': np.random.normal(80, 10, n_samples),
        'Heart_Rate': np.random.normal(72, 12, n_samples),
        'Gender': np.random.choice([0, 1], n_samples),
        'Medical_History': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Smoking': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Sporting': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    }
    
    X = pd.DataFrame(data)
    
    # Create target based on some rules (simplified hypertension risk)
    y = (
        (X['Age'] > 50) |
        (X['BMI'] > 30) |
        (X['Systolic_BP'] > 140) |
        (X['Diastolic_BP'] > 90) |
        (X['Medical_History'] == 1)
    ).astype(int)
    
    # Create and train model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    model.fit(X, y)
    
    # Save model
    os.makedirs('models_chpt', exist_ok=True)
    with open('models_chpt/test_hypertension_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("✅ Test model created at models_chpt/test_hypertension_model.pkl")
    return model

if __name__ == "__main__":
    create_test_model()