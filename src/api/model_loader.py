import pickle
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.expected_features = None
        self.project_root = Path(__file__).parent.parent.parent
        self.load_model()
        self.determine_expected_features()
    
    def get_absolute_path(self, relative_path):
        """Convert relative path to absolute path from project root"""
        return self.project_root / relative_path
    
    def load_model(self):
        """Load the model from the specified path"""
        # Try multiple possible model locations
        possible_model_paths = [
            "models/production/current/model/model.pkl",
            "models_chpt/hypertension_model_random_forest.pkl",
            "mlruns/2/models/m-*/artifacts/model.pkl"
        ]
        
        model_found = False
        for relative_path in possible_model_paths:
            model_path = self.get_absolute_path(relative_path)
            
            # Handle glob pattern
            if '*' in str(model_path):
                import glob
                matches = glob.glob(str(model_path))
                if matches:
                    model_path = Path(matches[0])
            
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    logger.info(f"✅ Model loaded successfully from: {model_path}")
                    self.model_type = type(self.model).__name__
                    logger.info(f"✅ Model type: {self.model_type}")
                    model_found = True
                    break
                except Exception as e:
                    logger.error(f"❌ Error loading model from {model_path}: {e}")
                    continue
        
        if not model_found:
            logger.error("❌ No model file found in any expected location")
            # Create a simple fallback model for testing
            self.create_fallback_model()
    
    def determine_expected_features(self):
        """Determine what features the model expects"""
        if self.model is None:
            self.expected_features = [
                'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
                'Gender', 'Medical_History', 'Smoking', 'Sporting'
            ]
            return
        
        # Try to get feature names from the model
        if hasattr(self.model, 'feature_names_in_'):
            self.expected_features = list(self.model.feature_names_in_)
            logger.info(f"🎯 Model expects features: {self.expected_features}")
        
        # For pipelines, check the final estimator
        elif hasattr(self.model, 'named_steps'):
            # Get the final estimator (classifier)
            final_step = None
            for step_name, step in self.model.named_steps.items():
                if hasattr(step, 'predict') or hasattr(step, 'predict_proba'):
                    final_step = step
                    break
            
            if final_step and hasattr(final_step, 'feature_names_in_'):
                self.expected_features = list(final_step.feature_names_in_)
                logger.info(f"🎯 Pipeline classifier expects features: {self.expected_features}")
            else:
                # Default to original features for pipelines
                self.expected_features = [
                    'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
                    'Gender', 'Medical_History', 'Smoking', 'Sporting'
                ]
                logger.info(f"🤔 Using default features for pipeline: {self.expected_features}")
        
        else:
            # Default feature set
            self.expected_features = [
                'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
                'Gender', 'Medical_History', 'Smoking', 'Sporting'
            ]
            logger.info(f"📝 Using default features: {self.expected_features}")
    
    def create_fallback_model(self):
        """Create a simple fallback model for testing"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            # simple modelfallback
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
            ])
            
            # Create some dummy training data
            X_dummy = pd.DataFrame({
                'Age': [45, 50, 35, 60, 40],
                'BMI': [25, 28, 22, 30, 26],
                'Systolic_BP': [120, 130, 115, 140, 125],
                'Diastolic_BP': [80, 85, 75, 90, 82],
                'Heart_Rate': [72, 75, 70, 80, 74],
                'Gender': [0, 1, 0, 1, 0],
                'Medical_History': [0, 1, 0, 1, 0],
                'Smoking': [0, 1, 0, 1, 0],
                'Sporting': [1, 0, 1, 0, 1]
            })
            y_dummy = [0, 1, 0, 1, 0]
            
            self.model.fit(X_dummy, y_dummy)
            self.model_type = "FallbackRandomForest"
            logger.info("✅ Created fallback model for testing")
            
        except Exception as e:
            logger.error(f"❌ Failed to create fallback model: {e}")
            self.model = None
    
    def preprocess_input(self, patient_data) -> pd.DataFrame:
        """
        Preprocess input data to match what the model expects
        Uses simple categorical encoding (0/1) without one-hot encoding
        """
        try:
            # Convert to dictionary and use model_dump for Pydantic v2
            input_dict = patient_data.model_dump()
            
            # Convert categorical variables to 0/1
            processed_dict = self.convert_categorical_input(input_dict)
            
            # Create DataFrame with expected features
            input_df = self.create_input_dataframe(processed_dict)
            
            logger.info(f"✅ Input prepared. Shape: {input_df.shape}")
            logger.info(f"✅ Input columns: {list(input_df.columns)}")
            logger.info(f"✅ Input values: {input_df.iloc[0].to_dict()}")
            
            return input_df # type: ignore
                
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            raise
    
    def convert_categorical_input(self, data_dict):
        """
        Convert human-readable categorical inputs to encoded values (0/1)
        """
        converted = data_dict.copy()
        
        # Convert categorical variables
        categorical_mappings = {
            'Gender': {'Female': 0, 'Male': 1},
            'Medical_History': {'No': 0, 'Yes': 1},
            'Smoking': {'No': 0, 'Yes': 1},
            'Sporting': {'No': 0, 'Yes': 1}
        }
        
        for field, mapping in categorical_mappings.items():
            if field in converted:
                value = converted[field]
                if value in mapping:
                    converted[field] = mapping[value]
                else:
                    # Default to 0 if value not recognized
                    converted[field] = 0
        
        return converted
    
    def create_input_dataframe(self, processed_data):
        """
        Create DataFrame from processed data with correct column order
        that matches what the model expects
        """
        # Use the expected features determined from the model
        feature_order = self.expected_features
        
        # Ensure all expected features are present
        for feature in feature_order:
            if feature not in processed_data:
                processed_data[feature] = 0  # Default value
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([processed_data])[feature_order]
        
        return df

# Initialize model loader
model_loader = ModelLoader()