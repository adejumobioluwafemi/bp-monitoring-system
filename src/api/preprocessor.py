# src/api/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import yaml
import pickle
import os

class InferencePreprocessor:
    """
    Handles preprocessing for inference that matches the training preprocessing
    """
    
    def __init__(self, params_path='params.yaml'):
        self.params_path = params_path
        self.preprocessor = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.load_preprocessing_config()
    
    def load_preprocessing_config(self):
        """Load preprocessing configuration from params.yaml"""
        try:
            if os.path.exists(self.params_path):
                with open(self.params_path, 'r') as f:
                    params = yaml.safe_load(f)
                
                # Get feature columns
                prepare_params = params.get('prepare', {})
                self.numerical_cols = prepare_params.get('numerical_features', 
                    ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate'])
                self.categorical_cols = prepare_params.get('categorical_features',
                    ['Gender', 'Medical_History', 'Smoking', 'Sporting'])
                
                # Get preprocessing parameters
                preprocessing_params = params.get('preprocessing', {})
                self.numerical_params = preprocessing_params.get('numerical', {})
                self.categorical_params = preprocessing_params.get('categorical', {})
                
                print(f"✅ Loaded preprocessing config from {self.params_path}")
                
            else:
                print(f"⚠️ params.yaml not found at {self.params_path}, using defaults")
                self.set_default_config()
            
        except Exception as e:
            print(f"⚠️ Could not load params.yaml: {e}. Using defaults.")
            self.set_default_config()
    
    def set_default_config(self):
        """Set default configuration if params.yaml is not available"""
        self.numerical_cols = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate']
        self.categorical_cols = ['Gender', 'Medical_History', 'Smoking', 'Sporting']
        self.numerical_params = {'imputer': 'median', 'scaler': 'standard'}
        self.categorical_params = {'imputer': 'most_frequent', 'encoder': 'onehot', 'handle_unknown': 'ignore'}
        print("✅ Using default preprocessing configuration")
    
    def create_preprocessor(self):
        """Create the preprocessing pipeline matching training"""
        # Numerical transformer
        numerical_imputer = SimpleImputer(strategy=self.numerical_params.get('imputer', 'median'))
        
        scaler_type = self.numerical_params.get('scaler', 'standard')
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', numerical_imputer),
            ('scaler', scaler)
        ])
        
        # Categorical transformer
        categorical_imputer = SimpleImputer(strategy=self.categorical_params.get('imputer', 'most_frequent'))
        categorical_encoder = OneHotEncoder(
            handle_unknown=self.categorical_params.get('handle_unknown', 'ignore'),
            sparse_output=False
        )
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('encoder', categorical_encoder)
        ])
        
        # Column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        
        print(f"✅ Created preprocessor with {len(self.numerical_cols)} numerical and {len(self.categorical_cols)} categorical features")
        return self.preprocessor
    
    def fit_preprocessor(self, X_train):
        """Fit the preprocessor with training data"""
        if self.preprocessor is None:
            self.create_preprocessor()
        
        self.preprocessor.fit(X_train)
        print("✅ Preprocessor fitted with training data")
        return self.preprocessor
    
    def transform_input(self, input_data):
        """
        Transform input data using the fitted preprocessor
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_preprocessor first.")
        
        return self.preprocessor.transform(input_data)
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            return None
        
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                # Get one-hot encoded feature names
                encoder = transformer.named_steps['encoder']
                encoded_features = encoder.get_feature_names_out(columns)
                feature_names.extend(encoded_features)
        
        return feature_names

def convert_categorical_input(data_dict):
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

def create_input_dataframe(processed_data, feature_order=None):
    """
    Create DataFrame from processed data with correct column order
    """
    # Default feature order if not provided
    if feature_order is None:
        feature_order = [
            'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
            'Gender', 'Medical_History', 'Smoking', 'Sporting'
        ]
    
    for feature in feature_order:
        if feature not in processed_data:
            processed_data[feature] = 0  # Default value
    
    df = pd.DataFrame([processed_data])[feature_order]
    
    return df