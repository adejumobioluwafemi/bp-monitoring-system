from pathlib import Path
import argparse
import os
import sys
import yaml

sys.path.append(str(Path(__file__).parent.parent))
try:
    from data_layer.data_connector import DataConnector
    from ml.data_processor import DataProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def load_params():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        print("✅ Loaded parameters from params.yaml")
        return params
    except FileNotFoundError:
        print("❌ params.yaml file not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing params.yaml: {e}")
        sys.exit(1)

def prepare_data():
    """Prepare and split data using parameters from params.yaml"""
    print("📊 Preparing data using parameters from params.yaml...")
    
    params = load_params()
    prepare_params = params.get('prepare', {})
    
    raw_data_path = prepare_params.get('rawdata_file_path', 'data/raw/Hypertension_Data_Set.csv')
    target_column = prepare_params.get('target_column', 'Hypertension_Tests')
    test_size = prepare_params.get('test_size', 0.2)
    val_size = prepare_params.get('val_size', 0.1)
    random_state = prepare_params.get('random_state', 42)
    
    numerical_features = prepare_params.get('numerical_features', [])
    categorical_features = prepare_params.get('categorical_features', [])
    
    # Print parameter summary
    # print(f"📋 Parameters:")
    # print(f"   Raw data path: {raw_data_path}")
    # print(f"   Target column: {target_column}")
    # print(f"   Test size: {test_size}")
    # print(f"   Validation size: {val_size}")
    # print(f"   Random state: {random_state}")
    # print(f"   Numerical features: {len(numerical_features)}")
    # print(f"   Categorical features: {len(categorical_features)}")
    
    # Validate parameters
    if not numerical_features:
        print("⚠️  No numerical features specified in params.yaml")
    if not categorical_features:
        print("⚠️  No categorical features specified in params.yaml")
    
    # Load raw data
    try:
        data = DataConnector.get_data(raw_data_path)
        print(f"✅ Loaded {len(data)} samples from {raw_data_path}")
    except Exception as e:
        print(f"❌ Error loading data from {raw_data_path}: {e}")
        sys.exit(1)
    
    # Check if target column exists
    if target_column not in data.columns:
        print(f"❌ Target column '{target_column}' not found in data")
        print(f"   Available columns: {list(data.columns)}")
        sys.exit(1)
    
    data_processor = DataProcessor(
        target_column=target_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    processed_data_path = "data/processed"
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
            data, save_path=processed_data_path, generate_metrics=True
        )
        
        # Print split summary
        print(f"📊 Data Split Summary:")
        print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
        print(f"   Validation set: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
        print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
        print(f"   Target distribution - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Error during data splitting: {e}")
        sys.exit(1)
    
    print("✅ Data preparation completed successfully")
    return True

def main():
    """Main function with argument parsing support"""
    parser = argparse.ArgumentParser(description='Prepare data for hypertension prediction')
    parser.add_argument('--config', type=str, default='params.yaml',
                       help='Path to configuration file (default: params.yaml)')
    args = parser.parse_args()
    
    # If a custom config path is provided, update it
    if args.config != 'params.yaml':
        global params_file
        params_file = args.config
    
    prepare_data()

if __name__ == '__main__':
    main()