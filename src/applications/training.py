"""
Training Application for Hypertension Prediction
With command-line argument support for DVC pipeline
"""


from pathlib import Path
import argparse
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))
try:
    from data_layer.data_connector import DataConnector
    from ml.data_processor import DataProcessor
    from ml.training_pipeline import TrainingPipeline
    from validate_data import validate_data
    from ml.cross_validator import CrossValidator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def prepare_data_only():
    """Only prepare and split data without training"""
    print("📊 Preparing data only...")

    # Load raw data
    raw_data_path = "data/raw/Hypertension_Data_Set.csv"
    data = DataConnector.get_data(raw_data_path)
    print(f"   Loaded {len(data)} samples")

    # Initialize data processor
    data_processor = DataProcessor(
        target_column='Hypertension_Tests',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    processed_data_path = "data/processed"
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        data, save_path=processed_data_path, generate_metrics=True
    )

    print("✅ Data preparation completed")
    return True


def train_only():
    """Train model using preprocessed data"""
    print("🎯 Training model only...")

    # Load preprocessed data
    train_data = DataConnector.load_data("data/processed/train_clean.csv")
    val_data = DataConnector.load_data("data/processed/val_clean.csv")

    # Prepare features and target
    X_train = train_data.drop(columns=['Hypertension_Tests'])
    y_train = train_data['Hypertension_Tests']
    X_val = val_data.drop(columns=['Hypertension_Tests'])
    y_val = val_data['Hypertension_Tests']

    # Define feature columns
    numerical_cols = ['Age', 'BMI', 'Systolic_BP',
                      'Diastolic_BP', 'Heart_Rate']
    categorical_cols = ['Gender', 'Medical_History', 'Smoking', 'Sporting']

    # Create and fit preprocessor
    data_processor = DataProcessor()
    preprocessor = data_processor.create_preprocessor(
        numerical_cols, categorical_cols)

    # Initialize and run training pipeline
    training_pipeline = TrainingPipeline(
        preprocessor=preprocessor,
        model_type='random_forest',
        random_state=42
    )

    # Train on training data
    training_pipeline.fit(X_train, y_train)

    # Validate on validation set
    val_accuracy = training_pipeline.evaluate(X_val, y_val)
    print(f"✅ Validation Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    models_path = "models_chpt"
    os.makedirs(models_path, exist_ok=True)
    training_pipeline.save_model(os.path.join(
        models_path, "hypertension_model.pkl"))

    print("✅ Model training completed")
    return True


def full_training():
    """Run complete training workflow (data prep + training)"""
    print(" Starting complete Hypertension Model Training...")

    # Prepare data
    prepare_data_only()

    # Data validation
    validate_data()

    # Train model
    train_only()

    # 4. Optional: Cross-validation and evaluation
    print("📊 Running cross-validation...")
    test_data = DataConnector.load_data("data/processed/test_clean.csv")
    X_test = test_data.drop(columns=['Hypertension_Tests'])
    y_test = test_data['Hypertension_Tests']

    # Load trained model for evaluation
    from data_layer.object_connector import ObjectConnector
    model = ObjectConnector.load_object("models_chpt/hypertension_model.pkl")

    # Initialize cross-validator
    cv = CrossValidator(pipeline=model)
    metrics = cv.evaluate_model(
        X_test, y_test, save_plots=True, plot_path='reports/plots/')

    print(
        f"🎉 Training completed! Test Accuracy: {metrics.get('accuracy', 0):.4f}")
    return True


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Hypertension Model Training')
    parser.add_argument('--prepare-data-only', action='store_true',
                        help='Only prepare and split data without training')
    parser.add_argument('--train-only', action='store_true',
                        help='Train model using preprocessed data')
    parser.add_argument('--full', action='store_true',
                        help='Run complete training workflow (default)')

    args = parser.parse_args()

    # Default behavior if no arguments provided
    if not any([args.prepare_data_only, args.train_only, args.full]):
        args.full = True

    try:
        if args.prepare_data_only:
            success = prepare_data_only()
        elif args.train_only:
            success = train_only()
        else:  # args.full or default
            success = full_training()

        if success:
            print("✅ Training script completed successfully")
            sys.exit(0)
        else:
            print("❌ Training script failed")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error in training script: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
