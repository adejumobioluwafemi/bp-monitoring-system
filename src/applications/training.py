"""
Training Application for Hypertension Prediction
With command-line argument support for DVC pipeline
"""


from pathlib import Path
import argparse
import os
import sys
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

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


def train_only_without_mlflow(model_type=None):
    """Train model with parameters from params.yaml"""
    
    # Load parameters from params.yaml
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        model_type = model_type or params['model']['model_type']
        print(f"🎯 Training {model_type} model using parameters from params.yaml...")
        
    except Exception as e:
        print(f"⚠️  Could not load params.yaml: {e}. Using defaults.")
        model_type = model_type or 'random_forest'
        params = {}
    
    train_data = DataConnector.load_data("data/processed/train_clean.csv")
    val_data = DataConnector.load_data("data/processed/val_clean.csv")

    target_col = params.get('prepare', {}).get('target_column', 'Hypertension_Tests')
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col]

    numerical_cols = params.get('prepare', {}).get('numerical_features', 
                   ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate'])
    categorical_cols = params.get('prepare', {}).get('categorical_features',
                    ['Gender', 'Medical_History', 'Smoking', 'Sporting'])

    data_processor = DataProcessor()
    preprocessor = data_processor.create_preprocessor(numerical_cols, categorical_cols)

    training_pipeline = TrainingPipeline(
        preprocessor=preprocessor,
        model_type=model_type,
        random_state=params.get('model', {}).get('random_state', 42)
    )

    training_pipeline.fit(X_train, y_train)

    models_path = "models_chpt"
    os.makedirs(models_path, exist_ok=True)
    model_spec_filename = f"hypertension_model_{model_type}.pkl"
    default_model_filename = "hypertension_model.pkl"
    training_pipeline.save_model(os.path.join(models_path, model_spec_filename))
    training_pipeline.save_model(os.path.join(models_path, default_model_filename))

    print("📈 Generating evaluation report...")
    comprehensive_report = training_pipeline.get_comprehensive_report(X_train, y_train, X_val, y_val)
    training_pipeline.save_training_report(comprehensive_report)


    print(f"✅ {model_type} model training completed")
    return True

def training_only(model_type=None):
    """
    Production-ready mlflow integration with consistent model naming
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("hypertension-risk-prediction")

    # Disable autologging to prevent conflicts
    mlflow.sklearn.autolog(disable=True)
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    model_type = model_type or params['model']['model_type']
    
    MODEL_NAME = "hypertension-classifier" 

    with mlflow.start_run(run_name=f"prod-{model_type}") as run:
        print(f"Starting MLflow-tracked training for {model_type}")

        mlflow.log_params(params['model'][model_type])
        mlflow.log_params({
            "test_size": params['prepare']['test_size'],
            "val_size": params['prepare']['val_size']
        })

        train_data = DataConnector.load_data("data/processed/train_clean.csv")
        val_data = DataConnector.load_data("data/processed/val_clean.csv")

        target_col = params.get('prepare', {}).get('target_column', 'Hypertension_Tests')
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_val = val_data.drop(columns=[target_col])
        y_val = val_data[target_col]

        numerical_cols = params.get('prepare', {}).get('numerical_features', 
                    ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate'])
        categorical_cols = params.get('prepare', {}).get('categorical_features',
                        ['Gender', 'Medical_History', 'Smoking', 'Sporting'])

        data_processor = DataProcessor()
        preprocessor = data_processor.create_preprocessor(numerical_cols, categorical_cols)

        training_pipeline = TrainingPipeline(
            preprocessor=preprocessor,
            model_type=model_type,
            random_state=params.get('model', {}).get('random_state', 42)
        )

        training_pipeline.fit(X_train, y_train)

        signature = infer_signature(X_train, training_pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=training_pipeline.pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature
        )
        print(f"💾 Logged model as '{MODEL_NAME}' with signature")

        comprehensive_report = training_pipeline.get_comprehensive_report(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        training_pipeline.save_training_report(comprehensive_report)

        import time
        current_time = int(time.time() * 1000)  # Current timestamp in milliseconds
        
        
        mlflow.log_metrics({
            "train_acc": comprehensive_report['training_metrics']['accuracy'],
            "val_acc": comprehensive_report['validation_metrics']['accuracy'],
            "train_f1": comprehensive_report['training_metrics']['f1'],
            "val_f1": comprehensive_report['validation_metrics']['f1'],
            "roc_auc": comprehensive_report['validation_metrics']['roc_auc']
        }, step=0)  # step=0 for all metrics to ensure they're batched together

        mlflow.log_metric("training_time", training_pipeline.training_history['training_time_seconds'])
        mlflow.log_param("feature_count", training_pipeline.training_history['features_count'])
        mlflow.log_param("model_type", training_pipeline.training_history['model_type'])
        mlflow.sklearn.log_model(training_pipeline.pipeline, "model")
        print("📊 Logged comprehensive metrics")
        
        mlflow.log_artifact("reports/training_metrics.json")
        mlflow.log_artifact("params.yaml")
        print("📎 Logged artifact files")

        mlflow.set_tags({
            "project": "bp-monitoring-system",
            "team": "data-science",
            "iteration": "v1",
            "data_source": "clinical",
            "model_family": model_type,
            "registered_model_name": MODEL_NAME
        })
        print("🏷️ Set organizational tags")

        print(f"✅ MLflow tracking complete! View at: http://127.0.0.1:8080/#/experiments/0/runs/{run.info.run_id}")
        
        return run.info.run_id, MODEL_NAME

def full_training_cv_with_test():
    """Run complete training workflow (data prep + training)"""
    print(" Starting complete Hypertension Model Training...")

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
    parser.add_argument('--train-only', action='store_true',
                        help='Train model using preprocessed data')
    parser.add_argument('--full', action='store_true',
                        help='Run complete training workflow (default)')

    args = parser.parse_args()

    # Default behavior if no arguments provided
    if not any([args.train_only, args.full]):
        args.full = True

    try:
        if args.train_only:
            success = training_only()
        else:  # args.full or default
            success = full_training_cv_with_test()

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
