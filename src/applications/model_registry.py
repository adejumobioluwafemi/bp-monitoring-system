# in the src/applications/model_registry.py
import mlflow
from mlflow.tracking import MlflowClient
import yaml
import json

class ModelRegistry:
    def __init__(self, tracking_uri="http://127.0.0.1:8080"):
        self.client = MlflowClient()
        mlflow.set_tracking_uri(tracking_uri)
        self.DEFAULT_MODEL_NAME = "hypertension-classifier"  # Consistent default
    
    def register_model(self, run_id, model_name=None):
        """Register a model from a specific run with consistent naming"""
        model_name = model_name or self.DEFAULT_MODEL_NAME
        
        try:
            # Register the model
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            
            print(f"✅ Model registered: {model_name} version {result.version}")
            return result
        except Exception as e:
            print(f"❌ Failed to register model {model_name}: {e}")
            return None
    
    def transition_stage(self, model_name=None, version=None, stage="Staging"):
        """Transition model to different stage (Staging, Production, Archived)"""
        model_name = model_name or self.DEFAULT_MODEL_NAME
        
        if version is None:
            print("❌ Version must be specified for stage transition")
            return False
            
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"✅ Model {model_name} v{version} transitioned to {stage}")
            return True
        except Exception as e:
            print(f"❌ Failed to transition model {model_name} v{version}: {e}")
            return False
    
    def set_alias(self, model_name=None, alias="champion", version=None):
        """Set alias for a model version"""
        model_name = model_name or self.DEFAULT_MODEL_NAME
        
        if version is None:
            print("❌ Version must be specified for alias")
            return False
            
        try:
            self.client.set_registered_model_alias(model_name, alias, version)
            print(f"✅ Alias '{alias}' set for {model_name} v{version}")
            return True
        except Exception as e:
            print(f"❌ Failed to set alias for {model_name} v{version}: {e}")
            return False
    
    def get_production_model(self, model_name=None):
        """Get the current production model"""
        model_name = model_name or self.DEFAULT_MODEL_NAME
        
        try:
            # Get model versions and find the production one
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return versions[0]
            print(f"⚠️ No production model found for {model_name}")
            return None
        except Exception as e:
            print(f"❌ Failed to get production model {model_name}: {e}")
            return None
    
    def get_latest_version(self, model_name=None):
        """Get the latest version of a model regardless of stage"""
        model_name = model_name or self.DEFAULT_MODEL_NAME
        
        try:
            versions = self.client.get_latest_versions(model_name)
            if versions:
                return max(versions, key=lambda x: x.version)
            return None
        except Exception as e:
            print(f"❌ Failed to get latest version for {model_name}: {e}")
            return None
    
    def compare_models(self, model_name=None, versions=[1, 2]):
        """Compare different model versions"""
        model_name = model_name or self.DEFAULT_MODEL_NAME
        
        comparison = {}
        for version in versions:
            try:
                mv = self.client.get_model_version(model_name, version)
                run = self.client.get_run(mv.run_id)
                comparison[version] = {
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'stage': mv.current_stage,
                    'model_type': run.data.tags.get('model_family', 'unknown')
                }
            except Exception as e:
                print(f"⚠️ Could not get version {version} of {model_name}: {e}")
                continue
        return comparison

def promote_to_staging(model_name="hypertension-classifier", version=None):
    """Promote the best model to staging"""
    registry = ModelRegistry()
    
    # If no version specified, get the latest
    if version is None:
        latest_model = registry.get_latest_version(model_name)
        if latest_model:
            version = latest_model.version
        else:
            print("❌ No model versions found. Please register the model first.")
            return False
    
    # Check if model exists before transition
    try:
        # This will throw an exception if model doesn't exist
        versions = registry.client.get_latest_versions(model_name)
        if not versions:
            print(f"❌ Model {model_name} not found in registry. Register it first.")
            return False
    except Exception as e:
        print(f"❌ Model {model_name} not found: {e}")
        return False
    
    # Load current metrics to determine best model (optional enhancement)
    try:
        with open('reports/training_metrics.json', 'r') as f:
            metrics = json.load(f)
        print(f"📊 Model metrics: Validation Accuracy = {metrics.get('validation_metrics', {}).get('accuracy', 'N/A')}")
    except:
        print("⚠️ Could not load training metrics")
    
    # Promote to staging
    success = registry.transition_stage(model_name, version, "Staging")
    if success:
        registry.set_alias(model_name, "champion", version)
        return True
    return False

def promote_to_production(model_name="hypertension-classifier", version=None):
    """Promote a specific version to production"""
    registry = ModelRegistry()
    
    if version is None:
        print("❌ Version must be specified for production promotion")
        return False
    
    # Archive current production model if exists
    current_prod = registry.get_production_model(model_name)
    if current_prod:
        registry.transition_stage(model_name, current_prod.version, "Archived")
        print(f"📦 Archived previous production model v{current_prod.version}")
    
    # Promote new version
    success = registry.transition_stage(model_name, version, "Production")
    if success:
        registry.set_alias(model_name, "champion", version)
        print(f"🚀 Model {model_name} version {version} promoted to production!")
        return True
    
    return False

def register_and_promote_model(run_id, model_name="hypertension-classifier", stage="Staging"):
    """Complete workflow: register model and promote to specified stage"""
    registry = ModelRegistry()
    
    # Step 1: Register the model
    registered_model = registry.register_model(run_id, model_name)
    if not registered_model:
        print("❌ Failed to register model")
        return False
    
    # Step 2: Promote to the desired stage
    version = registered_model.version
    if stage == "Staging":
        success = promote_to_staging(model_name, version)
    elif stage == "Production":
        success = promote_to_production(model_name, version)
    else:
        print(f"❌ Unknown stage: {stage}")
        return False
    
    return success

def promote_with_alias(model_name, version, alias="champion"):
    """Modern approach: use aliases instead of stages"""
    registry = ModelRegistry()
    
    try:
        # Set alias instead of transitioning stage
        registry.client.set_registered_model_alias(model_name, alias, version)
        print(f"✅ Alias '{alias}' set for {model_name} v{version}")
        
        # Add tag for documentation
        registry.client.set_model_version_tag(
            model_name, 
            version, 
            "deployment_status", 
            "staging" if alias == "champion" else alias
        )
        return True
    except Exception as e:
        print(f"❌ Failed to set alias: {e}")
        return False
# Example usage that ensures alignment
# def complete_training_and_registration():
#    """Complete workflow with aligned naming"""
#    # Train and get consistent model name
#    run_id, model_name = training_only_mlflow()
#    
#    # Register using the same model name
#    registry = ModelRegistry()
#    registered_model = registry.register_model(run_id, model_name)
#    
#    if registered_model:
#        # Promote to staging
#        promote_to_staging(model_name, registered_model.version)
#    
#    return registered_model

if __name__ == "__main__":
    # You need to provide the run_id from your training
    # run_id = "your_actual_run_id_here"  # Get this from your training script output
    
    # Example usage:
    # register_and_promote_model(run_id, "hypertension-classifier", "Staging")
    
    # For now, since you're testing, use this:
    print("Please run the training script first to get a run_id, then use:")
    print("python src/applications/model_registry.py --run-id <YOUR_RUN_ID>")
    
    # Or add argument parsing:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', help='MLflow Run ID to register and promote')
    args = parser.parse_args()
    
    if args.run_id:
        register_and_promote_model(args.run_id, "hypertension-classifier", "Staging")
    else:
        print("❌ No run_id provided. Please provide a run_id from training.")