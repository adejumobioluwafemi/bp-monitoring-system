import mlflow
from mlflow.tracking import MlflowClient
import json
import yaml
import logging
import os
import shutil
import subprocess
import tempfile
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPromotionWorkflow:
    def __init__(self, tracking_uri="http://127.0.0.1:8080", dvc_remote="mys3remote"):
        self.tracking_uri = tracking_uri
        self.dvc_remote = dvc_remote
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        logger.info(f"Initialized ModelPromotionWorkflow with tracking URI: {tracking_uri}")
        logger.info(f"DVC remote: {dvc_remote}")
        
        # Create necessary directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories for DVC tracking"""
        directories = [
            "models/production",
            "models/staging", 
            "models/backups",
            "models/dvc-tracked"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"✅ Ensured directory exists: {directory}")
    
    def _check_mlflow_connection(self):
        """Check if MLflow server is accessible"""
        try:
            experiments = self.client.search_experiments()
            logger.info(f"✅ Successfully connected to MLflow server. Found {len(experiments)} experiments.")
            return True
        except Exception as e:
            logger.error(f"❌ Cannot connect to MLflow server at {self.tracking_uri}: {e}")
            logger.info("💡 Make sure MLflow server is running: mlflow server --host 127.0.0.1 --port 8080")
            return False
    
    def _check_dvc_connection(self):
        """Check if DVC remote is accessible"""
        try:
            result = subprocess.run(
                ["dvc", "remote", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            if self.dvc_remote in result.stdout:
                logger.info(f"✅ DVC remote '{self.dvc_remote}' is configured")
                return True
            else:
                logger.warning(f"⚠️ DVC remote '{self.dvc_remote}' not found")
                return False
        except Exception as e:
            logger.error(f"❌ DVC check failed: {e}")
            return False
    
    def _run_dvc_command(self, command, description=""):
        """Run DVC command with error handling"""
        try:
            if description:
                logger.info(f"🔄 {description}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.getcwd()  # Ensure we're in the project root
            )
            logger.debug(f"✅ DVC command successful: {' '.join(command)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ DVC command failed: {' '.join(command)}")
            logger.error(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error running DVC command: {e}")
            return False
    
    def _download_model_from_mlflow(self, run_id, model_name="model"):
        """
        Download model artifacts from MLflow server to local directory
        """
        try:
            # Create temporary directory for model download
            temp_dir = tempfile.mkdtemp(prefix=f"mlflow_model_{run_id}_")
            logger.info(f"📥 Downloading model from MLflow run {run_id} to {temp_dir}")
            
            # Download the model using MLflow client
            model_uri = f"runs:/{run_id}/{model_name}"
            local_path = mlflow.artifacts.download_artifacts(
                model_uri, 
                dst_path=temp_dir
            )
            
            logger.info(f"✅ Model downloaded successfully to: {local_path}")
            return local_path, temp_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to download model from MLflow: {e}")
            return None, None
    
    def backup_model_to_dvc(self, run_id, version, stage="Staging"):
        """
        Backup model to DVC-tracked location by downloading from MLflow first
        """
        try:
            logger.info(f"💾 Backing up model version {version} to DVC...")
            
            # Step 1: Download model from MLflow server
            model_path, temp_dir = self._download_model_from_mlflow(run_id)
            if not model_path:
                return False
            
            # Step 2: Create version-specific backup directory
            backup_dir = f"models/dvc-tracked/v{version}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Step 3: Copy downloaded model to DVC directory
            dvc_model_path = f"{backup_dir}/model"
            
            if os.path.exists(dvc_model_path):
                shutil.rmtree(dvc_model_path)
            
            # Copy the entire model directory
            shutil.copytree(model_path, dvc_model_path)
            
            # Step 4: Create metadata file
            metadata = {
                "mlflow_run_id": run_id,
                "model_version": version,
                "stage": stage,
                "backup_timestamp": datetime.now().isoformat(),
                "model_uri": f"runs:/{run_id}/model",
                "promotion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(f"{backup_dir}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Step 5: Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Step 6: Add to DVC tracking
            dvc_success = self._run_dvc_command(
                ["dvc", "add", backup_dir],
                f"Adding model v{version} to DVC tracking"
            )
            
            if dvc_success:
                # Step 7: Push to remote storage
                push_success = self._run_dvc_command(
                    ["dvc", "push"],
                    f"Pushing model v{version} to DVC remote"
                )
                
                if push_success:
                    logger.info(f"✅ Model version {version} successfully backed up to DVC: {backup_dir}")
                    return True
                else:
                    logger.warning(f"⚠️ Model backed up locally but DVC push failed for v{version}")
                    return False
            else:
                logger.error(f"❌ Failed to add model v{version} to DVC tracking")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error backing up model to DVC: {e}")
            # Clean up temp directory on error
            if 'temp_dir' in locals() and temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False
    
    def create_production_symlink(self, version):
        """
        Create a symbolic link to the current production model for easy access
        """
        try:
            production_link = "models/production/current"
            
            # Remove existing symlink if it exists
            if os.path.exists(production_link) or os.path.islink(production_link):
                os.remove(production_link)
            
            # Create symlink to DVC-tracked model
            model_path = f"models/dvc-tracked/v{version}"
            
            if os.path.exists(model_path):
                # Use absolute path for symlink to avoid issues
                abs_model_path = os.path.abspath(model_path)
                os.symlink(abs_model_path, production_link)
                logger.info(f"🔗 Created production symlink: {production_link} -> {abs_model_path}")
                return True
            else:
                logger.warning(f"⚠️ DVC model path not found for symlink: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating production symlink: {e}")
            return False
    
    def evaluate_model_criteria(self, run_id):
        """Evaluate model against promotion criteria"""
        try:
            run = self.client.get_run(run_id)
            metrics = run.data.metrics
            
            # Use the actual metric names from your training
            criteria = {
                'min_accuracy': 0.75,
                'min_f1': 0.70,
                'max_accuracy_gap': 0.10,  # Prevent overfitting
                'min_roc_auc': 0.80
            }
            
            # Get metrics with fallbacks for different naming conventions
            val_accuracy = metrics.get('val_acc', metrics.get('val_accuracy', 0))
            val_f1 = metrics.get('val_f1', 0)
            train_accuracy = metrics.get('train_acc', metrics.get('train_accuracy', 0))
            roc_auc = metrics.get('roc_auc', 0)
            
            # Calculate accuracy gap if not directly available
            accuracy_gap = abs(train_accuracy - val_accuracy)
            
            passed = (
                val_accuracy >= criteria['min_accuracy'] and
                val_f1 >= criteria['min_f1'] and
                accuracy_gap <= criteria['max_accuracy_gap'] and
                roc_auc >= criteria['min_roc_auc']
            )
            
            logger.info(f"📊 Model evaluation for run {run_id}:")
            logger.info(f"   Validation Accuracy: {val_accuracy:.3f} (min: {criteria['min_accuracy']})")
            logger.info(f"   Validation F1: {val_f1:.3f} (min: {criteria['min_f1']})")
            logger.info(f"   Accuracy Gap: {accuracy_gap:.3f} (max: {criteria['max_accuracy_gap']})")
            logger.info(f"   ROC AUC: {roc_auc:.3f} (min: {criteria['min_roc_auc']})")
            logger.info(f"   Promotion: {'✅ PASSED' if passed else '❌ FAILED'}")
            
            return passed, criteria
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model criteria for run {run_id}: {e}")
            return False, {}
    
    def get_experiment_runs(self, experiment_name="hypertension-risk-prediction"):
        """Get runs from a specific experiment"""
        try:
            # Search for experiment by name
            experiments = self.client.search_experiments()
            target_experiment = None
            
            for exp in experiments:
                if experiment_name.lower() in exp.name.lower():
                    target_experiment = exp
                    break
            
            if target_experiment is None:
                logger.warning(f"⚠️ Experiment '{experiment_name}' not found")
                return []
            
            runs = self.client.search_runs(
                experiment_ids=[target_experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'"
            )
            
            logger.info(f"📁 Found {len(runs)} finished runs in experiment '{experiment_name}'")
            return runs
            
        except Exception as e:
            logger.error(f"❌ Error getting experiment runs: {e}")
            return []
    
    def automated_promotion_workflow(self):
        """Automated model promotion workflow with DVC integration"""
        logger.info("🔍 Running automated model promotion workflow with DVC...")
        
        # Check MLflow connection first
        if not self._check_mlflow_connection():
            return
        
        # Optional: Check DVC connection
        self._check_dvc_connection()
        
        # Get runs from hypertension experiment
        runs = self.get_experiment_runs("hypertension-risk-prediction")
        
        if not runs:
            logger.info("ℹ️ No runs found for promotion")
            return
        
        promoted_count = 0
        
        for run in runs:
            run_id = run.info.run_id
            logger.info(f"🔍 Evaluating run: {run_id}")
            
            # Check if model meets promotion criteria
            passed, criteria = self.evaluate_model_criteria(run_id)
            
            if passed:
                try:
                    # Check if model is already registered
                    model_uri = f"runs:/{run_id}/model"
                    
                    # Get or create registered model
                    try:
                        registered_model = mlflow.register_model(model_uri, "hypertension-classifier")
                        logger.info(f"📝 Registered new model version {registered_model.version}")
                    except Exception as e:
                        if "already exists" in str(e):
                            logger.info("ℹ️ Model already registered, checking versions...")
                            # Get the latest version and check if this run is already registered
                            versions = self.client.search_model_versions(f"run_id = '{run_id}'")
                            if versions:
                                registered_model = versions[0]
                                logger.info(f"ℹ️ Run already registered as version {registered_model.version}")
                                
                                # Still backup to DVC even if already registered
                                self.backup_model_to_dvc(run_id, registered_model.version, "Staging")
                                continue
                            else:
                                logger.warning(f"⚠️ Model exists but run {run_id} not found in versions")
                                continue
                        else:
                            raise
                    
                    # Transition to Staging using modern alias approach
                    self.client.set_registered_model_alias(
                        name="hypertension-classifier",
                        alias="champion",
                        version=registered_model.version
                    )
                    
                    # Also set staging alias
                    self.client.set_registered_model_alias(
                        name="hypertension-classifier",
                        alias="staging",
                        version=registered_model.version
                    )
                    
                    # Backup to DVC
                    dvc_success = self.backup_model_to_dvc(run_id, registered_model.version, "Staging")
                    
                    if dvc_success:
                        logger.info(f"🚀 Model version {registered_model.version} promoted to Staging + DVC backup")
                    else:
                        logger.info(f"🚀 Model version {registered_model.version} promoted to Staging (DVC backup failed)")
                    
                    promoted_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error promoting run {run_id}: {e}")
            else:
                logger.info(f"⏭️  Run {run_id} does not meet promotion criteria")
        
        logger.info(f"✅ Promotion workflow completed. Promoted {promoted_count} models.")
    
    def manual_promotion_check(self, version):
        """Manual promotion check for specific version"""
        try:
            model_version = self.client.get_model_version("hypertension-classifier", version)
            passed, criteria = self.evaluate_model_criteria(model_version.run_id)
            
            if passed:
                logger.info(f"✅ Version {version} meets production criteria")
                return True
            else:
                logger.info(f"❌ Version {version} fails criteria")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking version {version}: {e}")
            return False
    
    def promote_to_production(self, version):
        """Promote specific version to production using modern alias approach"""
        try:
            # Archive current production if exists
            try:
                # Use search_model_versions instead of deprecated get_latest_versions
                prod_versions = self.client.search_model_versions(
                    "name='hypertension-classifier' and status='READY'"
                )
                # Find versions with production alias
                current_prod_versions = []
                for v in prod_versions:
                    aliases = [alias for alias in v.aliases] if hasattr(v, 'aliases') else []
                    if 'production' in aliases or 'champion' in aliases:
                        current_prod_versions.append(v)
                
                for prod_version in current_prod_versions:
                    # Remove production aliases
                    try:
                        self.client.delete_registered_model_alias(
                            name="hypertension-classifier",
                            alias="production"
                        )
                    except:
                        pass
                    
                    try:
                        self.client.delete_registered_model_alias(
                            name="hypertension-classifier",
                            alias="champion"
                        )
                    except:
                        pass
                    
                    # Set archived alias
                    self.client.set_registered_model_alias(
                        name="hypertension-classifier",
                        alias="archived",
                        version=prod_version.version
                    )
                    logger.info(f"📦 Archived previous production version {prod_version.version}")
            except Exception as e:
                logger.info("ℹ️ No existing production model to archive")
            
            # Promote new version using aliases (modern approach)
            self.client.set_registered_model_alias(
                name="hypertension-classifier",
                alias="champion",
                version=version
            )
            
            # Also set production alias
            self.client.set_registered_model_alias(
                name="hypertension-classifier", 
                alias="production",
                version=version
            )
            
            logger.info(f"🚀 Version {version} promoted to Production using aliases!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error promoting version {version} to production: {e}")
            return False
    
    def promote_to_production_with_dvc(self, version):
        """Promote model to production with full DVC integration"""
        try:
            # Get model info for DVC backup
            model_version = self.client.get_model_version("hypertension-classifier", version)
            run_id = model_version.run_id
            
            logger.info(f"🚀 Starting production promotion for version {version} (run: {run_id})")
            
            # Promote in MLflow first
            mlflow_success = self.promote_to_production(version)
            
            if mlflow_success:
                logger.info(f"✅ MLflow promotion successful for v{version}")
                
                # Backup to DVC
                dvc_success = self.backup_model_to_dvc(run_id, version, "Production")
                
                # Create production symlink (only if DVC backup was successful)
                symlink_success = False
                if dvc_success:
                    symlink_success = self.create_production_symlink(version)
                else:
                    logger.warning("⚠️ Skipping symlink creation due to DVC backup failure")
                
                if dvc_success and symlink_success:
                    logger.info(f"🎉 Version {version} fully promoted to Production + DVC + Symlink")
                    return True
                elif dvc_success:
                    logger.info(f"🎉 Version {version} promoted to Production + DVC (symlink failed)")
                    return True
                else:
                    logger.warning(f"⚠️ Version {version} promoted to Production but DVC backup failed")
                    return True  # Still return True since MLflow promotion worked
            else:
                logger.error(f"❌ MLflow promotion failed for v{version}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in production promotion with DVC: {e}")
            return False
    
    def list_dvc_tracked_models(self):
        """List all DVC-tracked models"""
        try:
            dvc_models_dir = "models/dvc-tracked"
            if not os.path.exists(dvc_models_dir):
                logger.info("ℹ️ No DVC-tracked models found")
                return []
            
            models = []
            for item in os.listdir(dvc_models_dir):
                if item.startswith("v") and os.path.isdir(os.path.join(dvc_models_dir, item)):
                    version = item[1:]  # Remove 'v' prefix
                    metadata_file = os.path.join(dvc_models_dir, item, "metadata.json")
                    
                    model_info = {"version": version, "path": item}
                    
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            model_info.update(metadata)
                    
                    models.append(model_info)
            
            logger.info(f"📋 Found {len(models)} DVC-tracked models:")
            for model in models:
                logger.info(f"   - v{model['version']} ({model.get('stage', 'unknown')})")
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Error listing DVC models: {e}")
            return []
    
    def get_production_model_info(self):
        """Get information about current production model"""
        try:
            # Get MLflow production model using modern approach
            prod_versions = self.client.search_model_versions(
                "name='hypertension-classifier' and status='READY'"
            )
            
            # Find version with production alias
            production_version = None
            for version in prod_versions:
                aliases = [alias for alias in version.aliases] if hasattr(version, 'aliases') else []
                if 'production' in aliases or 'champion' in aliases:
                    production_version = version
                    break
            
            if not production_version:
                logger.info("ℹ️ No model in production")
                return None
            
            model_info = {
                "mlflow_version": production_version.version,
                "run_id": production_version.run_id,
                "aliases": production_version.aliases if hasattr(production_version, 'aliases') else [],
                "dvc_backup": False,
                "production_symlink": False
            }
            
            # Check if DVC backup exists
            dvc_path = f"models/dvc-tracked/v{production_version.version}"
            if os.path.exists(dvc_path):
                model_info["dvc_backup"] = True
                model_info["dvc_path"] = dvc_path
            
            # Check if production symlink exists
            symlink_path = "models/production/current"
            if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                model_info["production_symlink"] = True
                model_info["symlink_target"] = os.path.realpath(symlink_path) if os.path.exists(symlink_path) else "broken"
            
            logger.info("🏭 Production Model Info:")
            for key, value in model_info.items():
                logger.info(f"   {key}: {value}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Error getting production model info: {e}")
            return None
    
    def debug_mlflow_storage(self, run_id):
        """Debug method to check MLflow artifact storage"""
        try:
            run = self.client.get_run(run_id)
            logger.info(f"🔍 Debug info for run {run_id}:")
            logger.info(f"   Experiment ID: {run.info.experiment_id}")
            logger.info(f"   Artifact URI: {run.info.artifact_uri}")
            
            # List all artifacts in the run
            artifacts = self.client.list_artifacts(run_id)
            logger.info(f"   Artifacts found: {[a.path for a in artifacts]}")
            
            # Try to download a small artifact to test
            if artifacts:
                test_artifact = artifacts[0].path
                temp_dir = tempfile.mkdtemp()
                try:
                    downloaded_path = mlflow.artifacts.download_artifacts(
                        f"runs:/{run_id}/{test_artifact}",
                        dst_path=temp_dir
                    )
                    logger.info(f"   Test download successful: {downloaded_path}")
                    # Clean up
                    shutil.rmtree(temp_dir)
                except Exception as download_error:
                    logger.error(f"   Test download failed: {download_error}")
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            
            return True
        except Exception as e:
            logger.error(f"❌ Debug failed: {e}")
            return False


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Model Promotion Workflow with DVC Integration')
    parser.add_argument('--auto', action='store_true', help='Run automated promotion workflow')
    parser.add_argument('--check', type=int, help='Check specific version for promotion')
    parser.add_argument('--promote', type=int, help='Promote specific version to production (MLflow only)')
    parser.add_argument('--promote-dvc', type=int, help='Promote specific version to production with DVC integration')
    parser.add_argument('--list-dvc', action='store_true', help='List all DVC-tracked models')
    parser.add_argument('--production-info', action='store_true', help='Get current production model information')
    parser.add_argument('--debug-run', type=str, help='Debug MLflow storage for specific run ID')
    parser.add_argument('--tracking-uri', default="http://127.0.0.1:8080", help='MLflow tracking URI')
    parser.add_argument('--dvc-remote', default="mys3remote", help='DVC remote name')
    
    args = parser.parse_args()
    
    workflow = ModelPromotionWorkflow(
        tracking_uri=args.tracking_uri,
        dvc_remote=args.dvc_remote
    )
    
    try:
        if args.debug_run:
            # Debug specific run
            workflow.debug_mlflow_storage(args.debug_run)
        elif args.check:
            # Check specific version
            workflow.manual_promotion_check(args.check)
        elif args.promote:
            # Promote specific version (MLflow only)
            if workflow.manual_promotion_check(args.promote):
                workflow.promote_to_production(args.promote)
            else:
                logger.error("❌ Version does not meet promotion criteria")
                return 1
        elif args.promote_dvc:
            # Promote specific version with DVC integration
            if workflow.manual_promotion_check(args.promote_dvc):
                success = workflow.promote_to_production_with_dvc(args.promote_dvc)
                if not success:
                    return 1
            else:
                logger.error("❌ Version does not meet promotion criteria")
                return 1
        elif args.list_dvc:
            # List DVC-tracked models
            workflow.list_dvc_tracked_models()
        elif args.production_info:
            # Get production model info
            workflow.get_production_model_info()
        else:
            # Run automated workflow (default)
            workflow.automated_promotion_workflow()
            
    except Exception as e:
        logger.error(f"❌ Error in promotion workflow: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# Run automated promotion workflow (default)
# python src/applications/model_promotion.py --auto

# Check if version 2 meets promotion criteria
# python src/applications/model_promotion.py --check 2

# Promote version 2 to production (MLflow only)
# python src/applications/model_promotion.py --promote 2

# Promote version 2 to production with DVC integration
# python src/applications/model_promotion.py --promote-dvc 2

# List all DVC-tracked models
# python src/applications/model_promotion.py --list-dvc

# Get current production model information
# python src/applications/model_promotion.py --production-info

# Debug MLflow storage for a specific run
# python src/applications/model_promotion.py --debug-run b3d3df80644b44859ab621c502207979

# Use different MLflow tracking URI
# python src/applications/model_promotion.py --auto --tracking-uri http://127.0.0.1:8080