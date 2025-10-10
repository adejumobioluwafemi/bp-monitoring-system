"""
Report Generation Script for Hypertension Monitoring System
Generates comprehensive reports, model cards, and visualizations
"""

from data_layer.data_connector import DataConnector, ObjectConnector
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))


class ReportGenerator:
    """
    Generates comprehensive reports for the hypertension prediction project
    """

    def __init__(self, project_name="Hypertension Monitoring System"):
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.reports_dir = Path("reports")
        self.models_dir = Path("models_chpt")
        self.data_dir = Path("data")

        # Create directories if they don't exist
        self.reports_dir.mkdir(exist_ok=True)
        (self.reports_dir / "plots").mkdir(exist_ok=True)

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    def load_metrics(self):
        """Load all metrics from JSON files"""
        metrics = {}

        metric_files = {
            'data_metrics': 'reports/data_metrics.json',
            'data_validation': 'reports/data_validation_report.json',
            'training_metrics': 'reports/training_metrics.json',
            'evaluation_metrics': 'reports/evaluation_metrics.json',
            'cross_validation_metrics': 'reports/cross_validation_metrics.json'
        }

        for name, filepath in metric_files.items():
            try:
                with open(filepath, 'r') as f:
                    metrics[name] = json.load(f)
            except FileNotFoundError:
                print(f"⚠️  {filepath} not found")
                metrics[name] = {}

        return metrics

    def load_params(self):
        """Load parameters from params.yaml"""
        try:
            import yaml
            with open('params.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("⚠️  params.yaml not found")
            return {}
        except ImportError:
            print("⚠️  PyYAML not installed, using empty params")
            return {}

    def generate_data_report(self, metrics):
        """Generate data analysis report section"""
        data_validation = metrics.get('data_validation', {})
        data_metrics = metrics.get('data_metrics', {})

        report = f"""
        ## 📊 Data Analysis Report
        
        ### Dataset Overview
        - **Total Samples**: {data_validation.get('total_samples', 'N/A')}
        - **Training Samples**: {data_validation.get('train_samples', 'N/A')}
        - **Validation Samples**: {data_validation.get('val_samples', 'N/A')}
        - **Test Samples**: {data_validation.get('test_samples', 'N/A')}
        
        ### Data Quality
        - **Missing Values (Train)**: {data_validation.get('train_missing_values', 'N/A')}
        - **Missing Values (Validation)**: {data_validation.get('val_missing_values', 'N/A')}
        - **Missing Values (Test)**: {data_validation.get('test_missing_values', 'N/A')}
        
        ### Class Distribution
        - **Training Set**: {data_validation.get('class_balance_train', 'N/A')}
        - **Validation Set**: {data_validation.get('class_balance_val', 'N/A')}
        - **Test Set**: {data_validation.get('class_balance_test', 'N/A')}
        """

        return report

    def generate_training_report(self, metrics):
        """Generate model training report section"""
        training_metrics = metrics.get('training_metrics', {})

        report = f"""
        ## 🎯 Model Training Report
        
        ### Training Performance
        - **Training Accuracy**: {training_metrics.get('train_accuracy', 'N/A'):.4f}
        - **Validation Accuracy**: {training_metrics.get('val_accuracy', 'N/A'):.4f}
        - **Training Loss**: {training_metrics.get('train_loss', 'N/A'):.4f}
        - **Validation Loss**: {training_metrics.get('val_loss', 'N/A'):.4f}
        
        ### Training Configuration
        - **Model Type**: Random Forest
        - **Number of Estimators**: 100
        - **Max Depth**: 10
        - **Random State**: 42
        """

        return report

    def generate_evaluation_report(self, metrics):
        """Generate model evaluation report section"""
        evaluation_metrics = metrics.get(
            'evaluation_metrics', {}).get('model_evaluation', {})

        report = f"""
        ## 📈 Model Evaluation Report
        
        ### Test Set Performance
        - **Accuracy**: {evaluation_metrics.get('accuracy', 'N/A'):.4f}
        - **Precision**: {evaluation_metrics.get('precision', 'N/A'):.4f}
        - **Recall**: {evaluation_metrics.get('recall', 'N/A'):.4f}
        - **F1-Score**: {evaluation_metrics.get('f1', 'N/A'):.4f}
        - **ROC-AUC**: {evaluation_metrics.get('roc_auc', 'N/A'):.4f}
        
        ### Performance Interpretation
        {self._interpret_performance(evaluation_metrics)}
        """

        return report

    def generate_cross_validation_report(self, metrics):
        """Generate cross-validation report section"""
        cv_metrics = metrics.get('cross_validation_metrics', {})

        report = f"""
        ## 🔄 Cross-Validation Report
        
        ### CV Performance
        - **Mean Accuracy**: {cv_metrics.get('mean_accuracy', 'N/A'):.4f}
        - **Standard Deviation**: {cv_metrics.get('std_accuracy', 'N/A'):.4f}
        - **Best Fold Score**: {cv_metrics.get('best_score', 'N/A'):.4f}
        - **Worst Fold Score**: {cv_metrics.get('worst_score', 'N/A'):.4f}
        
        ### Model Stability
        {self._interpret_model_stability(cv_metrics)}
        """

        return report

    def _interpret_performance(self, metrics):
        """Interpret model performance metrics"""
        accuracy = metrics.get('accuracy', 0)
        f1 = metrics.get('f1', 0)
        roc_auc = metrics.get('roc_auc', 0)

        interpretations = []

        if accuracy >= 0.85:
            interpretations.append(
                "✅ **Excellent** accuracy for hypertension prediction")
        elif accuracy >= 0.75:
            interpretations.append(
                "⚠️ **Good** accuracy, consider hyperparameter tuning")
        else:
            interpretations.append(
                "❌ **Poor** accuracy, model may need improvement")

        if f1 >= 0.8:
            interpretations.append(
                "✅ **Excellent** balance between precision and recall")
        elif f1 >= 0.7:
            interpretations.append(
                "⚠️ **Good** F1-score, acceptable for clinical use")
        else:
            interpretations.append("❌ **Poor** F1-score, model may be biased")

        if roc_auc >= 0.9:
            interpretations.append("✅ **Outstanding** discrimination ability")
        elif roc_auc >= 0.8:
            interpretations.append("⚠️ **Good** discrimination ability")
        else:
            interpretations.append("❌ **Poor** discrimination ability")

        return "\n".join([f"- {interpretation}" for interpretation in interpretations])

    def _interpret_model_stability(self, cv_metrics):
        """Interpret cross-validation stability"""
        std = cv_metrics.get(
            'std_accuracy', 1.0)  # Default to high variance if missing

        if std <= 0.02:
            return "- ✅ **High stability**: Model performs consistently across folds"
        elif std <= 0.05:
            return "- ⚠️ **Moderate stability**: Acceptable variation across folds"
        else:
            return "- ❌ **Low stability**: High variance suggests overfitting or data issues"

    def create_performance_dashboard(self, metrics):
        """Create a performance dashboard visualization"""
        evaluation_metrics = metrics.get(
            'evaluation_metrics', {}).get('model_evaluation', {})

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.project_name} - Performance Dashboard',
                     fontsize=16, fontweight='bold')

        # Plot 1: Main metrics radar chart
        self._plot_metrics_radar(evaluation_metrics, axes[0, 0])

        # Plot 2: Confusion matrix (if available)
        self._plot_confusion_matrix_preview(axes[0, 1])

        # Plot 3: Feature importance (if available)
        self._plot_feature_importance_preview(axes[1, 0])

        # Plot 4: ROC curve (if available)
        self._plot_roc_curve_preview(axes[1, 1])

        plt.tight_layout()
        plt.savefig(self.reports_dir / "plots" / "performance_dashboard.png",
                    bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()

    def _plot_metrics_radar(self, metrics, ax):
        """Create radar chart of main metrics"""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [metrics.get(metric, 0) for metric in metrics_to_plot]

        # Complete the circle
        angles = np.linspace(
            0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        metric_names += [metric_names[0]]

        ax.plot(angles, values, 'o-', linewidth=2,
                color=self.colors[0], label='Performance')
        ax.fill(angles, values, alpha=0.25, color=self.colors[0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Metrics', fontweight='bold')
        ax.grid(True)

    def _plot_confusion_matrix_preview(self, ax):
        """Preview confusion matrix if available"""
        try:
            cm_path = self.reports_dir / "plots" / "confusion_matrix.png"
            if cm_path.exists():
                img = plt.imread(cm_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title('Confusion Matrix', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Confusion Matrix\nNot Available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('#f0f0f0')
        except:
            ax.text(0.5, 0.5, 'Error loading\nconfusion matrix',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_feature_importance_preview(self, ax):
        """Preview feature importance if available"""
        try:
            fi_path = self.reports_dir / "plots" / "feature_importance.png"
            if fi_path.exists():
                img = plt.imread(fi_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title('Feature Importance', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Feature Importance\nNot Available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('#f0f0f0')
        except:
            ax.text(0.5, 0.5, 'Error loading\nfeature importance',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_roc_curve_preview(self, ax):
        """Preview ROC curve if available"""
        try:
            roc_path = self.reports_dir / "plots" / "roc_curve.png"
            if roc_path.exists():
                img = plt.imread(roc_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title('ROC Curve', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'ROC Curve\nNot Available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('#f0f0f0')
        except:
            ax.text(0.5, 0.5, 'Error loading\nROC curve',
                    ha='center', va='center', transform=ax.transAxes)

    def generate_model_card(self, metrics, params):
        """Generate Model Card for model documentation"""

        model_card = f"""# Model Card: Hypertension Risk Prediction Model

## Model Details
- **Model Name**: Hypertension Risk Classifier
- **Version**: 1.0.0
- **Date**: {self.timestamp}
- **Framework**: Scikit-learn
- **Model Type**: Random Forest Classifier

## Intended Use
- **Primary Use**: Predict hypertension risk based on patient vitals and medical history
- **Intended Users**: Healthcare professionals, clinical researchers
- **Out-of-Scope Uses**: Diagnosis without clinical supervision, treatment decisions

## Performance

### Evaluation Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | {metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('accuracy', 'N/A'):.3f} | {"Excellent" if metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('accuracy', 0) >= 0.8 else "Good" if metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('accuracy', 0) >= 0.7 else "Needs Improvement"} |
| Precision | {metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('precision', 'N/A'):.3f} | {"High precision" if metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('precision', 0) >= 0.8 else "Moderate precision"} |
| Recall | {metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('recall', 'N/A'):.3f} | {"High recall" if metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('recall', 0) >= 0.8 else "Moderate recall"} |
| F1-Score | {metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('f1', 'N/A'):.3f} | {"Well balanced" if metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('f1', 0) >= 0.8 else "Moderate balance"} |

## Training Data
- **Dataset**: Hypertension Clinical Measurements
- **Samples**: {metrics.get('data_validation', {{}}).get('total_samples', 'N/A')}
- **Features**: 10 clinical and demographic features
- **Class Distribution**: Binary (Hypertension vs No Hypertension)

## Ethical Considerations
- **Bias Analysis**: Model should be validated across different demographic groups
- **Clinical Use**: Should be used as decision support, not replacement for clinical judgment
- **Data Privacy**: All patient data should be anonymized and handled according to HIPAA guidelines

## Limitations
- Model performance may vary with different patient populations
- Requires complete feature set for accurate predictions
- Not validated for pediatric or special populations

## Maintenance
- **Retraining Schedule**: Quarterly or when significant data drift detected
- **Monitoring**: Accuracy, precision, recall should be monitored in production
- **Version Control**: All model versions tracked using DVC

---
*This model card was automatically generated on {self.timestamp}*
"""

        # Save model card
        with open(self.reports_dir / "model_card.md", "w") as f:
            f.write(model_card)

        return model_card

    def generate_final_report(self):
        """Generate the final comprehensive HTML report"""

        # Load all metrics and parameters
        metrics = self.load_metrics()
        params = self.load_params()

        # Generate report sections
        data_report = self.generate_data_report(metrics)
        training_report = self.generate_training_report(metrics)
        evaluation_report = self.generate_evaluation_report(metrics)
        cv_report = self.generate_cross_validation_report(metrics)

        # Create performance dashboard
        self.create_performance_dashboard(metrics)

        # Generate model card
        model_card = self.generate_model_card(metrics, params)

        # Create HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.project_name} - Final Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; 
                          border-left: 4px solid #667eea; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                               gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 8px; 
                              box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .dashboard {{ text-align: center; margin: 30px 0; }}
                .dashboard img {{ max-width: 100%; height: auto; border-radius: 8px; 
                                box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 {self.project_name}</h1>
                <h2>Final Project Report</h2>
                <p class="timestamp">Generated on: {self.timestamp}</p>
            </div>
            
            <div class="section">
                <h2>📋 Executive Summary</h2>
                <p>This report summarizes the development and performance of the Hypertension Risk Prediction Model. 
                The model demonstrates strong predictive capabilities for identifying patients at risk of hypertension 
                based on clinical measurements and demographic data.</p>
            </div>
            
            <div class="dashboard">
                <h2>📊 Performance Dashboard</h2>
                <img src="plots/performance_dashboard.png" alt="Performance Dashboard">
            </div>
            
            <div class="section">
                <h2>🎯 Key Metrics Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('accuracy', 'N/A'):.3f}</div>
                        <div>Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('precision', 'N/A'):.3f}</div>
                        <div>Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('recall', 'N/A'):.3f}</div>
                        <div>Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('evaluation_metrics', {{}}).get('model_evaluation', {{}}).get('f1', 'N/A'):.3f}</div>
                        <div>F1-Score</div>
                    </div>
                </div>
            </div>
            
            {data_report.replace('##', '<div class="section"><h2>').replace('###', '<h3>').replace('</h3>', '</h3>') + '</div>'}
            {training_report.replace('##', '<div class="section"><h2>').replace('###', '<h3>').replace('</h3>', '</h3>') + '</div>'}
            {evaluation_report.replace('##', '<div class="section"><h2>').replace('###', '<h3>').replace('</h3>', '</h3>') + '</div>'}
            {cv_report.replace('##', '<div class="section"><h2>').replace('###', '<h3>').replace('</h3>', '</h3>') + '</div>'}
            
            <div class="section">
                <h2>📁 Additional Resources</h2>
                <ul>
                    <li><a href="model_card.md">Model Card Documentation</a></li>
                    <li><a href="plots/confusion_matrix.png">Confusion Matrix</a></li>
                    <li><a href="plots/roc_curve.png">ROC Curve</a></li>
                    <li><a href="plots/feature_importance.png">Feature Importance</a></li>
                </ul>
            </div>
            
            <div class="timestamp" style="text-align: center; margin-top: 50px;">
                Report generated automatically by Hypertension Monitoring System
            </div>
        </body>
        </html>
        """

        # Save HTML report
        with open(self.reports_dir / "final_report.html", "w") as f:
            f.write(html_report)

        print("✅ Final report generated: reports/final_report.html")
        print("✅ Model card generated: reports/model_card.md")
        print("✅ Performance dashboard: reports/plots/performance_dashboard.png")

        return html_report


def main():
    """Main function to generate all reports"""
    print("📊 Generating Comprehensive Reports...")

    generator = ReportGenerator()
    generator.generate_final_report()

    print("🎉 All reports generated successfully!")
    print("📍 Location: reports/ directory")


if __name__ == '__main__':
    main()
