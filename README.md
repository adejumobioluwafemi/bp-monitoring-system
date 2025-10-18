Here's the updated README.md with DVC workflow and conda environment information:

```markdown
# Hypertension Risk Prediction System

A comprehensive machine learning system for predicting hypertension risk based on clinical parameters and lifestyle factors. This system includes data processing, model training, MLflow tracking, model registry, FastAPI serving, and Streamlit web interface.

## 🏥 System Overview

This system provides:
- **Data Processing Pipeline**: Clean and preprocess clinical data
- **Machine Learning Training**: Train and evaluate hypertension prediction models
- **MLflow Integration**: Track experiments and manage model registry
- **FastAPI Service**: REST API for model inference
- **Streamlit Web App**: User-friendly interface for risk assessment
- **Model Registry**: Version control and promotion workflow
- **DVC Pipeline**: Reproducible data and model versioning

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda
- MLflow (for model tracking)
- DVC (for data versioning)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd bp-monitoring-system
   ```

2. **Set up Conda environment**
   ```bash
   # Create and activate conda environment from environment.yml
   conda env create -f environment.yml
   conda activate bp-monitoring-env
   ```

3. **Set up DVC, [My note to Setup DVC Remote](https://docs.google.com/document/d/13OdkTFmlJOf35u6fjISQU2DwgH7IHpSwar9N4SoRJag/edit?usp=sharing)**
   ```bash
   # Initialize DVC (if not already done)
   dvc init
   
   # Add remote storage (if using cloud storage)
   dvc remote add -d mys3remote s3://your-bucket/path
   
   # Pull data from remote storage
   dvc pull
   ```

## 📁 Project Structure

```
bp-monitoring-system/
├── data/
│   ├── raw/                    # Original dataset (DVC tracked)
│   └── processed/              # Processed train/val/test splits
├── src/
│   ├── applications/           # Training and evaluation scripts
│   │   ├── training.py         # Model training with MLflow
│   │   ├── model_registry.py   # Model registry management
│   │   └── model_promotion.py  # Model promotion workflows
│   ├── api/                    # FastAPI service
│   │   ├── app.py              # Main FastAPI application
│   │   ├── model_loader.py     # Model loading and preprocessing
│   │   └── preprocessor.py     # Data preprocessing utilities
│   └── ml/                     # ML pipeline components
│       ├── data_processor.py   # Data processing pipeline
│       ├── training_pipeline.py # Training pipeline
│       └── model.py            # Model definitions
├── models/                     # Model storage
│   ├── dvc-tracked/            # DVC versioned models
│   ├── production/             # Production models
│   └── staging/                # Staging models
├── mlruns/                     # MLflow experiment tracking
├── reports/                    # Training metrics and reports
├── app.py                      # Streamlit web application
├── params.yaml                 # Configuration parameters
├── environment.yml             # Conda environment configuration
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC pipeline lock file
└── metrics.json                # Evaluation metrics
```

## 🛠️ Usage

### Option 1: Complete DVC Pipeline (Recommended)

Run the entire reproducible pipeline with DVC:

```bash
# Run complete pipeline: data preparation → training → evaluation
dvc repro

# View pipeline structure
dvc dag

# Push results to remote storage
dvc push
```

### Option 2: Manual Step-by-Step Execution

#### 1. Data Preparation

```bash
# Prepare and split the data
python src/applications/prepare_data.py

# Validate data quality
python src/applications/validate_data.py
```

#### 2. Model Training

```bash
# Train model with MLflow tracking
python src/applications/training.py --train-only

```

#### 3. Start MLflow Server (Optional), [My Note on MLFlow](https://docs.google.com/document/d/1GVk13aiVWKfKty9ykJfkc86kE4OQkVglbpQcF6xkU2c/edit?usp=sharing)

```bash
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db
```

View experiments at: http://localhost:8080

#### 4. Model Serving

**Start FastAPI Service: [My note on FastAPI](https://docs.google.com/document/d/1ZkaxWxFpjgfr_L9k2PDkpfEEnpQQ8MlROIKeYkuIakw/edit?usp=sharing)**
```bash
python src/api/app.py
```
API Documentation: http://localhost:8000/docs

**Start Streamlit Web App:**
```bash
streamlit run app.py
```
Web Interface: http://localhost:8501

#### 5. Model Management

```bash
# Promote model to staging
python src/applications/model_promotion.py --promote-dvc <version>

# Check model versions
python src/applications/model_registry.py --list-dvc
```

## 🔄 DVC Pipeline Stages

The DVC pipeline (`dvc.yaml`) includes:

### Stage 1: Data Preparation
- **Script**: `src/applications/prepare_data.py`
- **Inputs**: `data/raw/Hypertension_Data_Set.csv`
- **Outputs**: `data/processed/` (train/val/test splits)
- **Metrics**: `reports/data_metrics.json`

### Stage 2: Data Validation
- **Script**: `src/applications/validate_data.py`
- **Inputs**: `data/processed/`
- **Outputs**: `reports/data_validation_report.json`

### Stage 3: Model Training
- **Script**: `src/applications/training.py`
- **Inputs**: `data/processed/`, `params.yaml`
- **Outputs**: `models_chpt/`, `reports/training_metrics.json`
- **Metrics**: `metrics.json`

### Stage 4: Model Evaluation
- **Script**: `src/applications/evaluate_model.py`
- **Inputs**: `models_chpt/`, `data/processed/`
- **Outputs**: `reports/evaluation/` (plots and reports)

### Running Individual DVC Stages

```bash
# Run specific stage
dvc repro prepare_data
dvc repro train_model

# Force reprocess specific stage
dvc repro --force train_model

# Check pipeline status
dvc status
```

## 📊 Model Features

The system uses the following clinical parameters for hypertension risk prediction:

### Numerical Features
- **Age**: Patient age in years
- **BMI**: Body Mass Index
- **Systolic_BP**: Systolic blood pressure (mmHg)
- **Diastolic_BP**: Diastolic blood pressure (mmHg)
- **Heart_Rate**: Resting heart rate (bpm)

### Categorical Features
- **Gender**: Male or Female
- **Medical_History**: Family history of hypertension (Yes/No)
- **Smoking**: Current smoking status (Yes/No)
- **Sporting**: Regular exercise 3-4 times/week (Yes/No)

## 🔧 API Endpoints

### FastAPI Service (http://localhost:8000)

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /model-info` - Model information
- `GET /preprocessor-info` - Preprocessing configuration
- `GET /debug-features` - Debug feature processing

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 45,
       "BMI": 25.0,
       "Systolic_BP": 120,
       "Diastolic_BP": 80,
       "Heart_Rate": 72,
       "Gender": "Female",
       "Medical_History": "No",
       "Smoking": "No",
       "Sporting": "Yes"
     }'
```

## 🎯 Model Promotion Workflow

1. **Training**: Models are trained and automatically registered in MLflow
2. **Evaluation**: Models are evaluated against predefined criteria
3. **Staging**: Qualified models are promoted to staging environment
4. **Production**: After validation, models are promoted to production
5. **Monitoring**: Production models are monitored for performance

```bash
# Automated promotion workflow
python src/applications/model_promotion.py --auto

# Manual promotion
python src/applications/model_promotion.py --promote-dvc 1

# Check production model
python src/applications/model_promotion.py --production-info
```

## 🔍 Monitoring and Debugging

### MLflow Tracking
```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### DVC Data Versioning
```bash
# Check data and model versions
dvc status
dvc diff

# View metrics history
dvc metrics show
dvc metrics diff

# Compare pipeline changes
dvc params diff
```

### Model Registry
```bash
# List all model versions
python src/applications/model_registry.py --list-dvc

# Get production model info
python src/applications/model_promotion.py --production-info
```

## 📈 Model Performance

The system includes comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion matrices
- Feature importance analysis
- Training/validation performance comparison

## 🚨 Important Notes

- **Medical Disclaimer**: This system is for educational and screening purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
- **Data Privacy**: Ensure compliance with healthcare data regulations when using real patient data.
- **Model Validation**: Always validate models with clinical experts before deployment in healthcare settings.
- **Reproducibility**: Use DVC to ensure reproducible experiments and model versions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update DVC pipeline if needed
6. Submit a pull request

### Adding New Pipeline Stages

When adding new stages to the DVC pipeline:

1. Update `dvc.yaml` with new stage definition
2. Run `dvc repro` to test the pipeline
3. Commit changes to DVC files:
   ```bash
   git add dvc.yaml dvc.lock .dvc/
   git commit -m "Add new pipeline stage"
   ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Port already in use**: 
   ```bash
   lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
   ```

2. **Model not found**:
   - Check if model files exist in `models_chpt/` or `models/`
   - Run training to generate models: `dvc repro train_model`

3. **Import errors**:
   - Ensure you're in the correct conda environment: `conda activate bp-monitoring-env`
   - Check Python path includes project root

4. **DVC pipeline issues**:
   ```bash
   # Reset and rerun pipeline
   dvc destroy
   dvc repro
   
   # Check DVC status
   dvc doctor
   ```

5. **Conda environment issues**:
   ```bash
   # Recreate environment
   conda deactivate
   conda env remove -n bp-monitoring-env
   conda env create -f environment.yml
   ```

### Getting Help

- Check the MLflow UI for experiment tracking
- Use the `/debug-features` API endpoint for preprocessing issues
- Review training logs in `mlruns/` directory
- Check DVC pipeline status with `dvc status` and `dvc dag`
```
