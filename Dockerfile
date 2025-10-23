# FROM continuumio/miniconda3:latest
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Copy and create environment first for better caching
#COPY environment.yml .
#RUN conda env create -f environment.yml && conda clean -a
#RUN pip install --no-cache-dir --timeout 100 "fastapi[standard]" uvicorn pydantic sqlalchemy
#RUN pip install --no-cache-dir --timeout 100 pandas numpy scikit-learn
#RUN pip install --no-cache-dir --timeout 100 python-multipart pyyaml requests
# Optional: Install MLflow and Plotly only if needed for inference
#RUN pip install --no-cache-dir --timeout 100 mlflow plotly

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY src/ ./src/
COPY models_chpt/ ./models_chpt/
RUN mkdir -p models/production/model
COPY models/production/model/ ./models/production/model/
COPY params.yaml ./

RUN mkdir -p /app/data/processed
#RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Note: curl is needed for the healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \ 
    CMD curl -f http://localhost:8000/health || exit 1 

#ENV PATH=/opt/conda/envs/bp-monitoring-env/bin:$PATH

# The SHELL directive ensures this runs in the conda environment
#CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "src/api/app.py"]