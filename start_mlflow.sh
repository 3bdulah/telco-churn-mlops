#!/bin/bash
# Start the MLflow tracking server with SQLite backend and local artifact storage.
# Run from the project root directory.
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 \
    --port 5500
