#!/bin/bash
# Start the MLflow tracking server with SQLite backend and local artifact storage.
# Run from the project root directory.
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 127.0.0.1 \
    --port 5500
