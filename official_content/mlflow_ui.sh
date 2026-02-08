#!/bin/bash
# MLflow UI起動スクリプト

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MLRUNS_PATH="$REPO_ROOT/mlruns"

echo "Starting MLflow UI..."
echo "Tracking URI: file://$MLRUNS_PATH"
echo "Access at: http://localhost:58000"
echo ""

mlflow ui --backend-store-uri "file://$MLRUNS_PATH" --port 58000
