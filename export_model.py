#!/usr/bin/env python3
"""
export_model.py — Export trained model from MLflow to pickle file
Run this after training to prepare model for serving.
"""

import os
import sys
import pickle
import logging
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def export_model(run_id=None):
    """Export model from MLflow run to pickle file."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if run_id:
        # Load from specific run
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Loading model from run: {run_id}")
    else:
        # Load latest run
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=["0"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            logger.error("No MLflow runs found!")
            sys.exit(1)
        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Loading model from latest run: {run_id}")

    # Load model
    model = mlflow.sklearn.load_model(model_uri)

    # Save as pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    file_size = os.path.getsize(MODEL_PATH)
    logger.info(f"Model exported to {MODEL_PATH} ({file_size} bytes)")
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Features: {model.n_features_in_}")

    return MODEL_PATH


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    export_model(run_id)
