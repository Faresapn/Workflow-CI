"""
inference.py — Model Serving + Prometheus Metrics Exporter
Mushroom Classification using trained Random Forest model.
Exposes /predict endpoint and /metrics for Prometheus.
"""

import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from sklearn.base import BaseEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Load Model
# ============================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model", "model.pkl"),
)

model: BaseEstimator = None


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {MODEL_PATH}")


# ============================================================
# Prometheus Metrics
# ============================================================

# Metric 1: Total inference requests
INFERENCE_REQUEST_TOTAL = Counter(
    "inference_request_total",
    "Total number of inference requests",
    ["endpoint", "status"],
)

# Metric 2: Inference request duration
INFERENCE_REQUEST_DURATION = Histogram(
    "inference_request_duration_seconds",
    "Time spent processing inference request",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Metric 3: Prediction class distribution
INFERENCE_PREDICTION_CLASS = Counter(
    "inference_prediction_class_total",
    "Count of predictions per class",
    ["predicted_class"],
)

# Metric 4: Model memory usage
MODEL_MEMORY_USAGE = Gauge(
    "model_memory_usage_bytes",
    "Approximate memory usage of the loaded model",
)

# Metric 5: Inference errors
INFERENCE_ERRORS_TOTAL = Counter(
    "inference_errors_total",
    "Total number of inference errors",
    ["error_type"],
)

# Info metric
MODEL_INFO = Info("model", "Information about the loaded model")


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)


@app.before_first_request
def initialize():
    load_model()
    # Estimate model memory usage
    model_size = os.path.getsize(MODEL_PATH)
    MODEL_MEMORY_USAGE.set(model_size)
    MODEL_INFO.info(
        {
            "name": "Mushroom_RF_Classifier",
            "version": "1.0.0",
            "features": str(model.n_features_in_) if hasattr(model, "n_features_in_") else "unknown",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        data = request.get_json(force=True)

        # Parse input
        if "features" in data:
            features = pd.DataFrame(data["features"])
        else:
            features = pd.DataFrame([data])

        # Predict
        prediction = model.predict(features)
        prediction_proba = (
            model.predict_proba(features).tolist()
            if hasattr(model, "predict_proba")
            else None
        )

        duration = time.time() - start_time

        # Update metrics
        INFERENCE_REQUEST_TOTAL.labels(endpoint="/predict", status="success").inc()
        INFERENCE_REQUEST_DURATION.labels(endpoint="/predict").observe(duration)

        for pred in prediction:
            INFERENCE_PREDICTION_CLASS.labels(predicted_class=str(pred)).inc()

        return jsonify(
            {
                "prediction": prediction.tolist(),
                "probability": prediction_proba,
                "duration_seconds": round(duration, 6),
            }
        )

    except Exception as e:
        INFERENCE_REQUEST_TOTAL.labels(endpoint="/predict", status="error").inc()
        INFERENCE_ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    INFERENCE_REQUEST_TOTAL.labels(endpoint="/health", status="success").inc()
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    load_model()
    model_size = os.path.getsize(MODEL_PATH)
    MODEL_MEMORY_USAGE.set(model_size)
    MODEL_INFO.info(
        {
            "name": "Mushroom_RF_Classifier",
            "version": "1.0.0",
            "features": str(model.n_features_in_) if hasattr(model, "n_features_in_") else "unknown",
        }
    )
    app.run(host="0.0.0.0", port=5001)
