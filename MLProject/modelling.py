import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Aktifkan autologging MLflow
    mlflow.autolog()

    # Load dataset
    data_path = os.path.join(
        os.path.dirname(__file__),
        "mushrooms_preprocessed",
        "mushrooms_preprocessed.csv",
    )
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    # Pisahkan fitur dan target
    X = df.drop(columns=["class"])
    y = df["class"]

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")

    # Mulai MLflow run
    with mlflow.start_run():
        # Train model
        logger.info("Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log metrik manual (autolog juga sudah log)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        # Print hasil
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1 Score:  {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Log confusion matrix sebagai artifact
        np.savetxt("confusion_matrix.csv", conf_matrix, delimiter=",")
        mlflow.log_artifact("confusion_matrix.csv")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
