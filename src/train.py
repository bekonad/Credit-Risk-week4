"""
train.py
---------
Task 5: Model Training & Experiment Tracking with MLflow
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/processed/processed.csv"
TARGET_COL = "is_high_risk"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# -----------------------------
# Load Data
# -----------------------------
def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at {path}")
    return pd.read_csv(path)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }


# -----------------------------
# Main
# -----------------------------
def main():

    print("Loading processed data...")
    df = load_data(DATA_PATH)

    X = df.drop(columns=[TARGET_COL, "CustomerId"])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training and testing data prepared.")

    mlflow.set_experiment("Credit_Risk_Week4")

    # =====================================================
    # Logistic Regression Run
    # =====================================================
    with mlflow.start_run(run_name="Logistic_Regression"):

        lr = LogisticRegression(
            max_iter=500,
            random_state=RANDOM_STATE
        )
        lr.fit(X_train, y_train)

        metrics = evaluate(lr, X_test, y_test)

        mlflow.log_params(lr.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            lr,
            name="logistic_regression",
            registered_model_name="CreditRisk_LogisticRegression"
        )

        print("Logged Logistic Regression")

    # =====================================================
    # Random Forest Run
    # =====================================================
    with mlflow.start_run(run_name="Random_Forest"):

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        )
        rf.fit(X_train, y_train)

        metrics = evaluate(rf, X_test, y_test)

        mlflow.log_params(rf.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            rf,
            name="random_forest",
            registered_model_name="CreditRisk_RandomForest"
        )

        print("Logged Random Forest")

    print("\nTraining completed successfully.")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
