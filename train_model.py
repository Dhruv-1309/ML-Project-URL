"""Train a phishing URL detector using handcrafted URL features."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from feature_extractor import extract_features


DATASET_PATH = "dataset.csv"
MODEL_PATH = "phishing_model.pkl"
METRICS_PATH = "model_metrics.json"


def main() -> None:
    # Load CSV from project root.
    df = pd.read_csv(DATASET_PATH)

    # Basic validation for required columns.
    required_columns = {"url", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Fill missing URL values and ensure string type.
    url_series = df["url"].fillna("").astype(str)

    # Fill missing labels with 0 and force integer labels.
    y = df["label"].fillna(0).astype(int)

    # Extract features from each URL into a DataFrame.
    feature_rows = url_series.apply(extract_features)
    X = pd.DataFrame(list(feature_rows))

    # Fill any missing feature values with 0.
    X = X.fillna(0)

    # Split into train/test sets (20% test).
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train a Random Forest model with stronger defaults for this tabular task.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate on test data.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    confusion = confusion_matrix(y_test, y_pred)
    feature_importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(8)
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion)
    print("\nTop Feature Importances:")
    print(feature_importance.round(4).to_string())

    # Save trained model for later predictions.
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    metrics_payload = {
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": confusion.tolist(),
        "feature_importance": [
            {"feature": feature, "importance": round(float(score), 4)}
            for feature, score in feature_importance.items()
        ],
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
