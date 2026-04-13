"""Shared prediction utilities for the phishing URL detector."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from feature_extractor import extract_features


MODEL_PATH = "phishing_model.pkl"
METRICS_PATH = "model_metrics.json"


@dataclass
class PredictionResult:
    """Structured phishing prediction details for CLI and web usage."""

    url: str
    normalized_url: str
    predicted_class: int
    label: str
    confidence: float
    features: dict[str, float | int]
    risk_band: str


def _risk_band(confidence: float) -> str:
    """Convert confidence into a human-friendly risk band."""
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.65:
        return "Medium"
    return "Low"


def load_model(model_path: str = MODEL_PATH):
    """Load a trained model from disk."""
    return joblib.load(model_path)


def predict_url(url: str, model_path: str = MODEL_PATH) -> PredictionResult:
    """Predict whether a URL is safe or phishing."""
    cleaned_url = (url or "").strip()
    if not cleaned_url:
        raise ValueError("Please enter a URL to analyze.")

    model = load_model(model_path)
    features = extract_features(cleaned_url)
    X_input = pd.DataFrame([features]).fillna(0)

    predicted_class = int(model.predict(X_input)[0])
    probabilities = model.predict_proba(X_input)[0]

    if predicted_class == 1:
        label = "PHISHING"
        confidence = float(probabilities[1])
    else:
        label = "SAFE"
        confidence = float(probabilities[0])

    return PredictionResult(
        url=cleaned_url,
        normalized_url=cleaned_url,
        predicted_class=predicted_class,
        label=label,
        confidence=confidence,
        features=features,
        risk_band=_risk_band(confidence),
    )


def predict_urls(urls: list[str], model_path: str = MODEL_PATH) -> list[PredictionResult]:
    """Predict multiple URLs, skipping blank lines."""
    cleaned_urls = [url.strip() for url in urls if url and url.strip()]
    if not cleaned_urls:
        raise ValueError("Please enter at least one URL to analyze.")

    return [predict_url(url, model_path=model_path) for url in cleaned_urls]


def load_model_metrics(metrics_path: str = METRICS_PATH) -> dict:
    """Load saved model evaluation metadata if available."""
    if not Path(metrics_path).exists():
        return {}

    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)
