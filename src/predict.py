from pathlib import Path
import json
import joblib
import pandas as pd

from .features import add_credit_features


def load_artifacts(model_path: Path, metadata_path: Path):
    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def prepare_input(raw_input: dict, excluded_predictors=None) -> pd.DataFrame:
    frame = pd.DataFrame([raw_input])
    frame = add_credit_features(frame)

    for col in excluded_predictors or []:
        if col in frame.columns:
            frame = frame.drop(columns=col)

    return frame


def predict_applicant(model, metadata, raw_input: dict, threshold=None):
    X = prepare_input(raw_input, metadata.get("excluded_predictors", []))
    probability = float(model.predict_proba(X)[0, 1])

    if threshold is None:
        threshold = float(metadata.get("default_threshold", 0.5))

    approved = probability >= threshold

    if probability >= 0.75:
        risk = "Low"
    elif probability >= 0.50:
        risk = "Moderate"
    elif probability >= 0.25:
        risk = "High"
    else:
        risk = "Very High"

    return {
        "probability": probability,
        "threshold": float(threshold),
        "approved": bool(approved),
        "decision": "Approved" if approved else "Rejected",
        "risk_level": risk,
        "prepared_input": X,
    }
