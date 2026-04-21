from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schema import PredictionRequest, PredictionResponse

app = FastAPI(title="Bank Churn Predictor")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "app" / "model.pkl"
_model: Any | None = None


def get_model() -> Any:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Model file not found at '{MODEL_PATH}'. "
                    "Please run `python training/train.py` from the repository root first."
                ),
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    model = get_model()
    features = pd.DataFrame([payload.model_dump()])

    try:
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    label = "churn" if prediction == 1 else "stay"
    return PredictionResponse(label=label, probability=probability)
