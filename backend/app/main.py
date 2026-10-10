"""
CARDIOVISION AI — FastAPI Backend
=================================
Production-style backend serving the trained ML model.
All predictions come from the actual trained pipeline.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─── Global State ───────────────────────────────────────────────────────────
model = None
metrics = None
input_schema = None
dataset_summary = None
demo_patients = None
preprocessing_info = None
audit_info = None


def load_resources():
    """Load all resources at startup."""
    global model, metrics, input_schema, dataset_summary, demo_patients, preprocessing_info, audit_info

    # Load model
    model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️  Model not found at {model_path}")

    # Load JSON resources
    json_files = {
        "metrics": "metrics.json",
        "input_schema": "input_schema.json",
        "dataset_summary": "dataset_summary.json",
        "demo_patients": "demo_patients.json",
        "preprocessing_info": "preprocessing.json",
        "audit_info": "audit.json",
    }
    for var_name, filename in json_files.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                globals()[var_name] = json.load(f)
            print(f"✅ Loaded {filename}")
        else:
            print(f"⚠️  {filename} not found")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources on startup."""
    load_resources()
    yield


# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CardioVision AI",
    description="Interactive Cardiovascular Risk Visualization & Prediction API. "
                "This is an AI-based educational/research prototype, NOT a medical diagnostic tool.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ───────────────────────────────────────────────────────
class PatientInput(BaseModel):
    """Patient data input schema — matches actual dataset features."""
    age: float = Field(..., ge=18, le=100, description="Age in years")
    gender: int = Field(..., ge=1, le=2, description="1=Female, 2=Male")
    height: float = Field(..., ge=100, le=250, description="Height in cm")
    weight: float = Field(..., ge=20, le=300, description="Weight in kg")
    ap_hi: int = Field(..., ge=60, le=300, description="Systolic BP in mmHg")
    ap_lo: int = Field(..., ge=30, le=200, description="Diastolic BP in mmHg")
    cholesterol: int = Field(..., ge=1, le=3, description="1=Normal, 2=Above Normal, 3=Well Above Normal")
    gluc: int = Field(..., ge=1, le=3, description="1=Normal, 2=Above Normal, 3=Well Above Normal")
    smoke: int = Field(..., ge=0, le=1, description="0=Non-smoker, 1=Smoker")
    alco: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    active: int = Field(..., ge=0, le=1, description="0=Not Active, 1=Active")

    @field_validator("ap_lo")
    @classmethod
    def diastolic_less_than_systolic(cls, v, info):
        if "ap_hi" in info.data and v >= info.data["ap_hi"]:
            raise ValueError("Diastolic BP must be less than systolic BP")
        return v


class PredictionResult(BaseModel):
    prediction: int
    prediction_label: str
    model_probability: float
    ai_risk_index: int
    risk_category: str
    top_contributors: list
    disclaimer: str


class ContributorInfo(BaseModel):
    feature: str
    display_name: str
    value: float
    contribution: float
    direction: str


# ─── Helper Functions ───────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Physical Activity",
    "bmi": "BMI",
    "pulse_pressure": "Pulse Pressure",
    "map": "Mean Arterial Pressure",
    "gender_2": "Male Gender",
    "cholesterol_2": "Cholesterol Above Normal",
    "cholesterol_3": "Cholesterol Well Above Normal",
    "gluc_2": "Glucose Above Normal",
    "gluc_3": "Glucose Well Above Normal",
}


def prepare_input(patient: PatientInput) -> pd.DataFrame:
    """Convert patient input to the DataFrame format expected by the model."""
    data = patient.model_dump()
    # Add engineered features
    data["bmi"] = round(data["weight"] / ((data["height"] / 100) ** 2), 1)
    data["pulse_pressure"] = data["ap_hi"] - data["ap_lo"]
    data["map"] = round(data["ap_lo"] + (data["ap_hi"] - data["ap_lo"]) / 3, 1)

    # Must match the exact feature order used during training
    feature_order = [
        "age", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure", "map",
        "smoke", "alco", "active",
        "gender", "cholesterol", "gluc"
    ]
    return pd.DataFrame([{k: data[k] for k in feature_order}])


def compute_risk(probability: float) -> tuple:
    """
    Compute AI Risk Index and category from model probability.

    AI Risk Index = probability × 100 (0–100 scale)
    Categories:
        LOW:      index < 35
        MODERATE: 35 ≤ index < 65
        HIGH:     index ≥ 65

    These thresholds are model-derived, NOT clinically validated.
    """
    risk_index = round(probability * 100)
    risk_index = max(0, min(100, risk_index))

    if risk_index < 35:
        category = "LOW"
    elif risk_index < 65:
        category = "MODERATE"
    else:
        category = "HIGH"

    return risk_index, category


def compute_contributions(patient: PatientInput, prediction_prob: float) -> list:
    """
    Compute feature contributions using permutation-based approach.
    For each feature, compares the prediction with the original value
    vs a baseline (median) value.
    """
    global model

    original_df = prepare_input(patient)
    original_prob = prediction_prob

    # Baseline values (approximate medians from the dataset)
    baselines = {
        "age": 54.0, "gender": 1, "height": 165, "weight": 72.0,
        "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1,
    }

    contributions = []
    input_fields = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                    "cholesterol", "gluc", "smoke", "alco", "active"]

    for feature in input_fields:
        if feature in baselines:
            # Create modified input with baseline value for this feature
            modified = patient.model_dump()
            modified[feature] = baselines[feature]

            # Recompute derived features
            modified["bmi"] = round(modified["weight"] / ((modified["height"] / 100) ** 2), 1)
            modified["pulse_pressure"] = modified["ap_hi"] - modified["ap_lo"]
            modified["map"] = round(modified["ap_lo"] + (modified["ap_hi"] - modified["ap_lo"]) / 3, 1)

            feature_order = [
                "age", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure", "map",
                "smoke", "alco", "active",
                "gender", "cholesterol", "gluc"
            ]
            modified_df = pd.DataFrame([{k: modified[k] for k in feature_order}])

            try:
                modified_prob = model.predict_proba(modified_df)[0][1]
                contribution = original_prob - modified_prob
            except Exception:
                contribution = 0.0

            raw_value = getattr(patient, feature)
            direction = "increases model risk" if contribution > 0 else "decreases model risk"

            contributions.append({
                "feature": feature,
                "display_name": FEATURE_DISPLAY_NAMES.get(feature, feature),
                "value": raw_value,
                "contribution": round(contribution, 4),
                "direction": direction,
            })

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


# ─── ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "metrics_loaded": metrics is not None,
        "version": "1.0.0",
    }


@app.get("/dataset-summary")
async def get_dataset_summary():
    """Return actual dataset statistics."""
    if dataset_summary is None:
        raise HTTPException(status_code=503, detail="Dataset summary not loaded")
    return dataset_summary


@app.get("/model-performance")
async def get_model_performance():
    """Return actual model metrics from metrics.json."""
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not loaded")
    return metrics


@app.get("/demo-patients")
async def get_demo_patients():
    """Return demo patient profiles derived from actual dataset rows."""
    if demo_patients is None:
        raise HTTPException(status_code=503, detail="Demo patients not loaded")
    return demo_patients


@app.get("/input-schema")
async def get_input_schema():
    """Return the input form schema derived from actual dataset columns."""
    if input_schema is None:
        raise HTTPException(status_code=503, detail="Input schema not loaded")
    return input_schema


@app.post("/predict", response_model=PredictionResult)
async def predict(patient: PatientInput):
    """
    Predict cardiovascular disease risk for a patient.

    Returns model prediction, probability, AI Risk Index, risk category,
    and top contributing factors.

    This is an educational/research prototype — NOT a medical diagnostic tool.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure the model has been trained.")

    try:
        input_df = prepare_input(patient)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        risk_index, risk_category = compute_risk(probability)
        contributors = compute_contributions(patient, probability)

        return PredictionResult(
            prediction=prediction,
            prediction_label="CVD Risk Detected" if prediction == 1 else "Lower CVD Risk",
            model_probability=round(probability, 4),
            ai_risk_index=risk_index,
            risk_category=risk_category,
            top_contributors=contributors,
            disclaimer="This is an AI-based educational/research prototype and is not a medical "
                       "diagnostic tool. Predictions should not be used to make medical decisions. "
                       "Consult a qualified healthcare professional for clinical assessment.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain")
async def explain(patient: PatientInput):
    """
    Provide detailed model explanation for a patient prediction.

    Feature contributions represent MODEL CONTRIBUTION to the prediction,
    NOT physical anatomical damage.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = prepare_input(patient)
        probability = float(model.predict_proba(input_df)[0][1])
        risk_index, risk_category = compute_risk(probability)
        contributors = compute_contributions(patient, probability)

        # Separate into risk-increasing and risk-decreasing
        increasing = [c for c in contributors if c["contribution"] > 0.001]
        decreasing = [c for c in contributors if c["contribution"] < -0.001]
        neutral = [c for c in contributors if -0.001 <= c["contribution"] <= 0.001]

        return {
            "model_probability": round(probability, 4),
            "ai_risk_index": risk_index,
            "risk_category": risk_category,
            "risk_increasing_factors": increasing,
            "risk_decreasing_factors": decreasing,
            "neutral_factors": neutral,
            "all_contributors": contributors,
            "explanation_method": "Perturbation-based feature contribution (baseline comparison)",
            "disclaimer": "Feature contributions represent MODEL CONTRIBUTION to the prediction, "
                          "NOT physical anatomical damage. This is an educational prototype.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


# ─── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
