"""
CARDIOVISION AI — Production WSGI Web Application (PythonAnywhere Ready)
========================================================================
Serves both the React 3D frontend (SPA) and the machine learning REST API
as a single unified WSGI application callable (`application = app`).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
RESULTS_DIR = os.path.join(BACKEND_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "dist")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

# ─── Global State & Resources ────────────────────────────────────────────────
model = None
metrics = None
input_schema = None
dataset_summary = None
demo_patients = None
preprocessing_info = None
audit_info = None

def load_resources():
    global model, metrics, input_schema, dataset_summary, demo_patients, preprocessing_info, audit_info

    model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️ Model not found at {model_path}")

    json_files = {
        "metrics": "metrics.json",
        "input_schema": "input_schema.json",
        "dataset_summary": "dataset_summary.json",
        "demo_patients": "demo_patients.json",
        "preprocessing_info": "preprocessing.json",
        "audit_info": "audit.json",
    }
    for var_name, filename in json_files.items():
        p = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(p):
            with open(p, "r") as f:
                globals()[var_name] = json.load(f)
            print(f"✅ Loaded {filename}")
        else:
            print(f"⚠️ {filename} not found at {p}")

load_resources()

# ─── Feature Mapping & Calculations ──────────────────────────────────────────
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
}

BASELINES = {
    "age": 54.0, "gender": 1, "height": 165, "weight": 72.0,
    "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1,
}

FEATURE_ORDER = [
    "age", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure", "map",
    "smoke", "alco", "active",
    "gender", "cholesterol", "gluc"
]

def prepare_input(data: dict) -> pd.DataFrame:
    payload = dict(data)
    payload["bmi"] = round(payload["weight"] / ((payload["height"] / 100) ** 2), 1)
    payload["pulse_pressure"] = payload["ap_hi"] - payload["ap_lo"]
    payload["map"] = round(payload["ap_lo"] + (payload["ap_hi"] - payload["ap_lo"]) / 3, 1)
    return pd.DataFrame([{k: payload[k] for k in FEATURE_ORDER}])

def compute_risk(probability: float):
    risk_index = round(probability * 100)
    risk_index = max(0, min(100, risk_index))
    if risk_index < 35:
        category = "LOW"
    elif risk_index < 65:
        category = "MODERATE"
    else:
        category = "HIGH"
    return risk_index, category

def compute_contributions(data: dict, prediction_prob: float):
    contributions = []
    input_fields = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                    "cholesterol", "gluc", "smoke", "alco", "active"]

    for feature in input_fields:
        if feature in BASELINES:
            modified = dict(data)
            modified[feature] = BASELINES[feature]
            modified_df = prepare_input(modified)
            try:
                mod_prob = model.predict_proba(modified_df)[0][1]
                contrib = prediction_prob - mod_prob
            except Exception:
                contrib = 0.0

            raw_val = data.get(feature)
            direction = "increases model risk" if contrib > 0 else "decreases model risk"
            contributions.append({
                "feature": feature,
                "display_name": FEATURE_DISPLAY_NAMES.get(feature, feature),
                "value": raw_val,
                "contribution": round(contrib, 4),
                "direction": direction,
            })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions

# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "metrics_loaded": metrics is not None,
        "version": "1.0.0",
        "server": "PythonAnywhere WSGI (Flask)",
    })

@app.route("/dataset-summary", methods=["GET"])
def get_dataset_summary():
    if dataset_summary is None:
        return jsonify({"detail": "Dataset summary not loaded"}), 503
    return jsonify(dataset_summary)

@app.route("/model-performance", methods=["GET"])
def get_model_performance():
    if metrics is None:
        return jsonify({"detail": "Metrics not loaded"}), 503
    return jsonify(metrics)

@app.route("/demo-patients", methods=["GET"])
def get_demo_patients():
    if demo_patients is None:
        return jsonify({"detail": "Demo patients not loaded"}), 503
    return jsonify(demo_patients)

@app.route("/input-schema", methods=["GET"])
def get_input_schema():
    if input_schema is None:
        return jsonify({"detail": "Input schema not loaded"}), 503
    return jsonify(input_schema)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"detail": "Model not loaded"}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"detail": "No JSON payload provided"}), 400

    # Validation
    if data.get("ap_lo", 0) >= data.get("ap_hi", 0):
        return jsonify({"detail": "Diastolic BP must be less than systolic BP"}), 422

    try:
        input_df = prepare_input(data)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        risk_index, risk_category = compute_risk(probability)
        contributors = compute_contributions(data, probability)

        return jsonify({
            "prediction": prediction,
            "prediction_label": "CVD Risk Detected" if prediction == 1 else "Lower CVD Risk",
            "model_probability": round(probability, 4),
            "ai_risk_index": risk_index,
            "risk_category": risk_category,
            "top_contributors": contributors,
            "disclaimer": (
                "This is an AI-based educational/research prototype and is not a medical "
                "diagnostic tool. Predictions should not be used to make medical decisions. "
                "Consult a qualified healthcare professional for clinical assessment."
            ),
        })
    except Exception as e:
        return jsonify({"detail": f"Prediction error: {str(e)}"}), 500

@app.route("/explain", methods=["POST"])
def explain():
    if model is None:
        return jsonify({"detail": "Model not loaded"}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"detail": "No JSON payload provided"}), 400

    try:
        input_df = prepare_input(data)
        probability = float(model.predict_proba(input_df)[0][1])
        risk_index, risk_category = compute_risk(probability)
        contributors = compute_contributions(data, probability)

        increasing = [c for c in contributors if c["contribution"] > 0.001]
        decreasing = [c for c in contributors if c["contribution"] < -0.001]
        neutral = [c for c in contributors if -0.001 <= c["contribution"] <= 0.001]

        return jsonify({
            "model_probability": round(probability, 4),
            "ai_risk_index": risk_index,
            "risk_category": risk_category,
            "risk_increasing_factors": increasing,
            "risk_decreasing_factors": decreasing,
            "neutral_factors": neutral,
            "all_contributors": contributors,
            "explanation_method": "Perturbation-based feature contribution (baseline comparison)",
            "disclaimer": "Feature contributions represent MODEL CONTRIBUTION to the prediction, NOT physical anatomical damage.",
        })
    except Exception as e:
        return jsonify({"detail": f"Explanation error: {str(e)}"}), 500

# ─── Static Files & SPA Catch-All ───────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    if path != "" and os.path.exists(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, "index.html")

# WSGI standard entry point required by PythonAnywhere
application = app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
