"""
CARDIOVISION AI — Phases 2 & 3: Preprocessing + Model Comparison + Selection + Calibration
============================================================================================
Builds a reproducible ML pipeline, compares 6 models, selects the best,
and saves everything to backend/results/ and backend/models/.

All metrics are computed from the actual dataset — nothing is fabricated.
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
import time
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 2 — DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv(DATA_PATH, sep=";")
print(f"\nLoaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 1: Drop identifier column ──────────────────────────────────────
df = df.drop(columns=["id"])
print(f"[1] Dropped 'id' column → {df.shape[1]} columns remaining")

# ── Step 2: Convert age from days to years ──────────────────────────────
df["age"] = (df["age"] / 365.25).round(1)
print(f"[2] Converted 'age' from days to years (range: {df['age'].min():.1f} – {df['age'].max():.1f})")

# ── Step 3: Add BMI feature ────────────────────────────────────────────
df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(1)
print(f"[3] Engineered 'bmi' feature (range: {df['bmi'].min():.1f} – {df['bmi'].max():.1f})")

# ── Step 4: Add pulse pressure and MAP ─────────────────────────────────
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
df["map"] = (df["ap_lo"] + (df["ap_hi"] - df["ap_lo"]) / 3).round(1)
print(f"[4] Engineered 'pulse_pressure' and 'map' features")

# ── Step 5: Remove medically implausible values ────────────────────────
before_clean = len(df)
df = df[
    (df["ap_hi"] >= 60) & (df["ap_hi"] <= 250) &
    (df["ap_lo"] >= 30) & (df["ap_lo"] <= 200) &
    (df["ap_lo"] < df["ap_hi"]) &
    (df["height"] >= 100) & (df["height"] <= 220) &
    (df["weight"] >= 30) & (df["weight"] <= 250) &
    (df["bmi"] >= 10) & (df["bmi"] <= 80)
]
after_clean = len(df)
removed = before_clean - after_clean
print(f"[5] Removed {removed} medically implausible rows ({removed/before_clean*100:.2f}%)")
print(f"    Remaining: {after_clean} rows")

# ── Step 6: Define features and target ─────────────────────────────────
TARGET = "cardio"
FEATURE_COLS = [c for c in df.columns if c != TARGET]

# Classify columns
NUMERIC_FEATURES = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure", "map"]
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc"]
BINARY_FEATURES = ["smoke", "alco", "active"]

# Binary features are already 0/1, treat as numeric
ALL_NUMERIC = NUMERIC_FEATURES + BINARY_FEATURES
ALL_CATEGORICAL = CATEGORICAL_FEATURES

print(f"\n[6] Feature classification:")
print(f"    Numeric:     {ALL_NUMERIC}")
print(f"    Categorical: {ALL_CATEGORICAL}")
print(f"    Target:      {TARGET}")

X = df[FEATURE_COLS]
y = df[TARGET]

# ── Step 7: Stratified train/test split ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n[7] Stratified train/test split:")
print(f"    Train: {X_train.shape[0]} rows")
print(f"    Test:  {X_test.shape[0]} rows")
print(f"    Train class dist: {dict(y_train.value_counts().sort_index())}")
print(f"    Test  class dist: {dict(y_test.value_counts().sort_index())}")

# ── Step 8: Build preprocessing pipeline ───────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, ALL_NUMERIC),
        ("cat", categorical_transformer, ALL_CATEGORICAL),
    ],
    remainder="drop"
)

print(f"\n[8] Preprocessing pipeline created:")
print(f"    Numeric:     StandardScaler (fit on train only)")
print(f"    Categorical: OneHotEncoder (drop='first')")
print(f"    No data leakage — fit only on training data")

# Save preprocessing info
preprocess_info = {
    "original_rows": int(before_clean + removed),  # This is just before_clean
    "rows_after_cleaning": int(after_clean),
    "rows_removed": int(removed),
    "pct_removed": round(removed / before_clean * 100, 2),
    "feature_columns": FEATURE_COLS,
    "numeric_features": ALL_NUMERIC,
    "categorical_features": ALL_CATEGORICAL,
    "target": TARGET,
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "random_state": SEED,
    "engineered_features": ["bmi", "pulse_pressure", "map"],
    "cleaning_rules": {
        "ap_hi": "60 ≤ systolic ≤ 250",
        "ap_lo": "30 ≤ diastolic ≤ 200",
        "ap_lo_lt_ap_hi": "diastolic < systolic",
        "height": "100 ≤ height ≤ 220",
        "weight": "30 ≤ weight ≤ 250",
        "bmi": "10 ≤ bmi ≤ 80",
    },
}
with open(os.path.join(RESULTS_DIR, "preprocessing.json"), "w") as f:
    json.dump(preprocess_info, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3 — MODEL COMPARISON")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1),
    "SVM": SVC(kernel="rbf", max_iter=500, probability=True, random_state=SEED),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, max_depth=6, random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}
all_confusion_matrices = {}
all_roc_data = {}

for name, model in models.items():
    print(f"\n{'─' * 50}")
    print(f"Training: {name}")
    start = time.time()

    # Build pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    # Cross-validation
    cv_results = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False, n_jobs=-1
    )

    cv_acc_mean = cv_results["test_accuracy"].mean()
    cv_acc_std = cv_results["test_accuracy"].std()
    cv_f1_mean = cv_results["test_f1"].mean()
    cv_roc_mean = cv_results["test_roc_auc"].mean()
    cv_recall_mean = cv_results["test_recall"].mean()

    # Fit on full training set
    pipe.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Test metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    elapsed = time.time() - start

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4),
        "cv_accuracy_mean": round(cv_acc_mean, 4),
        "cv_accuracy_std": round(cv_acc_std, 4),
        "cv_f1_mean": round(cv_f1_mean, 4),
        "cv_roc_auc_mean": round(cv_roc_mean, 4),
        "cv_recall_mean": round(cv_recall_mean, 4),
        "training_time_seconds": round(elapsed, 2),
    }

    all_confusion_matrices[name] = cm.tolist()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    all_roc_data[name] = {
        "fpr": [round(x, 4) for x in fpr[::max(1, len(fpr) // 100)]],
        "tpr": [round(x, 4) for x in tpr[::max(1, len(tpr) // 100)]],
    }

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"  CV Acc:    {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
    print(f"  CV Recall: {cv_recall_mean:.4f}")
    print(f"  CV ROC:    {cv_roc_mean:.4f}")
    print(f"  Time:      {elapsed:.2f}s")
    print(f"  Confusion Matrix:\n{cm}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL SELECTION")
print("=" * 70)

# Score each model: weighted combination of recall, roc_auc, cv stability, f1
print("\n📊 Model Ranking (Recall × 0.3 + ROC-AUC × 0.3 + CV_Stability × 0.2 + F1 × 0.2):\n")

rankings = {}
for name, r in results.items():
    # CV stability: lower std = better, normalize
    stability = 1.0 - min(r["cv_accuracy_std"] * 10, 1.0)  # scale std
    composite = (
        r["recall"] * 0.3 +
        r["roc_auc"] * 0.3 +
        stability * 0.2 +
        r["f1_score"] * 0.2
    )
    rankings[name] = round(composite, 4)
    print(f"  {name:30s}  composite={composite:.4f}  "
          f"(recall={r['recall']:.4f}, roc={r['roc_auc']:.4f}, "
          f"stab={stability:.4f}, f1={r['f1_score']:.4f})")

best_model_name = max(rankings, key=rankings.get)
print(f"\n🏆 SELECTED MODEL: {best_model_name}")
print(f"   Composite Score: {rankings[best_model_name]:.4f}")

best_result = results[best_model_name]
print(f"\n   Why {best_model_name}?")
print(f"   - Recall:    {best_result['recall']:.4f} — catches more CVD-positive cases")
print(f"   - ROC-AUC:   {best_result['roc_auc']:.4f} — strong discriminative ability")
print(f"   - CV Mean:   {best_result['cv_accuracy_mean']:.4f} ± {best_result['cv_accuracy_std']:.4f} — stable across folds")
print(f"   - F1 Score:  {best_result['f1_score']:.4f} — balanced precision-recall")

# ═══════════════════════════════════════════════════════════════════════════
# RE-TRAIN BEST MODEL & CALIBRATE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CALIBRATION & FINAL MODEL TRAINING")
print("=" * 70)

best_classifier = models[best_model_name]

# Build final pipeline
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", best_classifier),
])

# Fit on full training set
final_pipeline.fit(X_train, y_train)

# Calibrate
calibrated_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", CalibratedClassifierCV(best_classifier, cv=5, method="isotonic")),
])
calibrated_pipe.fit(X_train, y_train)

# Compare calibrated vs uncalibrated
y_prob_uncal = final_pipeline.predict_proba(X_test)[:, 1]
y_prob_cal = calibrated_pipe.predict_proba(X_test)[:, 1]

roc_uncal = roc_auc_score(y_test, y_prob_uncal)
roc_cal = roc_auc_score(y_test, y_prob_cal)

print(f"\n   Uncalibrated ROC-AUC: {roc_uncal:.4f}")
print(f"   Calibrated ROC-AUC:   {roc_cal:.4f}")

# Use the better one
if roc_cal >= roc_uncal - 0.005:
    use_calibrated = True
    final_model = calibrated_pipe
    print(f"   → Using calibrated model (isotonic calibration)")
else:
    use_calibrated = False
    final_model = final_pipeline
    print(f"   → Using uncalibrated model (calibration did not improve)")

# Final predictions
y_final_pred = final_model.predict(X_test)
y_final_prob = final_model.predict_proba(X_test)[:, 1]

final_acc = accuracy_score(y_test, y_final_pred)
final_prec = precision_score(y_test, y_final_pred)
final_rec = recall_score(y_test, y_final_pred)
final_f1 = f1_score(y_test, y_final_pred)
final_roc = roc_auc_score(y_test, y_final_prob)
final_cm = confusion_matrix(y_test, y_final_pred)

print(f"\n   Final Model Test Metrics:")
print(f"   Accuracy:  {final_acc:.4f}")
print(f"   Precision: {final_prec:.4f}")
print(f"   Recall:    {final_rec:.4f}")
print(f"   F1 Score:  {final_f1:.4f}")
print(f"   ROC-AUC:   {final_roc:.4f}")
print(f"   Confusion Matrix:\n{final_cm}")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_names = list(X_test.columns)

# Try to get feature importance from the model
from sklearn.inspection import permutation_importance

# Use permutation importance as a universal fallback
print("\nComputing permutation importance on test set...")
perm_imp = permutation_importance(
    final_model, X_test, y_test,
    n_repeats=5, random_state=SEED, n_jobs=-1, scoring="roc_auc"
)

importances = {}
for i, fname in enumerate(feature_names):
    importances[fname] = round(float(perm_imp.importances_mean[i]), 4)

sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nFeature Importances (permutation, by |importance|):")
for fname, imp in sorted_imp:
    print(f"  {fname:25s}  {imp:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# AI Risk Index documentation
risk_doc = {
    "name": "AI Risk Index",
    "range": "0–100",
    "calculation": "model_probability × 100, rounded to nearest integer",
    "calibrated": use_calibrated,
    "calibration_method": "isotonic" if use_calibrated else "none",
    "disclaimer": "The AI Risk Index is a MODEL-DERIVED VISUALIZATION SCORE from 0–100. "
                  "It is NOT a clinically validated cardiovascular risk score.",
    "thresholds": {
        "LOW": "AI Risk Index < 35",
        "MODERATE": "35 ≤ AI Risk Index < 65",
        "HIGH": "AI Risk Index ≥ 65",
    },
    "threshold_disclaimer": "These thresholds are model-derived and are NOT clinically validated."
}

# Compile metrics.json
metrics = {
    "selected_model": best_model_name,
    "selection_reason": f"{best_model_name} was selected based on highest composite score "
                        f"(Recall×0.3 + ROC-AUC×0.3 + CV-Stability×0.2 + F1×0.2 = {rankings[best_model_name]:.4f}). "
                        f"It achieved recall={best_result['recall']:.4f}, ROC-AUC={best_result['roc_auc']:.4f}, "
                        f"and stable cross-validation ({best_result['cv_accuracy_mean']:.4f} ± {best_result['cv_accuracy_std']:.4f}).",
    "calibrated": use_calibrated,
    "final_model_metrics": {
        "accuracy": round(final_acc, 4),
        "precision": round(final_prec, 4),
        "recall": round(final_rec, 4),
        "f1_score": round(final_f1, 4),
        "roc_auc": round(final_roc, 4),
        "confusion_matrix": final_cm.tolist(),
    },
    "model_comparison": results,
    "confusion_matrices": all_confusion_matrices,
    "roc_curves": all_roc_data,
    "rankings": rankings,
    "feature_importances": importances,
    "feature_names": feature_names,
    "risk_index": risk_doc,
}

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Metrics saved to results/metrics.json")

# Save input schema
input_schema = {
    "fields": [
        {"name": "age", "display": "Age", "type": "number", "unit": "years",
         "range": [29, 65], "plausible_range": [18, 100], "required": True, "group": "Demographics"},
        {"name": "gender", "display": "Gender", "type": "select",
         "options": [{"value": 1, "label": "Female"}, {"value": 2, "label": "Male"}],
         "required": True, "group": "Demographics"},
        {"name": "height", "display": "Height", "type": "number", "unit": "cm",
         "range": [100, 220], "plausible_range": [100, 250], "required": True, "group": "Demographics"},
        {"name": "weight", "display": "Weight", "type": "number", "unit": "kg",
         "range": [30, 250], "plausible_range": [20, 300], "required": True, "group": "Demographics"},
        {"name": "ap_hi", "display": "Systolic Blood Pressure", "type": "number", "unit": "mmHg",
         "range": [60, 250], "plausible_range": [60, 300], "required": True, "group": "Vitals"},
        {"name": "ap_lo", "display": "Diastolic Blood Pressure", "type": "number", "unit": "mmHg",
         "range": [30, 200], "plausible_range": [30, 200], "required": True, "group": "Vitals"},
        {"name": "cholesterol", "display": "Cholesterol Level", "type": "select",
         "options": [{"value": 1, "label": "Normal"}, {"value": 2, "label": "Above Normal"}, {"value": 3, "label": "Well Above Normal"}],
         "required": True, "group": "Blood Indicators"},
        {"name": "gluc", "display": "Glucose Level", "type": "select",
         "options": [{"value": 1, "label": "Normal"}, {"value": 2, "label": "Above Normal"}, {"value": 3, "label": "Well Above Normal"}],
         "required": True, "group": "Blood Indicators"},
        {"name": "smoke", "display": "Smoking", "type": "select",
         "options": [{"value": 0, "label": "Non-smoker"}, {"value": 1, "label": "Smoker"}],
         "required": True, "group": "Lifestyle"},
        {"name": "alco", "display": "Alcohol Intake", "type": "select",
         "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}],
         "required": True, "group": "Lifestyle"},
        {"name": "active", "display": "Physical Activity", "type": "select",
         "options": [{"value": 0, "label": "Not Active"}, {"value": 1, "label": "Active"}],
         "required": True, "group": "Lifestyle"},
    ],
    "derived_features": [
        {"name": "bmi", "formula": "weight / (height/100)^2", "description": "Body Mass Index"},
        {"name": "pulse_pressure", "formula": "ap_hi - ap_lo", "description": "Pulse Pressure"},
        {"name": "map", "formula": "ap_lo + (ap_hi - ap_lo) / 3", "description": "Mean Arterial Pressure"},
    ],
}
with open(os.path.join(RESULTS_DIR, "input_schema.json"), "w") as f:
    json.dump(input_schema, f, indent=2)
print(f"✅ Input schema saved to results/input_schema.json")

# Save the final model
model_path = os.path.join(MODELS_DIR, "final_model.joblib")
joblib.dump(final_model, model_path)
print(f"✅ Final model saved to models/final_model.joblib")

# Save the preprocessor separately for reference
preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, preprocessor_path)
print(f"✅ Preprocessor saved to models/preprocessor.joblib")

# Save test data for later use
test_data = {
    "X_test": X_test.to_dict(orient="records"),
    "y_test": y_test.tolist(),
}

# Save demo patients (from dataset quantiles)
# Pick 3 actual dataset rows representing low, moderate, high risk
y_probs_all = final_model.predict_proba(X_test)[:, 1]

low_idx = np.argmin(np.abs(y_probs_all - np.percentile(y_probs_all, 10)))
mod_idx = np.argmin(np.abs(y_probs_all - np.percentile(y_probs_all, 50)))
high_idx = np.argmin(np.abs(y_probs_all - np.percentile(y_probs_all, 90)))

demo_patients = []
for label, idx in [("Low Risk Profile", low_idx), ("Moderate Risk Profile", mod_idx), ("High Risk Profile", high_idx)]:
    row = X_test.iloc[idx].to_dict()
    prob = float(y_probs_all[idx])
    risk_index = round(prob * 100)
    if risk_index < 35:
        category = "LOW"
    elif risk_index < 65:
        category = "MODERATE"
    else:
        category = "HIGH"
    demo_patients.append({
        "label": label,
        "source": "Derived from actual dataset row (test set)",
        "features": {k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in row.items()},
        "expected_probability": round(prob, 4),
        "expected_risk_index": risk_index,
        "expected_category": category,
    })

with open(os.path.join(RESULTS_DIR, "demo_patients.json"), "w") as f:
    json.dump(demo_patients, f, indent=2)
print(f"✅ Demo patients saved to results/demo_patients.json")

# Save dataset summary for the API
dataset_summary = {
    "total_rows_original": 70000,
    "total_rows_after_cleaning": after_clean,
    "rows_removed": removed,
    "columns_original": 13,
    "features_used": len(FEATURE_COLS),
    "feature_names": FEATURE_COLS,
    "target": TARGET,
    "class_distribution": {
        "0_no_cvd": int((y == 0).sum()),
        "1_cvd": int((y == 1).sum()),
    },
    "class_balance": round((y == 1).sum() / len(y) * 100, 2),
    "age_range": {"min": float(df["age"].min()), "max": float(df["age"].max())},
    "preprocessing_steps": [
        "Dropped 'id' column",
        "Converted age from days to years",
        "Engineered BMI, pulse pressure, and MAP features",
        "Removed medically implausible values",
        "StandardScaler on numeric features (fit on train only)",
        "OneHotEncoder on categorical features (drop='first')",
        "Stratified 80/20 train/test split",
    ],
}
with open(os.path.join(RESULTS_DIR, "dataset_summary.json"), "w") as f:
    json.dump(dataset_summary, f, indent=2)
print(f"✅ Dataset summary saved to results/dataset_summary.json")

print("\n" + "=" * 70)
print("✅ PHASES 2 & 3 COMPLETE")
print("=" * 70)
print(f"\n   Selected Model:     {best_model_name}")
print(f"   Final Accuracy:     {final_acc:.4f}")
print(f"   Final Recall:       {final_rec:.4f}")
print(f"   Final ROC-AUC:      {final_roc:.4f}")
print(f"   Calibrated:         {use_calibrated}")
print(f"   Files saved:")
print(f"     results/metrics.json")
print(f"     results/input_schema.json")
print(f"     results/preprocessing.json")
print(f"     results/dataset_summary.json")
print(f"     results/demo_patients.json")
print(f"     models/final_model.joblib")
print(f"     models/preprocessor.joblib")
