"""
CARDIOVISION AI — Phase 1: Dataset Audit
=========================================
Inspects the actual CSV dataset before any modeling.
All outputs are real values from the dataset.
"""
import pandas as pd
import numpy as np
import json
import os

# ─── Load ───────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
df = pd.read_csv(DATA_PATH, sep=";")

print("=" * 70)
print("CARDIOVISION AI — PHASE 1: DATASET AUDIT")
print("=" * 70)

# ─── Basic Shape ────────────────────────────────────────────────────────────
print(f"\n📊 DATASET SHAPE")
print(f"   Rows   : {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")

# ─── Column Info ────────────────────────────────────────────────────────────
print(f"\n📋 COLUMNS ({df.shape[1]} total):")
print("-" * 60)
for col in df.columns:
    dtype = df[col].dtype
    nunique = df[col].nunique()
    nulls = df[col].isnull().sum()
    print(f"   {col:15s}  dtype={str(dtype):10s}  unique={nunique:6d}  nulls={nulls}")

# ─── Data Types ─────────────────────────────────────────────────────────────
print(f"\n🔢 DATA TYPES:")
print(df.dtypes.to_string())

# ─── Missing Values ─────────────────────────────────────────────────────────
missing = df.isnull().sum()
total_missing = missing.sum()
print(f"\n❓ MISSING VALUES: {total_missing}")
if total_missing > 0:
    print(missing[missing > 0].to_string())
else:
    print("   No missing values found.")

# ─── Duplicate Rows ─────────────────────────────────────────────────────────
dupes = df.duplicated().sum()
print(f"\n🔁 DUPLICATE ROWS: {dupes}")

# ─── Numeric vs Categorical ────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\n🔢 NUMERIC COLUMNS ({len(numeric_cols)}):")
for c in numeric_cols:
    print(f"   {c}")
if non_numeric_cols:
    print(f"\n📝 NON-NUMERIC COLUMNS ({len(non_numeric_cols)}):")
    for c in non_numeric_cols:
        print(f"   {c}")
else:
    print("\n📝 NON-NUMERIC COLUMNS: None")

# ─── Unique Values for Low-Cardinality Columns ─────────────────────────────
print(f"\n🔍 UNIQUE VALUES (columns with ≤20 unique):")
print("-" * 60)
for col in df.columns:
    uvals = sorted(df[col].unique())
    if len(uvals) <= 20:
        print(f"   {col}: {uvals}")

# ─── Target Column ─────────────────────────────────────────────────────────
TARGET = "cardio"
print(f"\n🎯 TARGET COLUMN: '{TARGET}'")
print(f"   Unique values: {sorted(df[TARGET].unique())}")
print(f"\n   CLASS DISTRIBUTION:")
dist = df[TARGET].value_counts().sort_index()
for val, cnt in dist.items():
    pct = cnt / len(df) * 100
    print(f"     {val}: {cnt:6d}  ({pct:.2f}%)")

# ─── Descriptive Stats ─────────────────────────────────────────────────────
print(f"\n📈 DESCRIPTIVE STATISTICS:")
print(df.describe().to_string())

# ─── Suspicious / Outlier Values ───────────────────────────────────────────
print(f"\n⚠️  SUSPICIOUS / MEDICALLY IMPLAUSIBLE VALUES:")
print("-" * 60)

# age is in days — convert for display
age_years = df["age"] / 365.25
print(f"   age (days): min={df['age'].min()}, max={df['age'].max()}")
print(f"   age (years): min={age_years.min():.1f}, max={age_years.max():.1f}")

# height
print(f"   height (cm): min={df['height'].min()}, max={df['height'].max()}")
h_suspicious = df[(df["height"] < 100) | (df["height"] > 220)]
print(f"     Rows with height < 100 or > 220: {len(h_suspicious)}")

# weight
print(f"   weight (kg): min={df['weight'].min()}, max={df['weight'].max()}")
w_suspicious = df[(df["weight"] < 30) | (df["weight"] > 250)]
print(f"     Rows with weight < 30 or > 250: {len(w_suspicious)}")

# blood pressure
print(f"   ap_hi (systolic):  min={df['ap_hi'].min()}, max={df['ap_hi'].max()}")
print(f"   ap_lo (diastolic): min={df['ap_lo'].min()}, max={df['ap_lo'].max()}")

bp_neg = df[(df["ap_hi"] <= 0) | (df["ap_lo"] <= 0)]
print(f"     Rows with BP ≤ 0: {len(bp_neg)}")

bp_extreme_hi = df[(df["ap_hi"] > 250) | (df["ap_hi"] < 60)]
print(f"     Rows with systolic > 250 or < 60: {len(bp_extreme_hi)}")

bp_extreme_lo = df[(df["ap_lo"] > 200) | (df["ap_lo"] < 30)]
print(f"     Rows with diastolic > 200 or < 30: {len(bp_extreme_lo)}")

bp_inverted = df[df["ap_lo"] >= df["ap_hi"]]
print(f"     Rows where diastolic ≥ systolic: {len(bp_inverted)}")

# ─── Identifier Columns ────────────────────────────────────────────────────
print(f"\n🆔 IDENTIFIER COLUMNS:")
print(f"   'id' — sequential identifier, should NOT be used as a predictor.")
print(f"     min={df['id'].min()}, max={df['id'].max()}, nunique={df['id'].nunique()}")

# ─── Target Interpretation ─────────────────────────────────────────────────
print(f"\n" + "=" * 70)
print("📌 TARGET INTERPRETATION")
print("=" * 70)
print("""
The dataset target column 'cardio' is a BINARY label (0 or 1).

This is the well-known Kaggle "Cardiovascular Disease dataset" collected
during medical examination. The target 'cardio' indicates presence (1) or
absence (0) of cardiovascular disease — based on patient examination data.

⚠️  The dataset target represents CARDIOVASCULAR DISEASE classification/risk,
   NOT confirmed coronary artery disease (CAD).

   It does NOT represent:
     - coronary blockage or stenosis
     - plaque burden
     - artery narrowing
     - specific cardiac events (MI, stroke)

   The features are examination-level variables (age, gender, height, weight,
   blood pressure, cholesterol level, glucose level, smoking, alcohol, physical
   activity), NOT imaging or angiography data.

   Therefore the system predicts:
     "Cardiovascular Disease Risk" — NOT "Coronary Artery Disease."
""")

# ─── Column Semantics ──────────────────────────────────────────────────────
print("=" * 70)
print("📖 COLUMN SEMANTICS")
print("=" * 70)
semantics = {
    "id":          "Unique row identifier (NOT a predictor)",
    "age":         "Age in DAYS (integer)",
    "gender":      "1 = female, 2 = male (categorical)",
    "height":      "Height in cm",
    "weight":      "Weight in kg",
    "ap_hi":       "Systolic blood pressure (mmHg)",
    "ap_lo":       "Diastolic blood pressure (mmHg)",
    "cholesterol": "1 = normal, 2 = above normal, 3 = well above normal",
    "gluc":        "1 = normal, 2 = above normal, 3 = well above normal",
    "smoke":       "0 = non-smoker, 1 = smoker",
    "alco":        "0 = no alcohol, 1 = alcohol intake",
    "active":      "0 = not physically active, 1 = physically active",
    "cardio":      "TARGET — 0 = no CVD, 1 = CVD present",
}
for col, desc in semantics.items():
    print(f"   {col:15s} → {desc}")

# ─── Save audit summary as JSON ────────────────────────────────────────────
audit = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "column_names": df.columns.tolist(),
    "dtypes": {c: str(df[c].dtype) for c in df.columns},
    "missing_values": {c: int(df[c].isnull().sum()) for c in df.columns},
    "total_missing": int(total_missing),
    "duplicate_rows": int(dupes),
    "target_column": TARGET,
    "target_values": sorted(df[TARGET].unique().tolist()),
    "class_distribution": {str(k): int(v) for k, v in dist.items()},
    "class_percentages": {str(k): round(v / len(df) * 100, 2) for k, v in dist.items()},
    "numeric_columns": numeric_cols,
    "identifier_columns": ["id"],
    "feature_columns": [c for c in df.columns if c not in ["id", "cardio"]],
    "column_semantics": semantics,
    "age_range_years": {"min": round(age_years.min(), 1), "max": round(age_years.max(), 1)},
    "suspicious_counts": {
        "height_outliers": int(len(h_suspicious)),
        "weight_outliers": int(len(w_suspicious)),
        "bp_negative_or_zero": int(len(bp_neg)),
        "systolic_extreme": int(len(bp_extreme_hi)),
        "diastolic_extreme": int(len(bp_extreme_lo)),
        "diastolic_gte_systolic": int(len(bp_inverted)),
    },
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
audit_path = os.path.join(RESULTS_DIR, "audit.json")
with open(audit_path, "w") as f:
    json.dump(audit, f, indent=2)
print(f"\n✅ Audit saved to {audit_path}")
print("=" * 70)
