# ============================================================
# config.py – Central configuration for the Healthcare Project
# ============================================================

import os

# ── Reproducibility ─────────────────────────────────────────
RANDOM_SEED   = 42
N_ESTIMATORS  = 400
CV_FOLDS      = 5
TEST_SIZE     = 0.20

# ── Paths ────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

DATASET_PATH = "E:/D_volume/Preethi/OneDrive/Desktop/desktopfiles/final year project dataset/final_integrated_dataset.csv"
MODEL_PATH    = os.path.join(MODELS_DIR, "healthcare_model.pkl")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")

# ── Feature order (must match training) ──────────────────────
FEATURE_COLUMNS = [
    "Age", "Sex", "BMI", "Family_History", "Hypertension",
    "Smoking_Status", "Physical_Activity", "Stress_Level", "Steps",
    "Sleep_Quality", "Cholesterol_mg/dL", "HDL_mg/dL", "LDL_mg/dL",
    "Glucose_Fasting_mg/dL", "Insulin_uIU/mL", "Systolic_BP",
]
TARGET_COLUMN = "Diagnosis"

# ── Clinical thresholds ───────────────────────────────────────
THRESHOLDS = {
    "Glucose_Fasting_mg/dL": {"warn": 100, "high": 126},
    "Cholesterol_mg/dL":     {"warn": 200, "high": 240},
    "LDL_mg/dL":             {"warn": 130, "high": 160},
    "HDL_mg/dL":             {"low":  40},
    "BMI":                   {"warn": 25,  "high": 30},
    "Systolic_BP":           {"warn": 120, "high": 140},
}

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
