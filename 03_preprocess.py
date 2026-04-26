"""
03_preprocess.py
================
Step 3 – Data preprocessing, feature engineering & selection.

Outputs (saved to models/):
    scaler.pkl   – fitted StandardScaler
    X_train.npy, X_test.npy, y_train.npy, y_test.npy

Run: python 03_preprocess.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.ensemble           import RandomForestClassifier

from config import (
    DATASET_PATH, TARGET_COLUMN, FEATURE_COLUMNS,
    RANDOM_SEED, TEST_SIZE, MODELS_DIR, SCALER_PATH,
)

NUMPY_DIR = os.path.join(MODELS_DIR, "splits")
os.makedirs(NUMPY_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1  Encode & clean
# ─────────────────────────────────────────────────────────────
def encode_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sex: Male → 1, Female → 0
    if "Sex" in df.columns:
        df["Sex"] = (
            df["Sex"]
            .astype(str)
            .replace({"Male": "1", "Female": "0"})
        )
        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")
        print("✅  Sex encoded: Male=1, Female=0")

    # Drop rows with any remaining NaN
    before = len(df)
    df = df.dropna()
    after  = len(df)
    if before != after:
        print(f"⚠️   Dropped {before - after} rows with missing values")

    return df


# ─────────────────────────────────────────────────────────────
# 2  Split & scale
# ─────────────────────────────────────────────────────────────
def split_and_scale(df: pd.DataFrame):
    # Keep only known feature columns that exist
    present_feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_feats = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_feats:
        print(f"⚠️   Columns missing in CSV (will be skipped): {missing_feats}")

    X = df[present_feats]
    y = df[TARGET_COLUMN]

    print(f"\n📐 Features used  : {X.shape[1]}")
    print(f"📐 Samples         : {X.shape[0]:,}")
    print(f"🎯 Class balance   : {y.value_counts().to_dict()}")

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\n✅  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # StandardScaler fitted on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("✅  Features scaled (StandardScaler)")

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾  Scaler saved → {SCALER_PATH}")

    # Save numpy arrays
    np.save(os.path.join(NUMPY_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(NUMPY_DIR, "X_test.npy"),  X_test_scaled)
    np.save(os.path.join(NUMPY_DIR, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(NUMPY_DIR, "y_test.npy"),  y_test.to_numpy())
    print(f"💾  Split arrays saved → {NUMPY_DIR}/")

    return X_train_scaled, X_test_scaled, y_train, y_test, present_feats


# ─────────────────────────────────────────────────────────────
# 3  Feature importance
# ─────────────────────────────────────────────────────────────
def feature_importance(X_train, y_train, feature_names: list) -> None:
    print("\n🎯 Computing feature importance (Random Forest, 100 trees) …")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    fi = (
        pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 Features:")
    print(fi.head(10).to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(fi["Feature"][:10], fi["Importance"][:10], color="#3b82f6")
    plt.xlabel("Importance", fontweight="bold")
    plt.title("Top 10 Feature Importance", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join("data", "eda_plots", "feature_importance.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {plot_path}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("🔧 DATA PREPROCESSING")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"✅  Loaded {len(df):,} rows")

    df = encode_and_clean(df)
    X_tr, X_te, y_tr, y_te, feats = split_and_scale(df)
    feature_importance(X_tr, y_tr, feats)

    print("\n" + "=" * 70)
    print("✅  Preprocessing complete.")
    print(f"    Run  python 04_train.py  next.")
    print("=" * 70)
