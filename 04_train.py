"""
04_train.py
===========
Step 4 – Train the stacking ensemble (RF + LightGBM + XGBoost + CatBoost)
         with SMOTE inside 5-fold cross-validation.

Output: models/healthcare_model.pkl

Run: python 04_train.py
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
import lightgbm  as lgb
import xgboost   as xgb
from catboost            import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from config import (
    RANDOM_SEED, N_ESTIMATORS, CV_FOLDS,
    MODELS_DIR, MODEL_PATH, SCALER_PATH,
)

NUMPY_DIR = os.path.join(MODELS_DIR, "splits")


# ─────────────────────────────────────────────────────────────
# Load splits
# ─────────────────────────────────────────────────────────────
def load_splits():
    X_train = np.load(os.path.join(NUMPY_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(NUMPY_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(NUMPY_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(NUMPY_DIR, "y_test.npy"))
    print(f"✅  Splits loaded – Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# Build estimators
# ─────────────────────────────────────────────────────────────
def build_stack():
    base_learners = [
        ("rf",  RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED,
            class_weight="balanced", n_jobs=-1)),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED,
            class_weight="balanced", verbose=-1)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED,
            eval_metric="logloss", verbosity=0)),
        ("cat", CatBoostClassifier(
            iterations=N_ESTIMATORS, random_state=RANDOM_SEED,
            auto_class_weights="Balanced", verbose=0)),
    ]
    meta = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta,
        cv=5, n_jobs=-1,
    )
    return stack


# ─────────────────────────────────────────────────────────────
# Cross-validation with SMOTE
# ─────────────────────────────────────────────────────────────
def cross_validate_with_smote(stack, X_train, y_train):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = {k: [] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}

    print(f"\n🔄 {CV_FOLDS}-Fold Cross-Validation with SMOTE:")
    print("-" * 60)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        t0 = time.time()
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # SMOTE applied only to training fold (no leakage)
        sm = SMOTE(random_state=RANDOM_SEED)
        X_tr_bal, y_tr_bal = sm.fit_resample(X_tr, y_tr)

        stack.fit(X_tr_bal, y_tr_bal)

        y_pred  = stack.predict(X_val)
        y_proba = stack.predict_proba(X_val)[:, 1]

        cv_scores["accuracy"].append(accuracy_score(y_val, y_pred))
        cv_scores["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        cv_scores["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        cv_scores["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        cv_scores["roc_auc"].append(roc_auc_score(y_val, y_proba))

        elapsed = time.time() - t0
        print(f"  Fold {fold}: Acc={cv_scores['accuracy'][-1]:.4f} | "
              f"F1={cv_scores['f1'][-1]:.4f} | "
              f"AUC={cv_scores['roc_auc'][-1]:.4f}  ({elapsed:.0f}s)")

    print("\n📊 CV Results (Mean ± Std):")
    for metric, scores in cv_scores.items():
        print(f"   {metric.upper():<12}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return cv_scores


# ─────────────────────────────────────────────────────────────
# Final fit & evaluate
# ─────────────────────────────────────────────────────────────
def final_fit_evaluate(stack, X_train, X_test, y_train, y_test):
    print("\n🎯 Final Training on Full Training Set (SMOTE applied) …")
    sm = SMOTE(random_state=RANDOM_SEED)
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    print(f"   Before SMOTE : {len(X_train):,}  |  After : {len(X_bal):,}")

    t0 = time.time()
    stack.fit(X_bal, y_bal)
    print(f"   Training time: {time.time()-t0:.0f}s")
    print("✅  Model trained!")

    y_pred  = stack.predict(X_test)
    y_proba = stack.predict_proba(X_test)[:, 1]

    print("\n🎯 TEST SET PERFORMANCE:")
    print(f"   Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"   Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"   F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"   ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    return stack


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 TRAINING STACKING ENSEMBLE")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_splits()
    stack = build_stack()
    cross_validate_with_smote(stack, X_train, y_train)
    trained = final_fit_evaluate(stack, X_train, X_test, y_train, y_test)

    # Save model
    joblib.dump(trained, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n💾  Model saved → {MODEL_PATH}  ({size_mb:.2f} MB)")

    print("\n" + "=" * 70)
    print("✅  Training complete.")
    print("    Run  python 05_evaluate.py  to see full evaluation.")
    print("=" * 70)
