"""
05_evaluate.py
==============
Step 5 – Full model evaluation:
    • Metrics bar chart
    • Confusion matrix (raw + normalised)
    • ROC curves (stacking vs baselines)
    • Prediction confidence distribution

Run: python 05_evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree         import DecisionTreeClassifier
from sklearn.svm          import SVC
from sklearn.naive_bayes  import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc,
)
import lightgbm  as lgb
import xgboost   as xgb
from catboost            import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from config import RANDOM_SEED, MODEL_PATH, MODELS_DIR

NUMPY_DIR  = os.path.join(MODELS_DIR, "splits")
PLOT_DIR   = os.path.join("data", "eval_plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────
def load():
    X_train = np.load(os.path.join(NUMPY_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(NUMPY_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(NUMPY_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(NUMPY_DIR, "y_test.npy"))
    model   = joblib.load(MODEL_PATH)
    print(f"✅  Loaded model ({type(model).__name__})")
    return X_train, X_test, y_train, y_test, model


# ─────────────────────────────────────────────────────────────
# 1  Metrics bar chart
# ─────────────────────────────────────────────────────────────
def plot_metrics(y_test, y_pred, y_proba):
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }

    print("\n📊 Stacking Ensemble – Test Metrics:")
    for k, v in metrics.items():
        print(f"   {k:<12}: {v:.4f}")

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors,
                  edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim([0.75, 1.0])
    ax.grid(axis="y", alpha=0.3)
    for bar, (_, v) in zip(bars, metrics.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "metrics_bar.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")
    return metrics


# ─────────────────────────────────────────────────────────────
# 2  Confusion matrix
# ─────────────────────────────────────────────────────────────
def plot_confusion(y_test, y_pred):
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["No Risk", "At Risk"],
                yticklabels=["No Risk", "At Risk"],
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax1.set_ylabel("True",      fontweight="bold")
    ax1.set_xlabel("Predicted", fontweight="bold")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Oranges", ax=ax2,
                xticklabels=["No Risk", "At Risk"],
                yticklabels=["No Risk", "At Risk"],
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax2.set_title("Normalised (%)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("True",      fontweight="bold")
    ax2.set_xlabel("Predicted", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")

    # Clinical impact
    fn = np.sum((y_test == 1) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn_rate = fn / np.sum(y_test == 1) * 100
    fp_rate = fp / np.sum(y_test == 0) * 100
    print(f"\n🏥 Clinical Impact:")
    print(f"   FN Rate : {fn_rate:.2f}%  → {fn} high-risk patients missed")
    print(f"   FP Rate : {fp_rate:.2f}%  → {fp} unnecessary interventions")


# ─────────────────────────────────────────────────────────────
# 3  Baseline comparison + ROC curves
# ─────────────────────────────────────────────────────────────
def model_comparison(X_train, X_test, y_train, y_test, stack):
    print("\n📊 Baseline Model Comparison …")

    sm = SMOTE(random_state=RANDOM_SEED)
    X_bal, y_bal = sm.fit_resample(X_train, y_train)

    baselines = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=10),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "Naive Bayes":         GaussianNB(),
        "SVM":                 SVC(kernel="rbf", random_state=RANDOM_SEED, probability=True),
        "LightGBM":            lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED, verbose=-1),
        "XGBoost":             xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                                  eval_metric="logloss", verbosity=0),
        "CatBoost":            CatBoostClassifier(iterations=100, random_state=RANDOM_SEED, verbose=0),
        "Stacking Ensemble":   stack,
    }

    rows = []
    plt.figure(figsize=(10, 8))
    for name, mdl in baselines.items():
        if name != "Stacking Ensemble":
            mdl.fit(X_bal, y_bal)
        yp   = mdl.predict(X_test)
        ypr  = mdl.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":     name,
            "Accuracy":  accuracy_score(y_test, yp),
            "Precision": precision_score(y_test, yp, zero_division=0),
            "Recall":    recall_score(y_test, yp, zero_division=0),
            "F1 Score":  f1_score(y_test, yp, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_test, ypr),
        })
        fpr, tpr, _ = roc_curve(y_test, ypr)
        ra = auc(fpr, tpr)
        lw = 3 if "Stack" in name else 1.5
        col = "#ef4444" if "Stack" in name else None
        plt.plot(fpr, tpr, linewidth=lw, color=col, alpha=0.8 if col else 0.6,
                 label=f"{name} (AUC={ra:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate",  fontsize=12, fontweight="bold")
    plt.title("ROC Curves – All Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")

    res_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    print("\n📊 Comparison Table:")
    print(res_df.to_string(index=False))

    best = res_df.iloc[0]
    print(f"\n🏆 Best: {best['Model']}  (F1={best['F1 Score']:.4f})")


# ─────────────────────────────────────────────────────────────
# 4  Confidence distribution
# ─────────────────────────────────────────────────────────────
def plot_confidence(y_test, y_proba):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(y_proba[y_test == 0], bins=50, alpha=0.6, label="Actual No Risk",  color="#10b981")
    ax.hist(y_proba[y_test == 1], bins=50, alpha=0.6, label="Actual At Risk",  color="#ef4444")
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Threshold 0.5")
    ax.set_xlabel("Predicted Probability",  fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency",              fontsize=12, fontweight="bold")
    ax.set_title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confidence_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("📈 MODEL EVALUATION")
    print("=" * 70)

    X_train, X_test, y_train, y_test, model = load()

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n[1/4] Metrics Bar Chart …")
    plot_metrics(y_test, y_pred, y_proba)

    print("\n[2/4] Confusion Matrix …")
    plot_confusion(y_test, y_pred)

    print("\n[3/4] Baseline Comparison + ROC Curves …")
    model_comparison(X_train, X_test, y_train, y_test, model)

    print("\n[4/4] Confidence Distribution …")
    plot_confidence(y_test, y_proba)

    print("\n" + "=" * 70)
    print(f"✅  Evaluation complete – plots saved to  {PLOT_DIR}/")
    print("    Run  streamlit run app.py  to launch the web app.")
    print("=" * 70)
