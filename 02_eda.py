"""
02_eda.py
=========
Step 2 – Exploratory Data Analysis.
Saves all figures to  data/eda_plots/ .

Run: python 02_eda.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import DATASET_PATH, TARGET_COLUMN

# ── Output directory ─────────────────────────────────────────
PLOT_DIR = os.path.join("data", "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11

KEY_FEATURES = [
    "Glucose_Fasting_mg/dL",
    "Cholesterol_mg/dL",
    "BMI",
    "Age",
]


# ─────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────
# 1  Class distribution
# ─────────────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#10b981", "#ef4444"]

    counts.plot(kind="bar", ax=ax1, color=colors, edgecolor="black", linewidth=1.5)
    ax1.set_title("Class Distribution (Count)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Diagnosis", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Patients", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(["No Risk", "At Risk"], rotation=0, fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(counts):
        ax1.text(i, v + 50, f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax2.pie(counts, labels=["No Risk", "At Risk"],
            autopct="%1.1f%%", colors=colors, startangle=90,
            explode=(0.05, 0), shadow=True,
            textprops={"fontsize": 12, "fontweight": "bold"})
    ax2.set_title("Class Proportion", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "class_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")


# ─────────────────────────────────────────────────────────────
# 2  Feature distributions
# ─────────────────────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    palette = {"0": "#10b981", "1": "#ef4444"}

    for i, feat in enumerate(KEY_FEATURES):
        if feat not in df.columns:
            continue
        sns.histplot(
            data=df, x=feat,
            hue=df[TARGET_COLUMN].astype(str),
            kde=True, ax=axes[i],
            palette=palette, alpha=0.6, edgecolor="black",
        )
        axes[i].set_title(f"{feat} Distribution", fontsize=13, fontweight="bold")
        axes[i].set_xlabel(feat, fontsize=11)
        axes[i].set_ylabel("Count", fontsize=11)
        axes[i].legend(["No Risk", "At Risk"], title="Diagnosis", fontsize=10)
        axes[i].grid(axis="y", alpha=0.3)

    plt.suptitle("Key Feature Distributions by Diagnosis",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")


# ─────────────────────────────────────────────────────────────
# 3  Box plots
# ─────────────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    palette = {"0": "#10b981", "1": "#ef4444"}

    for i, feat in enumerate(KEY_FEATURES):
        if feat not in df.columns:
            continue
        sns.boxplot(
            data=df, x=df[TARGET_COLUMN].astype(str), y=feat,
            ax=axes[i], palette=palette,
        )
        axes[i].set_title(f"{feat} by Diagnosis", fontsize=13, fontweight="bold")
        axes[i].set_xticklabels(["No Risk", "At Risk"], fontsize=11)
        axes[i].set_xlabel("Diagnosis", fontsize=11)
        axes[i].set_ylabel(feat, fontsize=11)
        axes[i].grid(axis="y", alpha=0.3)

    plt.suptitle("Outlier Analysis – Box Plots by Diagnosis",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "boxplots.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")


# ─────────────────────────────────────────────────────────────
# 4  Correlation heatmap
# ─────────────────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=[np.number])
    corr    = numeric.corr()
    mask    = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                mask=mask, vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix (Lower Triangle)",
              fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "correlation_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")

    if TARGET_COLUMN in corr:
        top = corr[TARGET_COLUMN].sort_values(ascending=False).iloc[1:6]
        print("\n🎯 Top 5 Features Correlated with Diagnosis:")
        print(top.to_string())


# ─────────────────────────────────────────────────────────────
# 5  Statistical significance
# ─────────────────────────────────────────────────────────────
def significance_tests(df: pd.DataFrame) -> None:
    print("\n📊 Statistical Significance Tests (t-test):")
    print("=" * 70)
    print(f"{'Feature':<35} | {'t-stat':>10} | {'p-value':>12} | Sig")
    print("-" * 70)
    for feat in KEY_FEATURES:
        if feat not in df.columns:
            continue
        g0 = df[df[TARGET_COLUMN] == 0][feat]
        g1 = df[df[TARGET_COLUMN] == 1][feat]
        t, p = stats.ttest_ind(g0, g1)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{feat:<35} | {t:>10.3f} | {p:>12.4e} | {sig:>3}")
    print("-" * 70)
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load()

    print("\n" + "=" * 70)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\n[1/5] Class Distribution …")
    plot_class_distribution(df)

    print("\n[2/5] Feature Distributions …")
    plot_feature_distributions(df)

    print("\n[3/5] Box Plots …")
    plot_boxplots(df)

    print("\n[4/5] Correlation Matrix …")
    plot_correlation(df)

    print("\n[5/5] Statistical Tests …")
    significance_tests(df)

    print("\n" + "=" * 70)
    print(f"✅  EDA complete – plots saved to  {PLOT_DIR}/")
    print("=" * 70)
