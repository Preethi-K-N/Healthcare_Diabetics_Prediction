"""
01_load_data.py
===============
Step 1 – Load the CSV dataset and perform an initial inspection.

Run: python 01_load_data.py
"""

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import DATASET_PATH, TARGET_COLUMN, FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        print("   Place your CSV at  data/healthcare_dataset.csv  and rerun.")
        sys.exit(1)


def inspect(df: pd.DataFrame) -> None:
    """Print a structured overview of the DataFrame."""
    print("\n" + "=" * 70)
    print("📋 DATASET OVERVIEW")
    print("=" * 70)

    print("\n── First 5 rows ──────────────────────────────────────────────")
    print(df.head().to_string(index=True))

    print("\n── Column dtypes ─────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Statistical Summary ───────────────────────────────────────")
    print(df.describe().T.to_string())

    print("\n── Missing Values ────────────────────────────────────────────")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅  No missing values detected – dataset is complete!")
    else:
        print(missing[missing > 0].to_string())

    if TARGET_COLUMN in df.columns:
        print("\n── Target Distribution ───────────────────────────────────────")
        counts = df[TARGET_COLUMN].value_counts()
        for label, n in counts.items():
            print(f"   Class {label}: {n:,}  ({n / len(df) * 100:.2f}%)")
        ratio = counts.min() / counts.max()
        print(f"   Imbalance ratio (minor/major): {ratio:.2f}:1")

    print("\n" + "=" * 70)
    print("✅  Data loading complete.")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset()
    inspect(df)
