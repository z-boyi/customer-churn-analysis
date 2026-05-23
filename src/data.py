"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from .config import RAW_DATA_PATH, CLEAN_DATA_PATH, CATEGORICAL_COLS, TARGET


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw churn dataset.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with raw data
    """
    df = pd.read_csv(filepath)
    return df


def load_clean_data(filepath: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load preprocessed churn dataset.

    Args:
        filepath: Path to the cleaned CSV file

    Returns:
        DataFrame with cleaned data
    """
    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Get basic information about the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with dataset statistics
    """
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.to_dict(),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw churn dataset.

    Args:
        df: Raw input DataFrame

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Remove CustomerID (non-informative)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric, handling non-numeric values
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill missing values with median
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target variable to binary
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)

    # Engineer tenure-based features
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, float("inf")],
            labels=["0-1 years", "1-2 years", "2-4 years", "4+ years"],
        )

    # Convert categorical columns to lowercase for consistency
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].str.lower()

    return df


def prepare_modeling_data(
    df: pd.DataFrame, target: str = TARGET
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target variable.

    Args:
        df: Input DataFrame
        target: Target column name

    Returns:
        Tuple of (features, target)
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y
