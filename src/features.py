"""Feature engineering and preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List
from .config import CATEGORICAL_COLS, NUMERICAL_COLS


def encode_categorical(X: pd.DataFrame, fit_encoders: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Encode categorical variables using label encoding.

    Args:
        X: Input features
        fit_encoders: If True, fit new encoders; if False, use existing ones

    Returns:
        Tuple of (encoded DataFrame, encoders dictionary)
    """
    X = X.copy()
    encoders = {}

    cat_cols = [col for col in CATEGORICAL_COLS if col in X.columns]

    for col in cat_cols:
        if X[col].dtype == "object":
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            encoders[col] = encoder

    return X, encoders


def scale_numerical(X: pd.DataFrame, scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize numerical features.

    Args:
        X: Input features
        scaler: Fitted scaler; if None, creates a new one

    Returns:
        Tuple of (scaled DataFrame, scaler)
    """
    X = X.copy()
    num_cols = [col for col in NUMERICAL_COLS if col in X.columns]

    if scaler is None:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    else:
        X[num_cols] = scaler.transform(X[num_cols])

    return X, scaler


def preprocess_features(
    X: pd.DataFrame,
    encoders: dict = None,
    scaler: StandardScaler = None,
) -> Tuple[pd.DataFrame, dict, StandardScaler]:
    """Apply all preprocessing steps to features.

    Args:
        X: Input features
        encoders: Pre-fitted encoders (for test set)
        scaler: Pre-fitted scaler (for test set)

    Returns:
        Tuple of (preprocessed DataFrame, encoders, scaler)
    """
    fit_mode = encoders is None and scaler is None

    X, encoders = encode_categorical(X, fit_encoders=fit_mode)
    X, scaler = scale_numerical(X, scaler=scaler)

    return X, encoders, scaler


def get_feature_names(X: pd.DataFrame) -> List[str]:
    """Get list of feature names after preprocessing.

    Args:
        X: Input DataFrame

    Returns:
        List of feature names
    """
    return X.columns.tolist()
