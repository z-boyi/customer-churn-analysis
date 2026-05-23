"""Configuration and constants for the churn analysis project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Data file paths
RAW_DATA_PATH = DATA_DIR / "raw_churn.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_churn.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# Categorical columns
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Numerical columns
NUMERICAL_COLS = [
    "MonthlyCharges",
    "TotalCharges",
    "tenure",
]

# Target variable
TARGET = "Churn"
