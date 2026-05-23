"""Prediction script for customer churn using trained models."""

import pandas as pd
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from src.config import MODELS_DIR, DATA_DIR, CLEAN_DATA_PATH
from src.data import load_clean_data, prepare_modeling_data
from src.features import preprocess_features, encode_categorical, scale_numerical


def load_model_and_metadata(model_name: str = "gradient_boosting") -> Tuple:
    """Load a trained model and its metadata.

    Args:
        model_name: Name of the model to load

    Returns:
        Tuple of (model, metadata)
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, metadata


def predict_single_customer(customer_data: Dict, model, feature_names: list) -> Dict:
    """Make a prediction for a single customer.

    Args:
        customer_data: Dictionary of customer features
        model: Trained model
        feature_names: List of feature names used in training

    Returns:
        Dictionary with prediction and probability
    """
    # Convert to DataFrame
    df = pd.DataFrame([customer_data])

    # Preprocess
    df_processed, _, _ = preprocess_features(df)

    # Ensure correct columns
    df_processed = df_processed[feature_names]

    # Make prediction
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability[1]),
        "no_churn_probability": float(probability[0]),
    }


def predict_batch(df: pd.DataFrame, model, feature_names: list) -> pd.DataFrame:
    """Make predictions for multiple customers.

    Args:
        df: DataFrame with customer features
        model: Trained model
        feature_names: List of feature names used in training

    Returns:
        DataFrame with predictions
    """
    # Preprocess
    df_processed, _, _ = preprocess_features(df)

    # Ensure correct columns
    df_processed = df_processed[feature_names]

    # Make predictions
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]

    results = df.copy()
    results["churn_prediction"] = predictions
    results["churn_probability"] = probabilities

    return results


def identify_high_risk_customers(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    top_n: int = 10,
) -> np.ndarray:
    """Identify high-risk churn customers.

    Args:
        probabilities: Array of churn probabilities
        threshold: Probability threshold for flagging as at-risk
        top_n: Number of top customers to return

    Returns:
        Array of indices for high-risk customers
    """
    high_risk_indices = np.where(probabilities >= threshold)[0]
    # Sort by probability (descending) and get top_n
    top_indices = high_risk_indices[np.argsort(probabilities[high_risk_indices])[::-1][:top_n]]
    return top_indices


def example_single_prediction():
    """Example: Make a prediction for a single customer."""

    print("\n" + "=" * 80)
    print("EXAMPLE: SINGLE CUSTOMER PREDICTION")
    print("=" * 80)

    # Load model
    model, metadata = load_model_and_metadata()

    # Example customer
    customer = {
        "gender": "male",
        "SeniorCitizen": 0,
        "Partner": "yes",
        "Dependents": "no",
        "tenure": 24,
        "PhoneService": "yes",
        "MultipleLines": "no",
        "InternetService": "fiber optic",
        "OnlineSecurity": "no",
        "OnlineBackup": "yes",
        "DeviceProtection": "no",
        "TechSupport": "no",
        "StreamingTV": "no",
        "StreamingMovies": "no",
        "Contract": "month-to-month",
        "PaperlessBilling": "yes",
        "PaymentMethod": "electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 2052.0,
    }

    prediction = predict_single_customer(customer, model, metadata["feature_names"])

    print(f"\nCustomer Characteristics:")
    print(f"  Tenure: {customer['tenure']} months")
    print(f"  Contract: {customer['Contract']}")
    print(f"  Monthly Charges: ${customer['MonthlyCharges']:.2f}")

    print(f"\nPrediction Results:")
    print(f"  Churn Prediction: {'Yes' if prediction['churn_prediction'] == 1 else 'No'}")
    print(f"  Churn Probability: {prediction['churn_probability']:.2%}")
    print(f"  Retention Probability: {prediction['no_churn_probability']:.2%}")


def example_batch_prediction():
    """Example: Make predictions for a batch of customers."""

    print("\n" + "=" * 80)
    print("EXAMPLE: BATCH PREDICTION")
    print("=" * 80)

    # Load model
    model, metadata = load_model_and_metadata()

    # Load test data
    df = load_clean_data(CLEAN_DATA_PATH)
    X, y = prepare_modeling_data(df)

    # Get a sample for prediction
    sample_df = X.head(50).copy()

    # Make predictions
    results = predict_batch(sample_df, model, metadata["feature_names"])

    # Identify high-risk customers
    high_risk_indices = identify_high_risk_customers(
        results["churn_probability"].values, threshold=0.4, top_n=10
    )

    print(f"\nAnalyzed {len(results)} customers")
    print(f"Predicted Churn: {(results['churn_prediction'] == 1).sum()} customers")
    print(f"High-Risk Customers (Probability >= 40%): {len(high_risk_indices)} customers")

    print(f"\nTop 10 High-Risk Customers:")
    print(results.iloc[high_risk_indices][["tenure", "Contract", "MonthlyCharges", "churn_probability"]])

    # Save predictions
    results.to_csv(MODELS_DIR / "batch_predictions.csv", index=False)
    print(f"\nPredictions saved to {MODELS_DIR / 'batch_predictions.csv'}")


def main():
    """Run prediction examples."""

    print("Customer Churn Prediction")

    # Example 1: Single prediction
    example_single_prediction()

    # Example 2: Batch prediction
    example_batch_prediction()


if __name__ == "__main__":
    main()
