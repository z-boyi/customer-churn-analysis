"""Advanced model training, evaluation, and comparison pipeline."""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import CLEAN_DATA_PATH, MODELS_DIR, RANDOM_STATE, TARGET
from src.data import load_clean_data, prepare_modeling_data
from src.features import preprocess_features, encode_categorical, scale_numerical
from src.modeling import ModelEvaluator, create_baseline_models
from src.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)


def train_models_with_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Train models with hyperparameter tuning using GridSearchCV."""

    print("\n" + "=" * 80)
    print("ADVANCED MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 80)

    # Parameter grids for each model
    param_grids = {
        "Logistic Regression": {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
    }

    models_dict = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    best_models = {}
    tuning_results = {}

    for model_name, model in models_dict.items():
        print(f"\nTuning {model_name}...")

        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        best_models[model_name] = grid_search.best_estimator_
        tuning_results[model_name] = {
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
        }

        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"Best Params: {grid_search.best_params_}")

    return best_models, tuning_results


def evaluate_and_compare_models(best_models, X_test, y_test):
    """Evaluate and compare tuned models."""

    print("\n" + "=" * 80)
    print("MODEL COMPARISON ON TEST SET")
    print("=" * 80)

    evaluator = ModelEvaluator(random_state=RANDOM_STATE)
    comparison_df = evaluator.compare_models(best_models, X_test, y_test)

    print("\nPerformance Metrics:")
    print(comparison_df.round(4))

    # Save comparison results
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv")

    return comparison_df, evaluator


def cross_validate_best_model(best_model, X_train, y_train, model_name):
    """Perform cross-validation on the best model."""

    print(f"\n{model_name} - Cross-Validation Results:")

    evaluator = ModelEvaluator(random_state=RANDOM_STATE)
    cv_scores = evaluator.cross_validate_model(best_model, X_train, y_train, cv=5)

    cv_summary = {}
    for metric, scores in cv_scores.items():
        if "test_" in metric:
            metric_name = metric.replace("test_", "")
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv_summary[metric_name] = {"mean": mean_score, "std": std_score}
            print(f"  {metric_name}: {mean_score:.4f} (+/- {std_score:.4f})")

    return cv_summary


def save_best_model(model, model_name, feature_names):
    """Save the best model and metadata."""

    model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
    metadata_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_metadata.json"

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "feature_names": feature_names,
        "training_date": pd.Timestamp.now().isoformat(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


def main():
    """Main pipeline for model training and evaluation."""

    # Load and prepare data
    print("Loading and preparing data...")
    df = load_clean_data(CLEAN_DATA_PATH)
    X, y = prepare_modeling_data(df, target=TARGET)

    # Preprocess features
    print("Preprocessing features...")
    X_processed, encoders, scaler = preprocess_features(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Churn rate (train): {y_train.mean():.2%}")
    print(f"Churn rate (test): {y_test.mean():.2%}")

    # Train models with hyperparameter tuning
    best_models, tuning_results = train_models_with_hyperparameter_tuning(
        X_train, y_train, X_test, y_test
    )

    # Evaluate and compare models
    comparison_df, evaluator = evaluate_and_compare_models(best_models, X_test, y_test)

    # Get the best performing model
    best_model_name = comparison_df["roc_auc"].idxmax()
    best_model = best_models[best_model_name]

    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
    print(f"{'=' * 80}")

    # Cross-validation for best model
    cv_summary = cross_validate_best_model(best_model, X_train, y_train, best_model_name)

    # Generate visualizations for best model
    print(f"\nGenerating visualizations for {best_model_name}...")

    # Confusion matrix
    cm = evaluator.get_confusion_matrix(best_model, X_test, y_test)
    plot_confusion_matrix(
        cm, save_path=MODELS_DIR / f"{best_model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    # ROC curve
    fpr, tpr, _ = evaluator.get_roc_curve(best_model, X_test, y_test)
    auc = comparison_df.loc[best_model_name, "roc_auc"]
    plot_roc_curve(fpr, tpr, auc, save_path=MODELS_DIR / f"{best_model_name.lower().replace(' ', '_')}_roc_curve.png")

    # Feature importance (for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(
            X_processed.columns.tolist(),
            best_model.feature_importances_,
            save_path=MODELS_DIR / f"{best_model_name.lower().replace(' ', '_')}_feature_importance.png",
        )

    # Save the best model
    save_best_model(best_model, best_model_name, X_processed.columns.tolist())

    # Save tuning results
    with open(MODELS_DIR / "hyperparameter_tuning_results.json", "w") as f:
        json.dump(tuning_results, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
