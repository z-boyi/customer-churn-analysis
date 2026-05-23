"""Model training, evaluation, and comparison utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from .config import RANDOM_STATE


class ModelEvaluator:
    """Evaluate and compare classification models."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.results = {}

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
    ) -> Dict[str, float]:
        """Evaluate a trained model on test set.

        Args:
            model: Trained sklearn model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for tracking

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        self.results[model_name] = metrics
        return metrics

    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict[str, np.ndarray]:
        """Perform cross-validation on a model.

        Args:
            model: Sklearn model
            X: Features
            y: Target
            cv: Number of folds

        Returns:
            Dictionary of cross-validation scores
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        scores = cross_validate(
            model,
            X,
            y,
            cv=skf,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=False,
        )

        return scores

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Compare multiple models on test set.

        Args:
            models: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with comparison metrics
        """
        results_list = []

        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            metrics["model"] = model_name
            results_list.append(metrics)

        return pd.DataFrame(results_list).set_index("model")

    def get_confusion_matrix(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> np.ndarray:
        """Get confusion matrix for a model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Confusion matrix
        """
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def get_roc_curve(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data for a model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        return fpr, tpr, thresholds

    def get_classification_report(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> str:
        """Get detailed classification report.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Formatted classification report
        """
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred)


def create_baseline_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Create a collection of baseline models for comparison.

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Dictionary of {model_name: model}
    """
    return {
        "Logistic Regression": LogisticRegression(
            random_state=random_state, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
    }
