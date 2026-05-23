"""Visualization utilities for exploratory and model analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_churn_distribution(df: pd.DataFrame, target: str = "Churn", save_path: Optional[Path] = None):
    """Plot churn distribution.

    Args:
        df: Input DataFrame
        target: Target column name
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Count plot
    churn_counts = df[target].value_counts()
    axes[0].bar(["No Churn", "Churn"], churn_counts.values, color=["#2ecc71", "#e74c3c"])
    axes[0].set_ylabel("Count")
    axes[0].set_title("Churn Distribution")
    axes[0].set_ylim([0, len(df)])

    # Percentage
    churn_pct = df[target].value_counts(normalize=True) * 100
    axes[1].pie(
        churn_pct.values,
        labels=[f"No Churn ({churn_pct.values[0]:.1f}%)", f"Churn ({churn_pct.values[1]:.1f}%)"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
    )
    axes[1].set_title("Churn Percentage")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_categorical_churn(
    df: pd.DataFrame,
    col: str,
    target: str = "Churn",
    save_path: Optional[Path] = None,
):
    """Plot churn rate by categorical variable.

    Args:
        df: Input DataFrame
        col: Categorical column name
        target: Target column name
        save_path: Path to save figure
    """
    churn_by_cat = df.groupby(col)[target].agg(["sum", "count"])
    churn_by_cat["churn_rate"] = churn_by_cat["sum"] / churn_by_cat["count"]

    fig, ax = plt.subplots(figsize=(10, 5))
    churn_by_cat["churn_rate"].sort_values().plot(kind="barh", ax=ax, color="#3498db")
    ax.set_xlabel("Churn Rate")
    ax.set_title(f"Churn Rate by {col}")
    ax.set_xlim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_numerical_churn(
    df: pd.DataFrame,
    col: str,
    target: str = "Churn",
    save_path: Optional[Path] = None,
):
    """Plot numerical variable by churn status.

    Args:
        df: Input DataFrame
        col: Numerical column name
        target: Target column name
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    data_no_churn = df[df[target] == 0][col]
    data_churn = df[df[target] == 1][col]

    ax.boxplot(
        [data_no_churn, data_churn],
        labels=["No Churn", "Churn"],
        patch_artist=True,
        boxprops=dict(facecolor="#3498db", alpha=0.7),
    )
    ax.set_ylabel(col)
    ax.set_title(f"{col} by Churn Status")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot correlation matrix heatmap.

    Args:
        df: Input DataFrame (numeric columns only)
        save_path: Path to save figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: Optional[Path] = None):
    """Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 15,
    save_path: Optional[Path] = None,
):
    """Plot feature importances.

    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to display
        save_path: Path to save figure
    """
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color="#3498db")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[Path] = None):
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
