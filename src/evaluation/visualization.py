"""
Visualization routines for evaluation reports and figures.
"""
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_confusion_matrix(
    confusion_matrix_data: List[List[int]],
    output_path: Path,
    title: str = "Confusion Matrix",
    labels: Optional[List[str]] = None,
) -> None:
    """
    Renders and saves a formatted confusion matrix heatmap.
    """
    labels = labels or ["Human (0)", "AI (1)"]
    cm = np.array(confusion_matrix_data)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Calculate percentages
    total = np.sum(cm)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / total) * 100 if total > 0 else 0
            annot[i, j] = f"{count}\n({pct:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )

    ax.set_title(title, fontsize=14, pad=15, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Ground Truth", fontsize=12, labelpad=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    auc_score: Optional[float],
    output_path: Path,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> None:
    """
    Renders and saves an ROC Curve plot.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    auc_label = f"AUC = {auc_score:.3f}" if auc_score is not None else "AUC = N/A"
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC Curve ({auc_label})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(
    y_true: List[int],
    y_scores: List[float],
    output_path: Path,
    title: str = "AI Detection Score Distribution",
) -> None:
    """
    Plots probability density histograms for Human vs AI classes.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    scores = np.array(y_scores)
    labels = np.array(y_true)
    
    human_scores = scores[labels == 0]
    ai_scores = scores[labels == 1]

    if len(human_scores) > 0:
        ax.hist(human_scores, bins=15, alpha=0.6, color="#10B981", label=f"Human (N={len(human_scores)})", density=True)
    if len(ai_scores) > 0:
        ax.hist(ai_scores, bins=15, alpha=0.6, color="#EF4444", label=f"AI-Generated (N={len(ai_scores)})", density=True)

    ax.set_xlabel("AI Detection Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.legend(frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
