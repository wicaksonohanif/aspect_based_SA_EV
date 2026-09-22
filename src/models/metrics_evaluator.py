"""
Metrics Evaluator & Confusion Matrix Visualization Module for ABSA
Spec Compliance: specs/03_model_training.spec.md
Computes Accuracy, Precision, Recall, Macro/Weighted F1, Exact Match Ratio,
Class Weights, and 4x4 Confusion Matrices per aspect.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

LABEL2ID = {'None': 0, 'none': 0, 'positif': 1, 'netral': 2, 'negatif': 3}
ID2LABEL = {0: 'None', 1: 'positif', 2: 'netral', 3: 'negatif'}
ASPECTS = ['infra', 'ekonomi', 'kualitas', 'purnajual']
CLASS_NAMES = ['None', 'Positif', 'Netral', 'Negatif']


def compute_aspect_class_weights(df: pd.DataFrame, aspects: list[str] = ASPECTS) -> dict[str, torch.Tensor]:
    """
    Menghitung loss class weights ter-balans per aspek untuk menangani class imbalance.

    Args:
        df: Dataframe pandas yang berisi kolom '{aspect}_sentiment'

    Returns:
        Dict mapping aspect -> torch.FloatTensor shape [4]
    """
    weights_dict = {}
    classes = np.array([0, 1, 2, 3])

    for aspect in aspects:
        col = f"{aspect}_sentiment" if f"{aspect}_sentiment" in df.columns else aspect
        if col not in df.columns:
            weights_dict[aspect] = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
            continue

        raw_vals = df[col].fillna("none").astype(str).str.lower().map(LABEL2ID).fillna(0).astype(int).values
        
        # Calculate balanced weights using sklearn
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=raw_vals)
        weights_dict[aspect] = torch.tensor(weights, dtype=torch.float32)

    return weights_dict


def evaluate_predictions(
    y_true_dict: dict[str, np.ndarray | list],
    y_pred_dict: dict[str, np.ndarray | list],
    aspects: list[str] = ASPECTS,
) -> dict:
    """
    Mengevaluasi hasil prediksi terhadap ground truth secara multi-aspek.

    Args:
        y_true_dict: Dict mapping aspect -> array 1D true labels [0, 1, 2, 3]
        y_pred_dict: Dict mapping aspect -> array 1D predicted labels [0, 1, 2, 3]

    Returns:
        Dict lengkap berisi metrik per aspek, mean metrics, exact match ratio, dan confusion matrices.
    """
    aspect_metrics = {}
    cm_dict = {}
    macro_f1s = []
    weighted_f1s = []
    accuracies = []

    for aspect in aspects:
        y_true = np.array(y_true_dict[aspect])
        y_pred = np.array(y_pred_dict[aspect])

        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        aspect_metrics[aspect] = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
        }
        cm_dict[aspect] = cm

        accuracies.append(acc)
        macro_f1s.append(f1_macro)
        weighted_f1s.append(f1_weighted)

    # Calculate Exact Match Ratio (Subset Accuracy: 100% match across all 4 aspects)
    n_samples = len(y_true_dict[aspects[0]])
    exact_matches = 0
    for i in range(n_samples):
        match = all(y_true_dict[asp][i] == y_pred_dict[asp][i] for asp in aspects)
        if match:
            exact_matches += 1
    exact_match_ratio = exact_matches / n_samples if n_samples > 0 else 0.0

    overall_metrics = {
        "mean_accuracy": np.mean(accuracies),
        "mean_macro_f1": np.mean(macro_f1s),
        "mean_weighted_f1": np.mean(weighted_f1s),
        "exact_match_ratio": exact_match_ratio,
    }

    return {
        "per_aspect": aspect_metrics,
        "overall": overall_metrics,
        "confusion_matrices": cm_dict,
    }


def plot_confusion_matrices(
    cm_dict: dict[str, np.ndarray],
    model_name: str,
    save_path: str = "confusion_matrices.png",
    aspects: list[str] = ASPECTS,
) -> str:
    """
    Visualisasi Grid 2x2 Heatmap Confusion Matrix 4x4 untuk 4 aspek dan menyimpan sebagai gambar.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Confusion Matrices 4x4 — {model_name}", fontsize=16, fontweight="bold", y=0.98)

    for idx, aspect in enumerate(aspects):
        ax = axes[idx // 2, idx % 2]
        cm = cm_dict[aspect]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )
        ax.set_title(f"Aspek: {aspect.capitalize()}", fontsize=14, fontweight="semibold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


if __name__ == "__main__":
    print("Metrics Evaluator module initialized successfully!")
