"""
Metrics Evaluator & Confusion Matrix Visualization Module for ABSA (v4 - Aspect-Customized Thresholds)
Spec Compliance: specs/03_model_training.spec.md
Computes Accuracy, Precision, Recall, Macro/Weighted F1 (All-Class & Active-Sentiment), Exact Match Ratio,
Smoothed Class Weights, and 4x4 Confusion Matrices per aspect.
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

# Customized aspect decision thresholds based on aspect frequency
ASPECT_THRESHOLDS = {
    "infra": 0.40,
    "ekonomi": 0.50,
    "kualitas": 0.50,
    "purnajual": 0.35,
}


def compute_aspect_class_weights(
    df: pd.DataFrame,
    aspects: list[str] = ASPECTS,
    smooth_factor: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Menghitung smoothed loss class weights dengan penyesuaian ekstra (targeted alpha)
    untuk kelas minoritas ekstrem seperti purnajual positif dan infra positif.
    """
    weights_dict = {}
    classes = np.array([0, 1, 2, 3])

    for aspect in aspects:
        col = f"{aspect}_sentiment" if f"{aspect}_sentiment" in df.columns else aspect
        if col not in df.columns:
            weights_dict[aspect] = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
            continue

        raw_vals = df[col].fillna("none").astype(str).str.lower().map(LABEL2ID).fillna(0).astype(int).values
        raw_weights = compute_class_weight(class_weight="balanced", classes=classes, y=raw_vals)

        # Smooth weights via power scaling (sqrt)
        smoothed = np.power(raw_weights, smooth_factor)

        # Apply targeted alpha boost for extreme minority classes
        if aspect == "purnajual":
            smoothed[1] *= 2.5  # positif purnajual boost
            smoothed[2] *= 1.5  # netral purnajual boost
        elif aspect == "infra":
            smoothed[1] *= 1.8  # positif infra boost

        smoothed = smoothed / np.mean(smoothed)
        weights_dict[aspect] = torch.tensor(smoothed, dtype=torch.float32)

    return weights_dict


def predict_with_threshold(
    logits: torch.Tensor | np.ndarray,
    none_threshold: float | dict[str, float] = ASPECT_THRESHOLDS,
    aspect: str | None = None,
) -> np.ndarray:
    """
    Prediksi label menggunakan Aspect-Customized Decision Thresholding untuk kelas 'None' (index 0).
    """
    if isinstance(none_threshold, dict) and aspect in none_threshold:
        thresh = none_threshold[aspect]
    elif isinstance(none_threshold, (int, float)):
        thresh = float(none_threshold)
    else:
        thresh = 0.50

    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    else:
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    preds = []
    for i in range(probs.shape[0]):
        p_none = probs[i, 0]
        if p_none > thresh:
            preds.append(0)
        else:
            active_probs = probs[i, 1:]  # index 1, 2, 3
            best_active_idx = int(np.argmax(active_probs)) + 1
            preds.append(best_active_idx)

    return np.array(preds)


def evaluate_predictions(
    y_true_dict: dict[str, np.ndarray | list],
    y_pred_dict: dict[str, np.ndarray | list],
    aspects: list[str] = ASPECTS,
) -> dict:
    """
    Mengevaluasi hasil prediksi terhadap ground truth secara multi-aspek.
    """
    aspect_metrics = {}
    cm_dict = {}
    macro_f1s_all = []
    macro_f1s_active = []
    accuracies = []

    for aspect in aspects:
        y_true = np.array(y_true_dict[aspect])
        y_pred = np.array(y_pred_dict[aspect])

        acc = accuracy_score(y_true, y_pred)
        
        # All-Class Metrics (0, 1, 2, 3)
        prec_m_all = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_m_all = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_m_all = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Active-Sentiment Metrics (1, 2, 3)
        prec_m_active = precision_score(y_true, y_pred, labels=[1, 2, 3], average="macro", zero_division=0)
        rec_m_active = recall_score(y_true, y_pred, labels=[1, 2, 3], average="macro", zero_division=0)
        f1_m_active = f1_score(y_true, y_pred, labels=[1, 2, 3], average="macro", zero_division=0)

        f1_w_all = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        aspect_metrics[aspect] = {
            "accuracy": acc,
            "f1_macro_all": f1_m_all,
            "f1_macro_active": f1_m_active,
            "precision_macro_all": prec_m_all,
            "recall_macro_all": rec_m_all,
            "f1_weighted_all": f1_w_all,
        }
        cm_dict[aspect] = cm

        accuracies.append(acc)
        macro_f1s_all.append(f1_m_all)
        macro_f1s_active.append(f1_m_active)

    n_samples = len(y_true_dict[aspects[0]])
    exact_matches = 0
    for i in range(n_samples):
        match = all(y_true_dict[asp][i] == y_pred_dict[asp][i] for asp in aspects)
        if match:
            exact_matches += 1
    exact_match_ratio = exact_matches / n_samples if n_samples > 0 else 0.0

    overall_metrics = {
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_macro_f1_all": float(np.mean(macro_f1s_all)),
        "mean_macro_f1_active": float(np.mean(macro_f1s_active)),
        "exact_match_ratio": float(exact_match_ratio),
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
    print("Metrics Evaluator v4 initialized successfully!")
