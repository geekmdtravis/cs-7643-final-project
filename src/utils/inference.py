"""
Run inference on a CXR model from a saved torch model.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tqdm
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    coverage_error,
    f1_score,
    fbeta_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ..models import CXRModel
from .config import Config
from .persistence import load_model

cfg = Config()


def run_inference(
    model: str | CXRModel,
    test_loader: DataLoader,
    device: torch.device | Literal["cuda", "cpu"] = "cuda",
):
    """
    Run inference using the provided model.

    Args:
        model (CXRModel | str): CXRModel or path to a saved model file.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device | str): Device to run the inference on. Default is "cuda".
    Returns:
        tuple: Tuple containing:
            - all_preds (np.ndarray): Array of predicted probabilities.
            - all_labels (np.ndarray): Array of true labels.
    """
    if isinstance(model, str):
        path = Path(model)
        if not path.exists():
            raise FileNotFoundError(f"Model path {model} does not exist.")
        if not path.is_file():
            raise ValueError(f"Model path {model} is not a file.")
        if not path.suffix == ".pth":
            logging.warning(
                f"Model path {model} does not have a .pth extension. "
                "This may not be a valid model file."
            )

        print(f"Loading model from {model}...")
        model = load_model(path)
        model = model.to(device)

    all_preds = []
    all_labels = []
    batch = 1
    with torch.no_grad():
        pbar = tqdm.tqdm(
            test_loader, desc="Running inference", unit="batch", total=len(test_loader)
        )
        for images, tabular, labels in pbar:
            batch += 1
            images: torch.Tensor = images.to(device)
            tabular: torch.Tensor = tabular.to(device)
            labels: torch.Tensor = labels.to(device)

            outputs = model(images, tabular)
            preds = torch.sigmoid(outputs)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_preds, all_labels


def find_optimal_thresholds(
    y_true: np.ndarray, y_pred_proba: np.ndarray, labels: list
) -> dict:
    """
    Find optimal classification thresholds for each label based on class prevalence.

    Args:
        y_true (np.ndarray): Array of true labels
        y_pred_proba (np.ndarray): Array of predicted probabilities
        labels (list): List of class labels

    Returns:
        dict: Dictionary mapping label names to their optimal thresholds
    """
    thresholds = {}

    for i, label in enumerate(labels):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred_proba[:, i]

        prevalence = np.mean(y_true_label)

        potential_thresholds = np.linspace(0.01, 0.99, 99)

        best_threshold = 0.5
        best_metric = 0

        if prevalence < 0.01:  # Very rare
            for threshold in potential_thresholds:
                y_pred_binary = (y_pred_label >= threshold).astype(int)
                # Beta=2 gives recall twice the importance of precision
                f2_score_val = fbeta_score(y_true_label, y_pred_binary, beta=2)

                if f2_score_val > best_metric:
                    best_metric = f2_score_val
                    best_threshold = threshold
        else:  # More common
            for threshold in potential_thresholds:
                y_pred_binary = (y_pred_label >= threshold).astype(int)
                f1_score_val = f1_score(y_true_label, y_pred_binary)

                if f1_score_val > best_metric:
                    best_metric = f1_score_val
                    best_threshold = threshold

        thresholds[label] = best_threshold

    return thresholds


def evaluate_model(preds: np.ndarray, labels: np.ndarray):
    """
    Evaluate the model using multiple metrics suitable for multi-label classification.

    Args:
        preds (np.ndarray): Array of predicted probabilities.
        labels (np.ndarray): Array of true labels.
    Returns:
        dict: Dictionary containing various evaluation metrics:
            - auc_scores: List of AUC scores for each class
            - report: Classification report as a dictionary
            - hamming_loss: Hamming loss score
            - jaccard_similarity: Jaccard similarity score
            - avg_precision: Average precision score
            - confusion_matrices: List of confusion matrices for each class
            - lrap: Label ranking average precision
            - coverage_error: Coverage error
            - ranking_loss: Ranking loss
    """
    thresholds = find_optimal_thresholds(labels, preds, cfg.class_labels)

    binary_preds = np.zeros_like(preds)
    for i, label in enumerate(cfg.class_labels):
        binary_preds[:, i] = (preds[:, i] >= thresholds[label]).astype(int)

    results = {"thresholds": thresholds}

    report = classification_report(
        labels, binary_preds, target_names=cfg.class_labels, output_dict=True
    )
    results["report"] = report

    auc_scores = []
    valid_indices = []

    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            auc_scores.append(auc)
            valid_indices.append(i)
        else:
            auc_scores.append(np.nan)

    results["auc_scores"] = auc_scores

    if valid_indices:
        results["micro_auc"] = roc_auc_score(
            labels[:, valid_indices], preds[:, valid_indices], average="micro"
        )
        results["macro_auc"] = roc_auc_score(
            labels[:, valid_indices], preds[:, valid_indices], average="macro"
        )
        results["weighted_auc"] = roc_auc_score(
            labels[:, valid_indices], preds[:, valid_indices], average="weighted"
        )
    else:
        results["micro_auc"] = np.nan
        results["macro_auc"] = np.nan
        results["weighted_auc"] = np.nan

    results["hamming_loss"] = hamming_loss(labels, binary_preds)

    results["jaccard_similarity"] = jaccard_score(
        labels, binary_preds, average="samples"
    )

    results["avg_precision"] = average_precision_score(
        labels, preds, average="weighted"
    )

    confusion_matrices = []
    for i in range(labels.shape[1]):
        cm = confusion_matrix(labels[:, i], binary_preds[:, i])
        confusion_matrices.append(cm)
    results["confusion_matrices"] = confusion_matrices

    results["lrap"] = label_ranking_average_precision_score(labels, preds)
    results["coverage_error"] = coverage_error(labels, preds)
    results["ranking_loss"] = label_ranking_loss(labels, preds)

    return results


def print_evaluation_results(
    results: dict,
    save_path: str = None,
):
    """
    Print the evaluation results and optionally save them to a file.

    Args:
        results (dict): Dictionary containing all evaluation metrics
        save_path (str, optional): Path to save the results.
            If None, results are only printed.
    """
    output = []

    output.append("Optimal Classification Thresholds:\n")
    for class_name, threshold in results["thresholds"].items():
        output.append(f"{class_name}: {threshold:.4f}")

    output.append("\nIndividual Class Performance:")
    output.append(
        "(AUC scores are binary classification metrics that handle class imbalance)\n"
    )

    for i, (class_name, auc) in enumerate(zip(cfg.class_labels, results["auc_scores"])):
        support = (
            results["report"][class_name]["support"]
            if class_name in results["report"]
            else 0
        )
        if not np.isnan(auc):
            output.append(f"{class_name}:")
            output.append(f"  AUC Score: {auc:.4f}")
            output.append(f"  Support: {support} samples")
        else:
            output.append(f"{class_name}:")
            output.append("  AUC Score: Not applicable (only one class present)")
            output.append(f"  Support: {support} samples")
        output.append("")

    output.append("Overall AUC Scores:")
    output.append(f"Weighted Average AUC: {results['weighted_auc']:.4f}")
    output.append("  (weights each class's AUC by its frequency)")
    output.append(f"Macro Average AUC: {results['macro_auc']:.4f}")
    output.append("  (gives equal weight to each class)")
    output.append(f"Micro Average AUC: {results['micro_auc']:.4f}")
    output.append("  (aggregates all predictions regardless of class)")

    output.append("\nOverall Metrics:")
    output.append(f"Hamming Loss: {results['hamming_loss']:.4f}")
    output.append(f"Jaccard Similarity: {results['jaccard_similarity']:.4f}")
    output.append(f"Average Precision: {results['avg_precision']:.4f}")
    output.append(f"Label Ranking Average Precision: {results['lrap']:.4f}")
    output.append(f"Coverage Error: {results['coverage_error']:.4f}")
    output.append(f"Ranking Loss: {results['ranking_loss']:.4f}")

    output.append("\nClassification Report:\n")
    for class_name, metrics in results["report"].items():
        if isinstance(metrics, dict):
            output.append(f"Class: {class_name}")
            output.append(f"  Precision: {metrics['precision']:.4f}")
            output.append(f"  Recall: {metrics['recall']:.4f}")
            output.append(f"  F1-score: {metrics['f1-score']:.4f}")
            output.append(f"  Support: {metrics['support']}")
            output.append("")

    output.append("\nConfusion Matrices (per class):")
    for i, (class_name, cm) in enumerate(
        zip(cfg.class_labels, results["confusion_matrices"])
    ):
        output.append(f"\n{class_name}:")
        output.append("[[TN FP]")
        output.append(" [FN TP]]")
        output.append(str(cm))

    results_str = "\n".join(output)

    print(results_str)

    if save_path:
        with open(save_path, "w") as f:
            f.write(results_str)
        print(f"Results saved to {save_path}")
