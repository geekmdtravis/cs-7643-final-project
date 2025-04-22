"""
Run inference on a CXR model from a saved torch model.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import tqdm
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    hamming_loss,
    jaccard_score,
    average_precision_score,
    confusion_matrix,
    label_ranking_average_precision_score,
    coverage_error,
    label_ranking_loss,
)
from torch.utils.data import DataLoader

from src.models import CXRModel
from src.utils import Config

cfg = Config()


def run_inference(
    state_dict_path: str,
    model: CXRModel,
    test_loader: DataLoader,
    device: torch.device | str = "cuda",
):
    """
    Run inference using the provided model.

    Args:
        model_path (str): Path to the saved model file.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device | str): Device to run the inference on. Default is "cuda".
    Returns:
        tuple: Tuple containing:
            - all_preds (np.ndarray): Array of predicted probabilities.
            - all_labels (np.ndarray): Array of true labels.
    """
    path = Path(state_dict_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path {state_dict_path} does not exist.")
    if not path.is_file():
        raise ValueError(f"Model path {state_dict_path} is not a file.")
    if not path.suffix == ".pth":
        logging.warning(
            f"Model path {state_dict_path} does not have a .pth extension. "
            "This may not be a valid model file."
        )
    if not isinstance(model, CXRModel):
        raise TypeError(
            f"Model should be an instance of CXRModel. Got {type(model)} instead."
        )

    print(f"Loading model from {state_dict_path}...")
    # Load the state dict
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    print(f"Model loaded successfully from {state_dict_path}.")
    # Move model to device
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
    # Convert probabilities to binary predictions
    binary_preds = (preds > 0.5).astype(int)

    # Initialize results dictionary
    results = {}

    # 1. AUC Scores (per class)
    auc_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(np.nan)
    results["auc_scores"] = auc_scores

    # 2. Classification Report
    report = classification_report(
        labels, binary_preds, target_names=cfg.class_labels, output_dict=True
    )
    results["report"] = report

    # 3. Hamming Loss
    results["hamming_loss"] = hamming_loss(labels, binary_preds)

    # 4. Jaccard Similarity (IoU)
    results["jaccard_similarity"] = jaccard_score(
        labels, binary_preds, average="samples"
    )

    # 5. Average Precision
    results["avg_precision"] = average_precision_score(
        labels, preds, average="weighted"
    )

    # 6. Confusion Matrices (one per class)
    confusion_matrices = []
    for i in range(labels.shape[1]):
        cm = confusion_matrix(labels[:, i], binary_preds[:, i])
        confusion_matrices.append(cm)
    results["confusion_matrices"] = confusion_matrices

    # 7. Label Ranking Metrics
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

    # AUC Scores section
    output.append("AUC Scores:\n")
    for i, (class_name, auc) in enumerate(zip(cfg.class_labels, results["auc_scores"])):
        if not np.isnan(auc):
            output.append(f"AUC for {class_name}: {auc:.4f}")
        else:
            output.append(
                f"AUC for {class_name}: Not applicable (only one class present)"
            )

    # Overall Metrics
    output.append("\nOverall Metrics:")
    output.append(f"Hamming Loss: {results['hamming_loss']:.4f}")
    output.append(f"Jaccard Similarity: {results['jaccard_similarity']:.4f}")
    output.append(f"Average Precision: {results['avg_precision']:.4f}")
    output.append(f"Label Ranking Average Precision: {results['lrap']:.4f}")
    output.append(f"Coverage Error: {results['coverage_error']:.4f}")
    output.append(f"Ranking Loss: {results['ranking_loss']:.4f}")

    # Classification Report section
    output.append("\nClassification Report:\n")
    for class_name, metrics in results["report"].items():
        if isinstance(metrics, dict):  # Skip the 'accuracy' and 'macro avg' entries
            output.append(f"Class: {class_name}")
            output.append(f"  Precision: {metrics['precision']:.4f}")
            output.append(f"  Recall: {metrics['recall']:.4f}")
            output.append(f"  F1-score: {metrics['f1-score']:.4f}")
            output.append(f"  Support: {metrics['support']}")
            output.append("")

    # Confusion Matrices
    output.append("\nConfusion Matrices (per class):")
    for i, (class_name, cm) in enumerate(
        zip(cfg.class_labels, results["confusion_matrices"])
    ):
        output.append(f"\n{class_name}:")
        output.append("[[TN FP]")
        output.append(" [FN TP]]")
        output.append(str(cm))

    # Join all lines into a single string
    results_str = "\n".join(output)

    # Print the results
    print(results_str)

    # Save to file if save_path is provided
    if save_path:
        with open(save_path, "w") as f:
            f.write(results_str)
        print(f"Results saved to {save_path}")
