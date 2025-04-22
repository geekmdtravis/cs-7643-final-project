"""
Run inference on a CXR model from a saved torch model.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tqdm
from sklearn.metrics import classification_report, roc_auc_score
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


def evaluate_model(preds: np.ndarray, labels: np.ndarray):
    """
    Evaluate the model using classification report and ROC AUC score.

    Args:
        preds (np.ndarray): Array of predicted probabilities.
        labels (np.ndarray): Array of true labels.
    Returns:
        tuple: Tuple containing:
            - auc_scores (list): List of AUC scores for each class.
            - report (dict): Classification report as a dictionary.
    """
    auc_scores = []
    auc_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(np.nan)

    binary_preds = (preds > 0.5).astype(int)
    report = classification_report(
        labels, binary_preds, target_names=cfg.class_labels, output_dict=True
    )
    return auc_scores, report


def print_evaluation_results(
    auc_scores: list,
    report: dict,
    save_path: str = None,
):
    """
    Print the evaluation results and optionally save them to a file.

    Args:
        auc_scores (list): List of AUC scores for each class.
        report (dict): Classification report as a dictionary.
        save_path (str, optional): Path to save the results.
            If None, results are only printed.
    """
    results = []

    # AUC Scores section
    results.append("AUC Scores:\n")
    for i, (class_name, auc) in enumerate(zip(cfg.class_labels, auc_scores)):
        if not np.isnan(auc):
            results.append(f"AUC for {class_name}: {auc:.4f}")
        else:
            results.append(
                f"AUC for {class_name}: Not applicable (only one class present)"
            )

    # Classification Report section
    results.append("\nClassification Report:\n")
    for class_name, metrics in report.items():
        results.append(f"Class: {class_name}")
        results.append(f"  Precision: {metrics['precision']:.4f}")
        results.append(f"  Recall: {metrics['recall']:.4f}")
        results.append(f"  F1-score: {metrics['f1-score']:.4f}")
        results.append(f"  Support: {metrics['support']}")
        results.append("")

    # Join all lines into a single string
    results_str = "\n".join(results)

    # Print the results
    print(results_str)

    # Save to file if save_path is provided
    if save_path:
        with open(save_path, "w") as f:
            f.write(results_str)
        print(f"Results saved to {save_path}")
