"""
Run inference using trained DenseNet-121
Embedded model on the NIH Chest X-ray test dataset.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from src.data.create_dataloader import create_dataloader
from src.models import DenseNet121Vanilla
from src.utils import Config

# Define class names for classification report
CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

cfg = Config()


def run_inference(model, test_loader, device):
    """Run inference using the provided model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, _, labels in tqdm(test_loader, desc="Running inference"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def evaluate_model(preds, labels):
    """Calculate and return evaluation metrics."""
    # Calculate AUC-ROC for each class
    auc_scores = []
    for i in range(labels.shape[1]):
        if (
            len(np.unique(labels[:, i])) > 1
        ):  # Check if class has both positive and negative samples
            auc = roc_auc_score(labels[:, i], preds[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(np.nan)

    # Convert predictions to binary (0/1) using 0.5 as threshold
    binary_preds = (preds > 0.5).astype(int)

    # Get classification report with explicit class names
    report = classification_report(
        labels, binary_preds, target_names=CLASSES, output_dict=True
    )

    return auc_scores, report


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 16

    # Create test dataloader
    test_loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.cxr_test_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalization_mode="imagenet",
    )

    # Initialize model
    model = DenseNet121Vanilla(num_classes=15).to(device)

    # Load and evaluate best model
    logger.info("Evaluating best model...")
    model.load_state_dict(torch.load(cfg.artifacts / "densenet121_embedded_best.pth"))
    best_preds, test_labels = run_inference(model, test_loader, device)
    best_auc_scores, best_report = evaluate_model(best_preds, test_labels)

    # Load and evaluate final model
    logger.info("Evaluating final model...")
    model.load_state_dict(torch.load(cfg.artifacts / "densenet121_embedded_final.pth"))
    final_preds, test_labels = run_inference(model, test_loader, device)
    final_auc_scores, final_report = evaluate_model(final_preds, test_labels)

    # Log results
    logger.info("\nBest Model Results:")
    logger.info(f"Average AUC-ROC: {np.nanmean(best_auc_scores):.4f}")
    for i, (class_name, auc) in enumerate(zip(CLASSES, best_auc_scores)):
        logger.info(f"{class_name}: AUC-ROC = {auc:.4f}")

    logger.info("\nFinal Model Results:")
    logger.info(f"Average AUC-ROC: {np.nanmean(final_auc_scores):.4f}")
    for i, (class_name, auc) in enumerate(zip(CLASSES, final_auc_scores)):
        logger.info(f"{class_name}: AUC-ROC = {auc:.4f}")

    # Save detailed results to file
    with open("results/inference_results_embedded.txt", "w") as f:
        f.write("Best Model Results:\n")
        f.write(f"Average AUC-ROC: {np.nanmean(best_auc_scores):.4f}\n")
        for class_name, auc in zip(CLASSES, best_auc_scores):
            f.write(f"{class_name}: AUC-ROC = {auc:.4f}\n")
        f.write("\nClassification Report:\n")
        for class_name, metrics in best_report.items():
            if isinstance(metrics, dict):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")

        f.write("\nFinal Model Results:\n")
        f.write(f"Average AUC-ROC: {np.nanmean(final_auc_scores):.4f}\n")
        for class_name, auc in zip(CLASSES, final_auc_scores):
            f.write(f"{class_name}: AUC-ROC = {auc:.4f}\n")
        f.write("\nClassification Report:\n")
        for class_name, metrics in final_report.items():
            if isinstance(metrics, dict):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")


if __name__ == "__main__":
    main()
