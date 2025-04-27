"""
Experiment 2: Testing the best models and analyzing their time and
epochs to convergence. This script loads the best models and measures
their performance metrics.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import CXRModel, CXRModelConfig
from src.utils import Config, load_model
from src.utils.trainer import __validate_one_epoch

# Define the models to test
MODELS = [
    "densenet121",
    "densenet121_mm",
    "vit_b_32",
    "vit_b_32_mm",
]

# Define the model paths (you'll need to provide these)
MODEL_PATHS = {
    "densenet121": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "densenet121_lr_0.0001_bs_32_do_0.2_hd_None_best.pth"
    ),
    "densenet121_mm": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "densenet121_mm_lr_0.0001_bs_32_do_0.2_hd_None_best.pth"
    ),
    "vit_b_32": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "vit_b_32_lr_0.0001_bs_32_do_0.2_hd_None_best.pth"
    ),
    "vit_b_32_mm": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "vit_b_32_mm_lr_0.0001_bs_32_do_0.2_hd_None_best.pth"
    ),
}

# Define the embedded model paths (you'll need to provide these)
EMBEDDED_MODEL_PATHS = {
    "densenet121": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "embd_densenet121_lr_1e-05_bs_32_do_0.2_hd_None_ms_32_best.pth"
    ),
    "vit_b_32": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "embd_vit_b_32_lr_1e-05_bs_32_do_0.2_hd_None_ms_32_best.pth"
    ),
}

# Define the training data paths
TRAINING_DATA_PATHS = {
    "densenet121": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "densenet121_lr_0.0001_bs_32_do_0.2_hd_None_tvdata.csv"
    ),
    "densenet121_mm": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "densenet121_mm_lr_0.0001_bs_32_do_0.2_hd_None_tvdata.csv"
    ),
    "vit_b_32": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "vit_b_32_lr_0.0001_bs_32_do_0.2_hd_None_tvdata.csv"
    ),
    "vit_b_32_mm": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "vit_b_32_mm_lr_0.0001_bs_32_do_0.2_hd_None_tvdata.csv"
    ),
    "densenet121_embedded": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "embd_densenet121_lr_1e-05_bs_32_do_0.2_hd_None_ms_32_tvdata.csv"
    ),
    "vit_b_32_embedded": (
        "/tmp/cs7643_final_share/travis_results/results/tuning/"
        "embd_vit_b_32_lr_1e-05_bs_32_do_0.2_hd_None_ms_32_tvdata.csv"
    ),
}

# Define the model names for display
MODEL_NAMES = {
    "densenet121": "DenseNet121",
    "densenet121_mm": "DenseNet121 (Multimodal)",
    "vit_b_32": "ViT-B/32",
    "vit_b_32_mm": "ViT-B/32 (Multimodal)",
    "densenet121_embedded": "DenseNet121 (Embedded)",
    "vit_b_32_embedded": "ViT-B/32 (Embedded)",
}

# Define the batch size for testing
BATCH_SIZE = 32

# Define the number of workers for the DataLoader
NUM_WORKERS = 4

# Define the criterion for testing
CRITERION = nn.BCEWithLogitsLoss()

# Define the config
cfg = Config()


def load_model_and_config(model_path: str) -> Tuple[CXRModel, CXRModelConfig]:
    """
    Load a model and its configuration from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        Tuple[CXRModel, CXRModelConfig]: The loaded model and its configuration.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    model = model.to(cfg.device)
    model.eval()

    # Create a new config from the model's attributes
    config = CXRModelConfig(
        model=model.model_name,
        hidden_dims=None,
        dropout=0.2,
        num_classes=15,
        tabular_features=4,
        freeze_backbone=True,
    )

    return model, config


def create_dataloader(model_name: str, use_embedded: bool = False) -> DataLoader:
    """
    Create a DataLoader for testing.

    Args:
        model_name (str): Name of the model.
        use_embedded (bool): Whether to use embedded images.

    Returns:
        DataLoader: The DataLoader for testing.
    """
    from src.data import create_dataloader

    # Determine which image directory to use
    if use_embedded:
        embedded_dir = cfg.embedded32_val_dir
        cxr_valid_img_dir = embedded_dir
    else:
        cxr_valid_img_dir = cfg.cxr_val_dir

    # Create the DataLoader
    val_loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_val,
        cxr_images_dir=cxr_valid_img_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalization_mode="imagenet",
    )

    return val_loader


def test_model(model: CXRModel, val_loader: DataLoader, model_name: str) -> Dict:
    """
    Test a model and measure its performance.

    Args:
        model (CXRModel): The model to test.
        val_loader (DataLoader): The DataLoader for testing.
        model_name (str): Name of the model for display.

    Returns:
        Dict: The test results.
    """
    print(f"Testing {model_name}...")

    # Measure inference time
    start_time = time.time()
    val_loss, val_auc = __validate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=CRITERION,
        device=cfg.device,
        pb_prefix=f"Testing {model_name}",
    )
    end_time = time.time()
    inference_time = end_time - start_time

    # Calculate average time per batch
    avg_time_per_batch = inference_time / len(val_loader)

    print(f"Test results for {model_name}:")
    print(f"  - Validation Loss: {val_loss:.4f}")
    print(f"  - Validation AUC: {val_auc:.4f}")
    print(f"  - Inference Time: {inference_time:.2f} seconds")
    print(f"  - Average Time per Batch: {avg_time_per_batch:.4f} seconds")

    return {
        "model_name": model_name,
        "val_loss": val_loss,
        "val_auc": val_auc,
        "inference_time": inference_time,
        "avg_time_per_batch": avg_time_per_batch,
    }


def analyze_convergence(model_name: str) -> Dict:
    """
    Analyze the convergence of a model from its training data.

    Args:
        model_name (str): Name of the model.

    Returns:
        Dict: The convergence analysis results.
    """
    # Get the path to the training data
    train_val_data_path = TRAINING_DATA_PATHS.get(model_name)

    # Check if the path is defined
    if train_val_data_path is None:
        print(f"No training data path defined for {model_name}.")
        return {
            "model_name": model_name,
            "epochs_to_convergence": None,
            "time_to_convergence": None,
        }

    # Check if the file exists
    if not os.path.exists(train_val_data_path):
        print(f"Training data file {train_val_data_path} not found.")
        return {
            "model_name": model_name,
            "epochs_to_convergence": None,
            "time_to_convergence": None,
        }

    # Load the training data
    try:
        training_data = pd.read_csv(train_val_data_path)
        print(f"Loaded training data from {train_val_data_path}")
    except Exception as e:
        print(f"Error reading {train_val_data_path}: {e}")
        return {
            "model_name": model_name,
            "epochs_to_convergence": None,
            "time_to_convergence": None,
        }

    # Find the epoch with the best validation loss
    best_epoch = training_data["val_loss"].idxmin() + 1

    # Calculate the time to convergence (assuming each epoch takes the same time)
    # This is an approximation since we don't have the actual time per epoch
    epochs_to_convergence = best_epoch

    # Estimate time to convergence based on the model type
    # These are rough estimates based on typical training times
    estimated_time_per_epoch = {
        "densenet121": 60,  # seconds
        "densenet121_mm": 70,
        "vit_b_32": 120,
        "vit_b_32_mm": 130,
        "densenet121_embedded": 80,
        "vit_b_32_embedded": 140,
    }

    # Get the estimated time per epoch for this model
    model_key = model_name
    if model_key not in estimated_time_per_epoch:
        # Default to a reasonable value if not specified
        estimated_time_per_epoch[model_key] = 100

    # Calculate estimated time to convergence in seconds
    estimated_time_to_convergence = (
        epochs_to_convergence * estimated_time_per_epoch[model_key]
    )

    # Convert to minutes and hours for readability
    time_in_minutes = estimated_time_to_convergence / 60
    time_in_hours = time_in_minutes / 60

    print(f"Convergence analysis for {model_name}:")
    print(f"  - Epochs to Convergence: {epochs_to_convergence}")
    print(
        f"  - Estimated Time to Convergence: {estimated_time_to_convergence:.2f} "
        f"seconds ({time_in_minutes:.2f} minutes, {time_in_hours:.2f} hours)"
    )
    print(f"  - Training data source: {train_val_data_path}")

    return {
        "model_name": model_name,
        "epochs_to_convergence": epochs_to_convergence,
        "time_to_convergence": estimated_time_to_convergence,
        "time_in_minutes": time_in_minutes,
        "time_in_hours": time_in_hours,
    }


def plot_results(test_results: List[Dict], convergence_results: List[Dict]) -> None:
    """
    Plot the test results.

    Args:
        test_results (List[Dict]): The test results.
        convergence_results (List[Dict]): The convergence analysis results.
    """
    # Create a DataFrame from the test results
    test_df = pd.DataFrame(test_results)

    # Create a DataFrame from the convergence results
    convergence_df = pd.DataFrame(convergence_results)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot validation AUC
    axes[0, 0].bar(test_df["model_name"], test_df["val_auc"])
    axes[0, 0].set_title("Validation AUC")
    axes[0, 0].set_xlabel("Model")
    axes[0, 0].set_ylabel("AUC")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Plot validation loss
    axes[0, 1].bar(test_df["model_name"], test_df["val_loss"])
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Model")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Plot inference time
    axes[1, 0].bar(test_df["model_name"], test_df["inference_time"])
    axes[1, 0].set_title("Inference Time")
    axes[1, 0].set_xlabel("Model")
    axes[1, 0].set_ylabel("Time (seconds)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Plot epochs to convergence
    if convergence_df["epochs_to_convergence"].notna().any():
        axes[1, 1].bar(
            convergence_df["model_name"], convergence_df["epochs_to_convergence"]
        )
        axes[1, 1].set_title("Epochs to Convergence")
        axes[1, 1].set_xlabel("Model")
        axes[1, 1].set_ylabel("Epochs")
        axes[1, 1].tick_params(axis="x", rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("results/experiment2/model_comparison.png")
    print("Results plot saved to results/experiment2/model_comparison.png")


def main():
    """Main function."""
    # Create results directory if it doesn't exist
    Path("results/experiment2").mkdir(parents=True, exist_ok=True)

    # Create a log file to save all output
    log_file_path = "results/experiment2/model_comparison_results.txt"
    with open(log_file_path, "w") as log_file:
        # Redirect stdout to both console and file
        import sys

        original_stdout = sys.stdout

        class TeeOutput:
            def __init__(self, file, original_stdout):
                self.file = file
                self.original_stdout = original_stdout

            def write(self, message):
                self.original_stdout.write(message)
                self.file.write(message)
                self.file.flush()

            def flush(self):
                self.original_stdout.flush()
                self.file.flush()

        sys.stdout = TeeOutput(log_file, original_stdout)

        try:
            # Test results
            test_results = []
            convergence_results = []

            # Test each model
            for model_name in MODELS:
                # Load the model and its configuration
                model_path = MODEL_PATHS[model_name]
                model, config = load_model_and_config(model_path)

                # Create the DataLoader
                val_loader = create_dataloader(model_name)

                # Test the model
                result = test_model(model, val_loader, MODEL_NAMES[model_name])
                test_results.append(result)

                # Analyze convergence
                convergence_result = analyze_convergence(model_name)
                convergence_results.append(convergence_result)

            # Test embedded models
            for model_name, model_path in EMBEDDED_MODEL_PATHS.items():
                # Load the model and its configuration
                model, config = load_model_and_config(model_path)

                # Create the DataLoader with embedded images
                val_loader = create_dataloader(model_name, use_embedded=True)

                # Test the model
                embedded_model_name = f"{model_name}_embedded"
                result = test_model(model, val_loader, MODEL_NAMES[embedded_model_name])
                test_results.append(result)

                # Analyze convergence
                convergence_result = analyze_convergence(embedded_model_name)
                convergence_results.append(convergence_result)

            # Plot the results
            plot_results(test_results, convergence_results)

            # Print a summary table
            print("\nSummary Table:")
            summary_df = pd.DataFrame(test_results)
            summary_df = summary_df.sort_values("val_auc", ascending=False)
            print(summary_df.to_string(index=False))

            # Print convergence summary
            print("\nConvergence Summary:")
            convergence_df = pd.DataFrame(convergence_results)
            convergence_df = convergence_df.sort_values("epochs_to_convergence")
            print(convergence_df.to_string(index=False))

            # Save the summary tables to the log file
            print(f"\nResults saved to {log_file_path}")

        finally:
            # Restore original stdout
            sys.stdout = original_stdout
            log_file.close()


if __name__ == "__main__":
    main()
