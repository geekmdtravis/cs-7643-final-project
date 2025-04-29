"""
Experiment 3: Compare the best models from previous experiments and output their results
in a readable format. This script loads the best models and evaluates them on the test
set.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np

from src.data import create_dataloader
from src.utils import Config, evaluate_model, load_model, run_inference

# Define base output directory
OUTPUT_DIR = Path("results/experiment3")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Define the models to compare
MODELS = [
    "densenet121",
    "densenet121_mm",
    "densenet121_embedded",
    "vit_b_32",
    "vit_b_32_mm",
    "vit_b_32_embedded",
]

# Define the model paths for both regular and CB focal loss versions
MODEL_PATHS = {
    # DenseNet121 models
    "densenet121": {
        "BCE": "results/models/densenet121_best.pth",
        "focal": "results/models/densenet121_focal_best.pth",
    },
    "densenet121_mm": {
        "BCE": "results/models/densenet121_mm_best.pth",
        "focal": "results/models/densenet121_mm_focal_best.pth",
    },
    "densenet121_embedded": {
        "BCE": "results/models/densenet121_embedded_best.pth",
        "focal": "results/models/densenet121_embedded_focal_best.pth",
    },
    # ViT-B/32 models
    "vit_b_32": {
        "BCE": "results/models/vit_b_32_best.pth",
        "focal": "results/models/vit_b_32_focal_best.pth",
    },
    "vit_b_32_mm": {
        "BCE": "results/models/vit_b_32_mm_best.pth",
        "focal": "results/models/vit_b_32_mm_focal_best.pth",
    },
    "vit_b_32_embedded": {
        "BCE": "results/models/vit_b_32_embedded_best.pth",
        "focal": "results/models/vit_b_32_embedded_focal_best.pth",
    },
}

# Define the model names for display
MODEL_NAMES = {
    "densenet121": "DenseNet121",
    "densenet121_mm": "DenseNet121 (Multimodal)",
    "densenet121_embedded": "DenseNet121 (Embedded)",
    "vit_b_32": "ViT-B/32",
    "vit_b_32_mm": "ViT-B/32 (Multimodal)",
    "vit_b_32_embedded": "ViT-B/32 (Embedded)",
}

# Define the config
cfg = Config()


def create_performance_plots(df: pd.DataFrame) -> None:
    """
    Create various plots to visualize model performance comparisons.

    Args:
        df (pd.DataFrame): DataFrame containing the evaluation results.
    """
    # Set the style
    plt.style.use("default")
    
    # Set font size for better readability
    plt.rcParams.update({'font.size': 12})
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. AUC Score Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="model_name", y="auc", hue="loss_type")
    plt.title("AUC Score Comparison by Model and Loss Type", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("AUC Score", fontsize=12)
    plt.legend(title="Loss Type", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "auc_comparison.pdf")
    plt.close()

    # 2. F1 Score Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="model_name", y="f1", hue="loss_type")
    plt.title("F1 Score Comparison by Model and Loss Type", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.legend(title="Loss Type", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "f1_comparison.pdf")
    plt.close()

    # 3. Combined Performance Metrics
    metrics = ["auc", "precision", "recall", "f1"]

    # Create a long-format DataFrame for the metrics
    plot_df = df.melt(
        id_vars=["model_name", "loss_type"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )

    # Create subplots for each metric
    g = sns.catplot(
        data=plot_df,
        x="model_name",
        y="score",
        hue="loss_type",
        col="metric",
        kind="bar",
        height=4,
        aspect=1.5,
    )

    # Customize the plot
    g.fig.suptitle("Performance Metrics Comparison by Model and Loss Type", y=1.02, fontsize=14)
    
    # Rotate x-axis labels for each subplot and improve readability
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.legend(title="Loss Type", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "combined_metrics.pdf")
    plt.close()

    # 4. Model Type Comparison (DenseNet vs ViT)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model_type", y="auc", hue="loss_type")
    plt.title("AUC Score Distribution by Model Type and Loss Type", fontsize=14)
    plt.xlabel("Model Type", fontsize=12)
    plt.ylabel("AUC Score", fontsize=12)
    plt.legend(title="Loss Type", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_type_comparison.pdf")
    plt.close()

    # 5. Architecture Features Impact
    features = ["is_multimodal", "is_embedded"]
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=feature, y="auc", hue="loss_type")
        feature_name = feature.replace("is_", "").title()
        plt.title(f"AUC Score Distribution by {feature_name} and Loss Type", fontsize=14)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel("AUC Score", fontsize=12)
        plt.legend(title="Loss Type", fontsize=10)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{feature}_impact.pdf")
        plt.close()


def evaluate_model_performance(
    model_path: str, model_name: str, loss_type: str
) -> Dict:
    """
    Evaluate a model's performance on the test set.

    Args:
        model_path (str): Path to the model file.
        model_name (str): Name of the model for display.
        loss_type (str): Type of loss used ('regular' or 'focal').

    Returns:
        Dict: The evaluation results.
    """
    print(f"\nEvaluating {model_name} ({loss_type} loss)...")

    # Load the model
    model = load_model(model_path)
    model = model.to(cfg.device)
    model.eval()

    # Determine which image directory to use based on whether the model is embedded
    is_embedded = "Embedded" in model_name
    cxr_test_dir = cfg.embedded32_test_dir if is_embedded else cfg.cxr_test_dir

    # Create test dataloader
    test_loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cxr_test_dir,
    )

    # Run inference
    preds, labels = run_inference(model=model, test_loader=test_loader)

    # Evaluate the model
    results = evaluate_model(preds, labels)

    # Convert all NumPy types to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results = convert_numpy(results)

    # Add model name to results
    results["model_name"] = model_name
    results["model_type"] = model_name.split(" ")[0]  # Base model name
    results["is_multimodal"] = "Multimodal" in model_name
    results["is_embedded"] = is_embedded
    results["loss_type"] = loss_type

    return results


def save_results(results: List[Dict]) -> None:
    """
    Save the evaluation results in multiple formats.

    Args:
        results (List[Dict]): List of evaluation results.
    """
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Create a DataFrame for easier analysis
    # Extract metrics from the results dictionary
    processed_results = []
    for result in results:
        # Get the weighted average metrics from the classification report
        weighted_avg = result["report"]["weighted avg"]
        processed_result = {
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "loss_type": result["loss_type"],
            "is_multimodal": result["is_multimodal"],
            "is_embedded": result["is_embedded"],
            "auc": result["macro_auc"],  # Using macro AUC as the main AUC metric
            "precision": weighted_avg["precision"],
            "recall": weighted_avg["recall"],
            "f1": weighted_avg["f1-score"],
        }
        processed_results.append(processed_result)

    df = pd.DataFrame(processed_results)

    # Save results as CSV
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    # Create a summary table
    summary = df[
        [
            "model_name",
            "loss_type",
            "auc",
            "precision",
            "recall",
            "f1",
            "is_multimodal",
            "is_embedded",
        ]
    ].sort_values(["model_name", "loss_type"])

    # Save summary as markdown
    with open(OUTPUT_DIR / "summary.md", "w") as f:
        f.write("# Model Comparison Summary\n\n")
        f.write("## Overall Performance\n\n")
        f.write(tabulate(summary, headers="keys", tablefmt="pipe", showindex=False))
        f.write("\n\n")

        # Add per-model analysis
        f.write("## Detailed Analysis\n\n")
        for model_type in ["DenseNet121", "ViT-B/32"]:
            f.write(f"### {model_type}\n\n")
            model_results = df[df["model_type"] == model_type].sort_values(
                ["model_name", "loss_type"]
            )
            f.write(
                tabulate(
                    model_results, headers="keys", tablefmt="pipe", showindex=False
                )
            )
            f.write("\n\n")

    # Create visualization plots
    create_performance_plots(df)


def main():
    """Main function to run the experiment."""
    print("Starting Experiment 3: Model Comparison...")

    # Evaluate all models
    results = []
    for model_key in MODELS:
        for loss_type in ["BCE", "focal"]:
            model_path = MODEL_PATHS[model_key][loss_type]
            model_name = MODEL_NAMES[model_key]
            try:
                model_results = evaluate_model_performance(
                    model_path, model_name, loss_type
                )
                results.append(model_results)
            except Exception as e:
                print(f"Error evaluating {model_name} ({loss_type} loss): {str(e)}")

    # Save results
    save_results(results)
    print(f"\nResults have been saved to {OUTPUT_DIR}/")
    print(f"Plots have been saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
