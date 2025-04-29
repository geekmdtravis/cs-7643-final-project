"""
Experiment 3: Compare the best models from previous experiments and output their results
in a readable format. This script loads the best models and evaluates them on the test
set.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.data import create_dataloader
from src.utils import (
    Config,
    evaluate_model,
    load_model,
    print_evaluation_results,
    run_inference,
)

OUTPUT_DIR = Path("results/experiment3")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"

MODELS = [
    "densenet121",
    "densenet121_mm",
    "densenet121_embedded",
    "vit_b_32",
    "vit_b_32_mm",
    "vit_b_32_embedded",
]

MODELS_DIR = "/tmp/cs7643_final_share/best_models"

MODEL_PATHS = {
    "densenet121": {
        "BCE": f"{MODELS_DIR}/densenet121_best.pth",
        "focal": f"{MODELS_DIR}/densenet121_focal_best.pth",
    },
    "densenet121_mm": {
        "BCE": f"{MODELS_DIR}/densenet121_mm_best.pth",
        "focal": f"{MODELS_DIR}/densenet121_mm_focal_best.pth",
    },
    "densenet121_embedded": {
        "BCE": f"{MODELS_DIR}/densenet121_embedded_best.pth",
        "focal": f"{MODELS_DIR}/densenet121_embedded_focal_best.pth",
    },
    "vit_b_32": {
        "BCE": f"{MODELS_DIR}/vit_b_32_best.pth",
        "focal": f"{MODELS_DIR}/vit_b_32_focal_best.pth",
    },
    "vit_b_32_mm": {
        "BCE": f"{MODELS_DIR}/vit_b_32_mm_best.pth",
        "focal": f"{MODELS_DIR}/vit_b_32_mm_focal_best.pth",
    },
    "vit_b_32_embedded": {
        "BCE": f"{MODELS_DIR}/vit_b_32_embedded_best.pth",
        "focal": f"{MODELS_DIR}/vit_b_32_embedded_focal_best.pth",
    },
}

MODEL_NAMES = {
    "densenet121": "DenseNet121",
    "densenet121_mm": "DenseNet121 (Multimodal)",
    "densenet121_embedded": "DenseNet121 (Embedded)",
    "vit_b_32": "ViT-B/32",
    "vit_b_32_mm": "ViT-B/32 (Multimodal)",
    "vit_b_32_embedded": "ViT-B/32 (Embedded)",
}

cfg = Config()


def create_performance_plots(df: pd.DataFrame) -> None:
    """
    Create various plots to visualize model performance comparisons.

    Args:
        df (pd.DataFrame): DataFrame containing the evaluation results.
    """
    plt.style.use("default")

    plt.rcParams.update({"font.size": 12})

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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

    metrics = ["auc", "precision", "recall", "f1"]

    plot_df = df.melt(
        id_vars=["model_name", "loss_type"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )

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

    g.fig.suptitle(
        "Performance Metrics Comparison by Model and Loss Type", y=1.02, fontsize=14
    )

    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.legend(title="Loss Type", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "combined_metrics.pdf")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model_type", y="auc", hue="loss_type")
    plt.title("AUC Score Distribution by Model Type and Loss Type", fontsize=14)
    plt.xlabel("Model Type", fontsize=12)
    plt.ylabel("AUC Score", fontsize=12)
    plt.legend(title="Loss Type", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_type_comparison.pdf")
    plt.close()

    features = ["is_multimodal", "is_embedded"]
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=feature, y="auc", hue="loss_type")
        feature_name = feature.replace("is_", "").title()
        plt.title(
            f"AUC Score Distribution by {feature_name} and Loss Type", fontsize=14
        )
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

    model = load_model(model_path)
    model = model.to(cfg.device)
    model.eval()

    is_embedded = "Embedded" in model_name
    cxr_test_dir = cfg.embedded32_test_dir if is_embedded else cfg.cxr_test_dir

    test_loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cxr_test_dir,
    )

    preds, labels = run_inference(model=model, test_loader=test_loader)

    results = evaluate_model(preds, labels)

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

    results["model_name"] = model_name
    results["model_type"] = model_name.split(" ")[0]
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    processed_results = []
    for result in results:
        weighted_avg = result["report"]["weighted avg"]
        processed_result = {
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "loss_type": result["loss_type"],
            "is_multimodal": result["is_multimodal"],
            "is_embedded": result["is_embedded"],
            "auc": result["macro_auc"],
            "precision": weighted_avg["precision"],
            "recall": weighted_avg["recall"],
            "f1": weighted_avg["f1-score"],
        }
        processed_results.append(processed_result)

    df = pd.DataFrame(processed_results)

    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

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

    with open(OUTPUT_DIR / "summary.md", "w") as f:
        f.write("# Model Comparison Summary\n\n")
        f.write("## Overall Performance\n\n")
        f.write(tabulate(summary, headers="keys", tablefmt="pipe", showindex=False))
        f.write("\n\n")

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

    create_performance_plots(df)


def main():
    """Main function to run the experiment."""
    print("Starting Experiment 3: Model Comparison...")

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
                print_evaluation_results(
                    model_results, f"results/experiment3/{model_key}results.txt"
                )
            except Exception as e:
                print(f"Error evaluating {model_name} ({loss_type} loss): {str(e)}")

    save_results(results)
    print(f"\nResults have been saved to {OUTPUT_DIR}/")
    print(f"Plots have been saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
