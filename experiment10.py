"""
Train models with Focal Loss using hyperparameters from best models.
"""

from pathlib import Path

from src.data import create_dataloader
from src.models import CXRModelConfig
from src.models.cxr_model import SupportedModels
from src.utils import (
    Config,
    evaluate_model,
    print_evaluation_results,
    run_inference,
    train_model,
)

cfg = Config()

MODELS = [
    "densenet121",
    "densenet121_mm",
    "vit_b_32",
    "vit_b_32_mm",
    "densenet121_embedded",
    "vit_b_32_embedded",
]


def train_and_evaluate(model_name: SupportedModels):
    """Train and evaluate a model with Focal Loss."""
    print(f"\nTraining {model_name} with Focal Loss...")

    is_embedded = "_embedded" in model_name

    base_model_name = model_name
    if is_embedded:
        base_model_name = model_name.replace("_embedded", "")

    model_config = CXRModelConfig(
        model=base_model_name, freeze_backbone=True, hidden_dims=()
    )

    loss, auc, epoch_time, total_time, epoch_count, trained_model = train_model(
        model_config=model_config,
        lr=1e-4,
        epochs=200,
        batch_size=32,
        focal_loss=True,
        focal_loss_rebal_beta=0.999999,
        focal_loss_gamma=5.0,
        plot_path=f"results/plots/training_curves_{model_name}_focal.png",
        best_model_path=f"results/models/best_model_{model_name}_focal.pth",
        last_model_path=f"results/models/last_model_{model_name}_focal.pth",
        train_val_data_path=f"results/train_val_data_{model_name}_focal.csv",
        use_embedded_imgs=is_embedded,
        matrix_size=32,
    )

    print(f"Training completed for {model_name} with Focal Loss:")
    print(f"- Best Loss: {loss:.4f}")
    print(f"- Best AUC: {auc:.4f}")
    print(f"- Avg Time per Epoch: {epoch_time:.2f} seconds")
    print(f"- Total Training Time: {total_time:.2f} seconds")
    print(f"- Epochs Run: {epoch_count}")

    if is_embedded:
        test_img_dir = cfg.embedded32_test_dir
    else:
        test_img_dir = cfg.cxr_test_dir

    test_loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=test_img_dir,
        batch_size=32,
        num_workers=4,
        normalization_mode="imagenet",
    )

    print("\nRunning inference on test set...")
    preds, labels = run_inference(model=trained_model, test_loader=test_loader)

    results = evaluate_model(preds=preds, labels=labels)

    results_path = f"results/experiment10/evaluation_{model_name}_focal.txt"
    print_evaluation_results(results=results, save_path=results_path)

    return results


def main():
    Path("results/experiment10").mkdir(exist_ok=True)
    Path("results/models").mkdir(exist_ok=True)
    Path("results/experiment10/plots").mkdir(exist_ok=True)

    for model_name in MODELS:
        results = train_and_evaluate(model_name)

        print(f"\nResults for {model_name}:")
        print("Metric          | Value")
        print("-" * 30)
        print(f"Weighted AUC   | {results['weighted_auc']:.4f}")
        print(f"Macro AUC      | {results['macro_auc']:.4f}")
        print(f"Micro AUC      | {results['micro_auc']:.4f}")
        print(f"Hamming Loss   | {results['hamming_loss']:.4f}")
        print(f"Jaccard Score  | {results['jaccard_similarity']:.4f}")
        print(f"Avg Precision  | {results['avg_precision']:.4f}")


if __name__ == "__main__":
    main()
