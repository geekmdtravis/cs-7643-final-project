from typing import Literal

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

EMBEDDED_MODELS: list[SupportedModels] = [
    "densenet121",
    "vit_b_32",
]

MATRIX_SIZES = [32, 16]

BASE_DIR = "results/experiment6"


def run_study(
    model: str,
    lr: float = 1e-4,
    bs: int = 32,
    dropout: float = 0.2,
    hidden_dims: tuple[int] | None = None,
    matrix_size: Literal[16, 32] = 32,
) -> None:
    print(f"Beginning model={model} ms={matrix_size}...")
    file_prefix = f"{BASE_DIR}/embd_{model}_ms={matrix_size}"
    model_config = CXRModelConfig(
        model=model, freeze_backbone=True, dropout=dropout, hidden_dims=hidden_dims
    )
    loss, auc, epoch_time, total_time, epoch_count, trained_model = train_model(
        model_config=model_config,
        lr=lr,
        batch_size=bs,
        epochs=200,
        focal_loss_gamma=5.0,
        focal_loss_rebal_beta=0.999999,
        plot_path=(f"{file_prefix}.png"),
        best_model_path=(f"{file_prefix}_best.pth"),
        last_model_path=(f"{file_prefix}_last.pth"),
        train_val_data_path=(f"{file_prefix}_tvdata.csv"),
        use_embedded_imgs=True,
        matrix_size=matrix_size,
    )

    print(
        f"Training completed for {model}. Best:\n"
        f"- Loss: {loss:.4f}\n"
        f"- AUC: {auc:.4f}\n"
        f"- Time: {epoch_time:.2f} seconds (avg, per epoch)\n"
        f"- Total Time: {total_time:.2f} seconds\n"
        f"- Epochs Run: {epoch_count}\n"
    )

    loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.cxr_test_dir,
        batch_size=bs,
        num_workers=32,
    )

    preds, labels = run_inference(model=trained_model, test_loader=loader)

    results = evaluate_model(preds, labels)

    print_evaluation_results(
        results=results,
        save_path=(f"{file_prefix}_eval.txt"),
    )
    print("Inference completed.")


if __name__ == "__main__":
    print("Beginning hyperparameter tuning...")

    model_names = ",".join(EMBEDDED_MODELS)
    print(f"Training {model_names} on the embedded CXR dataset...")
    for model in EMBEDDED_MODELS:
        for ms in MATRIX_SIZES:
            run_study(model=model, matrix_size=ms)

    print("Hyperparameter tuning completed.")
