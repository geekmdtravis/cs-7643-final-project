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


VANILLA_MODELS: list[SupportedModels] = [
    "densenet121",
    "densenet121_mm",
    "vit_b_32",
    "vit_b_32_mm",
]
EMBEDDED_MODELS: list[SupportedModels] = [
    "densenet121",
    "vit_b_32",
]

HIDDEN_DIMS = [
    (512,),
    (512, 256, 128),
    (512, 256, 128, 64, 32),
]

BASE_DIR = "results/experiment4"


def run_vanilla_cxr_study(
    model: str,
    lr: float = 1e-4,
    bs: int = 32,
    dropout: float = 0.2,
    hidden_dims: tuple[int] | None = None,
) -> None:
    print(f"Beginning model={model} hidden_dims={hidden_dims}...")
    file_prefix = f"{BASE_DIR}/{model}_hd_{hidden_dims}"
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


def run_embedded_study(
    model: str,
    lr: float = 1e-4,
    bs: int = 32,
    dropout: float = 0.2,
    hidden_dims: tuple[int] | None = None,
    matrix_size: Literal[16, 32] = 32,
) -> None:
    print(f"Beginning model={model} hidden_dims={hidden_dims}...")
    file_prefix = f"{BASE_DIR}/embd_{model}_hd_{hidden_dims}"
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

    model_names = ",".join(VANILLA_MODELS)
    print(f"Training {model_names} on the vanilla CXR dataset...")
    for model in VANILLA_MODELS:
        for hidden_dims in HIDDEN_DIMS:
            run_vanilla_cxr_study(model=model, hidden_dims=hidden_dims)

    model_names = ",".join(EMBEDDED_MODELS)
    print(f"Training {model_names} on the embedded CXR dataset...")
    for model in EMBEDDED_MODELS:
        for hidden_dims in HIDDEN_DIMS:
            run_embedded_study(model=model, hidden_dims=hidden_dims)

    print("Hyperparameter tuning completed.")
