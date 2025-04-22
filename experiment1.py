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

MODELS: list[SupportedModels] = [
    "densenet121",
    "densenet121_mm",
    "densenet201",
    "densenet201_mm",
    "vit_b_16",
    "vit_b_16_mm",
    "vit_b_32",
    "vit_b_32_mm",
    "vit_l_16",
    "vit_l_16_mm",
]

LR = 1e-4
EPOCHS = 1
EXP_BASE_PATH = "results/experiment1"


def run_trainer_on_model(model: SupportedModels) -> None:
    print(f"Training model: {model}...")

    model_config = CXRModelConfig(model=model, freeze_backbone=True)
    loss, auc, epoch_time, total_time, epoch_count, trained_model = train_model(
        model_config=model_config,
        lr=LR,
        epochs=EPOCHS,
        plot_path=f"{EXP_BASE_PATH}/loss_plot_{model}.png",
        best_model_path=f"{EXP_BASE_PATH}/model_checkpoint_{model}.pth",
        last_model_path=f"{EXP_BASE_PATH}/last_model_{model}.pth",
        train_val_data_path=f"{EXP_BASE_PATH}/train_val_data_{model}.pth",
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
    )

    print(f"Running inference on {model}...")
    preds, labels = run_inference(model=trained_model, test_loader=loader)

    auc_scores, report = evaluate_model(preds, labels)

    print_evaluation_results(
        auc_scores=auc_scores,
        report=report,
        save_path=f"{EXP_BASE_PATH}/evaluation_report_{model}.txt",
    )
    print("Inference completed.")


if __name__ == "__main__":
    for model in MODELS:
        run_trainer_on_model(model)
