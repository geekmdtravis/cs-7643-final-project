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

if __name__ == "__main__":
    print("Beginning training and inference...")

    model: SupportedModels = "densenet121"
    model_config = CXRModelConfig(model=model, freeze_backbone=True)
    loss, auc, epoch_time, total_time, epoch_count, trained_model = train_model(
        model_config=model_config,
        lr=1e-4,
        epochs=1,
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

    preds, labels = run_inference(model=trained_model, test_loader=loader)

    results = evaluate_model(preds, labels)

    print_evaluation_results(results=results, save_path="results/evaluation_report.txt")
    print("Inference completed.")
