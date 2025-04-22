from src.models import CXRModelConfig
from src.models.cxr_model import SupportedModels
from src.utils import train_model, run_inference

MODELS: list[SupportedModels] = [
    "densenet121",
    # "densenet121_mm",
    # "densenet201",
    # "densenet201_mm",
    # "vit_b_16",
    # "vit_b_16_mm",
    # "vit_b_32",
    # "vit_b_32_mm",
    # "vit_l_16",
    # "vit_l_16_mm",
]


def run_trainer(model: SupportedModels):
    model_config = CXRModelConfig(model=model, freeze_backbone=True)
    loss, auc, epoch_time, total_time, epoch_count, _trained_model = train_model(
        model_config=model_config,
        lr=1e-4,
        epochs=1,
        batch_size=128,
        focal_loss=False,
        plot_path=f"results/plots/training_curves_{model}.png",
        best_model_path=f"results/models/best_model_{model}.pth",
        last_model_path=f"results/models/last_model_{model}.pth",
        train_val_data_path=f"results/train_val_data_{model}.csv",
    )
    print(
        f"Training completed for {model}. Best:\n"
        f"- Loss: {loss:.4f}\n"
        f"- AUC: {auc:.4f}\n"
        f"- Time: {epoch_time:.2f} seconds (avg, per epoch)\n"
        f"- Total Time: {total_time:.2f} seconds\n"
        f"- Epochs Run: {epoch_count}\n"
    )
    run_inference()


if __name__ == "__main__":
    print(f"One epoch of training for models: \n\t-{"\n\t-".join(MODELS)}\n")
    for model in MODELS:
        run_trainer(model=model)
    print("Training completed for all models.")
