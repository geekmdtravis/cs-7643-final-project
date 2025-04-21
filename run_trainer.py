from src.models import CXRModel
from src.models.cxr_model import SupportedModels
from src.utils import train_model

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


def train_selected_model(model: SupportedModels):
    cxr_modal = CXRModel(model=model, freeze_backbone=True, hidden_dims=())
    loss, auc, epoch_time, total_time, epoch_count = train_model(
        model=cxr_modal,
        lr=1e-3,
        epochs=25,
        batch_size=128,
        focal_loss=True,
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


if __name__ == "__main__":
    print(f"One epoch of training for models: \n\t-{"\n\t-".join(MODELS)}\n")
    for model in MODELS:
        train_selected_model(model=model)
    print("Training completed for all models.")
