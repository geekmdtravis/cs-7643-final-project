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

LEARNING_RATES = [1e-3, 1e-4, 1e-5, 1e-6]
BATCH_SIZES = [16, 32, 64, 128]
DROPOUTS = [0.2, 0.3, 0.4, 0.5]
MODELS: list[SupportedModels] = [
    "densenet121",
    "vit_b_32",
    "densenet121_mm",
    "vit_b_32_mm",
]
HIDDEN_DIMS = [(512, 256, 128), (512, 256, 128, 64), (512, 256, 128, 64, 32)]


def run_study(
    model: str,
    lr: float = 1e-4,
    bs: int = 32,
    dropout: float = 0.2,
    hidden_dims: tuple[int] | None = None,
) -> None:
    print(
        f"Beginning model={model} lr={lr} bs={bs} "
        f"dropout={dropout} hidden_dims={hidden_dims}..."
    )

    model_config = CXRModelConfig(
        model=model, freeze_backbone=True, dropout=dropout, hidden_dims=hidden_dims
    )
    loss, auc, epoch_time, total_time, epoch_count, trained_model = train_model(
        model_config=model_config,
        lr=lr,
        batch_size=bs,
        epochs=50,
        focal_loss_gamma=5.0,
        focal_loss_rebal_beta=0.999999,
        plot_path=(
            f"results/tuning/{model}_lr_{lr}_bs_{bs}_do_"
            f"{dropout}_hd_{hidden_dims}.png"
        ),
        best_model_path=(
            f"results/tuning/{model}_lr_{lr}_bs_{bs}_do_"
            f"{dropout}_hd_{hidden_dims}_best.pth"
        ),
        last_model_path=(
            f"results/tuning/{model}_lr_{lr}_bs_{bs}_do_"
            f"{dropout}_hd_{hidden_dims}_last.pth"
        ),
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
        save_path=(
            f"results/tuning/{model}_lr_{lr}_bs_{bs}_do_"
            f"{dropout}_hd_{hidden_dims}_eval.txt"
        ),
    )
    print("Inference completed.")


if __name__ == "__main__":
    print("Beginning hyperparameter tuning...")

    for model in MODELS:
        for lr in LEARNING_RATES:
            run_study(model=model, lr=lr)
        for bs in BATCH_SIZES:
            run_study(model=model, bs=bs)
        for dropout in DROPOUTS:
            run_study(model=model, dropout=dropout)
        for hidden_dims in HIDDEN_DIMS:
            run_study(model=model, hidden_dims=hidden_dims)
    print("Hyperparameter tuning completed.")
