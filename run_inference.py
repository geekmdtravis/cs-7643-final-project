from src.data import create_dataloader
from src.models import CXRModel
from src.utils import Config, evaluate_model, print_evaluation_results, run_inference

cfg = Config()

model = CXRModel(model="densenet121", freeze_backbone=True, hidden_dims=())

path = "results/models/best_model_densenet121.pth"
loader = create_dataloader(
    clinical_data=cfg.tabular_clinical_test,
    cxr_images_dir=cfg.cxr_test_dir,
    num_workers=32,
    batch_size=32,
    normalization_mode="imagenet",
)
preds, labels = run_inference(
    state_dict_path=path,
    model=model,
    test_loader=loader,
    device="cuda",
)


auc_scores, report = evaluate_model(preds, labels)

print_evaluation_results(
    auc_scores=auc_scores, report=report, save_path="results/evaluation_report.txt"
)
